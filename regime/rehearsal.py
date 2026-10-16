"""Rehearsal isolation for the weekly-SF regime Lambdas (alpha-engine-config-I12351).

Twin of crucible-research ``data/rehearsal.py`` (same contract, same
namespace); the two are kept byte-identical below this docstring until the
module moves into nousergon_lib.

A rehearsal of ``ne-weekly-freshness-pipeline`` ran RegimeSubstrate with
``action: produce`` and wrote ``regime/latest.json`` (run_id ``2610160202``,
``rehearsal-2026-10-15-1``), the pointer the macro agent and the trading-day
brief read. Nothing in the handler could tell a rehearsal from Saturday's
production run.

**The marker.** The SF execution name, which both regime Task Payloads already
carry as ``execution_arn: $$.Execution.Id`` (its last ``:`` segment). Rehearsal
executions are named ``rehearsal-<date>-<n>`` by the rehearsal door, the prefix
crucible-backtester#790 keys on. ``RUN_TOKEN`` / ``AE_PIPELINE_ROLE=rehearsal``
cover a hand run.

**What isolation does.** Under the marker every S3 write any boto3 client in the
process makes is re-keyed into ``rehearsal/<execution name>/<original key>``
in the same bucket: no ``regime/latest.json``, no ``regime/<run_id>.json``, no
retrospective-eval pointer at a live key. Reads of keys the rehearsal wrote
(in this process or an earlier stage of the same execution) are served from
the namespace; every other read goes live. ``ListObjects*`` is not overlaid;
an S3 operation that is neither a known read nor a keyed write raises
:class:`RehearsalLiveWriteRefused`; only clients created inside
:func:`isolate` (or attached) are covered; non-S3 side effects are not
re-keyed. A production execution has no active context and is unchanged.

**Stale data.** :func:`tolerate_stale` degrades a freshness refusal on a
rehearsal (recorded under ``rehearsal.degradations``) and returns False on
every other run, where the caller raises as before.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

#: Execution-name prefix of weekly-SF rehearsals (crucible-backtester#790).
REHEARSAL_PREFIX = "rehearsal-"
#: Root of the rehearsal namespace inside each bucket.
NAMESPACE_ROOT = "rehearsal"
#: Env var ``krepis.ssm_log_capture`` sets to the SF execution name.
RUN_TOKEN_ENV = "RUN_TOKEN"  # noqa: S105 — an env var NAME, not a secret
#: Explicit override for a hand-launched rehearsal.
PIPELINE_ROLE_ENV = "AE_PIPELINE_ROLE"
#: Event fields that may carry the execution name or ARN.
_NAME_FIELDS = ("execution_name",)
_ARN_FIELDS = ("execution_arn", "execution_id")

#: Keyed S3 operations whose ``Key`` is re-keyed into the namespace.
_KEYED_WRITE_OPS = frozenset(
    {
        "PutObject",
        "CopyObject",
        "DeleteObject",
        "CreateMultipartUpload",
        "UploadPart",
        "UploadPartCopy",
        "CompleteMultipartUpload",
        "AbortMultipartUpload",
        "ListParts",
        "PutObjectTagging",
        "DeleteObjectTagging",
        "PutObjectAcl",
        "PutObjectRetention",
        "PutObjectLegalHold",
        "RestoreObject",
    }
)
#: Keyed S3 reads, served from the namespace when the rehearsal wrote the key.
_KEYED_READ_OPS = frozenset(
    {
        "GetObject",
        "HeadObject",
        "GetObjectTagging",
        "GetObjectAttributes",
        "GetObjectAcl",
        "SelectObjectContent",
    }
)
#: S3 reads that carry no object key; passed through unchanged.
_UNKEYED_READ_OPS = frozenset(
    {
        "ListObjects",
        "ListObjectsV2",
        "ListObjectVersions",
        "HeadBucket",
        "GetBucketLocation",
        "ListBuckets",
        "ListMultipartUploads",
    }
)
#: Copy operations whose SOURCE is a read.
_COPY_OPS = frozenset({"CopyObject", "UploadPartCopy"})


class RehearsalLiveWriteRefused(RuntimeError):
    """A rehearsal reached an S3 write this module cannot re-key. Raised, never
    swallowed: the write must not happen at a live key."""


def _name_from_arn(arn: str) -> str:
    return arn.rsplit(":", 1)[-1]


def rehearsal_name(
    event: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """The rehearsal's execution name, or ``None`` for any other run.

    The event wins over the environment: a Lambda's event names the execution
    that invoked it, while ``RUN_TOKEN`` is whatever the container was given.
    A non-rehearsal execution name in the event is final, so a stray env var
    can never turn a production invocation into a rehearsal (or the reverse).
    """
    env = os.environ if env is None else env
    event = event or {}
    named: str | None = None
    for field in _NAME_FIELDS:
        value = event.get(field)
        if isinstance(value, str) and value.strip():
            named = value.strip()
            break
    if named is None:
        for field in _ARN_FIELDS:
            value = event.get(field)
            if isinstance(value, str) and value.strip():
                named = _name_from_arn(value.strip())
                break
    if named is not None:
        return named if named.startswith(REHEARSAL_PREFIX) else None

    token = (env.get(RUN_TOKEN_ENV) or "").strip()
    if token.startswith(REHEARSAL_PREFIX):
        return token
    if (env.get(PIPELINE_ROLE_ENV) or "").strip().lower() == "rehearsal":
        return f"{REHEARSAL_PREFIX}manual-{token or 'run'}"
    return None


def namespace_prefix(name: str) -> str:
    """``rehearsal/<execution name>/`` — where every write of that run lands."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
    return f"{NAMESPACE_ROOT}/{safe}/"


class RehearsalContext:
    """One rehearsal invocation: its namespace, overlay and degradations."""

    def __init__(self, name: str, *, stage: str) -> None:
        self.name = name
        self.stage = stage
        self.prefix = namespace_prefix(name)
        self._lock = threading.Lock()
        #: (bucket, live key) pairs whose current value lives in the namespace.
        self._overlay: set[tuple[str, str]] = set()
        self.redirected: list[str] = []
        self.degradations: list[dict[str, str]] = []
        self._uid = f"rehearsal-isolation-{id(self)}"

    # ── key mapping ────────────────────────────────────────────────────────
    def ns_key(self, key: str) -> str:
        if key.startswith(self.prefix):
            return key
        return f"{self.prefix}{key}"

    def is_overlaid(self, bucket: str, key: str) -> bool:
        with self._lock:
            return (bucket, key) in self._overlay

    def prime(self, s3_client: Any, bucket: str) -> int:
        """Load the keys earlier stages of this execution wrote into
        ``bucket``'s namespace, so their reads are served from it."""
        n = 0
        token = None
        for _ in range(1000):
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": self.prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = s3_client.list_objects_v2(**kwargs)
            with self._lock:
                for obj in resp.get("Contents") or []:
                    k = obj.get("Key", "")
                    if k.startswith(self.prefix):
                        self._overlay.add((bucket, k[len(self.prefix) :]))
                        n += 1
            if resp.get("IsTruncated") is not True:
                break
            token = resp.get("NextContinuationToken")
            if not isinstance(token, str) or not token:
                break
        return n

    # ── the botocore hook ──────────────────────────────────────────────────
    def _hook(self, params: dict, model: Any, **_: Any) -> None:
        if _ACTIVE is not self:
            # A client created during a rehearsal invocation keeps a copy of
            # this hook. Outside that invocation it must do nothing, so a
            # cached client can never re-key a later production call.
            return
        op = getattr(model, "name", "")
        bucket = params.get("Bucket", "")
        if op in _KEYED_WRITE_OPS:
            if op in _COPY_OPS:
                self._rewrite_copy_source(params)
            key = params.get("Key")
            if not isinstance(key, str) or not key:
                raise RehearsalLiveWriteRefused(f"rehearsal {self.name}: S3 {op} without a Key cannot be re-keyed")
            if key.startswith(self.prefix):
                return
            params["Key"] = self.ns_key(key)
            with self._lock:
                if op == "DeleteObject":
                    self._overlay.discard((bucket, key))
                elif op in ("PutObject", "CopyObject", "CompleteMultipartUpload"):
                    self._overlay.add((bucket, key))
                    self.redirected.append(f"s3://{bucket}/{key}")
            return
        if op == "DeleteObjects":
            for obj in (params.get("Delete") or {}).get("Objects") or []:
                key = obj.get("Key", "")
                if key and not key.startswith(self.prefix):
                    obj["Key"] = self.ns_key(key)
                    with self._lock:
                        self._overlay.discard((bucket, key))
            return
        if op in _KEYED_READ_OPS:
            key = params.get("Key")
            if isinstance(key, str) and self.is_overlaid(bucket, key):
                params["Key"] = self.ns_key(key)
            return
        if op in _UNKEYED_READ_OPS:
            return
        raise RehearsalLiveWriteRefused(
            f"rehearsal {self.name}: S3 {op or '<unknown>'} is not a known read "
            "or keyed write, so it cannot be isolated; refusing it "
            "(alpha-engine-config-I12351)"
        )

    def _rewrite_copy_source(self, params: dict) -> None:
        src = params.get("CopySource")
        if isinstance(src, dict):
            b, k = src.get("Bucket", ""), src.get("Key", "")
            if k and self.is_overlaid(b, k):
                params["CopySource"] = {**src, "Key": self.ns_key(k)}
        elif isinstance(src, str):
            b, _, k = src.lstrip("/").partition("/")
            if k and self.is_overlaid(b, k):
                params["CopySource"] = f"{b}/{self.ns_key(k)}"

    def attach(self, client: Any) -> Any:
        """Register the hook on one already-created client. Idempotent: the
        unique id makes a second registration (or the inherited session
        registration) a no-op."""
        client.meta.events.register(
            "before-parameter-build.s3",
            self._hook,
            unique_id=self._uid,
        )
        return client

    # ── degradations ───────────────────────────────────────────────────────
    def degrade(self, message: str, *, where: str) -> None:
        with self._lock:
            self.degradations.append({"where": where, "refusal": message})
        logger.warning(
            "[rehearsal] %s: %s — staleness refusal DEGRADED and continuing "
            "(rehearsal only; production raises here; alpha-engine-config-I12351): %s",
            self.name,
            where,
            message,
        )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "execution_name": self.name,
                "stage": self.stage,
                "namespace": self.prefix,
                "writes_redirected": len(self.redirected),
                "redirected_sample": self.redirected[:25],
                "degraded": bool(self.degradations),
                "degradations": list(self.degradations),
            }


_ACTIVE: RehearsalContext | None = None


def active() -> RehearsalContext | None:
    """The rehearsal context of the running invocation, or None."""
    return _ACTIVE


def tolerate_stale(message: str, *, where: str) -> bool:
    """True (and recorded) when a rehearsal should continue past a
    freshness/staleness refusal; False on every other run, where the caller
    raises exactly as it did before."""
    ctx = _ACTIVE
    if ctx is None:
        return False
    ctx.degrade(message, where=where)
    return True


@contextmanager
def isolate(
    event: Mapping[str, Any] | None,
    *,
    stage: str,
    buckets: tuple[str, ...] = ("alpha-engine-research",),
    session: Any = None,
    env: Mapping[str, str] | None = None,
) -> Iterator[RehearsalContext | None]:
    """Activate isolation for the duration of one invocation.

    Yields None (and changes nothing) unless :func:`rehearsal_name` names a
    rehearsal. Otherwise registers the re-keying hook on the boto3 default
    session, so every S3 client created inside the block inherits it, primes
    the read overlay from ``buckets``, and unregisters on exit (a warm Lambda
    container must not carry one rehearsal's hook into the next invocation).
    """
    global _ACTIVE
    name = rehearsal_name(event, env)
    if name is None:
        yield None
        return

    import boto3

    sess = session or boto3._get_default_session()  # noqa: SLF001 — the session boto3.client() uses
    ctx = RehearsalContext(name, stage=stage)
    lister = sess.client("s3")
    for bucket in buckets:
        try:
            n = ctx.prime(lister, bucket)
        except Exception as exc:  # noqa: BLE001 — a missing overlay only means reads go live
            logger.warning(
                "[rehearsal] %s: could not list %s in %s (%s) — reads of keys "
                "earlier stages wrote will go to the live key",
                name,
                ctx.prefix,
                bucket,
                exc,
            )
            n = 0
        logger.info("[rehearsal] %s: %d prior namespace object(s) in %s", name, n, bucket)
    core = sess._session  # noqa: SLF001 — botocore session behind boto3's
    core.register("before-parameter-build.s3", ctx._hook, unique_id=ctx._uid)
    _ACTIVE = ctx
    logger.warning(
        "[rehearsal] %s: %s is a REHEARSAL — every S3 write is re-keyed under "
        "%s; no live key is written (alpha-engine-config-I12351)",
        name,
        stage,
        ctx.prefix,
    )
    try:
        yield ctx
    finally:
        _ACTIVE = None
        core.unregister("before-parameter-build.s3", unique_id=ctx._uid)
