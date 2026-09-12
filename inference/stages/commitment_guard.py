"""Stage: commitment_guard — does the bundle being SERVED today still match the
pre-registered out-of-sample commitment window?  (alpha-engine-config-I10259)

The paired detector
-------------------
``training/commitment.py`` + ``training/model_zoo.py``'s COMMITMENT gate stop
the weekly rotation from promoting over the freeze. That is the ACTUATOR, and it
covers exactly one path. This stage is the DETECTOR, and it covers every path:
a hand promotion (``python -m model.registry --promote``), an
``auto_promote_winner`` flip, a training box that never pulled the ruling, a
restore from an archived bundle — anything that changes what inference actually
SERVES. Registered on the daily pipeline, it asks the only question that decides
whether the window is still valid: **is the version being served today one of
the versions the window pinned?**

Why the actuator alone is not enough: on 2026-09-12 the window was voided by the
promotion loop it had suspended, and the only reason anybody knew is that a
human read a promotions artifact. Repairing the promoter without a detector
would leave the window's validity unobserved again — and an unobserved freeze
reports as a healthy one (`principles.md` §2.7: *no data* is never green).

What it reads, and why from S3
------------------------------
The ruling lives in the private ``alpha-engine-config`` repo. The inference
Lambda has no checkout of it, so the ruling is PUBLISHED to
``s3://alpha-engine-research/_commitment/ALPHA_EXPERIMENT_COMMITMENT.yaml`` on
merge (alpha-engine-config side of I10259), and that copy is what this stage
reads — key from ``cfg.ALPHA_EXPERIMENT_COMMITMENT_S3_KEY``, env-overridable.
The served version comes from ``cfg.META_MANIFEST_KEY``
(``predictor/weights/meta/manifest.json``) → ``served_version``, the same
pointer ``training/model_zoo.py`` treats as the identity of the incumbent.

What it writes
--------------
``predictor/commitment/window_state.json`` — ``state: VOID`` with the served
version, the pinned versions, the freeze date and the detecting execution. This
is the durable surface: a voided window must be visible to a later reader and to
the console without re-deriving it from two S3 prefixes. IDEMPOTENT — an
existing VOID record for the SAME served version is left alone, so a week of
daily runs under one void writes once and the ``detected_at`` keeps meaning *when
the void started*, not *when it was last looked at*.

Failure posture — reports, never halts
--------------------------------------
Non-critical in ``inference/pipeline.py::STAGES``. A commitment-window
measurement must not stand a trading book down: the window is an EVALUATION
commitment, and voiding it changes what the eventual verdict may claim, not
whether today's predictions are valid. So this stage never raises into the
pipeline and never mutates ``ctx``.

Non-critical is not silent. Like ``inference/stages/research_free.py``, the stage
handles its own failures rather than relying on ``run_pipeline``'s log-and-
continue:

- (a) failure mode swallowed: any exception reading either S3 object, parsing the
  YAML, or writing the verdict.
- (b) why the primary deliverable survives: this stage reads the model manifest
  and writes one observability artifact; the live predictions path neither
  depends on it nor is touched by it.
- (c) recording surface: ``ops_alerts.publish_ops_alert`` plus the ERROR log
  line, and — for the mismatch itself — the durable
  ``predictor/commitment/window_state.json``.

An ABSENT commitment file is reported at ``warning``, never treated as "no
window": the whole bug class here is a ruling nothing read, so unreadable is
UNREPORTED and says so out loud.

Alert registration
------------------
Source is ``alpha-engine-predictor-inference`` — the EXISTING
``predictor_inference`` row in nousergon-data
``infrastructure/overseer/playbooks.yaml::alert_classes``, one
``severities: [dynamic]`` row covering every failure shape inside this Lambda
(operator ruling 2026-08-21, alpha-engine-config-I7740). Deliberately NOT a new
``predictor-commitment-guard`` literal: that would need a companion
nousergon-data registry PR before ``.github/workflows/alert-class-pr-guard.yml``
would let this one go green, and this condition is one of the shapes the
existing row already covers. Severities differ by condition (``critical`` for a
void, ``warning`` for an unreadable ruling), which that dynamic row permits.

Dedup keys are CONDITION-keyed, not run-keyed (ALERT003): keyed on the served
version for a void (the condition is "THIS bundle is serving against the
freeze") and on the bare condition for an absent file. With krepis' 60-minute
default dedup window and one scheduled invocation per trading morning, that
lands at one page per day per distinct condition, and re-pages if the served
version changes again.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import config as cfg
from inference.pipeline import PipelineContext

log = logging.getLogger(__name__)

# See the module docstring — the registered `predictor_inference` class. Do NOT
# replace with a new literal without landing the companion nousergon-data row.
_ALERT_SOURCE = "alpha-engine-predictor-inference"

_SCHEMA_VERSION = 1


def _alert(message: str, *, severity: str, dedup_key: str) -> None:
    """Publish to the ops surface. Never raises — an alerting failure must not
    take down a stage that is only reporting."""
    try:
        from ops_alerts import publish_ops_alert

        publish_ops_alert(
            message, severity=severity, source=_ALERT_SOURCE, dedup_key=dedup_key,
        )
    except Exception:  # noqa: BLE001 — see (c) in the module docstring
        log.error(
            "commitment_guard: ops alert publish FAILED — the finding below is "
            "recorded in this log line (and, for a void, in "
            "predictor/commitment/window_state.json) only.", exc_info=True,
        )


def _commitment_s3_key() -> str:
    return getattr(
        cfg, "ALPHA_EXPERIMENT_COMMITMENT_S3_KEY",
        "_commitment/ALPHA_EXPERIMENT_COMMITMENT.yaml",
    )


def _window_state_key() -> str:
    return getattr(
        cfg, "COMMITMENT_WINDOW_STATE_KEY", "predictor/commitment/window_state.json",
    )


def _read_yaml(s3, bucket: str, key: str) -> dict | None:
    """Parsed YAML at ``key``, or None when the object does not exist."""
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:  # noqa: BLE001 — NoSuchKey and friends
        if _is_missing(exc):
            return None
        raise
    import yaml

    doc = yaml.safe_load(body)
    if not isinstance(doc, dict):
        raise RuntimeError(
            f"s3://{bucket}/{key} parsed to {type(doc).__name__}, expected a mapping"
        )
    return doc


def _read_json(s3, bucket: str, key: str) -> dict | None:
    """Parsed JSON at ``key``, or None when the object does not exist."""
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:  # noqa: BLE001
        if _is_missing(exc):
            return None
        raise
    doc = json.loads(body)
    return doc if isinstance(doc, dict) else None


def _is_missing(exc: Exception) -> bool:
    """True when the exception is a not-found rather than a real read failure.

    Covers boto3's ClientError codes and the KeyError raised by the repo's own
    ``_FakeS3`` test stub. A real failure (credentials, throttling, a 5xx) must
    NOT be mistaken for an absent object: absence is a reportable state with its
    own alert, and a read failure is a different one.
    """
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code")
        if str(code) in ("NoSuchKey", "404", "NotFound", "NoSuchBucket"):
            return True
        return False
    return isinstance(exc, (KeyError, FileNotFoundError))


def run(ctx: PipelineContext) -> None:
    if ctx.dry_run or ctx.local:
        # Mirrors inference.stages.research_free's guard: the deploy canary
        # invokes this pipeline with dry_run=True, off the Step Function and off
        # any calendar, under a stated "no S3 writes" contract. A dry run serves
        # nothing, so there is nothing here for the reading to be correct about.
        log.info("commitment_guard: skipped (dry_run/local)")
        return
    if getattr(ctx, "weights_prefix_override", None):
        # A cloned shadow context deliberately scores with a CHALLENGER's
        # weights. That is observe-only, trades nothing, and is not what the
        # window pins — grading it against the freeze would page on every
        # registered challenger, every day.
        log.info("commitment_guard: skipped on a shadow (challenger-weights) context.")
        return

    try:
        _check(ctx)
    except Exception as exc:  # noqa: BLE001 — see (a)/(b)/(c) in the docstring
        log.error(
            "commitment_guard: check FAILED for %s: %s — the commitment window's "
            "validity is UNMEASURED for this session; live predictions are "
            "unaffected.", ctx.date_str, exc, exc_info=True,
        )
        _alert(
            f"Predictor commitment-window guard FAILED for {ctx.date_str}\n"
            f"{type(exc).__name__}: {exc}\n"
            f"The pre-registered out-of-sample window (alpha-engine-config-I10259) is "
            f"UNMEASURED for this session — a promotion over the frozen version would "
            f"not have been detected today. Live predictor/predictions/{ctx.date_str}.json "
            f"is unaffected. Re-run is not needed; the next trading morning re-checks.",
            severity="warning",
            dedup_key="predictor_commitment_guard_failed",
        )


def _check(ctx: PipelineContext) -> None:
    import boto3

    from training import commitment as commit

    s3 = boto3.client("s3")
    bucket = ctx.bucket or cfg.S3_BUCKET
    key = _commitment_s3_key()

    doc = _read_yaml(s3, bucket, key)
    if doc is None:
        # NOT "no window". The bug class being closed here is a ruling nothing
        # read; an absent published copy means this detector is blind, and a
        # blind detector must say so rather than report a clean session.
        log.error(
            "commitment_guard: no commitment file at s3://%s/%s — the "
            "out-of-sample window is UNREPORTED.", bucket, key,
        )
        _alert(
            f"Predictor commitment window UNREPORTED — no commitment file at "
            f"s3://{bucket}/{key}\n"
            f"This detector grades the SERVED bundle against the pre-registered "
            f"out-of-sample freeze (alpha-engine-config-I10259). With the published "
            f"ruling absent it can grade nothing, so a promotion over the frozen "
            f"version would go undetected. Publish it by merging any change to "
            f"private-docs/ALPHA_EXPERIMENT_COMMITMENT.yaml in alpha-engine-config "
            f"(its merge workflow writes this key), or clear this by removing the "
            f"freeze deliberately.",
            severity="warning",
            dedup_key="predictor_commitment_file_absent",
        )
        return

    if not commit.window_active(doc, ctx.date_str):
        log.info(
            "commitment_guard: no active out-of-sample window on %s "
            "(freeze_date=%s, rotation=%s, state=%s) — nothing to grade.",
            ctx.date_str, commit.freeze_date(doc), doc.get("rotation"),
            commit.window_state(doc),
        )
        return

    manifest = _read_json(s3, bucket, cfg.META_MANIFEST_KEY)
    served_version = (manifest or {}).get("served_version")
    frozen = commit.frozen_version_ids(doc)

    if not served_version:
        # The live manifest is the ONLY statement of what is served; without it
        # the window cannot be graded and that is a finding, not a pass — the
        # same posture model_zoo._resolve_serving_champion takes (it raises).
        log.error(
            "commitment_guard: %s carries no served_version — the window is "
            "UNMEASURED.", cfg.META_MANIFEST_KEY,
        )
        _alert(
            f"Predictor commitment window UNMEASURED for {ctx.date_str} — "
            f"s3://{bucket}/{cfg.META_MANIFEST_KEY} carries no `served_version`, so "
            f"the bundle being served cannot be identified and cannot be graded "
            f"against the freeze pinning {frozen or '(none declared)'} "
            f"(alpha-engine-config-I10259). Restamp the live manifest with "
            f"`python -m model.registry --bucket {bucket} --promote <version_id>` for "
            f"the version that should be serving.",
            severity="warning",
            dedup_key="predictor_commitment_served_version_missing",
        )
        return

    if served_version in frozen:
        log.info(
            "commitment_guard: window ACTIVE (freeze_date %s) and served_version "
            "%s is pinned — intact.", commit.freeze_date(doc), served_version,
        )
        return

    # ── VOID ────────────────────────────────────────────────────────────────
    fd = commit.freeze_date(doc)
    state_key = _window_state_key()
    existing = _read_json(s3, bucket, state_key) or {}
    already = (
        str(existing.get("state", "")).upper() == commit.VOID_STATE
        and existing.get("served_version") == served_version
    )

    record = {
        "schema_version": _SCHEMA_VERSION,
        "state": commit.VOID_STATE,
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "served_version": served_version,
        "frozen_version_ids": frozen,
        "freeze_date": fd.isoformat() if fd else None,
        "execution": {
            "date": ctx.date_str,
            "detector": "inference/stages/commitment_guard.py",
            "commitment_key": key,
            "manifest_key": cfg.META_MANIFEST_KEY,
            "issue": "alpha-engine-config-I10259",
        },
    }

    if already:
        # Idempotent: the void is already recorded for THIS served version, so
        # `detected_at` keeps meaning when the void STARTED. Rewriting daily
        # would erase the one field that dates the breach.
        log.error(
            "commitment_guard: window already VOID for served_version %s "
            "(detected_at %s) — record left unchanged.",
            served_version, existing.get("detected_at"),
        )
    else:
        s3.put_object(
            Bucket=bucket, Key=state_key,
            Body=json.dumps(record, indent=2).encode(),
            ContentType="application/json",
        )
        log.error(
            "commitment_guard: out-of-sample window VOID — served_version %s is not "
            "among the pinned %s; recorded at s3://%s/%s",
            served_version, frozen, bucket, state_key,
        )

    _alert(
        f"Predictor out-of-sample commitment window VOID ({ctx.date_str})\n"
        f"  serving:  {served_version}\n"
        f"  pinned:   {', '.join(frozen) or '(none declared)'}\n"
        f"  freeze_date: {fd.isoformat() if fd else '(unset)'}\n"
        f"The pre-registered 252-session out-of-sample window (Brian rulings "
        f"alpha-engine-config-I9679 / -I10302) pins the serving champion, and the "
        f"bundle being served today is not it — every session from here counts "
        f"toward no valid window. Recorded at s3://{bucket}/{state_key}. Either "
        f"restore the pinned version with `python -m model.registry --bucket "
        f"{bucket} --promote {frozen[0] if frozen else '<version_id>'}`, or restart "
        f"the window by amending private-docs/ALPHA_EXPERIMENT_COMMITMENT.yaml in "
        f"alpha-engine-config with a new freeze_date.",
        severity="critical",
        dedup_key=f"predictor_commitment_window_void_{served_version}",
    )
