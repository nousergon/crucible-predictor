"""Pre-spend ``PIPELINE_CONTRACT.yaml`` preflight probe (L4595 / config#693).

Companion to ``lib_pin_drift.py``: where ``check_lib_pin_drift`` asserts the
cross-repo *lib-pin* invariant before the Saturday Step Function spends on any
spot launch, this probe asserts the cross-repo *pipeline-contract* invariant —
that ``PIPELINE_CONTRACT.yaml`` (the "production-coverage axis") is internally
self-consistent and every ``artifact_id`` it references exists in
``ARTIFACT_REGISTRY.yaml`` (the "freshness axis").

Why a runtime gate when a CI validator already exists. The config repo's
``scripts/validate_pipeline_contract.py`` + the per-repo contract tests catch a
producer/consumer field drift at PR TIME, but a config / SF-definition change
made OUTSIDE those repos' CI (e.g. editing the contract YAML on a branch that
never touches a code repo, or an ARTIFACT_REGISTRY edit that orphans a
contract artifact_id) can still spend a Saturday spot before failing. This
probe re-runs the same self-consistency invariants at SF start, against the
copies alpha-engine-config's CI publishes to S3, so a contract break halts the
run in seconds with a named violation instead of mid-pipeline.

Exposed as ``action=check_pipeline_contract`` on the predictor Lambda handler
(mirrors ``check_lib_pin_drift``) so the SF invokes it as an early Task → Choice
gate right after ``LibPinDriftCheck``. Returns a JSON-serializable dict the SF
Choice consumes via ``has_violation``.

Fail-open on the checker's own fragility (mirrors ``lib_pin_drift.py`` degraded
mode): an S3-read or YAML-parse failure for either file → ``has_violation``
is OMITTED from the result (not set to ``false``) + WARN with a reason naming
WHICH failure it was, so the probe never false-halts the weekly run. It halts ONLY
on a CONFIRMED structural violation (both files fetched + parsed AND a
self-consistency / dangling-artifact_id breach).

alpha-engine-config-I7048 (2026-08-12): previously this branch returned
``has_violation=False`` — a definite, present-key "no violation" verdict —
indistinguishable downstream from a real pass. Measured live on the
scheduled 2026-08-08 weekly run: ``{"has_violation": false, "violations":
[], "boundary_count": null, "reason": "fetch_failed"}`` — the check never
examined anything (``boundary_count: null`` is the honest field) but reported
green anyway. The SF's ``PipelineContractGate`` Choice (nousergon-data
infrastructure/step_function.json) already has a correctly-designed
IsPresent-guarded branch for exactly this case — an ABSENT ``has_violation``
routes to ``PipelineContractGateDegraded`` (visible, SNS-alerted, non-
blocking) rather than the silent-pass Default. Omitting the key here (instead
of adding a new SF state) is the surgical fix: it reuses machinery that
already existed and was already correct, mirroring ``check_lib_pin_drift``'s
identical correction in ``lib_pin_drift.py``. Do NOT default to
``has_violation=True`` on a fetch failure — a check that could not run is not
evidence of a violation either; inverting the lie does not make it true.

alpha-engine-config-I7277 (2026-08-13): omitting ``has_violation`` was
necessary but not sufficient. The fail-open branch still emitted
``violations: []`` alongside ``reason=fetch_failed`` — measured live on
``ne-weekly-freshness-pipeline`` executions ``watch-rerun-2026-08-13-2`` and
``-3``: ``{"violations": [], "boundary_count": null, "reason":
"fetch_failed"}``. ``violations`` is the field a consumer is shaped to read,
and an empty list reads as an authoritative "checked, found nothing". A
payload cannot be honest about its verdict key and lie in the evidence list
beside it. The unmeasured branch now carries an explicit ``status=UNKNOWN``
and OMITS ``violations`` entirely, so no field in it can be read as a clean
check; the measured branch carries ``status=MEASURED`` alongside its
``violations`` list. ``boundary_count: null`` never again coexists with an
authoritative-looking empty ``violations``.

alpha-engine-config-I7281 (2026-08-14): the fetch itself is fixed here.

Measured 2026-08-13 from ``/aws/lambda/alpha-engine-predictor-inference``:
``HTTP Error 404: Not Found`` on BOTH raw URLs, on **190 of 190** invocations
in the preceding 30 days. ``nousergon/alpha-engine-config`` is a PRIVATE
repository, so the unauthenticated ``raw.githubusercontent.com`` read this
module was built on could never succeed — the "public, no auth" premise in
the old ``_fetch_raw`` docstring was false for this repo, and the gate had
therefore never measured anything on any run since it shipped.

The read now comes from S3, where alpha-engine-config's CI publishes both
files on merge to main. That was chosen over putting a GitHub PAT in the
Lambda: no new secret, no rotation surface, no second auth path, no GitHub
rate limit — and the predictor role already holds ``s3:GetObject`` on the
bucket, so the fix needed no read-side grant at all. It also turns the read
into an ARTIFACT read, which a scheduled drift check can assert against and
which carries a publish timestamp this probe now reports.

The reason vocabulary is deliberately wider than before. GitHub answers an
unauthorized read of a private repo with 404 rather than 403, so a permanent
structural impossibility was reported with the same ``fetch_failed`` a
transient blip gets — and was read as one for the entire life of the gate.
``source_missing`` / ``source_forbidden`` / ``fetch_failed`` / ``parse_failed``
are four different operator actions and never collapse into one.

Validation invariants mirror ``validate_pipeline_contract.py`` (the human SoT
lives in the config repo; this re-implements the *rules* because the predictor
does not depend on the config repo as an importable package). Unlike the CI
validator — which ``sys.exit(1)``s on the FIRST failure — this collects ALL
violations into a list so the SF surfaces the full picture in one halt.
"""

from __future__ import annotations

import logging

import boto3
import yaml
from botocore.exceptions import BotoCoreError, ClientError

log = logging.getLogger(__name__)

# Measurement status (alpha-engine-config-I7277, sf-pipeline-policy.md §2.3a
# rule 2). UNKNOWN means the probe could not run; it is NOT a verdict, and a
# payload carrying it deliberately omits `has_violation` AND `violations` so
# no consumer keyed on either can read an unmeasured gate as a clean one.
STATUS_UNKNOWN = "UNKNOWN"
STATUS_MEASURED = "MEASURED"

# Where the two source-of-truth YAMLs are PUBLISHED (alpha-engine-config-I7281).
#
# Both live in `nousergon/alpha-engine-config`, which is PRIVATE. They are
# published to S3 on merge to main and read from there — never from GitHub.
# The contract key is written by alpha-engine-config's
# `sync-pipeline-contract.yml`; the registry key by its older sibling
# `sync-artifact-registry.yml`, which has fed the freshness monitor from that
# same key since 2026-06-05. The registry is deliberately NOT republished under
# a contract-specific prefix: two copies of one file is the drift this whole
# class of bug is made of.
_SOURCE_BUCKET = "alpha-engine-research"
_CONTRACT_KEY = "_pipeline_contract/PIPELINE_CONTRACT.yaml"
_REGISTRY_KEY = "_freshness_monitor/ARTIFACT_REGISTRY.yaml"

# Fail-open reasons, deliberately distinct (alpha-engine-config-I7281).
#
# The predecessor read raw.githubusercontent.com unauthenticated and reported
# every outcome as `fetch_failed`. Because the repo is private, GitHub answers
# an unauthorized read with 404 rather than 403 — so a PERMANENT, structural
# impossibility was indistinguishable from a transient network blip, and read
# as one for the whole life of the gate (190 of 190 invocations in the 30 days
# to 2026-08-13). A checker that cannot say WHY it could not measure invites
# exactly that misreading, so these three never collapse into one:
_REASON_MISSING = "source_missing"      # the object is not there (never published)
_REASON_FORBIDDEN = "source_forbidden"  # it is there and this role may not read it
_REASON_FETCH_FAILED = "fetch_failed"   # a genuine transport/endpoint failure
_REASON_PARSE_FAILED = "parse_failed"   # fetched fine, is not valid YAML

# Invariant set — kept in lockstep with scripts/validate_pipeline_contract.py.
_ALLOWED_FAIL_POSTURE = frozenset({"fail_soft", "fail_loud"})
_REQUIRED_BOUNDARY_KEYS = frozenset(
    {
        "boundary_id",
        "producer_stage",
        "producer_repo",
        "artifact_id",
        "required_top_level_fields",
        "required_per_item_fields",
        "per_item_collections",
        "consumers",
    }
)


def _s3():
    """The S3 client, resolved at call time so tests can patch it."""
    return boto3.client("s3")


def _fetch_source(key: str) -> tuple[str | None, str | None, object | None]:
    """Read ``s3://{_SOURCE_BUCKET}/{key}``.

    Returns ``(text, failure_reason, last_modified)``. Exactly one of ``text``
    and ``failure_reason`` is non-None. ``last_modified`` is the object's S3
    LastModified on success, so the caller can report HOW FRESH the thing it
    validated was — a successfully-fetched-but-stale contract is the next
    instance of this bug class, and ``boundary_count`` alone does not detect it.

    The read is fail-open by contract (the probe must never halt the weekly
    run on its own fragility), but it is NOT reason-flat: a missing object, a
    denied read and a transport failure are three different operator actions
    and get three different reasons (alpha-engine-config-I7281).
    """
    try:
        obj = _s3().get_object(Bucket=_SOURCE_BUCKET, Key=key)
        return obj["Body"].read().decode("utf-8"), None, obj.get("LastModified")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("NoSuchKey", "NoSuchBucket", "404") or status == 404:
            log.warning(
                "Pipeline-contract preflight: s3://%s/%s does not exist — it has "
                "never been published, or the publisher's key changed",
                _SOURCE_BUCKET, key,
            )
            return None, _REASON_MISSING, None
        if code in ("AccessDenied", "403") or status == 403:
            log.warning(
                "Pipeline-contract preflight: s3://%s/%s exists but this role may "
                "not read it — an IAM gap, not an outage",
                _SOURCE_BUCKET, key,
            )
            return None, _REASON_FORBIDDEN, None
        log.warning(
            "Pipeline-contract preflight: s3://%s/%s unreadable (%s)",
            _SOURCE_BUCKET, key, exc,
        )
        return None, _REASON_FETCH_FAILED, None
    except (BotoCoreError, OSError) as exc:  # endpoint/DNS/socket, not an API answer
        log.warning(
            "Pipeline-contract preflight: s3://%s/%s unreachable (%s)",
            _SOURCE_BUCKET, key, exc,
        )
        return None, _REASON_FETCH_FAILED, None


def _is_nonempty_str_list(value: object, *, allow_empty: bool = False) -> bool:
    if not isinstance(value, list) or (not value and not allow_empty):
        return False
    return all(isinstance(item, str) and item for item in value)


def _registry_artifact_ids(registry: object) -> set[str]:
    """Set of every ``artifact_id`` declared in ARTIFACT_REGISTRY.yaml."""
    if not isinstance(registry, dict):
        return set()
    return {
        e["artifact_id"]
        for e in registry.get("artifacts", [])
        if isinstance(e, dict) and "artifact_id" in e
    }


def _validate_boundary(b: object, index: int, registry_ids: set[str]) -> list[str]:
    """Collect (not raise) every invariant violation for one boundary entry."""
    bid = b.get("boundary_id", "<unknown>") if isinstance(b, dict) else "<non-mapping>"
    where = f"boundaries[{index}] (id={bid})"
    if not isinstance(b, dict):
        return [f"{where} is not a mapping (got {type(b).__name__})"]

    violations: list[str] = []

    missing = _REQUIRED_BOUNDARY_KEYS - b.keys()
    if missing:
        violations.append(f"{where} missing required keys: {sorted(missing)}")
        # Without the load-bearing keys the rest of the checks are noise.
        return violations

    if not isinstance(bid, str) or not bid:
        violations.append(f"{where} boundary_id must be a non-empty string")

    if b["artifact_id"] not in registry_ids:
        violations.append(
            f"{where} artifact_id={b['artifact_id']!r} is not declared in "
            f"ARTIFACT_REGISTRY.yaml (dangling reference)"
        )

    if not _is_nonempty_str_list(b["required_top_level_fields"]):
        violations.append(f"{where} required_top_level_fields must be a non-empty str list")
        top: set[str] = set()
    else:
        top = set(b["required_top_level_fields"])

    if not _is_nonempty_str_list(b["required_per_item_fields"], allow_empty=True):
        violations.append(f"{where} required_per_item_fields must be a str list")
        per_item: set[str] = set()
    else:
        per_item = set(b["required_per_item_fields"])

    if not _is_nonempty_str_list(b["per_item_collections"], allow_empty=True):
        violations.append(f"{where} per_item_collections must be a str list")
        collections: list[str] = []
    else:
        collections = b["per_item_collections"]
    for coll in collections:
        if coll not in top:
            violations.append(
                f"{where} per_item_collections entry {coll!r} is not in "
                f"required_top_level_fields"
            )

    declared = top | per_item
    consumers = b["consumers"]
    if not isinstance(consumers, list) or not consumers:
        violations.append(f"{where} consumers must be a non-empty list")
        return violations
    for j, c in enumerate(consumers):
        cwhere = f"{where} consumers[{j}]"
        if not isinstance(c, dict):
            violations.append(f"{cwhere} is not a mapping")
            continue
        if not isinstance(c.get("repo"), str) or not c["repo"]:
            violations.append(f"{cwhere} missing repo")
        if not _is_nonempty_str_list(c.get("reads")):
            violations.append(f"{cwhere} reads must be a non-empty str list")
            reads: list[str] = []
        else:
            reads = c["reads"]
        if c.get("fail_posture") not in _ALLOWED_FAIL_POSTURE:
            violations.append(
                f"{cwhere} (repo={c.get('repo')}) fail_posture="
                f"{c.get('fail_posture')!r} not in {sorted(_ALLOWED_FAIL_POSTURE)}"
            )
        undeclared = set(reads) - declared
        if undeclared:
            violations.append(
                f"{cwhere} (repo={c.get('repo')}) reads field(s) the contract "
                f"does not declare: {sorted(undeclared)}"
            )
    return violations


def _validate_contract(contract: object, registry_ids: set[str]) -> list[str]:
    """Collect every top-level + per-boundary invariant violation."""
    if not isinstance(contract, dict):
        return ["top-level PIPELINE_CONTRACT.yaml is not a mapping"]

    violations: list[str] = []
    if "schema_version" not in contract:
        violations.append("missing top-level 'schema_version'")

    boundaries = contract.get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        violations.append("'boundaries' must be a non-empty list")
        return violations

    ids: list[str] = []
    for i, b in enumerate(boundaries):
        violations.extend(_validate_boundary(b, i, registry_ids))
        if isinstance(b, dict) and isinstance(b.get("boundary_id"), str):
            ids.append(b["boundary_id"])
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        violations.append(f"duplicate boundary_id(s): {dupes}")
    return violations


def _iso(value: object) -> str | None:
    """S3 LastModified -> ISO-8601 string, or None. Never raises.

    The payload is JSON-serialized by the SF's Lambda integration, and a
    datetime is not JSON. Returning None on anything unexpected keeps a
    freshness FIELD from being the thing that fails the whole probe.
    """
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:  # pragma: no cover — defensive; never fail the probe
            return None
    return None


def _unknown_result(reason: str) -> dict:
    """The single unmeasured-gate payload (alpha-engine-config-I7277).

    Constructed in one place so a future edit cannot reintroduce an
    authoritative-looking field on one fail-open branch and not the other.
    Carries NO ``has_violation`` and NO ``violations``: a consumer keyed on
    either sees an absent key, never an empty-and-clean-looking one.
    """
    return {
        "status": STATUS_UNKNOWN,
        "boundary_count": None,
        "reason": reason,
    }


def check_pipeline_contract(*, probe: bool = False) -> dict:
    """Assert the pipeline-contract invariants; return a dict the SF Choice reads.

    Reads both YAMLs from S3 (alpha-engine-config-I7281). The former ``branch``
    parameter is gone: the source is now the PUBLISHED copy of main, not a
    ref this probe chooses, and a parameter implying otherwise would invite a
    caller to gate a production run on an unmerged branch.

    Two mutually exclusive shapes, distinguished by ``status``:

    * ``status=MEASURED`` — both files fetched and parsed. Carries
      ``has_violation`` (``true``/``false``), the full ``violations`` list and
      a non-null ``boundary_count``.
    * ``status=UNKNOWN`` — a fetch or YAML-parse miss. Carries ``reason=
      fetch_failed`` and ``boundary_count=None``, and OMITS BOTH
      ``has_violation`` and ``violations`` (alpha-engine-config-I7048 +
      I7277: unmeasured is not measured-clean, and an empty evidence list is
      just as much a claim as a false verdict key).

    ``has_violation=true`` only when a confirmed self-consistency /
    dangling-artifact_id breach is found. Shape mirrors ``check_lib_pin_drift``.

    ``probe=True`` marks a SYNTHETIC invocation — ``infrastructure/deploy.sh``'s
    deploy-time canary, which gates on the PRESENCE of ``has_violation``, never
    on its value. The payload is identical either way; only the log severity of
    a detected violation drops to WARNING so it is recorded without paging
    (alpha-engine-config-I7954). Mirrors ``check_lib_pin_drift(probe=...)``.
    This action ALREADY passed ``dry_run: true`` from the canary before I7954 —
    which is exactly why threading ``dry_run`` alone was not the fix: nothing
    carried the synthetic-invocation fact down to the log severity.
    """
    contract_raw, contract_reason, contract_mtime = _fetch_source(_CONTRACT_KEY)
    registry_raw, registry_reason, registry_mtime = _fetch_source(_REGISTRY_KEY)

    # Fail-open: if either file is unreachable, do NOT halt the weekly run.
    # has_violation is OMITTED (not set False) — alpha-engine-config-I7048:
    # this state was never measured, so it must not report a definite
    # verdict. The SF's IsPresent-guarded Choice routes an absent key to
    # the visible PipelineContractGateDegraded path. `violations` is omitted
    # too (I7277): an empty list is the same claim in a different field.
    #
    # The CONTRACT's reason wins when both failed: it is this gate's subject,
    # and the registry is only the set it is checked against. Reporting the
    # registry's reason for a missing contract would send the operator to the
    # wrong publisher (alpha-engine-config-I7281).
    if contract_raw is None or registry_raw is None:
        reason = contract_reason or registry_reason or _REASON_FETCH_FAILED
        log.warning(
            "Pipeline-contract preflight: source file(s) unresolved "
            "(contract=%s registry=%s) reason=%s — proceeding (fail-open)",
            contract_raw is not None,
            registry_raw is not None,
            reason,
        )
        return _unknown_result(reason)

    try:
        contract = yaml.safe_load(contract_raw)
        registry = yaml.safe_load(registry_raw)
    except yaml.YAMLError as exc:
        # Distinct from a fetch miss: the publish worked and put something
        # unparseable in the bucket. The publisher validates before it copies,
        # so this means the bucket was written by something else.
        log.warning(
            "Pipeline-contract preflight: YAML parse failed (%s) — proceeding (fail-open)",
            exc,
        )
        return _unknown_result(_REASON_PARSE_FAILED)

    registry_ids = _registry_artifact_ids(registry)
    violations = _validate_contract(contract, registry_ids)
    boundary_count = (
        len(contract["boundaries"])
        if isinstance(contract, dict) and isinstance(contract.get("boundaries"), list)
        else None
    )

    has_violation = bool(violations)
    if has_violation:
        log.log(
            logging.WARNING if probe else logging.ERROR,
            "Pipeline-contract preflight VIOLATION(S): %s", "; ".join(violations),
        )

    return {
        "status": STATUS_MEASURED,
        "has_violation": has_violation,
        "violations": violations,
        "boundary_count": boundary_count,
        "reason": "violation_detected" if has_violation else "in_sync",
        # WHICH published copies were judged. A gate that says "in_sync"
        # without saying what it read cannot distinguish a current contract
        # from one published months ago (alpha-engine-config-I7281).
        "contract_published_at": _iso(contract_mtime),
        "registry_published_at": _iso(registry_mtime),
    }
