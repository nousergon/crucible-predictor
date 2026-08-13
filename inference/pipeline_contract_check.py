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
probe re-runs the same self-consistency invariants at SF start, fetched from
GitHub raw, so a contract break halts the run in seconds with a named violation
instead of mid-pipeline.

Exposed as ``action=check_pipeline_contract`` on the predictor Lambda handler
(mirrors ``check_lib_pin_drift``) so the SF invokes it as an early Task → Choice
gate right after ``LibPinDriftCheck``. Returns a JSON-serializable dict the SF
Choice consumes via ``has_violation``.

Fail-open on the checker's own fragility (mirrors ``lib_pin_drift.py`` degraded
mode): a GitHub-fetch or YAML-parse failure for either file → ``has_violation``
is OMITTED from the result (not set to ``false``) + WARN (``reason=
fetch_failed``), so the probe never false-halts the weekly run. It halts ONLY
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

Root cause of the fetch failure itself (measured 2026-08-13 from
``/aws/lambda/alpha-engine-predictor-inference``, live alias v473): ``HTTP
Error 404: Not Found`` on BOTH raw URLs. ``nousergon/alpha-engine-config`` is
a PRIVATE repository, so the unauthenticated ``raw.githubusercontent.com``
read this module was built on can never succeed — the "public, no auth"
premise in ``_fetch_raw``'s docstring is false for this repo. This gate has
therefore never measured anything in production. Fixing the fetch needs an
authenticated read (a credential in the Lambda plus the IAM grant to reach
it) and is tracked separately; this module's job is to stop the failure
reading as a pass.

Validation invariants mirror ``validate_pipeline_contract.py`` (the human SoT
lives in the config repo; this re-implements the *rules* because the predictor
does not depend on the config repo as an importable package). Unlike the CI
validator — which ``sys.exit(1)``s on the FIRST failure — this collects ALL
violations into a list so the SF surfaces the full picture in one halt.
"""

from __future__ import annotations

import logging
import urllib.request

import yaml

log = logging.getLogger(__name__)

# Measurement status (alpha-engine-config-I7277, sf-pipeline-policy.md §2.3a
# rule 2). UNKNOWN means the probe could not run; it is NOT a verdict, and a
# payload carrying it deliberately omits `has_violation` AND `violations` so
# no consumer keyed on either can read an unmeasured gate as a clean one.
STATUS_UNKNOWN = "UNKNOWN"
STATUS_MEASURED = "MEASURED"

# Config repo + default branch the contract SoT lives on.
_CONFIG_REPO = "nousergon/alpha-engine-config"
_CONTRACT_PATH = "private-docs/PIPELINE_CONTRACT.yaml"
_REGISTRY_PATH = "private-docs/ARTIFACT_REGISTRY.yaml"

_RAW_URL = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"

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


def _fetch_raw(path: str, branch: str = "main", timeout: float = 5.0) -> str | None:
    """Fetch ``{config_repo}@branch/{path}`` raw text, or None on any error.

    Reads the raw file from GitHub (public, no auth). Returns ``None`` on any
    network error — the probe treats that as "unknown, proceed with warning"
    (fail-open), mirroring ``lib_pin_drift._fetch_repo_pin``.
    """
    url = _RAW_URL.format(repo=_CONFIG_REPO, branch=branch, path=path)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except OSError as exc:  # URLError/HTTPError + bare read-phase TimeoutError
        log.warning("Pipeline-contract preflight: %s unreachable (%s)", path, exc)
        return None


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


def check_pipeline_contract(branch: str = "main") -> dict:
    """Assert the pipeline-contract invariants; return a dict the SF Choice reads.

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
    """
    contract_raw = _fetch_raw(_CONTRACT_PATH, branch=branch)
    registry_raw = _fetch_raw(_REGISTRY_PATH, branch=branch)

    # Fail-open: if either file is unreachable, do NOT halt the weekly run.
    # has_violation is OMITTED (not set False) — alpha-engine-config-I7048:
    # this state was never measured, so it must not report a definite
    # verdict. The SF's IsPresent-guarded Choice routes an absent key to
    # the visible PipelineContractGateDegraded path. `violations` is omitted
    # too (I7277): an empty list is the same claim in a different field.
    if contract_raw is None or registry_raw is None:
        log.warning(
            "Pipeline-contract preflight: source file(s) unresolved "
            "(contract=%s registry=%s) — proceeding (fail-open)",
            contract_raw is not None,
            registry_raw is not None,
        )
        return _unknown_result("fetch_failed")

    try:
        contract = yaml.safe_load(contract_raw)
        registry = yaml.safe_load(registry_raw)
    except yaml.YAMLError as exc:  # malformed YAML on the probed branch
        log.warning(
            "Pipeline-contract preflight: YAML parse failed (%s) — proceeding (fail-open)",
            exc,
        )
        return _unknown_result("fetch_failed")

    registry_ids = _registry_artifact_ids(registry)
    violations = _validate_contract(contract, registry_ids)
    boundary_count = (
        len(contract["boundaries"])
        if isinstance(contract, dict) and isinstance(contract.get("boundaries"), list)
        else None
    )

    has_violation = bool(violations)
    if has_violation:
        log.error(
            "Pipeline-contract preflight VIOLATION(S): %s", "; ".join(violations)
        )

    return {
        "status": STATUS_MEASURED,
        "has_violation": has_violation,
        "violations": violations,
        "boundary_count": boundary_count,
        "reason": "violation_detected" if has_violation else "in_sync",
    }
