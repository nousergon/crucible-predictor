"""Deploy-drift probe: confirm Step Function + CloudFormation deployed SHAs
match `origin/main` HEAD for their source repos.

Exposed as `action=check_deploy_drift` on the predictor Lambda handler so
the weekday Step Function can invoke it as a Task state before any real
work runs. Keeps the check-surface in one Lambda rather than deploying a
new one for a ~100-line concern; architectural split can happen later if
this Lambda's surface grows.

Returns a JSON-serializable dict the SF consumes via a Choice state.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Union

# Canonical helper lives in the lib so the catch-tuple bug-fix lands in
# one place (see lib v0.5.5). Re-imported as a module-level attribute so
# `patch.object(deploy_drift, "_fetch_origin_main_sha", ...)` keeps
# working in tests.
from nousergon_lib.preflight import _fetch_origin_main_sha  # noqa: F401

log = logging.getLogger(__name__)

_SF_ARN_TEMPLATE = (
    "arn:aws:states:{region}:{account}:stateMachine:{name}"
)

# ── alpha-engine-config-I8142 · sf-pipeline-policy §7a.4 staging ─────────────
#
# The I8142 change alters WHEN the halt branch of the live preopen
# `DeployDriftGate` is taken: an unreadable live definition used to emit a
# present `sf_drift: false` (pass) and now withdraws the verdict (halt). That
# is "a change to the error classification of an existing halting check" — §7a.4
# exactly — so it carries §7a's obligations even though the Choice state is
# untouched.
#
# It cannot be staged by suppressing the consequence: the consequence IS the
# ruling (I7799, "missing or unfetchable → UNKNOWN → fail closed"), and an
# observe mode that kept emitting `sf_drift: false` would be the defect itself
# for the length of the window. What is staged is the CONFIRMATION, per I8142's
# deliverable 4: the withdrawal fires only on inputs that were never measured,
# so on a healthy fleet it must never fire at all, and that is the observable.
#
# OBSERVE WINDOW — declared here, in the guard's own module (§7a obligation 2):
#   Criterion : `sf_drift` PRESENT on every real preopen execution for 5
#               consecutive preopen executions, OR through 2026-09-05,
#               whichever comes first.
#   Surface   : the probe payload's `sf_drift_state` on each preopen run
#               (`no_drift` on a healthy run; `unmeasured` is the withdrawal),
#               plus the ERROR log lines this module emits on withdrawal.
#   Promotion : nothing to flip — reaching the criterion RETIRES this comment
#               block and closes the tracked item. The behaviour is enforcing
#               from merge because a fail-open on the trading gate has no safe
#               observe form.
#   Rollback  : a withdrawal on a run whose live definition was in fact
#               readable is a defect in THIS change, not a drift finding —
#               revert the commit rather than widening the fallback.
#   Tracked   : alpha-engine-config-I8144, Re-exam: 2026-09-05.
#
# Note: the postclose `DeployDriftGate` (nousergon-data-PR1505) is itself in
# §7a observe mode until 2026-09-08 (alpha-engine-config-I8141), so the
# postclose side cannot halt on this during the window; only the preopen gate
# enforces it.
_I8142_OBSERVE_WINDOW = {
    "criterion": "sf_drift present on 5 consecutive preopen executions",
    "deadline": "2026-09-05",
    "tracked": "alpha-engine-config-I8144",
}

# alpha-engine-config-I7799 (Brian ruling 2026-08-20, option b).
#
# Until 2026-08-20 `sf_drift` was a comparison of the deployed `[git:<sha>]`
# STAMP against `repo@main` HEAD. That answers "has anything in this repo
# merged without deploying" — strictly broader than the question
# sf-pipeline-policy.md §3 halts on, which is whether "the live SF definition
# has drifted so orchestration semantics are unknown".
#
# Measured 2026-08-20: an illegal ASL escape in the WEEKLY definition failed
# the all-or-nothing deploy preflight for three consecutive merges. None of
# those commits altered the preopen definition — the live preopen orchestration
# was byte-identical to what `main` describes, and correct. `DeployDriftGate`
# halted anyway and the session went unmanaged: no exit management, no risk
# checks, no planner, open positions carried (§1.2).
#
# So for the WEEKDAY pipelines `sf_drift` now compares the deployed definition
# BODY against the repo's, canonicalised. The stamp comparison survives as
# `deploy_stamp_stale` — a real signal (a merged Lambda/collector/script change
# this pipeline INVOKES but does not embed is genuinely undeployed) that
# degrades and pages rather than halting. The weekly pipeline keeps stamp
# semantics: it has no market-open deadline, and a lost run there costs a
# rerun rather than a session.
_SF_DEFINITION_PATHS = {
    "ne-preopen-trading-pipeline": "infrastructure/step_function_daily.json",
    "ne-postclose-trading-pipeline": "infrastructure/step_function_eod.json",
}

#: State machines this probe answers for on STAMP semantics alone — no
#: definition body is compared, because I7799's relaxation is scoped to the
#: weekday pipelines. Named rather than left implicit in
#: ``_SF_DEFINITION_PATHS.get()`` returning ``None``: that fall-through cannot
#: tell "the weekly pipeline, deliberately stamp-only" from "a name nobody
#: declared", and the second must never quietly receive the first's semantics.
_STAMP_ONLY_SF_NAMES = frozenset({"ne-weekly-freshness-pipeline"})

#: The complete set of ``sf_name`` values a CALLER may ask for. The handler
#: validates against this before invoking, so an unrecognised name is a loud
#: error at the edge rather than a silent downgrade to stamp semantics deep
#: inside the probe — the latter would answer a question about a state machine
#: that does not exist, with a verdict a halting gate reads.
INVOKABLE_SF_NAMES = frozenset(_SF_DEFINITION_PATHS) | _STAMP_ONLY_SF_NAMES

# Pinned to the RESOLVED upstream SHA, never to `main`: raw.githubusercontent
# serves a CDN-cached copy of a branch ref, so fetching `@main` can return a
# body that is not the SHA we just compared against — which would make the
# comparison report drift that does not exist, or miss drift that does.
_RAW_FILE_URL = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"

# `[git:<40 hex>]` prefix injected by alpha-engine-data/infrastructure/
# deploy-infrastructure.sh into the SF Comment field at deploy time.
_GIT_PREFIX_RE = re.compile(r"^\[git:([0-9a-f]{7,40})\]")

# CloudFormation terminal / failed states. A stack sitting in any of these
# cannot be trusted to actually hold the resources its template declares,
# so the drift probe must distinguish this condition from "stack exists
# with no git-sha tag" (legacy deploy warn-path) and fire the hard-fail
# branch of the SF DeployDriftGate.
#
# Keep in sync with CloudFormation's published state machine
# (https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/using-cfn-describing-stacks.html).
_CFN_TERMINAL_FAILED_STATES = frozenset({
    "CREATE_FAILED",
    "ROLLBACK_COMPLETE",       # ← 2026-04-20 incident mode
    "ROLLBACK_FAILED",
    "DELETE_FAILED",
    "UPDATE_ROLLBACK_FAILED",
    "UPDATE_ROLLBACK_COMPLETE",
    "IMPORT_ROLLBACK_FAILED",
    "IMPORT_ROLLBACK_COMPLETE",
})


@dataclass(frozen=True)
class StackStateError:
    """Sentinel for ``_read_stack_tag`` return when we can't trust the stack.

    Distinguishes:
      - ``stack_not_exists`` — describe_stacks found nothing (or raised
        ValidationError "does not exist"). Stack is absent entirely.
      - ``stack_in_terminal_state`` — stack exists but is in a terminal
        failed state (ROLLBACK_COMPLETE etc.) and cannot be trusted to
        hold its declared resources. This is what tripped the
        2026-04-20 orchestration-stack recovery.
      - ``describe_stacks_error`` — boto3 itself raised (IAM, network).
        Treated as "can't prove healthy" → route to hard-fail.

    The ``no_git_sha_tag_legacy`` case is NOT a StackStateError — that's
    a legitimate legacy-deploy warn path where the stack is healthy and
    tagged correctly by older tooling; returned as a plain ``None``.
    """
    reason: str           # stack_not_exists | stack_in_terminal_state | describe_stacks_error
    detail: str = ""      # stack status name or exception class name
    stack_status: Optional[str] = None  # populated for terminal-state

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "detail": self.detail,
            "stack_status": self.stack_status,
        }


def _extract_sf_sha(comment: str) -> Optional[str]:
    """Pull `<sha>` out of a `[git:<sha>] rest…` Comment string."""
    if not comment:
        return None
    m = _GIT_PREFIX_RE.match(comment.strip())
    return m.group(1) if m else None


@dataclass(frozen=True)
class LiveDefinitionError:
    """Sentinel for ``_read_live_definition`` when the live machine is unreadable.

    The reason this is a SENTINEL and not ``None`` is alpha-engine-config-I8142.
    ``describe_state_machine`` supplies BOTH verdicts this probe emits about the
    live state machine — the ``[git:<sha>]`` stamp (``deploy_stamp_stale``) and
    the definition body (``sf_drift``). When that one call fails, ``None``
    cannot say whether the stamp is genuinely absent (a measured legacy-deploy
    state) or simply could not be read (nothing measured at all), and the second
    was silently rendered as the first: ``sf_sha=None`` makes the stamp verdict
    ``False`` by construction, so an AccessDenied emitted
    ``sf_drift: false, deploy_stamp_stale: false`` — present, definite, and
    describing an authorization failure. Measured live 2026-08-21 against
    ``ne-postclose-trading-pipeline``.

    ``reason`` is always ``describe_state_machine_error``; ``detail`` carries the
    exception class and message so the page can name the actual denial.
    """
    reason: str = "describe_state_machine_error"
    detail: str = ""

    def to_dict(self) -> dict:
        return {"reason": self.reason, "detail": self.detail}


def _read_live_definition(
    state_machine_arn: str,
) -> Union[dict, LiveDefinitionError]:
    """describe-state-machine → the parsed live definition document.

    Returns the parsed document, or a ``LiveDefinitionError`` when the call
    failed or the payload did not parse. There is no return value that can be
    mistaken for a successful read of an empty machine.

    ONE call answers both live-side questions (stamp and body) deliberately:
    they are properties of the same document, and two calls could observe two
    different deploys either side of an ``update-state-machine``. The cost of
    the single call is that its failure takes out both verdicts — which is
    exactly why the failure is typed rather than folded into ``None``, so both
    are marked UNMEASURED and neither is silently answered (I8142).
    """
    import boto3
    try:
        sfn = boto3.client("stepfunctions")
        resp = sfn.describe_state_machine(stateMachineArn=state_machine_arn)
        return json.loads(resp["definition"])
    except Exception as exc:  # noqa: BLE001 — any failure is one UNKNOWN
        log.warning("describe_state_machine failed for %s: %s",
                    state_machine_arn, exc)
        return LiveDefinitionError(
            detail=f"{exc.__class__.__name__}: {str(exc)[:200]}",
        )


def _read_stack_tag(
    stack_name: str, tag_key: str = "git-sha",
) -> Union[str, StackStateError, None]:
    """describe-stacks → Tags[git-sha], distinguishing four states:

    - ``str``                       stack healthy, tag present → return SHA
    - ``None``                      stack healthy, tag absent → legacy-deploy
                                    warn path (NOT drift)
    - ``StackStateError`` (reason=stack_not_exists)
                                    stack absent entirely — SF must hard-fail
    - ``StackStateError`` (reason=stack_in_terminal_state)
                                    stack in ROLLBACK_COMPLETE / *_FAILED →
                                    cannot be trusted; SF must hard-fail
    - ``StackStateError`` (reason=describe_stacks_error)
                                    boto3 raised (IAM, network); can't prove
                                    healthy — SF must hard-fail

    Previously this function conflated all four into ``None``, which made
    the SF's DeployDriftGate silently pass when the stack was catastrophically
    broken — exactly the 2026-04-20 orchestration-stack ROLLBACK_COMPLETE
    scenario that required an emergency recovery.
    """
    import boto3
    from botocore.exceptions import ClientError
    try:
        cfn = boto3.client("cloudformation")
        resp = cfn.describe_stacks(StackName=stack_name)
    except ClientError as exc:
        err_code = exc.response.get("Error", {}).get("Code", "")
        err_msg = str(exc)
        # boto3 surfaces "stack does not exist" as ValidationError with this
        # exact substring. Distinguish it from generic API failures so the
        # SF can report the clearer "stack_not_exists" reason code.
        if (err_code == "ValidationError" and "does not exist" in err_msg):
            log.warning("describe_stacks: %s does not exist", stack_name)
            return StackStateError(
                reason="stack_not_exists",
                detail=f"{stack_name} not found",
            )
        log.warning("describe_stacks failed for %s: %s", stack_name, exc)
        return StackStateError(
            reason="describe_stacks_error",
            detail=f"{exc.__class__.__name__}: {err_msg[:200]}",
        )
    except Exception as exc:  # noqa: BLE001 — non-ClientError (SDK bug, etc.)
        log.warning("describe_stacks failed for %s: %s", stack_name, exc)
        return StackStateError(
            reason="describe_stacks_error",
            detail=f"{exc.__class__.__name__}: {str(exc)[:200]}",
        )

    stacks = resp.get("Stacks") or []
    if not stacks:
        # describe_stacks returned 0 stacks without raising — shouldn't happen
        # with a StackName lookup, but preserve the distinct reason.
        return StackStateError(
            reason="stack_not_exists",
            detail=f"{stack_name} not returned by describe_stacks",
        )

    stack = stacks[0]
    stack_status = stack.get("StackStatus", "")
    if stack_status in _CFN_TERMINAL_FAILED_STATES:
        log.error(
            "Stack %s is in terminal state %s — describe_stacks resources "
            "cannot be trusted. Probe will report cf_drift=true with "
            "reason=stack_in_terminal_state so the SF DeployDriftGate fires.",
            stack_name, stack_status,
        )
        return StackStateError(
            reason="stack_in_terminal_state",
            detail=f"{stack_name} is in {stack_status}",
            stack_status=stack_status,
        )

    for tag in stack.get("Tags") or []:
        if tag.get("Key") == tag_key:
            return tag.get("Value")
    # Stack healthy, tag absent — legacy-deploy warn path, NOT drift.
    return None


def _shas_match(deployed: Optional[str], upstream: Optional[str]) -> bool:
    """Compare a deployed SHA stamp (may be 7-40 chars) to full upstream SHA.
    Missing either side → return True (can't prove drift → don't raise).
    """
    if not deployed or not upstream:
        return True
    if len(deployed) < 7:
        return True  # malformed stamp — warn elsewhere, don't block
    return upstream.startswith(deployed) or deployed.startswith(upstream)


def _canonical_definition(doc: dict) -> str:
    """Canonical JSON for a definition, with the deploy stamp normalised out.

    `deploy-infrastructure.sh` sets ``Comment = f'[git:{sha}] {orig}'.rstrip()``
    on the copy it uploads, so the live document and the repo document differ by
    that prefix by construction. Strip it from either side, drop the key when
    nothing else remains (the rstrip case where the repo Comment is empty), and
    serialise with sorted keys so key ORDER — which neither AWS nor the deploy
    preserves — is not read as a semantic difference.
    """
    normalised = json.loads(json.dumps(doc))
    comment = normalised.get("Comment")
    if isinstance(comment, str):
        stripped = comment.strip()
        match = _GIT_PREFIX_RE.match(stripped)
        if match:
            stripped = stripped[match.end():].strip()
        if stripped:
            normalised["Comment"] = stripped
        else:
            normalised.pop("Comment", None)
    return json.dumps(normalised, sort_keys=True, separators=(",", ":"))


#: Bucket the deploy uploads the stamped definitions to. Declared here rather
#: than read from an env var so the probe cannot be pointed somewhere else by an
#: ambient value — the class of defect alpha-engine-config-I7925 removed from
#: this module's GitHub credential.
_DEFINITION_BUCKET = "alpha-engine-research"


def _fetch_s3_definition(
    path: str, bucket: str = _DEFINITION_BUCKET,
) -> Optional[dict]:
    """Fetch the definition the LAST DEPLOY published, from S3.

    `deploy-infrastructure.sh` uploads each stamped definition to
    ``s3://{bucket}/{path}`` using the same key as its repo path, and uploads
    exactly the bytes it hands to ``update-state-machine``. So this is the
    deploy's own record of what it deployed — not a re-derivation of it from
    the upstream the deploy was built from.

    None on any failure — missing, unreadable, or unparseable. The caller
    treats None as "could not compare from S3" and falls back; there is no
    outcome of this function that can grant the guarantee.
    """
    import boto3
    try:
        s3 = boto3.client("s3")
        body = s3.get_object(Bucket=bucket, Key=path)["Body"].read()
        return json.loads(body)
    except Exception as exc:  # noqa: BLE001 — any failure routes to None
        log.warning("S3 definition read failed for s3://%s/%s: %s",
                    bucket, path, exc)
        return None


def _fetch_repo_definition(
    repo: str, ref: str, path: str, timeout: float = 5.0,
) -> Optional[dict]:
    """Fetch and parse one SF definition from GitHub at an exact ref.

    None on any failure — unreachable, non-200, or unparseable. The caller
    treats None as "could not compare", never as "no drift"; there is no
    outcome of this function that can grant the guarantee.
    """
    import urllib.request
    url = _RAW_FILE_URL.format(repo=repo, ref=ref, path=path)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except OSError as exc:  # URLError/HTTPError + bare read-phase TimeoutError
        log.warning("SF definition unreachable: %s (%s)", url, exc)
        return None
    except (ValueError, TypeError) as exc:
        log.warning("SF definition at %s did not parse: %s", url, exc)
        return None


def check_deploy_drift(
    region: str,
    account_id: str,
    sf_name: str = "ne-preopen-trading-pipeline",
    stack_name: str = "alpha-engine-orchestration",
    repo: str = "nousergon/nousergon-data",
    branch: str = "main",
) -> dict:
    """Compare deployed SF + CF SHAs against GitHub `repo@branch` HEAD.

    Returns a dict with per-artifact stamps, the upstream SHA, and
    booleans flagging drift. The Step Function's `DeployDriftGate` Choice
    (nousergon-data infrastructure/step_function_daily.json) reads
    `sf_drift`/`cf_drift` directly and is IsPresent-guarded on each,
    per Brian's 2026-08-09 ruling (config#6615): sf_drift is
    trading-correctness-critical, so an ABSENT sf_drift or cf_drift
    already routes to `HandleFailure` (fail CLOSED on unknown) — the
    OPPOSITE convention from the Saturday-SF preflight gates
    (`check_lib_pin_drift`/`check_pipeline_contract`, alpha-engine-config-
    I7048), which fail open-but-visible. Either way, the shared invariant
    is the same: a field this function could not actually MEASURE must
    never be a present, defaulted `False` — that reads as "checked, no
    drift" to the SF's IsPresent guard exactly as if it had been verified.

    alpha-engine-config-I7048 sibling fix (2026-08-12): `sf_drift`/
    `cf_drift` are now OMITTED (not `False`) specifically when a needed
    comparison SHA could not be fetched — i.e. `upstream is None` — while a
    local stamp to compare it against DOES exist. A MISSING local stamp
    (`sf_sha`/`stack_sha` is `None`) is a distinct, already-measured state
    (first deploy before stamping shipped, or a legacy CF stack with no
    git-sha tag) and correctly stays `False` regardless of `upstream` —
    there is nothing to compare, which is a confirmed non-drift verdict,
    not an unmeasured one.

    alpha-engine-config-I8142 (2026-08-21): the same invariant, applied to the
    LIVE side. `describe_state_machine` answers both live-side questions — the
    `[git:]` stamp behind `deploy_stamp_stale` and the definition body behind
    `sf_drift` — so its failure took out both and was reported as
    `sf_drift: false` / `deploy_stamp_stale: false`, present and definite, off
    an AccessDenied. Both are now WITHDRAWN when the live read fails, the
    failure is typed (`LiveDefinitionError`, surfaced as
    `live_definition_error`), and the three states are stated positively in
    `sf_drift_state` / `cf_drift_state` / `deploy_stamp_stale_state` so no
    consumer can collapse "no" into "no answer" with a `.get()`.

    Previously `test_github_outage_is_no_drift`
    pinned the bug this fixes: stamps present, GitHub unreachable, yet
    `has_drift=False` — a live GitHub outage during the weekday preflight
    would have silently traded through an unverified deploy instead of
    halting per the fail-closed ruling this Choice was built to enforce.
    """
    sf_arn = _SF_ARN_TEMPLATE.format(
        region=region, account=account_id, name=sf_name,
    )

    # alpha-engine-config-I8142: ONE describe_state_machine, and its failure is
    # a typed UNKNOWN rather than a None both live-side verdicts silently
    # answer from. `live_definition` is the document or None; `live_read_error`
    # is non-None exactly when nothing about the live machine was measured.
    live_read = _read_live_definition(sf_arn)
    live_read_error: Optional[LiveDefinitionError] = (
        live_read if isinstance(live_read, LiveDefinitionError) else None
    )
    live_definition: Optional[dict] = None if live_read_error else live_read
    live_definition_unreadable = live_read_error is not None
    sf_comment = (
        "" if live_definition is None else (live_definition.get("Comment") or "")
    )
    # None means two different things and only one of them is measured, so the
    # distinction is carried in `live_definition_unreadable`, never inferred
    # from `sf_sha` being None.
    sf_sha = None if live_definition is None else _extract_sf_sha(sf_comment)
    stack_read = _read_stack_tag(stack_name)
    # alpha-engine-config-I7924: a rejected GITHUB_TOKEN no longer stops the
    # fetch (nousergon-lib v0.124.79 retries unauthenticated against these
    # public repos), but it is still a real defect that must reach the
    # operator. Carried in the payload so the SF's degrade alert can NAME it,
    # rather than surviving silently on the anonymous fallback.
    github_stats: dict = {}
    upstream = _fetch_origin_main_sha(repo, branch=branch, stats=github_stats)

    # sf_drift is UNMEASURED (omitted below) when there IS a local stamp to
    # compare and upstream could not be fetched, OR when the live machine could
    # not be read at all (I8142) — the second case is not "no stamp", it is "no
    # answer", and only the first is a confirmed non-drift verdict.
    sf_drift_unmeasured = live_definition_unreadable or (
        sf_sha is not None and upstream is None
    )
    stamp_drift = (
        upstream is not None
        and sf_sha is not None
        and not _shas_match(sf_sha, upstream)
    )

    # ── config-I7799: content comparison for the weekday pipelines ──────────
    # The stamp verdict is kept under its own name whatever happens below, so
    # a genuinely undeployed repo is still SAID, just not halted on.
    #
    # alpha-engine-config-I7924: like sf_drift/cf_drift, this is OMITTED when
    # it could not be measured. On 2026-08-21 the probe emitted
    # `deploy_stamp_stale: false` with `upstream_sha: null` — nothing had been
    # compared, yet the field read to every consumer exactly as a verified
    # "this repo has deployed everything it merged". That is the same
    # unmeasured-rendered-as-clean shape the sf_drift omission exists to
    # prevent (sf-pipeline-policy 2.3a rule 2); it did not cause that morning's
    # halt only because sf_drift's omission fired one branch earlier.
    #
    # alpha-engine-config-I8142 deliverable 2: it is ALSO omitted when the live
    # machine was unreadable, because the stamp it compares is read from that
    # same document. `deploy_stamp_stale: false, reason: no_git_sha_stamp_legacy`
    # off an AccessDenied reads as a verified "everything merged is deployed";
    # it is a describe call nobody was allowed to make.
    deploy_stamp_stale = stamp_drift
    deploy_stamp_stale_unmeasured = live_definition_unreadable or (
        sf_sha is not None and upstream is None
    )
    sf_definition_path = _SF_DEFINITION_PATHS.get(sf_name)
    sf_definition_compared = False
    sf_definition_reason = "stamp_only_pipeline"

    sf_definition_source = "none"
    # The `[git:<sha>]` stamp carried INSIDE the S3 copy, i.e. the deploy that
    # wrote it. None until an S3 read succeeds. config-I7927.
    s3_deploy_sha: Optional[str] = None
    deploy_stamp_stale_reason_override: Optional[str] = None

    if sf_definition_path is None:
        # Weekly / unknown pipeline: stamp semantics, unchanged.
        sf_drift = stamp_drift
    else:
        # `live_definition` was read ONCE at the top of this function, together
        # with the stamp it carries (I8142) — a second describe_state_machine
        # here could observe a different deploy than the stamp did.
        # ── alpha-engine-config-I7927 ──────────────────────────────────────
        # The EXPECTED definition comes from S3 first, and GitHub only as a
        # fallback. This is what takes a third party off the critical path of
        # the trading day.
        #
        # sf-pipeline-policy 2.4 supplies the reason it is sound: the repo is
        # the SOLE WRITER of the live definitions, via
        # nousergon-data/infrastructure/deploy-infrastructure.sh. That script
        # already stamps each definition and uploads THE SAME BYTES it feeds to
        # update-state-machine (its own comment: "the stamped JSON is what gets
        # uploaded to S3 AND fed to update-state-machine, so S3 copy and live
        # definition stay in lockstep"). So the deploy already wrote down the
        # answer this probe was re-deriving from the source of truth's UPSTREAM.
        # Measured 2026-08-21: the S3 object and the deployed definition were
        # byte-identical, Comment included.
        #
        # Reading it back is same-region, inside the trust path, and needs no
        # new grant — the predictor role already holds s3:GetObject on the whole
        # alpha-engine-research bucket.
        #
        # What this does NOT do, deliberately: `deploy_stamp_stale` still asks
        # GitHub for main's HEAD, because "has this repo merged something it has
        # not deployed" is a question only the upstream can answer. That verdict
        # DEGRADES rather than halts (I7799), and is omitted when unmeasured, so
        # GitHub stays on the degrading path and leaves the halting one.
        repo_definition = None
        if live_definition is not None:
            repo_definition = _fetch_s3_definition(sf_definition_path)
            if repo_definition is not None:
                sf_definition_source = "s3"
                # alpha-engine-config-I7927 deliverable 4 — the S3 copy's own
                # freshness. The deploy stamps `[git:<sha>] ` into the Comment of
                # the very bytes it uploads, so the artifact SAYS which deploy
                # wrote it. Read it out here: it is an in-region expected SHA,
                # which is the one thing cf_drift was still going to GitHub for.
                s3_deploy_sha = _extract_sf_sha(
                    repo_definition.get("Comment") or "",
                )
            elif upstream is not None:
                # S3 unreadable — fall back to the pre-I7927 GitHub path rather
                # than dropping straight to the stamp. Strictly more information
                # than the stamp, and it is what ran until today.
                repo_definition = _fetch_repo_definition(
                    repo, upstream, sf_definition_path,
                )
                if repo_definition is not None:
                    sf_definition_source = "github"

        if live_definition is None or repo_definition is None:
            # Could not compare.
            #
            # ── alpha-engine-config-I8142 ──────────────────────────────────
            # Until 2026-08-21 this fell back to the STAMP verdict on EVERY
            # unreadable path, justified as "the stamp halts strictly more
            # often, so the fallback can only be more conservative". That
            # reasoning holds only while the stamp is a MEASURED quantity, and
            # on the `live_definition is None` path it is not: the stamp and
            # the body come out of the same describe_state_machine call, so one
            # AccessDenied leaves `sf_sha=None`, which makes `stamp_drift`
            # False BY CONSTRUCTION. The gate then read a present, definite
            # `sf_drift: false` off an authorization failure and passed —
            # measured live 2026-08-21 against ne-postclose-trading-pipeline,
            # and reachable on the preopen halting gate by throttling, a
            # transient states: API failure, or any future scoping of the
            # DeployDriftProbe grant.
            #
            # I7799's own ruling text is explicit: "Missing or unfetchable →
            # UNKNOWN → fail closed, unchanged" (sf-pipeline-policy §2.3a rule
            # 2). So the stamp fallback survives ONLY where the stamp was
            # actually read; otherwise the verdict is withdrawn and the
            # IsPresent-guarded DeployDriftGate halts.
            sf_drift = stamp_drift
            if live_definition is None:
                sf_definition_reason = "live_definition_unreadable"
                # Already true via `live_definition_unreadable`; restated so
                # this branch's own postcondition is local and cannot be
                # loosened by a change above it.
                sf_drift_unmeasured = True
                log.error(
                    "The live definition of %s could not be read (%s) — "
                    "sf_drift and deploy_stamp_stale are WITHDRAWN as "
                    "unmeasured, not reported false. DeployDriftGate's "
                    "IsPresent guard will halt this run (I8142).",
                    sf_name,
                    live_read_error.detail if live_read_error else "unknown",
                )
            elif sf_sha is None:
                # I8142 deliverable 1, second half: the expected definition is
                # unavailable AND there is no stamp on the live side either —
                # nothing was measured on EITHER side, so there is no verdict
                # to report. (`sf_sha is None` here means the live machine was
                # read and carries no stamp; the unreadable case is the branch
                # above.)
                sf_drift_unmeasured = True
                sf_definition_reason = (
                    "expected_definition_unavailable" if upstream is None
                    else "repo_definition_unreachable"
                )
                log.error(
                    "No expected definition for %s (%s) and the live machine "
                    "carries no [git:] stamp to fall back on — sf_drift "
                    "withdrawn as unmeasured (I8142).",
                    sf_name, sf_definition_reason,
                )
            elif upstream is None:
                # Both sources gone: S3 unreadable AND no upstream SHA to fetch
                # a GitHub copy at. This is the 2026-08-21 shape plus an S3
                # failure — strictly rarer than it was, and still fail-closed.
                sf_definition_reason = "expected_definition_unavailable"
            else:
                sf_definition_reason = "repo_definition_unreachable"
        else:
            sf_definition_compared = True
            sf_drift = (
                _canonical_definition(live_definition)
                != _canonical_definition(repo_definition)
            )
            sf_definition_reason = (
                "definition_mismatch" if sf_drift else "definition_identical"
            )
            # The comparison IS the measurement; a fetched-and-compared pair
            # is never unmeasured, whatever the stamp says.
            sf_drift_unmeasured = False
            if deploy_stamp_stale and not sf_drift:
                log.warning(
                    "Deploy stamp is stale (live %s vs main %s) but the "
                    "%s definition is identical to the repo's — reporting "
                    "deploy_stamp_stale=true and sf_drift=false so the "
                    "pipeline degrades and pages instead of halting "
                    "(alpha-engine-config-I7799).",
                    sf_sha, upstream, sf_name,
                )

    # ── config-I7927 deliverable 4: is the S3 expectation the one that
    # produced the live machine? ────────────────────────────────────────────
    # The S3 copy is now load-bearing for a verdict that HALTS the trading day,
    # and nothing yet asserted it describes the deploy that is actually live.
    # The two are written by the same script but in different steps — upload in
    # step 2, `update-state-machine` in step 3 — and `check-definition-drift.py`
    # exists precisely because a third party can write to that key (out-of-band
    # console edit, aborted deploy, drive-by S3 write).
    #
    # The assertion needs no clock and no third party: both artifacts carry the
    # SHA of the deploy that wrote them, so they can be asked whether they agree.
    #
    # A disagreement is a MEASURED deploy incoherence, not an unmeasurable
    # state, so it does NOT withdraw the verdict — withdrawing it would turn a
    # rare inconsistency into a halted trading session, which is the harm I7799
    # exists to prevent. It routes to the DEGRADE channel instead, via
    # `deploy_stamp_stale`, whose DeployDriftGate branch pages without halting.
    # Same halt-vs-degrade split, one layer down; strictly additive, and it
    # weakens nothing: `sf_drift` is exactly what it was one line above.
    s3_copy_stamp_mismatch = (
        sf_definition_source == "s3"
        and sf_sha is not None
        and s3_deploy_sha is not None
        and not _shas_match(s3_deploy_sha, sf_sha)
    )
    if s3_copy_stamp_mismatch:
        deploy_stamp_stale = True
        deploy_stamp_stale_unmeasured = False
        deploy_stamp_stale_reason_override = "s3_copy_stamp_mismatch"
        log.error(
            "The S3 definition for %s was written by deploy %s but the live "
            "state machine is stamped %s — the expectation this probe just "
            "compared against is not the one that produced the running "
            "orchestration. Degrading and paging rather than halting "
            "(alpha-engine-config-I7927).",
            sf_name, s3_deploy_sha, sf_sha,
        )

    # Interpret stack_read tri-state:
    #   str               → healthy stack with tag; compare to upstream
    #   None              → healthy stack, tag absent (legacy-deploy warn)
    #   StackStateError   → stack not usable; cf_drift=true with reason code
    cf_drift_source: Optional[str] = None
    if isinstance(stack_read, StackStateError):
        stack_sha = None
        stack_stamp_present = False
        cf_drift = True
        cf_drift_unmeasured = False
        cf_drift_reason = stack_read.reason
        cf_drift_detail = stack_read.detail
        cf_stack_status = stack_read.stack_status
    else:
        stack_sha = stack_read  # str | None
        stack_stamp_present = stack_sha is not None
        cf_drift_detail = ""
        cf_stack_status = None
        if stack_sha is not None and s3_deploy_sha is not None:
            # ── config-I7927: cf_drift measured IN-REGION ──────────────────
            # This is the branch without which GitHub was still on the critical
            # path after the sf_drift half landed (crucible-predictor#538).
            # DeployDriftGate carries
            #   Not IsPresent($.drift_result.Payload.cf_drift) -> HandleFailure
            # so a GitHub outage omitted `cf_drift` and HALTED the trading day
            # regardless of how `sf_drift` was obtained. Both had to become
            # answerable without a third party, or neither did.
            #
            # The reference point moves from "main HEAD" to "the SHA the last
            # deploy stamped", which is what the CFN stack tag is written from
            # in the same run. A mismatch is therefore a partial or out-of-band
            # deploy — real drift, degrading exactly as before.
            #
            # The case this reference point drops is "the stack is consistent
            # with the last deploy but main has moved on". That is NOT lost: it
            # is `deploy_stamp_stale`, which routes to the SAME
            # SetDeployDriftDegradedFlag and pages the same way. Same outcome,
            # one honest cause each, per I7799's split.
            cf_drift_unmeasured = False
            cf_drift = not _shas_match(stack_sha, s3_deploy_sha)
            cf_drift_source = "s3"
            cf_drift_reason = "sha_mismatch" if cf_drift else "in_sync"
        else:
            # Pre-I7927 path, unchanged. Same unmeasured rule as sf_drift: a
            # real stack tag exists but no expected SHA could be obtained.
            cf_drift_unmeasured = stack_sha is not None and upstream is None
            cf_drift = (
                upstream is not None
                and stack_sha is not None
                and not _shas_match(stack_sha, upstream)
            )
            cf_drift_source = None if stack_sha is None else "github"
            cf_drift_reason = (
                "fetch_failed" if cf_drift_unmeasured else
                "sha_mismatch" if cf_drift else
                ("no_git_sha_tag_legacy" if stack_sha is None else "in_sync")
            )

    if sf_definition_compared:
        sf_drift_reason = sf_definition_reason
    else:
        sf_drift_reason = (
            # I8142: name the ACTUAL unmeasured cause. "fetch_failed" meant the
            # upstream fetch; an unreadable live machine is a different fault
            # with a different remedy (an IAM grant, not GitHub), and the page
            # has to say which.
            "live_definition_unreadable" if live_definition_unreadable else
            sf_definition_reason if (
                sf_drift_unmeasured and sf_definition_path is not None
            ) else
            "fetch_failed" if sf_drift_unmeasured else
            "sha_mismatch" if sf_drift else
            ("no_git_sha_stamp_legacy" if sf_sha is None else "in_sync")
        )

    result = {
        "repo": repo,
        "branch": branch,
        "upstream_sha": upstream,
        "sf_sha": sf_sha,
        "sf_stamp_present": sf_sha is not None,
        "sf_drift_reason": sf_drift_reason,
        # config-I7799. `sf_drift` answers "is the orchestration about to run
        # different from what main describes" and HALTS. `deploy_stamp_stale`
        # answers "has this repo merged something it has not deployed" and
        # DEGRADES — a merged Lambda, collector or script this pipeline invokes
        # but does not embed is genuinely undeployed, and dropping that signal
        # to fix the halt would be trading one blind spot for another. It is
        # emitted conditionally below (I7924), for the same reason sf_drift is.
        "sf_definition_path": sf_definition_path,
        "sf_definition_compared": sf_definition_compared,
        "sf_definition_reason": sf_definition_reason,
        # alpha-engine-config-I7927: WHICH source supplied the expected
        # definition — "s3" (the deploy's own record, the intended path and the
        # one with no third party on it), "github" (fallback), or "none" (no
        # comparison happened; sf_definition_reason says why). Emitted always,
        # because "the halting verdict quietly started depending on GitHub
        # again" is exactly the regression this field exists to make visible.
        "sf_definition_source": sf_definition_source,
        "stack_sha": stack_sha,
        "stack_stamp_present": stack_stamp_present,
        "cf_drift_reason": cf_drift_reason,
        # config-I7927: WHICH expectation cf_drift was measured against —
        # "s3" (in-region, the deploy's own record), "github" (the pre-I7927
        # fallback), or None when there was no stack tag to compare. Emitted so
        # "the halting gate's other field quietly went back to needing GitHub"
        # is visible on the payload rather than inferred from a source read.
        "cf_drift_source": cf_drift_source,
        # The `[git:<sha>]` stamp inside the S3 expectation, i.e. the deploy
        # that wrote it. Compared against `sf_sha` (the live machine's stamp) to
        # assert the expectation describes the deploy that is actually running.
        "s3_deploy_sha": s3_deploy_sha,
        "cf_drift_detail": cf_drift_detail,
        "cf_stack_status": cf_stack_status,
        # alpha-engine-config-I7048: OMIT sf_drift/cf_drift entirely when
        # unmeasured — see the module/function docstrings. The SF's
        # DeployDriftGate Choice is already IsPresent-guarded to fail
        # CLOSED on an absent key (Brian's 2026-08-09 ruling, config#6615);
        # a present `False` here would silently defeat that guard.
        "has_drift": bool(
            (not sf_drift_unmeasured and sf_drift)
            or (not cf_drift_unmeasured and cf_drift)
        ),
    }
    if not sf_drift_unmeasured:
        result["sf_drift"] = sf_drift
    if not cf_drift_unmeasured:
        result["cf_drift"] = cf_drift
    # alpha-engine-config-I7924: same omit-when-unmeasured rule. The
    # DeployDriftGate branch reading this is NOT IsPresent-guarded toward
    # failure by design (see step_function_daily.json), so an absent key
    # correctly declines to degrade rather than degrading on a fact nobody
    # established — while `deploy_stamp_stale_reason` still says why.
    if not deploy_stamp_stale_unmeasured:
        result["deploy_stamp_stale"] = deploy_stamp_stale
    result["deploy_stamp_stale_reason"] = (
        # config-I7927: an S3 expectation written by a different deploy than the
        # live machine is a distinct fault from "main has moved", and the alert
        # must say which one fired.
        deploy_stamp_stale_reason_override
        if deploy_stamp_stale_reason_override is not None else
        "live_definition_unreadable" if live_definition_unreadable else
        "fetch_failed" if deploy_stamp_stale_unmeasured else
        "sha_mismatch" if deploy_stamp_stale else
        ("no_git_sha_stamp_legacy" if sf_sha is None else "in_sync")
    )
    # ── alpha-engine-config-I8142 deliverable 3 ──────────────────────────────
    # THREE states, said in one field each, always present. The booleans above
    # keep their omit-when-unmeasured contract because the SF Choice states are
    # IsPresent-guarded on them and that is the halting mechanism; but a
    # boolean that carries "no" and "no answer" in the same value is the shape
    # that produced every instance of this defect class, and an omitted key is
    # only as safe as the next consumer's `.get()`. These fields make the third
    # state POSITIVE — a consumer that reads them cannot collapse it by
    # accident, and one that renders a run's state has something to render.
    result["sf_drift_state"] = (
        "unmeasured" if sf_drift_unmeasured else ("drift" if sf_drift else "no_drift")
    )
    result["cf_drift_state"] = (
        "unmeasured" if cf_drift_unmeasured else ("drift" if cf_drift else "no_drift")
    )
    result["deploy_stamp_stale_state"] = (
        "unmeasured" if deploy_stamp_stale_unmeasured
        else ("stale" if deploy_stamp_stale else "in_sync")
    )
    # Present only when the live machine could not be read, so its absence is
    # never mistaken for a describe call that succeeded.
    if live_read_error is not None:
        result["live_definition_error"] = live_read_error.to_dict()
    # Present ONLY when the credential was actually rejected, so its absence
    # is never mistaken for a token that was checked and found healthy.
    if github_stats.get("github_credential_rejected"):
        result["github_credential_rejected"] = True
        result["github_credential_status"] = github_stats.get(
            "github_credential_status",
        )
    return result
