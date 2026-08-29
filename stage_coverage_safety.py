"""Safe wiring for `krepis.stage_coverage.assert_stage_coverage` calls.

alpha-engine-config-I8155. Three Lambda handlers in this repo
(`inference.handler`, `regime.handler`, `regime.retrospective_eval_handler`)
each call `assert_stage_coverage` immediately before returning, merging its
verdict into the response under `stage_coverage`. This module is the ONE
place that logic lives (`policy-shared-code`: third adoption is the lift
trigger, and all three call sites need the identical fix) rather than a
hand-copied version of it in each handler.

**The fleet mechanism.** The agreed fix for alpha-engine-config-I8155 is
`EXECUTION_RUN_DATE` — the Step Functions execution's own `run_date`, taken
from the event BEFORE any trading-day/cycle normalization, never a value
this process computed itself (`datetime.now()`, `now_dual().trading_day`,
a calendar-derived `calendar_date`).

Measured against `nousergon-data/infrastructure/step_function.json`
(2026-08-22): the SF Payloads for WeeklyRunDayGate, LibPinDriftCheck,
PipelineContractCheck, RegimeSubstrate and RegimeRetrospectiveEval carried
ONLY `action` — no date field at all. Every one of those five nonetheless
wrote a plausible-looking run_date, because each handler substituted
`datetime.now(timezone.utc).date()` or a calendar date derived from it,
which equals the execution's `$.run_date` exactly while the stage runs on
the same UTC day the execution started. Right for the wrong reason: an
execution beginning 23:50 UTC, a stage crossing midnight, or any redrive of
an older run_date splits them, invisibly, because the verdict still lands
under a plausible prefix.

`nousergon-data-PR1510` threads `"run_date.$": "$.run_date"` into all five
Payloads, so `resolve_event_run_date` below finds a real value on the live
path. It returns `None` only for a manual or off-SF invocation that supplied
none — and the correct behaviour there is to record UNMEASURED loudly rather
than fabricate a stand-in, which is the substitution this whole arc removes.

**Synthetic invocations carry no execution identity, by design.**
`infrastructure/deploy.sh`'s canary matrix invokes three of these handlers
(`check_weekly_run_day`, `check_lib_pin_drift`, `check_pipeline_contract`)
against a freshly published Lambda VERSION, off the Step Function entirely.
Such an invocation has no `$.run_date` to thread and must never acquire one:
it is not a run, so it has no run to be attributed to. Measured on the live
log group 2026-08-25: every predictor deploy emitted three `log.error` lines
from the `run_date is None` branch below, one per canaried gate action
(versions 528, 529, 530, 531 — one burst per deploy). `handler.py` attaches
flow-doctor's handler at ERROR, so each burst became three pages against a
`max_alerts_per_day: 10` budget (`flow-doctor.yaml`) — a detector spending
30% of the day's alert budget on invocations that are healthy by
construction, and training its reader to ignore the one shape it exists to
catch.

The fix is a DECLARATION, not a silence: `run_canary_action` stamps
`invocation_kind: "canary"` into every canary payload (one chokepoint, so no
call site can drift out of it), and this module refuses to assert coverage
for any synthetic invocation — before it even looks for a run_date, so a
synthetic invocation that somehow carried one still cannot write a verdict
into a real execution's prefix. A real SF invocation missing its run_date is
untouched by this and still pages, which is the half that had to survive.

First adoption of the marker is this repo. `policy-shared-code`'s trigger
is the SECOND adoption: when another repo's canary needs to declare itself,
the stamp lifts into `krepis.aws.invoke-canary` — which is the only place
that knows an invocation is a canary without being told — rather than being
copied into a second `deploy.sh`.

**A SECOND synthetic caller arrived, and it declared nothing.**
`alpha-engine-config/scripts/check_prespend_gate_arm.py` (merged 2026-08-28,
`alpha-engine-config-I9168`) probes each pre-spend gate by invoking its
Lambda with **the Task's own Payload and `$.run_date` filled from
`next_run_date()`** (`_payload_for` at `:455`, `probe_gate` at `:496`). That
is precisely the shape `is_synthetic_invocation`'s docstring warned about —
a caller with a real-looking run_date and no execution behind it — so its
probes wrote real verdict objects into a real execution's prefix. Measured
2026-08-29 on `s3://alpha-engine-research/_stage_coverage/2026-08-28/`:
`LibPinDriftCheck`, `PipelineContractCheck` and `EvaluatorDeployDriftCheck`
rewritten at 03:18:34–39Z and `WeeklyRunDayGate` /
`EvaluatorDirectorDeployDriftCheck` at 02:14Z, by no execution at all. That
matters because `alpha-engine-config-I8155`'s own gate predicate is an
OBJECT COUNT over that prefix: a probe write inflates the denominator's
numerator and lets the gate clear green on verdicts no run produced.

Two halves land here. `"probe"` joins the closed synthetic set (the config
script stamps it at its own single payload chokepoint), which stops the
write. And every verdict this module causes to be written now carries
`execution_arn` and `invocation_kind` — because until it did, a probe write
and a real write were **indistinguishable in the artifact**: every object in
that partition read `execution_arn: None`, including `SignalsEnvelope`,
which no probe touched. A verdict nobody can attribute cannot be excluded
from a count by anyone, ever, which is why the marker alone was not enough.

`policy-shared-code`'s lift trigger is now met twice over
(`crucible-backtester-PR752` keys the same exemption on `dry_run`; this repo
keys it on `invocation_kind`). The SOTA home for both halves is
`krepis.stage_coverage` itself: attribution as first-class `StageVerdict`
fields, and a refusal to record any verdict lacking an execution ARN — the
layer that knows an invocation is synthetic without being told. This
module's per-caller marker is the stopgap, not the answer.

**krepis is changing in the same arc**: `run_date` becomes a required
keyword and a blank value raises `krepis.stage_coverage
.StageCoverageContractError`. krepis merges LAST, so this repo must be
correct against BOTH the current (optional, blank-tolerant) and the new
(required, raising) shape. `safe_assert_stage_coverage` below never lets
either shape's failure escape the handler it is only supposed to be
observing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

#: The stage_coverage `status` this module's own degrade path reports.
#: Mirrors `krepis.stage_coverage.STATUS_UNMEASURED` without importing it
#: (the pinned SHA may predate the module entirely — see
#: `resolve_event_run_date`'s ImportError path).
STATUS_UNMEASURED = "UNMEASURED"

#: Event key carrying the KIND of this invocation. Absent on every Step
#: Functions Task Payload (the SF threads `run_date`, not this) and stamped
#: by `infrastructure/deploy.sh::run_canary_action` on every canary payload.
INVOCATION_KIND_KEY = "invocation_kind"

#: Invocation kinds that are synthetic — a probe of the deployed artifact
#: rather than a run of the pipeline. Closed set: anything else, including an
#: absent/blank value, is treated as a real invocation, so a new probe kind
#: has to be added here deliberately rather than inheriting the exemption.
#: `"probe"` is stamped by `alpha-engine-config/scripts/check_prespend_gate_arm.py`
#: at its own `_payload_for` chokepoint; `"canary"` by
#: `infrastructure/deploy.sh::run_canary_action`.
SYNTHETIC_INVOCATION_KINDS = frozenset({"canary", "probe"})

#: Event key carrying the Step Functions execution ARN (`$$.Execution.Id`),
#: the fleet's established name for it — `nousergon-data`'s
#: `step_function.json` already threads `"execution_arn.$": "$$.Execution.Id"`
#: at four states. `execution_id` is accepted as an alias because two of those
#: states spell it that way.
EXECUTION_ARN_KEY = "execution_arn"
EXECUTION_ARN_ALIASES = ("execution_arn", "execution_id")

#: S3 key prefix `krepis.stage_coverage` writes verdict records under.
#: Mirrored rather than imported because every krepis import in this module is
#: defensive (the pinned SHA may predate the module) and this constant is read
#: on a path that must work regardless. `policy-shared-code` §4 sanctions a
#: mirrored constant ONLY with a contract test proving the copies match —
#: `tests/test_stage_coverage_attribution.py::test_the_mirrored_verdict_prefix_matches_krepis`.
VERDICT_KEY_PREFIX = "_stage_coverage/"


class StageCoverageAttributionError(RuntimeError):
    """A verdict record was about to be written that could not be attributed.

    Raised only from :class:`_AttributingS3Client`, and only when the body it
    was handed is not the JSON object every verdict record is. Deliberately
    NOT swallowed here: `record_verdict` already wraps its `put_object` in a
    fail-soft `except` that logs at ERROR, so raising surfaces the contract
    break loudly on the handler's log group without destroying the stage's
    own deliverable.
    """


def invocation_kind(event: dict[str, Any]) -> str:
    """Return `event`'s declared invocation kind, normalized (`""` if absent)."""
    value = event.get(INVOCATION_KIND_KEY)
    return str(value).strip().lower() if isinstance(value, str) else ""


def is_synthetic_invocation(event: dict[str, Any]) -> bool:
    """True when `event` DECLARES itself a probe of the deployed artifact.

    Read before `resolve_event_run_date`, deliberately: the rule is not "a
    canary may skip the assertion when it happens to lack a run_date" but
    "a synthetic invocation never produces a coverage verdict at all". A
    canary that somehow carried a run_date would otherwise write a verdict
    into a real execution's prefix and count toward its denominator.
    """
    return invocation_kind(event) in SYNTHETIC_INVOCATION_KINDS


def resolve_execution_arn(event: dict[str, Any]) -> str | None:
    """Return the SF execution ARN `event` carries, or `None`.

    Never fabricated and never derived: an invocation that did not arrive
    with an execution identity does not have one. `None` is the honest and
    load-bearing answer — it is what marks a verdict as unattributable, which
    is the whole point of recording the field.
    """
    for key in EXECUTION_ARN_ALIASES:
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def invocation_attribution(event: dict[str, Any]) -> dict[str, Any]:
    """Return the attribution fields stamped onto `event`'s verdict record.

    `execution_arn` answers "which run produced this verdict"; `invocation_kind`
    answers "what kind of caller produced it". Both are `None` when absent —
    never `""`, so a consumer's `IS NOT NULL` / `.execution_arn != null` filter
    reads the same in S3 Select, jq and Athena.
    """
    kind = invocation_kind(event)
    return {
        EXECUTION_ARN_KEY: resolve_execution_arn(event),
        INVOCATION_KIND_KEY: kind or None,
    }


class _AttributingS3Client:
    """An S3 client that stamps attribution onto every verdict record written.

    `krepis.stage_coverage.record_verdict` serialises
    `StageVerdict.to_dict()` and `put_object`s it; `StageVerdict` has no
    attribution fields, and this repo cannot add them to another repo's
    dataclass. Injecting at the client is the one seam that reaches the
    ACTUAL written bytes rather than only the dict this module returns — and
    the artifact is what a gate predicate reads, not the Lambda response.

    Every other call is proxied through untouched, so the registry read and
    the per-artifact `head_object` probes behave exactly as before.

    This is a STOPGAP. See the module docstring: the fields belong on
    `StageVerdict` in krepis, and the refusal-to-record belongs beside them.
    """

    def __init__(self, inner: Any, attribution: dict[str, Any]) -> None:
        self._inner = inner
        self._attribution = attribution

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def put_object(self, **kwargs: Any) -> Any:
        key = kwargs.get("Key") or ""
        if isinstance(key, str) and key.startswith(VERDICT_KEY_PREFIX):
            kwargs["Body"] = self._stamp(key, kwargs.get("Body"))
        return self._inner.put_object(**kwargs)

    def _stamp(self, key: str, body: Any) -> bytes:
        import json

        if isinstance(body, bytes):
            raw = body.decode("utf-8")
        elif isinstance(body, str):
            raw = body
        else:
            raise StageCoverageAttributionError(
                f"verdict record {key!r} was handed a {type(body).__name__} body; "
                "attribution cannot be stamped onto it and an unattributable "
                "verdict must not be written silently (alpha-engine-config-I8155)"
            )
        try:
            record = json.loads(raw)
        except ValueError as exc:
            raise StageCoverageAttributionError(
                f"verdict record {key!r} is not JSON ({exc}); attribution "
                "cannot be stamped onto it"
            ) from exc
        if not isinstance(record, dict):
            raise StageCoverageAttributionError(
                f"verdict record {key!r} deserialised to "
                f"{type(record).__name__}, not an object"
            )
        record.update(self._attribution)
        return json.dumps(record, indent=2, default=str).encode()


def _attributing_s3_client(attribution: dict[str, Any], log: logging.Logger) -> Any:
    """Build the attributing S3 client, or `None` if boto3/krepis cannot.

    Returning `None` hands `assert_stage_coverage` its own default client —
    the verdict is still written, just without attribution.
    """
    try:
        import boto3

        from krepis.aws_region import resolve_region

        return _AttributingS3Client(
            boto3.client("s3", region_name=resolve_region()), attribution
        )
    except Exception:  # noqa: BLE001
        # Recorded, not swallowed (fleet "fail loud" rule): (a) the failure
        # mode is boto3 or `krepis.aws_region` being unimportable, or client
        # construction failing outright; (b) the handler's own deliverable —
        # the gate verdict — is unaffected, and the coverage verdict itself is
        # still written by krepis's own client, only without attribution;
        # (c) recorded on the handler's log group at ERROR, which flow-doctor
        # pages from.
        log.error(
            "stage-coverage attribution unavailable: could not build the "
            "attributing S3 client, so this verdict will be written with no "
            "execution_arn/invocation_kind and will be indistinguishable from "
            "a probe write (alpha-engine-config-I8155).",
            exc_info=True,
        )
        return None


def resolve_event_run_date(event: dict[str, Any]) -> str | None:
    """Return the SF execution's own run_date from `event`, un-normalized.

    `event["run_date"]` FIRST: that is the key the SF wires directly to
    `$.run_date` (nousergon-data-PR1510) and it means exactly one thing —
    the execution's own identity. `event["date"]` is the fallback, this
    repo's older convention for the event's raw, un-normalized date (see
    `inference.handler`'s top-level `date_str = event.get("date", None)` and
    the WeeklyRunDayGate branch's `check_weekly_run_day(event.get("date"))`,
    both already threading the raw event value through unchanged). The
    precedence matters: `date` is overloaded across this repo's call paths
    and can carry a cycle date on some of them, while `run_date` never does.

    Never a value this process computed — no `now()`, no cycle/calendar-date
    substitution. Returns `None` when the event carries neither, which the
    caller must treat as "execution identity absent" rather than fabricate a
    stand-in date.
    """
    value = event.get("run_date") or event.get("date")
    return value or None


def resolve_coverage_partition_date(event: dict[str, Any]) -> str | None:
    """Return the S3 PARTITION date for `event`'s coverage verdict.

    This is `resolve_event_run_date` normalized to the cycle's TRADING day —
    the one partition family `_stage_coverage/` has
    (`alpha-engine-config-I8809`, `partition_family: trading_day`).

    **Why the two dates had to be separated (alpha-engine-config-I8984).**
    The weekly Step Function normalizes `$.run_date` once, at
    `NormalizeRunDates`. `WeeklyRunDayGate` runs strictly BEFORE that state
    by construction — its only entry is `CheckWeeklyRunDayGate`, whose
    `Default` IS the normalizer — so the event reaching this Lambda on that
    path still carries the CALENDAR date. That is correct and deliberate for
    the gate's own arithmetic: it asks "was YESTERDAY the week's last trading
    session", and the trading day is the answer it computes, never its input.
    Its S3 prefix is a different question, and the payload conflated them.
    Measured on the 2026-08-22 cycle: `WeeklyRunDayGate.json` under BOTH
    `_stage_coverage/2026-08-21/` and `_stage_coverage/2026-08-22/`.

    **Why this is not the config-I8155 substitution.** I8155 forbids a value
    this PROCESS computed — `datetime.now()`, a self-invented stand-in for an
    execution identity that was never supplied. This is a pure, total,
    deterministic function OF the event's own date; it invents nothing, and
    `None` in still yields `None` out. `resolve_event_run_date` keeps its
    verbatim meaning for every non-partition reader.

    **Why at this chokepoint rather than in the WeeklyRunDayGate branch.**
    `krepis.dates.resolve_trading_day` is IDEMPOTENT BY CONTRACT — a
    trading-day input returns unchanged — so every already-normalized call
    site (LibPinDriftCheck, PipelineContractCheck, RegimeSubstrate,
    RegimeRetrospectiveEval, all downstream of `NormalizeRunDates`) is a
    no-op, and a fifth handler adopting `safe_assert_stage_coverage` inherits
    the partition family instead of re-deciding it. Fixing only the one
    branch would fix the instance and leave the class.

    Degrades to the raw value, loudly, if krepis cannot normalize it — never
    to today's date, which is the failure mode this whole arc removes.
    """
    run_date = resolve_event_run_date(event)
    if run_date is None:
        return None
    try:
        from krepis.dates import resolve_trading_day

        return resolve_trading_day(run_date)
    except Exception:  # noqa: BLE001
        # Recorded, not swallowed (fleet "fail loud" rule): (a) the failure
        # mode is an unparseable/uncalendarable event date or a krepis SHA
        # predating `resolve_trading_day`; (b) the handler's own deliverable
        # — the gate verdict — is unaffected, this is an observability side
        # effect; (c) recorded on the handler's log group at ERROR, which
        # flow-doctor pages from.
        logging.getLogger(__name__).error(
            "coverage partition normalization failed for run_date=%r "
            "(alpha-engine-config-I8984); using the raw event date. The "
            "verdict may land in the calendar partition and read as `absent`.",
            run_date, exc_info=True,
        )
        return run_date


def unmeasured_stage_coverage(
    stage: str,
    reason: str,
    *,
    window_start: datetime | None = None,
    attribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the `stage_coverage` UNMEASURED payload without calling krepis.

    Used when the event carries no execution run_date to assert against, and
    as the degrade target when `assert_stage_coverage` itself raises. Mirrors
    the field shape of `krepis.stage_coverage.StageVerdict.to_dict()` so a
    downstream reader of the `stage_coverage` key sees the same shape
    regardless of which path produced it.
    """
    return {
        "stage": stage,
        "status": STATUS_UNMEASURED,
        "stage_class": "",
        "declared_output": "",
        "expected": [],
        "missing": [],
        "stale": [],
        "covered": [],
        "unmeasured": [],
        "reason": reason,
        "run_date": "",
        "cycle_date": "",
        "window_start": window_start.astimezone(timezone.utc).isoformat()
        if window_start
        else None,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "enforce": False,
        "is_finding": False,
        # Attribution travels with the degrade path too: a reader of the
        # handler response must be able to answer "which run, which kind of
        # caller" on EVERY shape, not only the one that reached S3.
        **(attribution or {EXECUTION_ARN_KEY: None, INVOCATION_KIND_KEY: None}),
    }


def safe_assert_stage_coverage(
    stage: str,
    *,
    event: dict[str, Any],
    window_start: datetime | None,
    log: logging.Logger,
) -> dict[str, Any] | None:
    """Call `krepis.stage_coverage.assert_stage_coverage` without ever
    raising out of the handler it observes.

    Returns the verdict dict to merge into the handler's response under
    `stage_coverage`, an UNMEASURED dict (see `unmeasured_stage_coverage`)
    when the invocation declares itself synthetic (see
    `is_synthetic_invocation`), when the event carries no run_date, or when
    the call itself raised, or `None`
    when `krepis.stage_coverage` is not importable at all (the pinned SHA
    predates it) — in which case the caller should log and leave
    `stage_coverage` out of the response entirely, matching the existing
    ImportError convention at every call site.
    """
    attribution = invocation_attribution(event)

    if is_synthetic_invocation(event):
        # Not a defect and not a run: a deploy-time canary probing the freshly
        # published Lambda version. INFO, not ERROR — `handler.py` attaches
        # flow-doctor at ERROR, and this path fires on every deploy by
        # construction (see the module docstring's measurement).
        log.info(
            "stage-coverage assertion not applicable for %s: invocation_kind=%r "
            "is synthetic (alpha-engine-config-I8155) — a probe of the deployed "
            "artifact, not a pipeline run, so there is no execution to attribute "
            "a verdict to. No verdict written.",
            stage, invocation_kind(event),
        )
        return unmeasured_stage_coverage(
            stage,
            f"synthetic invocation (invocation_kind="
            f"{invocation_kind(event)!r}) for stage {stage!r} — a probe of the "
            "deployed artifact carries no execution identity by design "
            "(alpha-engine-config-I8155)",
            window_start=window_start,
            attribution=attribution,
        )

    # alpha-engine-config-I8984: the PARTITION date, not the event's raw
    # date. See `resolve_coverage_partition_date` for why they differ on the
    # WeeklyRunDayGate path, and why normalizing here is not the I8155
    # substitution this module exists to prevent.
    run_date = resolve_coverage_partition_date(event)
    if run_date is None:
        log.error(
            "stage-coverage assertion skipped for %s: event carries no "
            "run_date (alpha-engine-config-I8155) — the SF execution's own "
            "identity is absent, so no verdict can be attributed to a run. "
            "Recording UNMEASURED rather than fabricating a date.",
            stage,
        )
        return unmeasured_stage_coverage(
            stage,
            f"event carried no run_date for stage {stage!r} — execution "
            "identity absent (alpha-engine-config-I8155)",
            window_start=window_start,
            attribution=attribution,
        )

    try:
        from krepis.stage_coverage import assert_stage_coverage
    except ImportError as exc:
        log.error("stage-coverage assertion unavailable for %s: %s", stage, exc)
        return None

    # alpha-engine-config-I8155: krepis is changing `run_date` from optional
    # to a required keyword that raises `StageCoverageContractError` on a
    # blank value, and this repo must work against both the pre- and
    # post-change shape (krepis merges LAST in this arc). Import the
    # exception defensively — the pinned SHA may predate it — and fall back
    # to catching broadly: the whole point of this wrapper is that NOTHING
    # this call can raise may kill the stage it is only supposed to observe.
    try:
        from krepis.stage_coverage import StageCoverageContractError as _ContractError
    except ImportError:
        _ContractError = Exception  # pinned krepis predates the typed exception

    try:
        verdict = assert_stage_coverage(
            stage,
            run_date=run_date,
            window_start=window_start,
            s3_client=_attributing_s3_client(attribution, log),
        )
    except _ContractError as exc:
        log.error(
            "stage-coverage assertion raised for %s (run_date=%r): %s",
            stage, run_date, exc, exc_info=True,
        )
        return unmeasured_stage_coverage(
            stage,
            f"assert_stage_coverage raised: {exc}",
            window_start=window_start,
            attribution=attribution,
        )

    if isinstance(verdict, dict):
        verdict.update(attribution)
    return verdict
