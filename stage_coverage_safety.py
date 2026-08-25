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
copied into a second `deploy.sh`. Checked 2026-08-25: no other fleet repo
calls `safe_assert_stage_coverage` or an equivalent, so there is nothing to
lift for yet.

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
SYNTHETIC_INVOCATION_KINDS = frozenset({"canary"})


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


def unmeasured_stage_coverage(
    stage: str, reason: str, *, window_start: datetime | None = None
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
        )

    run_date = resolve_event_run_date(event)
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
        return assert_stage_coverage(stage, run_date=run_date, window_start=window_start)
    except _ContractError as exc:
        log.error(
            "stage-coverage assertion raised for %s (run_date=%r): %s",
            stage, run_date, exc, exc_info=True,
        )
        return unmeasured_stage_coverage(
            stage, f"assert_stage_coverage raised: {exc}", window_start=window_start
        )
