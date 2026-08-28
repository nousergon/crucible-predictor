"""The coverage partition family contract (alpha-engine-config-I8984).

`_stage_coverage/{date}/` has ONE partition family for a whole cycle: the
TRADING day (`alpha-engine-config-I8809`, `partition_family: trading_day`).
The weekly Step Function normalizes `$.run_date` to that family at
`NormalizeRunDates` — but `WeeklyRunDayGate` runs strictly BEFORE the
normalizer by construction, so the event this Lambda receives on that path
still carries the CALENDAR date. That is correct for the gate's own
arithmetic ("was YESTERDAY the week's last trading session"), and wrong for
the S3 prefix its coverage verdict is written under.

Measured on the 2026-08-22 cycle before this fix: `WeeklyRunDayGate.json`
present in BOTH `_stage_coverage/2026-08-21/` and `_stage_coverage/
2026-08-22/`. Invisible while `nousergon_lib.pipeline_status.coverage
.read_coverage_sweep` unions the two families; a genuine weekly `absent` —
the row state that pages with no threshold — from the 2026-09-05 cutover
(`alpha-engine-config-I8983`) onward.

These tests pin the resolution at this repo's single chokepoint, so a fourth
handler adopting `safe_assert_stage_coverage` inherits it rather than
re-deciding.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest
from krepis.dates import resolve_trading_day

from stage_coverage_safety import (
    resolve_coverage_partition_date,
    resolve_event_run_date,
    safe_assert_stage_coverage,
)


# ── The resolver itself ────────────────────────────────────────────────────

def test_a_calendar_saturday_resolves_to_the_cycles_trading_day() -> None:
    """The exact 2026-08-22 split this issue was filed for."""
    assert resolve_coverage_partition_date({"run_date": "2026-08-22"}) == "2026-08-21"


def test_the_resolution_is_idempotent_for_every_already_normalized_stage() -> None:
    """Post-`NormalizeRunDates` stages are untouched — this is why the fix
    can live at the shared chokepoint instead of one branch of the handler."""
    for trading_day in ("2026-08-21", "2026-08-28", "2026-08-27"):
        assert resolve_coverage_partition_date({"run_date": trading_day}) == trading_day


def test_the_event_date_fallback_is_normalized_too() -> None:
    """`event["date"]` is this repo's older raw-date convention and reaches
    the same S3 prefix, so it cannot be exempt."""
    assert resolve_coverage_partition_date({"date": "2026-08-22"}) == "2026-08-21"


def test_an_absent_run_date_stays_absent_and_is_never_fabricated() -> None:
    """config-I8155's rule survives: no self-computed stand-in date."""
    assert resolve_coverage_partition_date({}) is None
    assert resolve_coverage_partition_date({"run_date": ""}) is None


def test_an_unparseable_run_date_degrades_to_the_raw_value_not_to_now() -> None:
    """A garbage date must not silently become today's partition."""
    assert resolve_coverage_partition_date({"run_date": "not-a-date"}) == "not-a-date"


def test_it_never_disagrees_with_the_fleets_canonical_normalizer() -> None:
    """`resolve_coverage_partition_date` is `krepis.dates.resolve_trading_day`
    applied to the event's own date — never a second implementation of the
    NYSE calendar (`policy-shared-code`)."""
    for raw in ("2026-08-22", "2026-08-21", "2026-09-05", "2026-07-04"):
        assert resolve_coverage_partition_date({"run_date": raw}) == resolve_trading_day(raw)


# ── The chokepoint that writes the verdict ─────────────────────────────────

def test_the_verdict_is_asserted_against_the_trading_day_partition(monkeypatch) -> None:
    """The regression: `safe_assert_stage_coverage` must hand krepis the
    TRADING day, so exactly one `WeeklyRunDayGate.json` is written per cycle."""
    seen: dict[str, object] = {}

    def _fake_assert(stage, *, run_date, window_start):
        seen["stage"] = stage
        seen["run_date"] = run_date
        return {"stage": stage, "status": "COVERED_NO_OUTPUT", "run_date": run_date}

    import krepis.stage_coverage as ksc
    monkeypatch.setattr(ksc, "assert_stage_coverage", _fake_assert)

    verdict = safe_assert_stage_coverage(
        "WeeklyRunDayGate",
        event={"action": "check_weekly_run_day", "run_date": "2026-08-22"},
        window_start=datetime.now(timezone.utc),
        log=logging.getLogger(__name__),
    )

    assert seen["stage"] == "WeeklyRunDayGate"
    assert seen["run_date"] == "2026-08-21", (
        "WeeklyRunDayGate's verdict must land in the cycle's trading-day "
        "partition (alpha-engine-config-I8984); writing it under the calendar "
        "date is a genuine weekly `absent` after the I8983 cutover."
    )
    assert verdict is not None and verdict["run_date"] == "2026-08-21"


def test_a_synthetic_invocation_still_writes_no_verdict_at_all(monkeypatch) -> None:
    """config-I8155's canary carve-out is upstream of the resolution and
    stays that way — normalizing a date is not a licence to record one."""
    def _explode(*a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("a canary must never reach assert_stage_coverage")

    import krepis.stage_coverage as ksc
    monkeypatch.setattr(ksc, "assert_stage_coverage", _explode)

    verdict = safe_assert_stage_coverage(
        "WeeklyRunDayGate",
        event={"run_date": "2026-08-22", "invocation_kind": "canary"},
        window_start=None,
        log=logging.getLogger(__name__),
    )
    assert verdict is not None and verdict["status"] == "UNMEASURED"


def test_resolve_event_run_date_is_left_un_normalized(monkeypatch) -> None:
    """The two dates stay separable: `resolve_event_run_date` remains the
    execution's own identity, verbatim. `check_weekly_run_day`'s calendar
    arithmetic reads `event["date"]` directly and MUST NOT be normalized —
    normalizing it inverts the gate and the weekly run silently skips."""
    assert resolve_event_run_date({"run_date": "2026-08-22"}) == "2026-08-22"


def test_the_gates_calendar_arithmetic_is_unchanged_by_this_fix() -> None:
    """`WeeklyRunDayGateChoice` must still route THU/FRI to the skip branch."""
    from inference.trading_day_gate import check_weekly_run_day

    # 2026-08-22 is a Saturday: yesterday (Fri 08-21) WAS the week's last session.
    assert check_weekly_run_day("2026-08-22")["is_weekly_run_day"] is True
    # 2026-08-21 is a Friday: yesterday (Thu 08-20) was not.
    assert check_weekly_run_day("2026-08-21")["is_weekly_run_day"] is False
