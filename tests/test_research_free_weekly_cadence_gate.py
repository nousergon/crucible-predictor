"""``inference.stages.research_free`` must not page for a MISSING
candidates.json on an ordinary (non-canary, non-dry_run) weekday that is not
the weekly-SF run day (alpha-engine-config-I10301).

Distinct from ``tests/test_research_free_stage_dry_run.py`` (alpha-engine-
config-I10146), which fixed the deploy-canary/dry_run instance of this same
underlying mismatch. That fix only guarded the ``dry_run``/``local`` path.
It did nothing for a REAL, scheduled, non-dry_run weekday invocation — and
the weekday preopen SF (``ne-preopen-trading-pipeline``) invokes
``PredictorInference`` (and this stage, wired into it since
alpha-engine-config-I10067) Mon-Fri every trading morning, while
``candidates/{date}/candidates.json`` is written only on the weekly
scanner's self-selected THU-SAT run day (config#1824's WeeklyRunDayGate
predicate, ``ne-weekly-freshness-pipeline`` — a SEPARATE state machine this
stage is never part of). Measured: every non-run-day weekday invocation of
this stage since it was wired (2026-08-22 onward per the Flow Doctor alert
history) raised ``RuntimeError`` on the expected-absent artifact and paged
at ERROR — including the ordinary 2026-09-09 scheduled run, which is not a
deploy canary and shares no cause with I10146.

The fix resolves "is today a scanner-run day" from the SAME pure-calendar
predicate the weekly SF's own WeeklyRunDayGate state evaluates
(``inference/trading_day_gate.py::check_weekly_run_day``) — not a bare
try/except around the S3 NoSuchKey. This pins both branches: a non-run-day
miss must be a silent, logged skip; a run-day miss (or any other precondition
failure) must still reach the producer and, on failure, still page.
"""
from __future__ import annotations

import logging
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.pipeline import PipelineContext
from inference.stages import research_free
from inference.trading_day_gate import check_weekly_run_day

# 2026-09-09 (Wednesday): the ordinary scheduled weekday firing named in
# alpha-engine-config-I10301 — NOT the weekly-SF run day.
_NOT_RUN_DAY = "2026-09-09"
# 2026-09-05 (Saturday): previous day 2026-09-04 (Friday) is that week's
# last trading session — the weekly-SF run day.
_RUN_DAY = "2026-09-05"


def _ctx(**overrides) -> PipelineContext:
    kwargs = dict(date_str=_NOT_RUN_DAY, bucket="alpha-engine-research",
                  dry_run=False, local=False)
    kwargs.update(overrides)
    return PipelineContext(**kwargs)


def test_calendar_predicate_sanity():
    # Guards the two literals above against a future calendar/holiday-table
    # change silently invalidating this test's premise.
    assert check_weekly_run_day(_NOT_RUN_DAY)["is_weekly_run_day"] is False
    assert check_weekly_run_day(_RUN_DAY)["is_weekly_run_day"] is True


def test_non_run_day_skips_before_touching_the_producer():
    with patch(
        "inference.research_free_inference.run_research_free_inference"
    ) as mock_run:
        research_free.run(_ctx(date_str=_NOT_RUN_DAY))
    mock_run.assert_not_called()


def test_non_run_day_skip_produces_no_error_or_alert(caplog):
    # The whole point: an ordinary non-run-day weekday invocation must not
    # page. No ERROR log, and no ops alert publish.
    with patch("inference.stages.research_free._alert") as mock_alert:
        with caplog.at_level(logging.INFO, logger="inference.stages.research_free"):
            research_free.run(_ctx(date_str=_NOT_RUN_DAY))
    mock_alert.assert_not_called()
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not error_records, (
        f"non-run-day invocation logged at ERROR: "
        f"{[r.getMessage() for r in error_records]}"
    )
    assert any(
        "not the weekly-SF run day" in r.getMessage() for r in caplog.records
    )


def test_run_day_still_reaches_the_producer():
    with patch(
        "inference.research_free_inference.run_research_free_inference"
    ) as mock_run:
        mock_run.return_value = {"status": "ok", "n_written": 3, "n_errors": 0}
        research_free.run(_ctx(date_str=_RUN_DAY))
    mock_run.assert_called_once()


def test_run_day_missing_artifact_still_pages():
    # On the actual weekly run day, a missing/malformed candidates.json is a
    # genuine scanner-side failure, not a legitimate skip — this stage must
    # still reach the producer, catch its RuntimeError, and alert loud
    # (the existing I10067 failure-handling path, unchanged by this fix).
    with patch(
        "inference.research_free_inference.run_research_free_inference",
        side_effect=RuntimeError("failed to load candidates artifact: NoSuchKey"),
    ):
        with patch("inference.stages.research_free._alert") as mock_alert:
            research_free.run(_ctx(date_str=_RUN_DAY))
    mock_alert.assert_called_once()


def test_dry_run_skip_precedes_the_cadence_gate_regardless_of_date():
    # dry_run/local still short-circuits first (alpha-engine-config-I10146)
    # even on a date the cadence gate would itself have let through —
    # ordering must not regress either fix.
    with patch(
        "inference.trading_day_gate.check_weekly_run_day"
    ) as mock_gate:
        research_free.run(_ctx(date_str=_RUN_DAY, dry_run=True))
    mock_gate.assert_not_called()
