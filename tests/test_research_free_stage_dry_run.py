"""``inference.stages.research_free`` must skip on dry_run/local, matching
``inference.stages.regime_fast_signal``'s existing guard.

alpha-engine-config-I10141 investigation, 2026-09-07: the deploy-time canary
(``infrastructure/deploy.sh``'s ``predict(dry_run)`` action) invokes the daily
pipeline with ``dry_run=True``, off the Step Function, off ``TradingDayGate``,
and off any calendar. ``run_research_free_inference``'s precondition is
``candidates/{date}/candidates.json`` existing for that literal date — an
artifact the weekly Scanner writes only on trading days it has actually run
for. A canary landing on Labor Day (2026-09-07) or any date the Scanner
hasn't covered raised ``RuntimeError`` on the expected-absent artifact and
paged, twice from one invocation (the sibling failure is the dispersion-gate
downgrade in ``tests/test_write_predictions_inference_gate.py``).

This pins the fix: ``run()`` returns before ever touching
``run_research_free_inference``/S3 when ``ctx.dry_run`` or ``ctx.local`` is
set, exactly like ``regime_fast_signal.run()`` already does.
"""
from __future__ import annotations

import logging
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.pipeline import PipelineContext
from inference.stages import research_free


def _ctx(**overrides) -> PipelineContext:
    kwargs = dict(date_str="2026-09-07", bucket="alpha-engine-research",
                  dry_run=False, local=False)
    kwargs.update(overrides)
    return PipelineContext(**kwargs)


def test_dry_run_skips_before_touching_the_producer(caplog):
    with patch(
        "inference.research_free_inference.run_research_free_inference"
    ) as mock_run:
        with caplog.at_level(logging.INFO, logger="inference.stages.research_free"):
            research_free.run(_ctx(dry_run=True))
    mock_run.assert_not_called()
    assert any(
        "skipped (dry_run/local)" in r.getMessage() for r in caplog.records
    )


def test_local_skips_before_touching_the_producer():
    with patch(
        "inference.research_free_inference.run_research_free_inference"
    ) as mock_run:
        research_free.run(_ctx(local=True))
    mock_run.assert_not_called()


def test_dry_run_skip_produces_no_error_log(caplog):
    # The whole point: a canary invocation must not page. Confirm no ERROR
    # (or higher) log record is emitted at all on the dry_run path, even
    # though the real candidates artifact for this date does not exist and
    # the un-guarded path would have raised RuntimeError -> log.error().
    with caplog.at_level(logging.WARNING, logger="inference.stages.research_free"):
        research_free.run(_ctx(dry_run=True))
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not error_records, (
        f"dry_run invocation logged at ERROR: {[r.getMessage() for r in error_records]}"
    )


def test_real_invocation_still_calls_the_producer():
    # The half that had to survive: a real (non-dry-run, non-local) run must
    # still reach the producer. Not exercising a real S3/ArcticDB round trip
    # here — just confirming the new guard doesn't also swallow the live path.
    with patch(
        "inference.research_free_inference.run_research_free_inference"
    ) as mock_run:
        mock_run.return_value = {"status": "ok", "n_written": 0, "n_errors": 0}
        research_free.run(_ctx(dry_run=False, local=False))
    mock_run.assert_called_once()
