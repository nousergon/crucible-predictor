"""The collapse record must name the ARM it fired on, and the number that
actually discriminates the cause.

alpha-engine-config: on 2026-09-08 the pre-open run emitted three ERROR pages —

    VARIANCE FALLBACK ENGAGED: calibrator outputs collapsed to 2 unique p_up
    bins across 29 tickers (threshold=3). ... Falling through to linear
    heuristic rescale to recover variance for today's batch.

— attributed to component `predictor-inference`, on a day the LIVE traded batch
was healthy (29 distinct predicted_alpha, 8 unique p_up bins,
`calibration_degradation.degraded == false` in
`predictor/predictions/2026-09-08.json`). All three came from the shadow runner
re-scoring OBSERVE-ONLY challengers, which trade on none. CloudWatch
(/aws/lambda/alpha-engine-predictor-inference, 12:54:06Z-12:54:49Z) shows the
live pass logging `unique_p_up_bins=8` and the fallback firing only on the
shadow passes at 12:54:33, 12:54:38 and 12:54:43.

Two defects, both fixed here:

1. The record and the log are arm-blind, so an observe-only challenger's
   collapse pages as a live-trading ERROR. That is a false page and it cost the
   whole investigation.
2. `n_unique_p_up` conflates a model collapse with a calibrator's resolution.
   The number that names the cause is the count of distinct `predicted_alpha`:
   the shadow arms produced 6 across 29 tickers, 24 of them identical, while
   `momentum_20d` spanned 24 distinct values.

Neither is a licence to be quiet: a shadow collapse is a real finding about a
promotable arm and must still be recorded and logged.
"""

from __future__ import annotations

import logging

import pytest

from inference.stages import run_inference as ri


class _Collapsed:
    """Two plateaus — the measured live shape."""

    method = "isotonic"
    is_fitted = True
    _ece_after = 3e-06


class _Ctx:
    def __init__(self, predictions, *, shadow_version_id=None):
        self.predictions = predictions
        self.calibrator = _Collapsed()
        self.shadow_calibrator = None
        self.level_neutralization = None
        self.calibration_degradation = None
        self.shadow_version_id = shadow_version_id


def _collapsed_batch(n=29):
    """24 names on one alpha, 5 distinct — the measured 2026-09-08 shape."""
    preds = [{"ticker": f"T{i}", "predicted_alpha": 0.000402, "p_up": 0.504}
             for i in range(n - 5)]
    for i, a in enumerate((-0.000625, -0.001788, -0.002177, -0.002400, -0.002775)):
        preds.append({"ticker": f"O{i}", "predicted_alpha": a, "p_up": 0.4938})
    return preds


def _healthy_batch(n=29):
    return [{"ticker": f"T{i}", "predicted_alpha": 0.001 * (i - n / 2),
             "p_up": round(0.40 + 0.007 * i, 4)} for i in range(n)]


def test_live_collapse_still_logs_error_and_says_champion(caplog):
    ctx = _Ctx(_collapsed_batch())
    with caplog.at_level(logging.INFO, logger="inference.stages.run_inference"):
        ri._rescale_cross_sectional(ctx)
    errs = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errs) == 1, "the live path must still page loudly"
    assert "VARIANCE FALLBACK ENGAGED" in errs[0].getMessage()
    deg = ctx.calibration_degradation
    assert deg["degraded"] is True
    assert deg["arm"] == "champion"
    assert deg["shadow_version_id"] is None


def test_shadow_collapse_is_recorded_and_logged_but_not_as_a_live_error(caplog):
    vid = "spec-sota-combine-2026-07-24-8578f8ae"
    ctx = _Ctx(_collapsed_batch(), shadow_version_id=vid)
    with caplog.at_level(logging.INFO, logger="inference.stages.run_inference"):
        ri._rescale_cross_sectional(ctx)
    assert not [r for r in caplog.records if r.levelno == logging.ERROR], (
        "an observe-only challenger that trades on none must not raise a "
        "live-trading ERROR page"
    )
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1, "but it must NOT be silent — it is a real finding"
    msg = warns[0].getMessage()
    assert vid in msg, "the log must name the arm or it cannot be acted on"
    assert "shadow" in msg.lower()
    deg = ctx.calibration_degradation
    assert deg["degraded"] is True
    assert deg["arm"] == "shadow"
    assert deg["shadow_version_id"] == vid


def test_the_record_carries_the_number_that_names_the_cause():
    ctx = _Ctx(_collapsed_batch())
    ri._rescale_cross_sectional(ctx)
    deg = ctx.calibration_degradation
    # 24 identical + 5 distinct = 6 distinct alphas across 29 tickers.
    assert deg["n_distinct_predicted_alpha"] == 6
    assert deg["batch_size"] == 29


def test_the_healthy_record_carries_the_same_fields_so_a_reader_can_compare():
    """A field present only on the failure path cannot be trended, and its
    absence reads as healthy."""
    ctx = _Ctx(_healthy_batch())
    ri._rescale_cross_sectional(ctx)
    deg = ctx.calibration_degradation
    assert deg["degraded"] is False
    assert deg["arm"] == "champion"
    assert deg["shadow_version_id"] is None
    assert deg["n_distinct_predicted_alpha"] == 29


def test_a_healthy_shadow_arm_is_attributed_too():
    vid = "spec-sota-combine-2026-07-31-0c5140d5"
    ctx = _Ctx(_healthy_batch(), shadow_version_id=vid)
    ri._rescale_cross_sectional(ctx)
    assert ctx.calibration_degradation["arm"] == "shadow"
    assert ctx.calibration_degradation["shadow_version_id"] == vid


def test_champion_collapse_writes_an_explicitly_unactionable_batch():
    """alpha-engine-config-I10179 (Brian ruling 2026-09-08, option (c)): a
    collapsed calibrator on the LIVE champion must force
    prediction_confidence to 0.0 on every row so the executor's existing
    MIN_CONFIDENCE = 0.30 veto declines the whole book by its own rule — the
    discriminator is this same arm attribution from -I10178."""
    ctx = _Ctx(_collapsed_batch())
    ri._rescale_cross_sectional(ctx)
    assert ctx.calibration_degradation["arm"] == "champion"
    assert ctx.calibration_degradation["degraded"] is True
    for p in ctx.predictions:
        assert p["prediction_confidence"] == 0.0
        assert p["calibration_basis"] == "linear_heuristic_fallback"
    # Ranking is preserved even though the batch is unactionable — the linear
    # heuristic's p_up ordering still reflects the underlying alpha ordering,
    # per the recorded delta in the ruling (a rank-ordering measurement this
    # PR does not attempt).
    assert len({p["p_up"] for p in ctx.predictions}) > 1


def test_shadow_collapse_keeps_todays_confidence_behaviour():
    """The champion-only branch above must NOT engage for an observe-only
    shadow arm — it trades on nothing, so today's confidence computation
    (not forced to 0.0) is preserved (the fallback fired three times on
    shadow arms in -I10178 and nothing traded on it)."""
    vid = "spec-sota-combine-2026-07-24-8578f8ae"
    ctx = _Ctx(_collapsed_batch(), shadow_version_id=vid)
    ri._rescale_cross_sectional(ctx)
    assert ctx.calibration_degradation["arm"] == "shadow"
    confidences = {p["prediction_confidence"] for p in ctx.predictions}
    assert confidences != {0.0}, (
        "a shadow-arm collapse must not be forced unactionable — only the "
        "live champion path is neutered"
    )
    for p in ctx.predictions:
        assert p["calibration_basis"] == "linear_heuristic_fallback"


def test_a_context_without_the_attribute_defaults_to_champion():
    """Old contexts (and any caller that predates the field) must read as the
    live path, never as an unattributed arm that quietly downgrades a page."""
    class _Bare:
        predictions = _collapsed_batch()
        calibrator = _Collapsed()
        shadow_calibrator = None
        level_neutralization = None
        calibration_degradation = None

    ctx = _Bare()
    ri._rescale_cross_sectional(ctx)
    assert ctx.calibration_degradation["arm"] == "champion"
