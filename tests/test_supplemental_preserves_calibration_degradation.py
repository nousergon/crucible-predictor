"""A variance fallback must stay on the day's artifact after the top-up run.

alpha-engine-config-I10489 (Brian ruling 2026-10-25, option (b)): whenever the
variance fallback engages, `predictor/predictions/{date}.json` must carry
`calibration_degradation.degraded: true`.

The weekday SF can invoke PredictorInference twice. The coverage-gap
re-invocation scores only the few buy_candidates run 1 missed, which sits
below `_MIN_BATCH_SIZE_FOR_VARIANCE_GATE`, so its own verdict is
`degraded: false` even with the very same collapsed calibrator. It then
rewrites the dated artifact. Before the fix, that rewrite replaced run 1's
block wholesale. The full batch's fallback disappeared from the record, while
run 1's rows still carried the fallback's output.

The first test is RED against the pre-fix tree.
"""
from __future__ import annotations

from unittest.mock import patch

import inference.stages.write_output as wo
from tests.test_calibration_finalization_order import (
    _CollapsedCalibrator,
    _Ctx,
    _live_20260828_batch,
)


def _finalize(preds, monkeypatch):
    from inference.stages import run_inference as ri

    monkeypatch.setattr(ri.cfg, "XSEC_DEMEAN_ALPHA_ENABLED", True, raising=False)
    ctx = _Ctx(preds, _CollapsedCalibrator())
    ri._finalize_calibration(ctx)
    return ctx


def _topup_batch():
    return [
        {"ticker": "NEW1", "predicted_alpha": 0.012, "p_up": 0.5658,
         "p_down": 0.4342, "predicted_direction": "UP",
         "prediction_confidence": 0.1316},
        {"ticker": "NEW2", "predicted_alpha": -0.008, "p_up": 0.4637,
         "p_down": 0.5363, "predicted_direction": "DOWN",
         "prediction_confidence": 0.0726},
    ]


def _run_supplemental(topup_ctx, existing_day):
    """Drive write_output.run on the supplemental path; return what it wrote."""
    from inference.pipeline import PipelineContext

    ctx = PipelineContext(
        date_str="2026-09-03",
        bucket="bucket",
        dry_run=False,
        predictions=list(topup_ctx.predictions),
        signals_data={},
        explicit_tickers=[p["ticker"] for p in topup_ctx.predictions],
    )
    ctx.calibration_degradation = topup_ctx.calibration_degradation
    with patch.object(wo, "get_veto_threshold", return_value=0.65), \
            patch.object(wo, "_load_gbm_meta", return_value={}), \
            patch.object(wo, "_read_existing_day", return_value=existing_day), \
            patch.object(wo, "send_predictor_email", return_value=True), \
            patch("inference.coverage_check.compute_coverage_delta"), \
            patch.object(wo, "write_predictions") as mock_write:
        wo.run(ctx)
    return mock_write.call_args.kwargs["calibration_degradation"]


def test_first_run_fallback_survives_the_supplemental_rewrite(monkeypatch):
    first = _finalize(_live_20260828_batch(), monkeypatch)
    assert first.calibration_degradation["degraded"] is True  # fallback engaged

    topup = _finalize(_topup_batch(), monkeypatch)
    # The same collapsed calibrator, but too few names for the variance gate.
    assert topup.calibration_degradation["degraded"] is False

    written = _run_supplemental(topup, {
        "predictions": first.predictions,
        "calibration_degradation": first.calibration_degradation,
    })

    assert written is not None, "the day's artifact lost its calibration verdict"
    assert written["degraded"] is True, (
        "Run 1 engaged the variance fallback, but the supplemental top-up "
        "overwrote the day's calibration_degradation with its own "
        "degraded=false verdict (alpha-engine-config-I10489)."
    )
    assert written["reason"] == "calibrator_collapse"
    assert written["basis"] == "linear_heuristic_fallback"
    assert written["n_unique_p_up_pre_fallback"] == 2
    assert written["batch_size"] == 30
    # The top-up's own verdict is kept, not discarded.
    assert written["supplemental_run"] == topup.calibration_degradation


def test_healthy_day_keeps_the_supplemental_verdict_unchanged(monkeypatch):
    topup = _finalize(_topup_batch(), monkeypatch)
    existing = {"predictions": [], "calibration_degradation": {
        "degraded": False, "basis": "calibrator", "reason": None,
    }}
    written = _run_supplemental(topup, existing)
    assert written == topup.calibration_degradation
    assert "supplemental_run" not in written


def test_degraded_supplemental_run_stands_on_a_healthy_day(monkeypatch):
    topup = _finalize(_topup_batch(), monkeypatch)
    topup.calibration_degradation = {
        "degraded": True, "basis": "linear_heuristic_fallback",
        "reason": "no_calibrator_loaded",
    }
    existing = {"predictions": [], "calibration_degradation": {
        "degraded": False, "basis": "calibrator", "reason": None,
    }}
    assert _run_supplemental(topup, existing) == topup.calibration_degradation


def test_merge_helper_with_no_existing_artifact():
    new = {"degraded": False, "basis": "calibrator"}
    assert wo._merge_calibration_degradation(new, None) == new
    assert wo._merge_calibration_degradation(None, None) is None
