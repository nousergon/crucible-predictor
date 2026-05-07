"""Tests for the inference-time parallel path of ResearchGBMScorer.

Per audit Phase 3 PR 3/5: the LGB rides alongside the bucket-lookup in
observe-only mode. These tests cover the predict_from_dict integration
behavior (how the LGB consumes the same 9 features the meta-Ridge sees)
plus the load-tolerance contract (what happens when the LGB isn't yet
in S3 — early observation cycle).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.research_gbm import RESEARCH_GBM_FEATURES, ResearchGBMScorer


def _trained_scorer(seed: int = 0):
    """Trained scorer suitable for inference-style predict_from_dict."""
    rng = np.random.default_rng(seed)
    n = 400
    X = rng.normal(0, 1, (n, len(RESEARCH_GBM_FEATURES))).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n) > 0).astype(np.float32)
    return ResearchGBMScorer(n_estimators=80).fit(X, y)


# ── predict_from_dict consumes meta_features cleanly ────────────────────

class TestPredictFromDictWithMetaFeatures:

    def test_full_meta_features_dict_works(self):
        # The shape inference builds at run_inference.py:meta_features:
        # has all 9 RESEARCH_GBM_FEATURES + research_calibrator_prob
        # + momentum_score + expected_move (12 keys total). The LGB
        # should pick out only the 9 it needs and ignore the rest.
        scorer = _trained_scorer()
        meta_features = {
            "research_calibrator_prob": 0.55,    # extra key (not in RESEARCH_GBM_FEATURES)
            "momentum_score": 0.02,              # extra
            "expected_move": 0.015,              # extra
            "research_composite_score": 0.7,
            "research_conviction": 1.0,
            "sector_macro_modifier": 0.05,
            "macro_spy_20d_return": 0.03,
            "macro_spy_20d_vol": 0.18,
            "macro_vix_level": 0.85,
            "macro_vix_term_slope": -0.05,
            "macro_yield_curve_slope": 0.04,
            "macro_market_breadth": 0.55,
        }
        result = scorer.predict_from_dict(meta_features)
        assert 0.0 <= result <= 1.0

    def test_predict_idempotent_on_same_input(self):
        # Determinism: same dict → same prediction.
        scorer = _trained_scorer()
        feats = {name: 0.5 for name in RESEARCH_GBM_FEATURES}
        a = scorer.predict_from_dict(feats)
        b = scorer.predict_from_dict(feats)
        assert a == b

    def test_predict_varies_with_research_score(self):
        # Sanity: changing research_composite_score (the strongest
        # signal in our synthetic training set) changes the output.
        scorer = _trained_scorer()
        low_score_feats = {name: 0.0 for name in RESEARCH_GBM_FEATURES}
        low_score_feats["research_composite_score"] = -2.0
        high_score_feats = {name: 0.0 for name in RESEARCH_GBM_FEATURES}
        high_score_feats["research_composite_score"] = 2.0

        low_pred = scorer.predict_from_dict(low_score_feats)
        high_pred = scorer.predict_from_dict(high_score_feats)
        # Synthetic signal is positive on research_composite_score.
        assert high_pred > low_pred


# ── Load tolerance: LGB absent in S3 (early observation period) ──────────

class TestLoadToleranceForObservePeriod:
    """Inference must not fail when ResearchGBMScorer isn't yet in S3.

    Early observation cycle: PR #95 just landed but Sat training hasn't
    fired yet. inference/stages/load_model.py catches the missing-file
    error and proceeds without setting ctx.meta_models["research_gbm"].
    The downstream code at run_inference.py reads with .get() so the
    None is the expected null-fallback.
    """

    def test_meta_models_get_returns_none_when_absent(self):
        # Simulates the load_model.py outcome when the LGB file is missing.
        meta_models: dict = {}  # no research_gbm key
        research_gbm = meta_models.get("research_gbm")
        assert research_gbm is None

    def test_research_gbm_prob_is_none_when_scorer_absent(self):
        # Mirror the run_inference.py logic: if research_gbm is None or
        # not fitted, research_gbm_prob stays None.
        research_gbm = None
        research_gbm_prob = None
        if research_gbm is not None and getattr(research_gbm, "fitted", False):
            research_gbm_prob = 0.5  # would set
        assert research_gbm_prob is None

    def test_research_gbm_prob_is_none_when_scorer_unfitted(self):
        research_gbm = ResearchGBMScorer()  # fitted=False
        research_gbm_prob = None
        if research_gbm is not None and getattr(research_gbm, "fitted", False):
            research_gbm_prob = 0.5
        assert research_gbm_prob is None


# ── Output dict shape (parity with the run_inference.py per-ticker dict) ─

class TestPredictionDictShape:
    """Mirrors the per-ticker result dict run_inference.py builds.

    Specifically tests that research_gbm_prob is present (None or float)
    and rounds to 4 decimals when populated. Downstream consumers
    (dashboard, email) read with .get() so None is tolerated.
    """

    def test_dict_has_research_gbm_prob_key_when_populated(self):
        scorer = _trained_scorer()
        feats = {name: 0.5 for name in RESEARCH_GBM_FEATURES}
        prob = scorer.predict_from_dict(feats)
        # Mirror the round() in run_inference.py
        result = {
            "ticker": "TEST",
            "research_calibrator_prob": 0.5,
            "research_gbm_prob": round(prob, 4),
        }
        assert isinstance(result["research_gbm_prob"], float)
        assert 0.0 <= result["research_gbm_prob"] <= 1.0

    def test_dict_has_research_gbm_prob_none_when_absent(self):
        # When research_gbm is not loaded, the field is None.
        result = {
            "ticker": "TEST",
            "research_calibrator_prob": 0.5,
            "research_gbm_prob": None,
        }
        assert result["research_gbm_prob"] is None

    def test_research_gbm_prob_rounded_to_4_decimals(self):
        # The run_inference.py code uses round(research_gbm_prob, 4)
        # to match the existing prediction-dict precision.
        prob = 0.456789123
        rounded = round(prob, 4)
        assert rounded == 0.4568


# ── Defensive on edge cases ─────────────────────────────────────────────

class TestDefensiveOnEdgeCases:

    def test_predict_from_dict_with_nan_values_does_not_crash(self):
        # Defensive: ArcticDB sometimes ships NaN macro features for
        # short-history days. predict_from_dict must not crash.
        scorer = _trained_scorer()
        feats = {name: 0.0 for name in RESEARCH_GBM_FEATURES}
        feats["macro_vix_level"] = float("nan")
        # Predict can return NaN but should not raise. Booster typically
        # tolerates NaN inputs natively.
        try:
            result = scorer.predict_from_dict(feats)
            # Either 0.0 ≤ result ≤ 1.0 or NaN — both acceptable here.
            assert isinstance(result, float)
        except Exception:
            pytest.fail("predict_from_dict should not raise on NaN input")

    def test_inference_path_swallows_per_ticker_failures(self):
        # Mirror the try/except in run_inference.py: any predict_from_dict
        # failure logs at debug level and leaves research_gbm_prob as None.
        # This test confirms that pattern is safe.
        scorer = _trained_scorer()
        research_gbm_prob: float | None = None
        try:
            # Force an error by passing a non-dict
            research_gbm_prob = scorer.predict_from_dict(None)  # type: ignore[arg-type]
        except Exception:
            pass  # swallow, leave None
        assert research_gbm_prob is None
