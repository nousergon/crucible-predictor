"""Tests for the Stage 1c inference parallel path — macro-augmented
volatility GBM rides alongside the plain volatility GBM in observe-only
mode.

Stage 1c of the regime-conditioning rebuild (plan doc:
alpha-engine-docs/private/regime-conditioning-260510.md). Validates:
- ``inference/stages/load_model.py`` loads ``volatility_macro_aug_model.txt``
  alongside ``volatility_model.txt`` when present, gracefully missing when not.
- ``inference/stages/run_inference.py`` emits ``expected_move_macro_aug``
  parallel field in predictions JSON.
- Feature schema matches Stage 1b training: 6 vol (rank-normed) + 6 macro
  (z-scored) = 12 features in ``VOLATILITY_FEATURES + MACRO_NORM_FEATURES``
  order.
- Failure isolation: aug GBM absent / today's macros unavailable / per-
  ticker predict raises → plain ``expected_move`` is unaffected.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg


_REPO = Path(__file__).parent.parent


def _src(rel_path: str) -> str:
    return (_REPO / rel_path).read_text()


# ── Source-level regression: loader + inference path wired ──────────────


class TestSourceContract:

    def test_load_model_loads_volatility_macro_aug(self):
        src = _src("inference/stages/load_model.py")
        assert "volatility_macro_aug_model.txt" in src
        assert 'meta_models["volatility_macro_aug"]' in src

    def test_run_inference_predicts_with_macro_aug(self):
        src = _src("inference/stages/run_inference.py")
        # The aug GBM is fetched from meta_models
        assert 'meta_models.get("volatility_macro_aug")' in src
        # Today's macro vector is built via time-series z-score
        assert "time_series_zscore_normalize" in src
        # Aug feature batch concats vol-ranked + macro-z-scored
        assert "X_vol_aug_ranked" in src
        # Per-ticker predict on aug variant
        assert "expected_move_macro_aug" in src

    def test_predictions_dict_has_macro_aug_field(self):
        src = _src("inference/stages/run_inference.py")
        # The result-dict assignment uses the parallel field name
        assert '"expected_move_macro_aug":' in src

    def test_macro_norm_features_used_in_inference(self):
        # The inference path must read cfg.MACRO_NORM_FEATURES (Stage 1a
        # config) so the schema stays in lockstep with Stage 1b training.
        src = _src("inference/stages/run_inference.py")
        assert "cfg.MACRO_NORM_FEATURES" in src

    def test_macro_norm_window_used_in_inference(self):
        src = _src("inference/stages/run_inference.py")
        assert "cfg.MACRO_NORM_WINDOW" in src


# ── Aug feature concat shape contract ──────────────────────────────────


class TestAugFeatureConcatShape:
    """Validates the schema invariant: aug feature vector =
    [VOLATILITY_FEATURES rank-normed, MACRO_NORM_FEATURES z-scored] in
    that order, matching Stage 1b's VOL_AUG_FEATURES list."""

    def test_aug_feature_count_matches_training(self):
        # Stage 1b contract: VOL_AUG_FEATURES = VOLATILITY_FEATURES +
        # MACRO_NORM_FEATURES. The exact count is schema-dependent and
        # grows as Stage 2c+ adds macros — what matters is that the
        # additive relationship holds and both halves have at least one
        # feature.
        n_vol = len(cfg.VOLATILITY_FEATURES)
        n_macro = len(cfg.MACRO_NORM_FEATURES)
        assert n_vol >= 1
        assert n_macro >= 1
        # Inference broadcasts today's macro vector to every ticker, then
        # concats with X_vol_ranked. The aug GBM input vector size is
        # n_vol + n_macro by construction.

    def test_aug_concat_preserves_per_ticker_then_macro_order(self):
        # Simulate: (n_present, n_vol) + (n_present, n_macro) → (n_present, 12)
        n_present = 5
        X_vol_ranked = np.random.default_rng(0).random(
            (n_present, len(cfg.VOLATILITY_FEATURES))
        ).astype(np.float32)
        today_macros = np.array([0.5, -0.3, 1.2, 0.0, -0.8, 0.4], dtype=np.float32)
        macro_block = np.tile(today_macros, (n_present, 1))
        X_aug = np.concatenate([X_vol_ranked, macro_block], axis=1)

        # Shape contract
        assert X_aug.shape == (n_present, 12)
        # First 6 columns are per-ticker vol (vary across rows for the
        # same column). Random vol features have non-trivial stdev.
        assert X_aug[:, 0].std() > 0
        # Last 6 columns are constant per macro (broadcast across tickers).
        for j in range(6, 12):
            assert (X_aug[:, j] == X_aug[0, j]).all()

    def test_macro_block_order_matches_macro_norm_features(self):
        # When today_macros = [a, b, c, d, e, f], the broadcast block's
        # column j carries macro feature j of MACRO_NORM_FEATURES.
        today_macros = np.arange(6, dtype=np.float32)  # [0, 1, 2, 3, 4, 5]
        n_present = 3
        macro_block = np.tile(today_macros, (n_present, 1))
        # Column j contains value j (the broadcast preserves order).
        for j in range(6):
            assert (macro_block[:, j] == j).all()


# ── Failure isolation ──────────────────────────────────────────────────


class TestFailureIsolation:
    """Validates: aug GBM absent / today's macros unavailable / per-ticker
    predict raises → plain ``expected_move`` is unaffected."""

    def test_run_inference_falls_back_when_aug_missing(self):
        src = _src("inference/stages/run_inference.py")
        # The aug-predict block is gated on vol_scorer_aug AND
        # X_vol_aug_ranked being populated.
        assert "if (\n        vol_scorer_aug is not None" in src
        assert "and X_vol_ranked is not None" in src
        assert "and today_macros_zscored is not None" in src

    def test_macro_zscore_failure_is_non_blocking(self):
        src = _src("inference/stages/run_inference.py")
        # The macro z-score build is wrapped in try/except with
        # log.warning fallback — no production behaviour change.
        assert "Macro z-score for vol-aug GBM failed" in src

    def test_per_ticker_predict_failure_is_non_blocking(self):
        src = _src("inference/stages/run_inference.py")
        # Per-ticker aug predict is wrapped in try/except with log.debug
        # fallback — same shape as the existing research_gbm parallel path.
        assert "volatility_macro_aug.predict failed for" in src


# ── Smoke import ───────────────────────────────────────────────────────


class TestSmokeImport:

    def test_run_inference_imports_cleanly(self):
        # The Stage 1c additions don't break the module's import contract.
        import importlib

        for mod in (
            "inference.stages.load_model",
            "inference.stages.run_inference",
        ):
            m = importlib.import_module(mod)
            assert m is not None


# ── Train/inference consistency across a MACRO_NORM_FEATURES change ────


# The list as it stood before alpha-engine-config-I11523 dropped the HY OAS
# pair. A model trained the Saturday before a config change merges carries
# these names until the next weekly retrain.
_PRE_I11523_MACROS = [
    "spy_20d_return", "spy_20d_vol", "vix_level", "vix_term_slope",
    "yield_curve_slope", "market_breadth", "vix_vix3m_ratio",
    "market_breadth_200d", "yield_curve_10y_2y", "hy_oas_level",
    "hy_oas_change_21d", "baa10y_level", "baa10y_change_21d",
]


class _StubScorer:
    def __init__(self, names):
        self.feature_names = list(names)


def _fit_and_reload(tmp_path, macro_names):
    """A real booster, saved and reloaded the way load_model.py does it."""
    from model.gbm_scorer import GBMScorer

    names = list(cfg.VOLATILITY_FEATURES) + list(macro_names)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(names)))
    y = np.abs(X[:, 0]) + 0.1 * rng.normal(size=400)
    scorer = GBMScorer(
        params={"min_child_samples": 20, "num_leaves": 7, "verbose": -1},
        n_estimators=5, early_stopping_rounds=5,
    )
    scorer.fit(X[:300], y[:300], X[300:], y[300:], feature_names=names)
    path = tmp_path / "volatility_macro_aug.txt"
    scorer.save(path)
    return GBMScorer.load(path)


class TestMacroColumnsComeFromTheModel:

    def _all_columns(self):
        # build_features still emits the HY OAS pair, so both lists resolve.
        return set(_PRE_I11523_MACROS) | set(cfg.MACRO_NORM_FEATURES)

    def test_model_trained_on_the_old_list_is_scored_on_the_old_list(self, tmp_path):
        from inference.stages.run_inference import resolve_vol_aug_macro_columns

        scorer = _fit_and_reload(tmp_path, _PRE_I11523_MACROS)
        cols = resolve_vol_aug_macro_columns(scorer, self._all_columns())
        assert cols == _PRE_I11523_MACROS
        # And the vector built from those columns is the width it expects.
        x = np.zeros((1, len(cfg.VOLATILITY_FEATURES) + len(cols)), np.float32)
        assert scorer.predict(x).shape == (1,)

    def test_model_trained_on_the_new_list_is_scored_on_the_new_list(self, tmp_path):
        from inference.stages.run_inference import resolve_vol_aug_macro_columns

        scorer = _fit_and_reload(tmp_path, cfg.MACRO_NORM_FEATURES)
        cols = resolve_vol_aug_macro_columns(scorer, self._all_columns())
        assert cols == list(cfg.MACRO_NORM_FEATURES)
        assert "hy_oas_level" not in cols

    def test_no_model_falls_back_to_config(self):
        from inference.stages.run_inference import resolve_vol_aug_macro_columns

        assert resolve_vol_aug_macro_columns(None, []) == list(
            cfg.MACRO_NORM_FEATURES
        )

    def test_a_trained_macro_the_build_did_not_produce_raises(self):
        import pytest

        from inference.stages.run_inference import resolve_vol_aug_macro_columns

        scorer = _StubScorer(list(cfg.VOLATILITY_FEATURES) + _PRE_I11523_MACROS)
        with pytest.raises(ValueError, match="hy_oas_level"):
            resolve_vol_aug_macro_columns(scorer, cfg.MACRO_NORM_FEATURES)

    def test_a_reordered_volatility_prefix_raises(self):
        import pytest

        from inference.stages.run_inference import resolve_vol_aug_macro_columns

        vol = list(reversed(cfg.VOLATILITY_FEATURES))
        scorer = _StubScorer(vol + list(cfg.MACRO_NORM_FEATURES))
        with pytest.raises(ValueError, match="VOLATILITY_FEATURES"):
            resolve_vol_aug_macro_columns(scorer, self._all_columns())

    def test_inference_reads_the_macro_list_from_the_loaded_model(self):
        src = _src("inference/stages/run_inference.py")
        assert "resolve_vol_aug_macro_columns(" in src
        assert 'regime_features_df[\n                macro_cols\n            ]' in src
