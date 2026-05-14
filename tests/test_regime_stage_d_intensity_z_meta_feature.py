"""Tests for regime-v3 Stage D — predictor L2 META_FEATURES gains
``regime_intensity_z``.

Covers:
- META_FEATURES contains the new entry + position is stable
- REGIME_DERIVED_FEATURE_META_MAP wires the source column name to the
  L2 row name (intensity_z → regime_intensity_z)
- ``compute_intensity_z_series`` returns a date-indexed Series matching
  the substrate Lambda's risk-on orientation (positive = risk-on)
- Early rows (< min_history) default to 0.0 neutral
- ``RegimePredictor.build_features`` adds the ``intensity_z`` column
  to the returned regime_features_df
- Helper is graceful when ``regime/composite`` import fails
  (defensive default: column populated as 0.0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


from model.meta_model import (
    META_FEATURES,
    MACRO_FEATURE_META_MAP,
    REGIME_DERIVED_FEATURE_META_MAP,
)
from regime.composite import compute_intensity_z_series


# ---------------------------------------------------------------------------
# META_FEATURES schema
# ---------------------------------------------------------------------------


class TestMetaFeaturesSchema:
    def test_regime_intensity_z_present_in_meta_features(self):
        """Pin the contract — L2 Ridge fit + inference expect this
        feature in the META_FEATURES list. Removal would mismatch
        deployed model weights against the inference feature vector."""
        assert "regime_intensity_z" in META_FEATURES

    def test_regime_intensity_z_is_last_feature(self):
        """Stage D adds intensity_z as the trailing META_FEATURE so the
        column ordering of older deployed weights stays valid for the
        first N-1 features (Ridge re-fit catches up on next training).
        Pinning the position prevents an accidental insertion in the
        middle that would silently re-order weights."""
        assert META_FEATURES[-1] == "regime_intensity_z"

    def test_regime_derived_map_wires_intensity_z(self):
        """The source column on regime_features_df is ``intensity_z``
        (lowercase, no prefix — matches what compute_intensity_z_series
        emits). The L2 row name has the ``regime_`` prefix for
        namespace separation."""
        assert REGIME_DERIVED_FEATURE_META_MAP == {"intensity_z": "regime_intensity_z"}

    def test_no_overlap_between_macro_map_and_derived_map(self):
        """The two maps populate different meta-row keys — pin no
        collision so wiring one doesn't silently clobber the other."""
        macro_names = set(MACRO_FEATURE_META_MAP.values())
        derived_names = set(REGIME_DERIVED_FEATURE_META_MAP.values())
        assert macro_names.isdisjoint(derived_names)


# ---------------------------------------------------------------------------
# compute_intensity_z_series
# ---------------------------------------------------------------------------


def _synthetic_feature_panel(n_rows: int = 60, seed: int = 7) -> pd.DataFrame:
    """Macro feature panel with realistic distributions, indexed weekly."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "vix_level": rng.normal(18.0, 5.0, n_rows).clip(min=9.0),
        "hy_oas_bps": rng.normal(400.0, 100.0, n_rows).clip(min=200.0),
        "yield_curve_slope": rng.normal(0.5, 0.4, n_rows),
        "spy_20d_return": rng.normal(0.01, 0.04, n_rows),
        "market_breadth": rng.normal(0.55, 0.15, n_rows).clip(0.0, 1.0),
        "vix_term_slope": rng.normal(1.0, 0.3, n_rows),
    }, index=pd.date_range("2024-01-05", periods=n_rows, freq="W-FRI"))


class TestComputeIntensityZSeries:
    def test_returns_series_indexed_by_input(self):
        df = _synthetic_feature_panel()
        result = compute_intensity_z_series(df)
        assert isinstance(result, pd.Series)
        assert (result.index == df.index).all()
        assert len(result) == len(df)

    def test_early_rows_default_neutral(self):
        """The first 4 rows have insufficient history for a meaningful
        z-score — helper returns 0.0 (neutral)."""
        df = _synthetic_feature_panel()
        result = compute_intensity_z_series(df)
        assert (result.iloc[:4] == 0.0).all()

    def test_later_rows_have_finite_values(self):
        df = _synthetic_feature_panel(n_rows=100)
        result = compute_intensity_z_series(df)
        # Rows past min_history should have finite z-scores
        assert np.isfinite(result.iloc[10:]).all()

    def test_inverted_to_risk_on_orientation(self):
        """Substrate convention: positive intensity_z = risk-on. The
        helper INVERTS the raw composite (which is stress-orientation)
        so the META_FEATURE value matches the substrate.json field.

        Construction: a row with high VIX + high HY OAS + negative
        SPY return (clear risk-off) → raw composite gives positive
        stress z → helper returns NEGATIVE (risk-off in risk-on
        orientation)."""
        # Start with calm baseline history, then a stressed final row
        rng = np.random.default_rng(0)
        n = 60
        df = pd.DataFrame({
            "vix_level": rng.normal(14.0, 1.0, n).clip(min=9.0),
            "hy_oas_bps": rng.normal(280.0, 25.0, n).clip(min=200.0),
            "yield_curve_slope": rng.normal(1.0, 0.2, n),
            "spy_20d_return": rng.normal(0.025, 0.015, n),
            "market_breadth": rng.normal(0.75, 0.08, n).clip(0.0, 1.0),
            "vix_term_slope": rng.normal(1.3, 0.15, n),
        }, index=pd.date_range("2024-01-05", periods=n, freq="W-FRI"))
        # Inject extreme risk-off values into the last row
        df.iloc[-1] = {
            "vix_level": 45.0,
            "hy_oas_bps": 900.0,
            "yield_curve_slope": -0.5,
            "spy_20d_return": -0.10,
            "market_breadth": 0.20,
            "vix_term_slope": -0.5,
        }
        result = compute_intensity_z_series(df)
        # Risk-off row → strongly negative intensity_z (risk-on orientation)
        assert result.iloc[-1] < -1.0

    def test_empty_panel_returns_empty_series(self):
        result = compute_intensity_z_series(pd.DataFrame())
        assert result.empty
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Integration with RegimePredictor.build_features
# ---------------------------------------------------------------------------


def _make_macro_series(n_days: int = 600, seed: int = 11):
    """Synthesize the macro time series needed by
    RegimePredictor.build_features. ~2.5 years of daily data so
    rolling 20d windows + the dropna at the end leave enough rows
    for compute_intensity_z_series to produce non-neutral values."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    spy = pd.Series(400 + 0.05 * np.arange(n_days) + rng.normal(0, 2.0, n_days), index=dates)
    vix = pd.Series(18.0 + rng.normal(0, 3.0, n_days), index=dates).clip(lower=9.0)
    vix3m = pd.Series(19.0 + rng.normal(0, 2.5, n_days), index=dates).clip(lower=10.0)
    tnx = pd.Series(4.0 + rng.normal(0, 0.1, n_days), index=dates)
    irx = pd.Series(3.5 + rng.normal(0, 0.08, n_days), index=dates)
    return spy, vix, vix3m, tnx, irx


class TestBuildFeaturesAddsIntensityZ:
    def test_intensity_z_column_present_after_build(self):
        """End-to-end: after RegimePredictor.build_features runs, the
        returned regime_features_df has an ``intensity_z`` column.
        This pins the wiring that both training (meta_trainer) and
        inference (run_inference) depend on."""
        from model.regime_predictor import RegimePredictor

        spy, vix, vix3m, tnx, irx = _make_macro_series()
        rp = RegimePredictor()
        df = rp.build_features(spy, vix, vix3m, tnx, irx, all_close_prices={})
        assert "intensity_z" in df.columns, (
            "RegimePredictor.build_features must add the intensity_z "
            "column — Stage D META_FEATURES wiring depends on it. "
            "Without it, training would default regime_intensity_z to "
            "0.0 for every row (Ridge learns nothing) and inference "
            "would do the same (silent feature dropout)."
        )

    def test_intensity_z_values_are_finite_after_warmup(self):
        from model.regime_predictor import RegimePredictor

        spy, vix, vix3m, tnx, irx = _make_macro_series()
        rp = RegimePredictor()
        df = rp.build_features(spy, vix, vix3m, tnx, irx, all_close_prices={})
        # Past the early-row warmup, intensity_z should be finite
        # (may be near zero on synthetic-stable data — that's fine)
        late_rows = df["intensity_z"].iloc[20:]
        assert np.isfinite(late_rows).all()


# ---------------------------------------------------------------------------
# Defensive — helper graceful when underlying composite call fails
# ---------------------------------------------------------------------------


class TestBuildFeaturesGracefulOnIntensityFailure:
    def test_build_features_does_not_crash_on_intensity_helper_failure(self, monkeypatch):
        """If compute_intensity_z_series raises for any reason
        (corrupted state, unexpected NaN, etc.), build_features must
        not crash — it logs a warning and defaults intensity_z=0.0.
        Pins the defensive try/except shape so a future refactor
        that strips the except doesn't silently regress the
        production-Lambda fail-graceful semantics."""
        from model import regime_predictor as rp_module

        def _explode(*args, **kwargs):
            raise RuntimeError("synthetic failure for test")

        monkeypatch.setattr(
            "regime.composite.compute_intensity_z_series", _explode
        )

        spy, vix, vix3m, tnx, irx = _make_macro_series()
        df = rp_module.RegimePredictor().build_features(
            spy, vix, vix3m, tnx, irx, all_close_prices={}
        )
        assert "intensity_z" in df.columns
        assert (df["intensity_z"] == 0.0).all()
