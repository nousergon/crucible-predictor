"""Tests for Stage 2c-full of the regime-conditioning rebuild —
full-set macros in ``RegimePredictor.build_features()``:

- ``yield_curve_10y_2y``: recession-canonical curve (TNX - TWO) / 10
- ``hy_oas_level``: ICE BofA US HY Index OAS, license-gated to 2023+
- ``hy_oas_change_21d``: 21d change in HY OAS (regime-shift detector)
- ``baa10y_level``: Moody's BAA Corporate - 10Y Treasury spread, full
  40y FRED history
- ``baa10y_change_21d``: 21d change in BAA10Y (full-corpus credit
  regime-shift signal)

All five flow into ``cfg.MACRO_NORM_FEATURES`` so Stage 1b's macro-aug
volatility GBM consumes them via the time-series z-score pipeline.

Plan doc: ~/Development/alpha-engine-docs/private/regime-conditioning-260510.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from model.regime_predictor import RegimePredictor


def _spy_series(n: int = 300, start: str = "2018-01-01", seed: int = 0):
    rng = np.random.default_rng(seed)
    daily = rng.normal(0, 0.008, n)
    close = 300.0 * np.exp(np.cumsum(daily))
    return pd.Series(close, index=pd.date_range(start, periods=n, freq="B"))


def _flat_series(n: int = 300, level: float = 4.0, start: str = "2018-01-01"):
    return pd.Series(
        np.full(n, level, dtype=float),
        index=pd.date_range(start, periods=n, freq="B"),
    )


# ── Schema contract ────────────────────────────────────────────────────


class TestFullMacroSchema:

    def test_macro_norm_features_includes_full_set(self):
        for name in (
            "yield_curve_10y_2y",
            "hy_oas_level",
            "hy_oas_change_21d",
            "baa10y_level",
            "baa10y_change_21d",
        ):
            assert name in cfg.MACRO_NORM_FEATURES, (
                f"{name} missing from cfg.MACRO_NORM_FEATURES"
            )

    def test_build_features_emits_full_set(self):
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(),
            tnx_series=_flat_series(level=4.0),
            two_series=_flat_series(level=2.0),
            hyoas_series=_flat_series(level=3.5),
            baa10y_series=_flat_series(level=2.5),
        )
        for name in (
            "yield_curve_10y_2y",
            "hy_oas_level",
            "hy_oas_change_21d",
            "baa10y_level",
            "baa10y_change_21d",
        ):
            assert name in out.columns

    def test_macro_norm_schema_subset_of_build_features_output(self):
        # Stage 1b's _build_macro_array reads cfg.MACRO_NORM_FEATURES from
        # the regime_features_df. Every name in MACRO_NORM_FEATURES must
        # appear as a column in build_features output, or training will
        # raise KeyError.
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(),
            vix_series=_flat_series(level=18.0),
            vix3m_series=_flat_series(level=20.0),
            tnx_series=_flat_series(level=4.0),
            irx_series=_flat_series(level=2.0),
            two_series=_flat_series(level=3.0),
            hyoas_series=_flat_series(level=3.5),
            baa10y_series=_flat_series(level=2.5),
        )
        for name in cfg.MACRO_NORM_FEATURES:
            assert name in out.columns, (
                f"cfg.MACRO_NORM_FEATURES has {name}, but build_features "
                f"output does not — Stage 1b training would KeyError"
            )


# ── yield_curve_10y_2y ────────────────────────────────────────────────


class TestYieldCurve10y2y:

    def test_normal_curve_positive(self):
        # Normal yield curve: TNX (10Y) > TWO (2Y) → positive slope.
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(),
            tnx_series=_flat_series(level=4.5),
            two_series=_flat_series(level=2.0),
        )
        # (4.5 - 2.0) / 10 = 0.25 across all rows.
        assert np.allclose(out["yield_curve_10y_2y"].iloc[20:], 0.25)

    def test_inverted_curve_negative(self):
        # Inverted: 2Y > 10Y → negative slope (recession-precursor).
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(),
            tnx_series=_flat_series(level=3.5),
            two_series=_flat_series(level=4.5),
        )
        assert np.allclose(out["yield_curve_10y_2y"].iloc[20:], -0.10)

    def test_neutral_when_either_series_missing(self):
        # Need both to compute; missing → neutral 0.0.
        rp = RegimePredictor()
        out_no_two = rp.build_features(
            spy_series=_spy_series(),
            tnx_series=_flat_series(level=4.5),
            two_series=None,
        )
        assert (out_no_two["yield_curve_10y_2y"] == 0.0).all()


# ── hy_oas_level + hy_oas_change_21d ──────────────────────────────────


class TestHyOasFeatures:

    def test_level_passes_through(self):
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(),
            hyoas_series=_flat_series(level=3.50),
        )
        assert np.allclose(out["hy_oas_level"].iloc[20:], 3.50)

    def test_change_21d_zero_for_flat_series(self):
        # Flat series → 21d change is 0 once the lag has filled.
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(),
            hyoas_series=_flat_series(level=3.50),
        )
        assert np.allclose(out["hy_oas_change_21d"].iloc[40:], 0.0)

    def test_change_21d_captures_widening(self):
        # Step-up: HY OAS jumps from 3.0 to 6.0 at the date partway through.
        # 21 business days after the step the 21d change should reflect
        # the full +3.0 widening.
        n = 200
        idx = pd.date_range("2018-01-01", periods=n, freq="B")
        prices = np.concatenate([np.full(100, 3.0), np.full(n - 100, 6.0)])
        step_date = idx[100]
        # Pick a date 10 trading days post-step — the 21d lookback still
        # reaches into the pre-step regime → change = 6.0 - 3.0 = 3.0.
        target_date = idx[110]
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(n=n),
            hyoas_series=pd.Series(prices, index=idx),
        )
        # Use .loc so we don't need to reason about dropna offsets.
        assert target_date in out.index
        assert abs(out.loc[target_date, "hy_oas_change_21d"] - 3.0) < 0.01

    def test_neutral_when_series_missing(self):
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(),
            hyoas_series=None,
        )
        assert (out["hy_oas_level"] == 0.0).all()
        assert (out["hy_oas_change_21d"] == 0.0).all()


# ── baa10y_level + baa10y_change_21d ───────────────────────────────────


class TestBaa10yFeatures:

    def test_level_passes_through(self):
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(),
            baa10y_series=_flat_series(level=2.10),
        )
        assert np.allclose(out["baa10y_level"].iloc[20:], 2.10)

    def test_change_21d_captures_widening(self):
        # BAA10Y widens from 1.5 to 3.0 (recession scenario).
        n = 200
        idx = pd.date_range("2018-01-01", periods=n, freq="B")
        spread = np.concatenate([np.full(100, 1.5), np.full(n - 100, 3.0)])
        target_date = idx[110]  # 10 BDays post-step; 21d lookback hits pre-step
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(n=n),
            baa10y_series=pd.Series(spread, index=idx),
        )
        assert target_date in out.index
        assert abs(out.loc[target_date, "baa10y_change_21d"] - 1.5) < 0.01

    def test_neutral_when_series_missing(self):
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(),
            baa10y_series=None,
        )
        assert (out["baa10y_level"] == 0.0).all()
        assert (out["baa10y_change_21d"] == 0.0).all()


# ── Stage 1b integration sanity ───────────────────────────────────────


class TestStage1bIntegrationSanity:
    """Validates that the full macro set flows into Stage 1b's
    ``_build_macro_array`` without raising on the new column names."""

    def test_build_macro_array_with_full_macros(self):
        from training.meta_trainer import _build_macro_array

        rp = RegimePredictor()
        regime_df = rp.build_features(
            spy_series=_spy_series(),
            vix_series=_flat_series(level=18.0),
            vix3m_series=_flat_series(level=20.0),
            tnx_series=_flat_series(level=4.0),
            irx_series=_flat_series(level=2.0),
            two_series=_flat_series(level=3.0),
            hyoas_series=_flat_series(level=3.5),
            baa10y_series=_flat_series(level=2.5),
        )
        all_dates = []
        for d in regime_df.index[:50]:
            for _ in range(5):
                all_dates.append(d)

        X_macro = _build_macro_array(
            all_dates, regime_df, list(cfg.MACRO_NORM_FEATURES),
        )
        assert X_macro.shape == (250, len(cfg.MACRO_NORM_FEATURES))
        # Stage 2c-full additions: indexes for the 5 new macros.
        for name in (
            "yield_curve_10y_2y",
            "hy_oas_level",
            "hy_oas_change_21d",
            "baa10y_level",
            "baa10y_change_21d",
        ):
            idx = cfg.MACRO_NORM_FEATURES.index(name)
            # Column is finite (synthetic flat data has no NaN warmup).
            assert np.isfinite(X_macro[:, idx]).all()


# ── Backwards compatibility ──────────────────────────────────────────


class TestBackwardsCompat:
    """build_features() still works without the new optional parameters
    (existing callers in tests + production unchanged)."""

    def test_legacy_call_signature_works(self):
        rp = RegimePredictor()
        # Original 5 positional + all_close_prices. No new args.
        out = rp.build_features(
            spy_series=_spy_series(),
            vix_series=_flat_series(level=18.0),
            vix3m_series=_flat_series(level=20.0),
            tnx_series=_flat_series(level=4.0),
            irx_series=_flat_series(level=2.0),
        )
        # New columns exist with neutral 0.0 fallback (no series → 0).
        assert (out["yield_curve_10y_2y"] == 0.0).all()
        assert (out["hy_oas_level"] == 0.0).all()
        assert (out["baa10y_level"] == 0.0).all()
