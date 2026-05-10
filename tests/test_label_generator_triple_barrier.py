"""Tests for ``data.label_generator.compute_triple_barrier_alpha_labels``.

Validates the Stage 3 PR 1 vol-scaled label producer:

- Sector-neutral residual log-returns drive the barrier walk
- EWMA-20 vol with min_periods gate produces NaN labels at front-of-history
- Tail rows past data end are NaN (delegated to ``triple_barrier_alpha_labels``)
- Original ``compute_labels`` behavior is unaffected by this addition
- Vol-scaled barriers actually scale: doubling vol roughly doubles the
  barrier-cap value at vol-stationary regimes
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.label_generator import (
    compute_labels,
    compute_triple_barrier_alpha_labels,
)


def _make_price_df(daily_log_ret: float, n: int, start: str = "2024-01-02") -> pd.DataFrame:
    """Construct a DataFrame with a Close series compounding at fixed log-return."""
    idx = pd.bdate_range(start=start, periods=n)
    close = 100.0 * np.exp(np.cumsum(np.full(n, daily_log_ret)))
    return pd.DataFrame({"Close": close}, index=idx)


def _make_benchmark(daily_log_ret: float, n: int, start: str = "2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=n)
    bench = 100.0 * np.exp(np.cumsum(np.full(n, daily_log_ret)))
    return pd.Series(bench, index=idx, name="benchmark")


class TestComputeTripleBarrierAlphaLabels:

    def test_empty_df_returns_empty_with_column(self):
        df = pd.DataFrame({"Close": pd.Series(dtype=float)})
        out = compute_triple_barrier_alpha_labels(df)
        assert "triple_barrier_alpha_21d" in out.columns
        assert out.empty

    def test_front_of_history_is_nan_due_to_vol_gate(self):
        # min_periods=10 → first ~10 rows have NaN σ_t → NaN labels.
        df = _make_price_df(daily_log_ret=0.001, n=80)
        out = compute_triple_barrier_alpha_labels(
            df,
            benchmark_returns=None,
            forward_window=21,
            vol_window=20,
            vol_multiplier=2.0,
            min_periods=10,
        )
        labels = out["triple_barrier_alpha_21d"].to_numpy()
        # First few rows MUST be NaN (vol still warming up)
        assert np.isnan(labels[:5]).all()

    def test_tail_rows_are_nan(self):
        df = _make_price_df(daily_log_ret=0.001, n=80)
        out = compute_triple_barrier_alpha_labels(df, forward_window=21)
        labels = out["triple_barrier_alpha_21d"].to_numpy()
        # Last forward_window rows must be NaN (past data end)
        assert np.isnan(labels[-21:]).all()

    def test_sector_neutral_residual_is_zero_when_stock_eq_benchmark(self):
        # When stock and benchmark log-returns are identical, residual is
        # exactly 0 every step → label = 0 (uncapped time-out).
        n = 80
        df = _make_price_df(daily_log_ret=0.005, n=n)
        bench = _make_benchmark(daily_log_ret=0.005, n=n)
        out = compute_triple_barrier_alpha_labels(
            df,
            benchmark_returns=bench,
            forward_window=21,
            vol_window=20,
            min_periods=10,
        )
        labels = out["triple_barrier_alpha_21d"].to_numpy()
        finite = labels[~np.isnan(labels)]
        # All finite labels are exactly 0 — residual is 0 every step;
        # vol of zero series is zero → barriers are zero → BUT the
        # walk hits up_barrier=0 immediately on a non-strictly-negative
        # cum, capping at 0. Either way: label is 0.
        assert np.allclose(finite, 0.0, atol=1e-12)

    def test_outperforming_stock_yields_positive_residual_label(self):
        # Stock returns 0.005/day, benchmark returns 0.001/day → residual
        # = 0.004/day. Over 21 days residual cum = 0.084. Barrier scales
        # to vol of residual; since residual is constant 0.004, EWMA std
        # of a constant is ~0 → barriers ~0 → first-step touch.
        # We instead test on a noisy stock with positive drift to confirm
        # directionality without the constant-series edge.
        rng = np.random.default_rng(42)
        n = 200
        idx = pd.bdate_range(start="2024-01-02", periods=n)
        stock_log = rng.normal(0.002, 0.01, n)  # positive drift, vol ~1%
        bench_log = rng.normal(0.0, 0.01, n)
        stock_close = 100.0 * np.exp(np.cumsum(stock_log))
        bench_close = 100.0 * np.exp(np.cumsum(bench_log))
        df = pd.DataFrame({"Close": stock_close}, index=idx)
        bench = pd.Series(bench_close, index=idx)
        out = compute_triple_barrier_alpha_labels(
            df,
            benchmark_returns=bench,
            forward_window=21,
            vol_window=20,
            vol_multiplier=2.0,
            min_periods=10,
        )
        labels = out["triple_barrier_alpha_21d"].to_numpy()
        finite = labels[~np.isnan(labels)]
        # Positive-drift residual should produce mean-positive labels
        # (touched up barrier more often than down barrier).
        assert finite.mean() > 0

    def test_higher_vol_yields_wider_barrier_caps(self):
        # Doubling realized vol (with vol_multiplier fixed) should roughly
        # double the absolute barrier values, observable as a wider |label|
        # spread for hit rows. Use two stationary regimes: low-vol vs high-vol.
        rng = np.random.default_rng(0)
        n = 300
        idx = pd.bdate_range(start="2024-01-02", periods=n)
        # Low-vol stock: σ=0.005/day. High-vol stock: σ=0.015/day. Same drift.
        low_vol_log = rng.normal(0.0, 0.005, n)
        high_vol_log = rng.normal(0.0, 0.015, n)
        low_close = 100.0 * np.exp(np.cumsum(low_vol_log))
        high_close = 100.0 * np.exp(np.cumsum(high_vol_log))
        low_df = pd.DataFrame({"Close": low_close}, index=idx)
        high_df = pd.DataFrame({"Close": high_close}, index=idx)

        low_labels = compute_triple_barrier_alpha_labels(
            low_df,
            benchmark_returns=None,
            forward_window=21,
            vol_multiplier=2.0,
        )["triple_barrier_alpha_21d"].to_numpy()
        high_labels = compute_triple_barrier_alpha_labels(
            high_df,
            benchmark_returns=None,
            forward_window=21,
            vol_multiplier=2.0,
        )["triple_barrier_alpha_21d"].to_numpy()
        low_finite = low_labels[~np.isnan(low_labels)]
        high_finite = high_labels[~np.isnan(high_labels)]
        # High-vol mean-abs-label should be materially larger than low-vol's.
        assert np.mean(np.abs(high_finite)) > 2.0 * np.mean(np.abs(low_finite))

    def test_does_not_drop_rows(self):
        # Unlike compute_labels (which drops tail rows), this function
        # preserves the input row count — caller filters NaN downstream.
        df = _make_price_df(daily_log_ret=0.001, n=100)
        out = compute_triple_barrier_alpha_labels(df, forward_window=21)
        assert len(out) == len(df)


class TestComputeLabelsUnaffected:
    """compute_labels is unchanged in PR 1 — the new triple-barrier
    function is a sibling, not a modification of the existing API."""

    def test_compute_labels_signature_unchanged(self):
        df = _make_price_df(daily_log_ret=0.002, n=100)
        out = compute_labels(df, forward_days=21, up_threshold=0.05, down_threshold=-0.05)
        # Original columns still emitted
        assert "forward_return_5d" in out.columns
        assert "direction" in out.columns
        assert "direction_int" in out.columns
        # Triple-barrier column NOT added by compute_labels
        assert "triple_barrier_alpha_21d" not in out.columns
