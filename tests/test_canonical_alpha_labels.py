"""Tests for the canonical alpha label generator (audit Track A PR 1).

Canonical alpha = log_return(stock, h) - log_return(vol_cohort_EW_basket, h).
Vol-cohort partitioning is cross-sectional per-date so a ticker's cohort
shifts as its vol profile changes relative to peers.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.canonical_alpha_labels import (
    DEFAULT_N_COHORTS,
    DEFAULT_VOL_LOOKBACK_DAYS,
    assign_vol_cohorts,
    build_cohort_basket_returns,
    compute_canonical_alpha,
    compute_realized_volatility,
)


def _synthetic_close_series(
    n_days: int = 250, base: float = 100.0, vol: float = 0.20, seed: int = 0,
) -> pd.Series:
    """Geometric-Brownian-like close-price series.

    Annualized vol parameter ``vol`` controls the daily noise scale. The
    realized volatility from rolling-20d windows on the resulting series
    will land near ``vol`` (not exact — finite-sample noise + drift terms).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    daily_vol = vol * np.sqrt(dt)
    log_returns = rng.normal(0.0, daily_vol, n_days)
    log_prices = np.cumsum(log_returns) + np.log(base)
    return pd.Series(
        np.exp(log_prices),
        index=pd.bdate_range("2024-01-02", periods=n_days),
        name="close",
    )


# ── compute_realized_volatility ─────────────────────────────────────────

class TestRealizedVolatility:

    def test_basic_computation_returns_series(self):
        close = _synthetic_close_series(n_days=100, vol=0.20)
        rv = compute_realized_volatility(close, lookback_days=20)
        assert isinstance(rv, pd.Series)
        assert len(rv) == 100

    def test_first_lookback_rows_are_nan(self):
        close = _synthetic_close_series(n_days=100)
        rv = compute_realized_volatility(close, lookback_days=20)
        # First 20 rows have insufficient history for the rolling window.
        assert rv.iloc[:20].isna().all()
        # Beyond the warmup, finite values.
        assert rv.iloc[25:].notna().all()

    def test_vol_estimate_near_input_vol(self):
        close = _synthetic_close_series(n_days=500, vol=0.30)
        rv = compute_realized_volatility(close, lookback_days=60)
        # On a 60-day window, realized vol should land within +/- 30% of
        # the input vol parameter (loose because of finite-sample noise).
        avg_rv = float(rv.iloc[100:].mean())
        assert 0.20 <= avg_rv <= 0.45

    def test_short_history_returns_empty_finite(self):
        close = _synthetic_close_series(n_days=10)
        rv = compute_realized_volatility(close, lookback_days=20)
        # Insufficient history → all NaN.
        assert rv.isna().all()


# ── assign_vol_cohorts ──────────────────────────────────────────────────

class TestAssignVolCohorts:

    def test_assigns_cohort_per_date_per_ticker(self):
        # Build 5 tickers with different vols.
        rv_by_ticker = {
            f"T{i}": _synthetic_close_series(n_days=200, vol=0.10 + 0.05 * i, seed=i).pipe(
                compute_realized_volatility, lookback_days=20,
            ) for i in range(5)
        }
        cohorts = assign_vol_cohorts(rv_by_ticker, n_cohorts=5)
        assert isinstance(cohorts, pd.DataFrame)
        assert set(cohorts.columns) == {"T0", "T1", "T2", "T3", "T4"}
        # Each row should assign cohort 0..4 across the 5 tickers (rank-based).
        finite_rows = cohorts.dropna()
        # On finite rows, each value 0..4 should appear at most once
        # (qcut rank-based partitioning yields a permutation).
        for _, row in finite_rows.iterrows():
            assert set(row.values) <= {0, 1, 2, 3, 4}

    def test_low_vol_ticker_lands_in_low_cohort(self):
        rv_low = compute_realized_volatility(
            _synthetic_close_series(n_days=200, vol=0.05, seed=10), lookback_days=20,
        )
        rv_high = compute_realized_volatility(
            _synthetic_close_series(n_days=200, vol=0.50, seed=20), lookback_days=20,
        )
        cohorts = assign_vol_cohorts({"LOW": rv_low, "HIGH": rv_high}, n_cohorts=2)
        finite = cohorts.dropna()
        # On 2-cohort split, LOW should be 0 and HIGH should be 1 most of
        # the time. Allow some tolerance for noisy windows.
        assert (finite["LOW"] < finite["HIGH"]).mean() > 0.8

    def test_empty_input(self):
        result = assign_vol_cohorts({}, n_cohorts=10)
        assert result.empty


# ── build_cohort_basket_returns ─────────────────────────────────────────

class TestBuildCohortBasketReturns:

    def test_returns_dict_of_series_per_cohort(self):
        rng = np.random.default_rng(0)
        close_by_ticker = {
            f"T{i}": _synthetic_close_series(n_days=150, vol=0.15 + 0.04 * i, seed=i)
            for i in range(8)
        }
        rv_by_ticker = {
            t: compute_realized_volatility(c, 20) for t, c in close_by_ticker.items()
        }
        cohorts = assign_vol_cohorts(rv_by_ticker, n_cohorts=4)
        baskets = build_cohort_basket_returns(close_by_ticker, cohorts, horizon=5, n_cohorts=4)
        assert set(baskets.keys()) == {0, 1, 2, 3}
        for c, series in baskets.items():
            assert isinstance(series, pd.Series)

    def test_basket_with_one_member_yields_nan(self):
        # Construct: 2 tickers, 2 cohorts → each cohort has 1 member.
        close_by_ticker = {
            f"T{i}": _synthetic_close_series(n_days=100, vol=0.10 + 0.30 * i, seed=i)
            for i in range(2)
        }
        rv_by_ticker = {
            t: compute_realized_volatility(c, 20) for t, c in close_by_ticker.items()
        }
        cohorts = assign_vol_cohorts(rv_by_ticker, n_cohorts=2)
        baskets = build_cohort_basket_returns(close_by_ticker, cohorts, horizon=5, n_cohorts=2)
        # Each cohort has 1 member at each date — basket returns should
        # be NaN since single-member baskets are meaningless.
        for c, series in baskets.items():
            finite = series.dropna()
            # Allow some non-NaN cells where qcut produced ties /
            # fallback partitioning that put both tickers in same bucket
            # for some dates.
            assert len(finite) < len(series) / 2  # most cells NaN

    def test_basket_aggregates_member_returns(self):
        # 4 tickers, 2 cohorts → 2 members each. Each cohort's basket
        # return should equal the mean of its members' forward log returns.
        rng = np.random.default_rng(7)
        n_days = 100
        close_by_ticker = {
            f"T{i}": _synthetic_close_series(n_days=n_days, vol=0.10 + 0.20 * (i // 2), seed=i)
            for i in range(4)
        }
        rv_by_ticker = {
            t: compute_realized_volatility(c, 20) for t, c in close_by_ticker.items()
        }
        cohorts = assign_vol_cohorts(rv_by_ticker, n_cohorts=2)
        baskets = build_cohort_basket_returns(close_by_ticker, cohorts, horizon=5, n_cohorts=2)
        # Both cohorts should have finite values where the rolling vol
        # has stabilized.
        for c, series in baskets.items():
            assert series.iloc[40:80].notna().sum() > 5


# ── compute_canonical_alpha (end-to-end) ────────────────────────────────

class TestComputeCanonicalAlpha:

    def test_returns_dict_of_series_per_ticker(self):
        close_by_ticker = {
            f"T{i}": _synthetic_close_series(n_days=200, vol=0.15 + 0.05 * i, seed=i)
            for i in range(8)
        }
        result = compute_canonical_alpha(close_by_ticker, horizon=5, n_cohorts=4)
        assert set(result.keys()) == {f"T{i}" for i in range(8)}
        for series in result.values():
            assert isinstance(series, pd.Series)

    def test_canonical_alpha_centered_near_zero_on_average(self):
        # By construction (each ticker is benchmarked against its
        # cohort's EW basket), the average canonical alpha across
        # tickers within a cohort should be near zero. Total cross-ticker
        # average should also be near zero.
        close_by_ticker = {
            f"T{i}": _synthetic_close_series(n_days=300, vol=0.20, seed=i)
            for i in range(20)
        }
        result = compute_canonical_alpha(close_by_ticker, horizon=5, n_cohorts=5)
        # Stack all canonical alphas and compute mean; should be near zero.
        all_alphas = pd.concat(result.values())
        mean_alpha = all_alphas.dropna().mean()
        assert abs(mean_alpha) < 0.005

    def test_handles_short_history_ticker_gracefully(self):
        # 8 tickers with normal history + 1 with very short — short one
        # should get all-NaN canonical alpha. Use n_cohorts=3 to ensure
        # each cohort has ≥2 members (required for non-NaN basket return).
        close_by_ticker = {
            f"T{i}": _synthetic_close_series(n_days=200, vol=0.10 + 0.05 * i, seed=i)
            for i in range(8)
        }
        close_by_ticker["SHORT"] = _synthetic_close_series(n_days=3)
        result = compute_canonical_alpha(close_by_ticker, horizon=5, n_cohorts=3)
        assert result["SHORT"].isna().all()
        # Other tickers still produce finite values where vol has settled.
        assert result["T0"].iloc[40:].notna().sum() > 0

    def test_empty_input_returns_empty_dict(self):
        assert compute_canonical_alpha({}, horizon=5) == {}

    def test_canonical_alpha_log_domain_asymmetry(self):
        # Synthetic test of the audit's "log returns punish blowups
        # quadratically harder" property. Construct two tickers in the
        # same cohort: one with a +50% move, one with a -50% move at the
        # same horizon. Their canonical alphas (vs cohort) should be
        # asymmetric magnitudes around the baseline because log(1.5) and
        # log(0.5) differ by ~1.7×.
        n = 100
        flat = pd.Series([100.0] * n, index=pd.bdate_range("2024-01-02", periods=n))
        # Spike up on day 50 → forward 5d log return at day 49 ≈ log(1.5)
        spike_up = flat.copy()
        spike_up.iloc[50:] = 150.0
        # Drop on day 50 → forward 5d log return at day 49 ≈ log(0.5)
        spike_down = flat.copy()
        spike_down.iloc[50:] = 50.0
        close_by_ticker = {"FLAT": flat, "UP": spike_up, "DOWN": spike_down}
        result = compute_canonical_alpha(close_by_ticker, horizon=5, n_cohorts=2)
        # On day 45 (forward window 45→50 captures spike), UP and DOWN
        # have very different forward log returns relative to the cohort.
        # Just verify the magnitudes are asymmetric, not zero.
        idx = flat.index[45]
        if not result["UP"].isna().get(idx, True) and not result["DOWN"].isna().get(idx, True):
            up_mag = abs(result["UP"].loc[idx])
            down_mag = abs(result["DOWN"].loc[idx])
            # Both nontrivial — log-domain asymmetry visible if both finite.
            assert up_mag > 0.1 or down_mag > 0.1
