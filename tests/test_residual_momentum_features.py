"""Tests for residual-momentum feature construction (W2, L4469).

The load-bearing tests: known-beta recovery (residual tracks idio, not market),
strict no-look-ahead / point-in-time, the 12-1 skip-month convention,
front-of-history NaN, and graceful all-NaN on short history.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.residual_momentum_features import compute_residual_momentum_features
from model.residual_momentum_scorer import RESIDUAL_MOMENTUM_FEATURES

_PARAMS = dict(beta_window=60, mom_window=252, skip_window=21, vol_window=20, mom_change_window=21)


def _closes_from_returns(r: np.ndarray, start: float = 100.0) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(r), freq="B")
    return pd.Series(start * np.cumprod(1.0 + r), index=idx)


def test_columns_and_index_contract():
    close = _closes_from_returns(np.random.default_rng(0).normal(0, 0.01, 400))
    bench = _closes_from_returns(np.random.default_rng(1).normal(0, 0.01, 400))
    out = compute_residual_momentum_features(close, bench, bench, **_PARAMS)
    assert list(out.columns) == RESIDUAL_MOMENTUM_FEATURES
    assert out.index.equals(close.index)


def test_pure_beta_stock_has_zero_residual_momentum():
    # Stock return is EXACTLY beta * market return (no idiosyncratic component),
    # even with a strong market trend → residual is ~0 → resid_mom_vol_scaled ~0.
    rng = np.random.default_rng(42)
    r_bench = 0.001 + rng.normal(0, 0.01, 500)  # strong positive market drift
    r = 1.5 * r_bench                            # pure beta, idio == 0
    close = _closes_from_returns(r)
    bench = _closes_from_returns(r_bench)
    out = compute_residual_momentum_features(close, bench, bench, **_PARAMS)
    last = out["resid_mom_vol_scaled"].iloc[-1]
    assert np.isfinite(last)
    assert abs(last) < 1e-6  # residual identically zero up to float error


def test_residual_momentum_tracks_idio_not_raw_price():
    # Market trends DOWN, idiosyncratic component trends UP (beta=1). Then raw
    # price momentum (mom_12_1) is NEGATIVE but RESIDUAL momentum is POSITIVE —
    # the exact distinction W2.1 exists to capture.
    rng = np.random.default_rng(7)
    n = 500
    r_bench = -0.001 + rng.normal(0, 0.001, n)   # market drifts down
    idio = 0.0008 + rng.normal(0, 0.001, n)      # stock-specific drift up
    r = r_bench + idio                           # beta_true = 1
    close = _closes_from_returns(r)
    bench = _closes_from_returns(r_bench)
    out = compute_residual_momentum_features(close, bench, bench, **_PARAMS)
    assert out["resid_mom_vol_scaled"].iloc[-1] > 0   # residual momentum up
    assert out["mom_12_1"].iloc[-1] < 0               # raw price momentum down


def test_no_look_ahead_point_in_time():
    # Mutating prices STRICTLY AFTER date t must not change any feature at/<= t.
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 400)
    close = _closes_from_returns(r)
    bench = _closes_from_returns(rng.normal(0, 0.01, 400))
    feats1 = compute_residual_momentum_features(close, bench, bench, **_PARAMS)

    t = 350  # well past the warmup horizon
    close2 = close.copy()
    close2.iloc[t + 1:] *= 1.5  # perturb only the future
    feats2 = compute_residual_momentum_features(close2, bench, bench, **_PARAMS)

    a = feats1.iloc[: t + 1].to_numpy()
    b = feats2.iloc[: t + 1].to_numpy()
    # equal_nan: warmup NaNs must match too.
    assert np.array_equal(a, b, equal_nan=True)


def test_skip_month_excludes_recent_window():
    # mom_12_1 must NOT depend on the most-recent skip_window days; mom_1m must.
    rng = np.random.default_rng(5)
    close = _closes_from_returns(rng.normal(0, 0.01, 400))
    bench = _closes_from_returns(rng.normal(0, 0.01, 400))
    feats1 = compute_residual_momentum_features(close, bench, bench, **_PARAMS)

    skip = _PARAMS["skip_window"]
    close2 = close.copy()
    close2.iloc[-skip:] *= 1.3  # mutate only the most-recent month
    feats2 = compute_residual_momentum_features(close2, bench, bench, **_PARAMS)

    assert feats1["mom_12_1"].iloc[-1] == feats2["mom_12_1"].iloc[-1]   # unchanged
    assert feats1["mom_1m"].iloc[-1] != feats2["mom_1m"].iloc[-1]       # changed


def test_front_of_history_is_nan_not_filled():
    rng = np.random.default_rng(9)
    close = _closes_from_returns(rng.normal(0, 0.01, 400))
    bench = _closes_from_returns(rng.normal(0, 0.01, 400))
    out = compute_residual_momentum_features(close, bench, bench, **_PARAMS)
    w = _PARAMS["mom_window"]
    # The 252-window columns are entirely NaN before the window warms up.
    assert out["mom_12_1"].iloc[:w].isna().all()
    assert out["sector_mom"].iloc[:w].isna().all()
    assert out["resid_mom_vol_scaled"].iloc[: _PARAMS["beta_window"]].isna().all()


def test_short_history_yields_all_nan_long_window_cols_no_crash():
    # < window rows → the long-lookback columns are fully NaN (neutralized
    # downstream), and the call does not crash.
    rng = np.random.default_rng(11)
    close = _closes_from_returns(rng.normal(0, 0.01, 100))  # < 252
    bench = _closes_from_returns(rng.normal(0, 0.01, 100))
    out = compute_residual_momentum_features(close, bench, bench, **_PARAMS)
    assert list(out.columns) == RESIDUAL_MOMENTUM_FEATURES
    assert out["mom_12_1"].isna().all()
    assert out["resid_mom_vol_scaled"].isna().all()


def test_none_benchmark_falls_back_to_spy():
    rng = np.random.default_rng(13)
    close = _closes_from_returns(rng.normal(0, 0.01, 400))
    spy = _closes_from_returns(rng.normal(0, 0.01, 400))
    out = compute_residual_momentum_features(close, None, spy, **_PARAMS)
    assert list(out.columns) == RESIDUAL_MOMENTUM_FEATURES
    # sector_mom should be populated from SPY (the fallback), not all-NaN late.
    assert np.isfinite(out["sector_mom"].iloc[-1])


def test_no_benchmark_at_all_returns_all_nan():
    rng = np.random.default_rng(15)
    close = _closes_from_returns(rng.normal(0, 0.01, 400))
    out = compute_residual_momentum_features(close, None, None, **_PARAMS)
    assert list(out.columns) == RESIDUAL_MOMENTUM_FEATURES
    assert out.isna().all().all()
