"""
data/label_generator.py — Forward-return label computation.

Given a DataFrame of OHLCV + computed features, this module computes the
5-trading-day forward return for each row and bins it into three classes:

    UP   — forward_return > up_threshold   → label 2
    FLAT — forward_return in [down, up]    → label 1
    DOWN — forward_return < down_threshold → label 0

The integer mapping (DOWN=0, FLAT=1, UP=2) matches CLASS_LABELS in config.py
and the model output neuron indices.

When benchmark_returns (a benchmark Close series) is provided, the label is
based on the relative return vs the benchmark rather than the absolute return:

    relative_return_5d = stock_forward_return_5d - benchmark_forward_return_5d

Pass SPY Close for market-relative alpha, or a sector ETF Close (e.g. XLK) for
sector-neutral (idiosyncratic) alpha. Sector-neutral labels remove sector-level
noise and target the stock-specific signal more directly.

The column is still named forward_return_5d for backward compatibility
with IC computation downstream, but its value is the relative return.

Rows where the forward return cannot be computed (i.e., the last `forward_days`
rows of the series) are dropped — they have no valid label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeling.triple_barrier import triple_barrier_alpha_labels


def compute_labels(
    df: pd.DataFrame,
    forward_days: int = 5,
    up_threshold: float = 0.01,
    down_threshold: float = -0.01,
    benchmark_returns: pd.Series | None = None,
    adaptive_thresholds: bool = False,
    adaptive_window: int = 63,
    adaptive_up_pct: float = 65,
    adaptive_down_pct: float = 35,
) -> pd.DataFrame:
    """
    Append forward-return labels to a featured DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at minimum a 'Close' column.
        Typically output of feature_engineer.compute_features().
    forward_days : int
        Number of trading days ahead to compute the forward return.
        Default is 5 (one calendar week).
    up_threshold : float
        Minimum return to classify as UP. Default +1% (0.01).
        Ignored when adaptive_thresholds=True.
    down_threshold : float
        Maximum return to classify as DOWN. Default -1% (-0.01).
        Ignored when adaptive_thresholds=True.
    benchmark_returns : pd.Series or None
        Benchmark Close prices with a DatetimeIndex. When provided, labels are
        based on stock return minus benchmark return (relative alpha), rather
        than the absolute return. The forward_return_5d column stores the
        relative return value for IC computation.

        Pass SPY Close for market-relative alpha (original behaviour).
        Pass a sector ETF Close (e.g. XLK for Information Technology) for
        sector-neutral (idiosyncratic) alpha — recommended when sector_map.json
        is available, as it removes sector-level noise from the training target.
    adaptive_thresholds : bool
        When True, UP/DOWN thresholds are computed per-row as rolling percentiles
        of forward returns, adapting to the current volatility regime. This avoids
        the problem of fixed ±1% being too easy in high-vol and too hard in low-vol.
    adaptive_window : int
        Rolling window (trading days) for computing adaptive percentiles.
        Default 63 (~3 months).
    adaptive_up_pct : float
        Percentile for UP threshold (e.g. 65 = top 35% of returns → UP).
    adaptive_down_pct : float
        Percentile for DOWN threshold (e.g. 35 = bottom 35% of returns → DOWN).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with three new columns appended:
        - forward_return_5d  (float) : absolute or relative 5-day forward return
        - direction          (str)   : "UP", "FLAT", or "DOWN"
        - direction_int      (int)   : 2, 1, or 0 respectively
        Rows where forward_return_5d is NaN (last forward_days rows) are
        dropped, since they have no label.
    """
    if df.empty:
        df = df.copy()
        df["forward_return_5d"] = pd.Series(dtype=float)
        df["direction"] = pd.Series(dtype=str)
        df["direction_int"] = pd.Series(dtype=int)
        return df

    df = df.copy()
    close = df["Close"].astype(float)

    # Stock forward return: (future price / current price) - 1
    future_close = close.shift(-forward_days)
    stock_fwd_return = (future_close / close) - 1.0

    if benchmark_returns is not None:
        # Relative return: stock alpha vs benchmark over the same forward window.
        # benchmark_returns is a Close series; align to ticker's date index.
        benchmark_aligned = benchmark_returns.reindex(df.index)
        benchmark_future = benchmark_aligned.shift(-forward_days)
        benchmark_fwd_return = (benchmark_future / benchmark_aligned) - 1.0
        df["forward_return_5d"] = stock_fwd_return - benchmark_fwd_return
    else:
        df["forward_return_5d"] = stock_fwd_return

    # Drop rows where the forward return is undefined (end of series)
    df = df.dropna(subset=["forward_return_5d"])

    if adaptive_thresholds and len(df) >= adaptive_window:
        # Volatility-adaptive thresholds: rolling percentiles of forward returns.
        # In low-vol periods, thresholds tighten; in high-vol, they widen.
        rolling_up = df["forward_return_5d"].rolling(
            window=adaptive_window, min_periods=adaptive_window // 2
        ).quantile(adaptive_up_pct / 100.0)
        rolling_down = df["forward_return_5d"].rolling(
            window=adaptive_window, min_periods=adaptive_window // 2
        ).quantile(adaptive_down_pct / 100.0)

        # Fall back to fixed thresholds where rolling data is insufficient
        rolling_up = rolling_up.fillna(up_threshold)
        rolling_down = rolling_down.fillna(down_threshold)

        conditions = [
            df["forward_return_5d"] > rolling_up,
            df["forward_return_5d"] < rolling_down,
        ]
    else:
        # Fixed thresholds (original behavior)
        conditions = [
            df["forward_return_5d"] > up_threshold,
            df["forward_return_5d"] < down_threshold,
        ]

    choices_str = ["UP", "DOWN"]
    df["direction"] = np.select(conditions, choices_str, default="FLAT")

    # Integer labels: DOWN=0, FLAT=1, UP=2 (matches CLASS_LABELS order in config)
    label_map = {"DOWN": 0, "FLAT": 1, "UP": 2}
    df["direction_int"] = df["direction"].map(label_map).astype(int)

    return df


def compute_multi_horizon_labels(
    df: pd.DataFrame,
    horizons: list[int] = (1, 5, 10, 20),
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute forward returns for multiple horizons.

    Adds columns `forward_return_{h}d` for each horizon h. The primary label
    (forward_return_5d) is always included. Other horizons are auxiliary.

    Does NOT add direction/direction_int columns — those are computed
    by compute_labels() for the primary horizon only.

    Parameters
    ----------
    df : DataFrame with 'Close' column.
    horizons : list of forward-day horizons.
    benchmark_returns : Benchmark Close series for relative returns.

    Returns
    -------
    DataFrame with forward_return_{h}d columns added. Rows where the
    longest horizon return is NaN are dropped.
    """
    if df.empty:
        df = df.copy()
        for h in horizons:
            df[f"forward_return_{h}d"] = pd.Series(dtype=float)
        return df

    df = df.copy()
    close = df["Close"].astype(float)

    if benchmark_returns is not None:
        bench_aligned = benchmark_returns.reindex(df.index)

    for h in horizons:
        future_close = close.shift(-h)
        stock_fwd = (future_close / close) - 1.0
        if benchmark_returns is not None:
            bench_future = bench_aligned.shift(-h)
            bench_fwd = (bench_future / bench_aligned) - 1.0
            df[f"forward_return_{h}d"] = stock_fwd - bench_fwd
        else:
            df[f"forward_return_{h}d"] = stock_fwd

    # Drop rows where the longest horizon is NaN
    max_h = max(horizons)
    df = df.dropna(subset=[f"forward_return_{max_h}d"])

    return df


def compute_triple_barrier_alpha_labels(
    df: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    forward_window: int = 21,
    vol_window: int = 20,
    vol_multiplier: float = 2.0,
    min_periods: int = 10,
    column_name: str = "triple_barrier_alpha_21d",
) -> pd.DataFrame:
    """Append vol-scaled triple-barrier alpha labels to a featured DataFrame.

    Per LdP *Advances in Financial Machine Learning* Ch. 3.4: for each row,
    walk forward ``forward_window`` trading days using sector-neutral
    residual log-returns and label by which barrier the path touches first
    (profit-take, stop-loss, or time-out). Barriers are scaled to recent
    realized vol (``barrier = vol_multiplier × σ_t``) so the label adapts
    to each ticker's vol regime continuously instead of using static
    constants that produce asymmetrically degenerate labels across
    high-vol vs low-vol names.

    Sector-neutral residual log-returns are computed per-step:
    ``residual_t = log(close_t / close_{t-1}) - log(bench_t / bench_{t-1})``
    σ_t is the EWMA std of the residual log-return series with span
    ``vol_window``. Rows with insufficient EWMA history (< ``min_periods``)
    receive NaN labels and are NOT dropped — caller is responsible for
    handling NaN rows downstream (typical pattern: filter at training
    array assembly).

    The returned label is the realized residual log-return at first touch
    (capped at the touched barrier) or window-end residual log-return
    (uncapped) for time-out rows. Tail rows past data end are NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Close`` column with DatetimeIndex.
    benchmark_returns : pd.Series or None
        Benchmark Close prices (e.g. sector ETF for sector-neutral alpha,
        SPY for market-relative alpha). When None, labels are based on
        absolute log-returns instead of residuals.
    forward_window : int
        Time-out barrier in trading days. Default 21 (matches the
        post-2026-05-09 horizon migration).
    vol_window : int
        EWMA span for vol estimate. Default 20 (≈ 1 trading month).
    vol_multiplier : float
        Barrier width as multiple of σ_t. Default 2.0 (≈ 2-sigma barriers,
        LdP-standard).
    min_periods : int
        Minimum EWMA-window observations before σ_t is non-NaN. Below
        this, label is NaN. Default 10.
    column_name : str
        Output column name. Default ``triple_barrier_alpha_21d``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``column_name`` column appended. NaN at
        front-of-history (insufficient vol history) and tail (past
        data end). Rows are NOT dropped — use this column alongside
        the legacy ``forward_return_5d`` from ``compute_labels``.
    """
    if df.empty:
        df = df.copy()
        df[column_name] = pd.Series(dtype=float)
        return df

    df = df.copy()
    close = df["Close"].astype(float)
    stock_log_ret = np.log(close / close.shift(1))

    if benchmark_returns is not None:
        bench_aligned = benchmark_returns.reindex(df.index, method="ffill").astype(float)
        bench_log_ret = np.log(bench_aligned / bench_aligned.shift(1))
        residual_log_ret = stock_log_ret - bench_log_ret
    else:
        residual_log_ret = stock_log_ret

    # First row is NaN by construction (shift(1)) — replace with 0.0 for
    # the cum-walk array (label at front-of-history will be NaN anyway
    # via the vol-history gate below).
    log_ret_arr = residual_log_ret.fillna(0.0).to_numpy(dtype=np.float64)

    sigma = residual_log_ret.ewm(span=vol_window, min_periods=min_periods).std()
    sigma_arr = sigma.to_numpy(dtype=np.float64)
    barrier_arr = vol_multiplier * sigma_arr  # NaN where sigma is NaN

    labels = triple_barrier_alpha_labels(
        log_ret_arr,
        forward_window=forward_window,
        up_barrier_pct=barrier_arr,
        down_barrier_pct=barrier_arr,
    )
    df[column_name] = labels
    return df


def label_distribution(df: pd.DataFrame) -> dict[str, float]:
    """
    Return the class distribution as proportions.
    Useful for checking class imbalance before training.

    Returns
    -------
    dict with keys "UP", "FLAT", "DOWN" and float proportion values.
    """
    if "direction" not in df.columns or df.empty:
        return {"UP": 0.0, "FLAT": 0.0, "DOWN": 0.0}

    counts = df["direction"].value_counts(normalize=True)
    return {
        "UP": float(counts.get("UP", 0.0)),
        "FLAT": float(counts.get("FLAT", 0.0)),
        "DOWN": float(counts.get("DOWN", 0.0)),
    }
