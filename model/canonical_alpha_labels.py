"""
model/canonical_alpha_labels.py — canonical alpha label generator (audit Track A PR 1).

Implements the audit §7.3 canonical alpha definition:

  canonical_alpha(stock, t, h) = log_return(stock, h) - log_return(vol_cohort_EW_basket, h)

Where ``vol_cohort_EW_basket`` is the equal-weight basket of tickers in
the same trailing-volatility decile as ``stock`` at time ``t``. This
benchmarks the ticker against other tickers with similar volatility
profile at the same point in time, removing the volatility-aversion
bias that the legacy ``stock_5d_return - sector_etf_5d_return`` arithmetic
label has (LightGBM regression on arithmetic alpha gets pulled toward
smoother, lower-vol forecasts because high realized variance translates
to high MSE penalty).

Per audit §7.4: log returns punish blowups quadratically harder than
equivalent gains reward (``log(0.5)=-0.693`` vs ``log(1.5)=+0.405``,
~1.7× asymmetry). Drawdown aversion falls out for free without a
separate risk metric or asymmetric loss bolt-on.

This is **PR 1 of N** in the Track A sequence — module + tests, no
training/inference integration yet:

- PR 1 (this): canonical-label generator module
- PR 2: training integration in meta_trainer (compute canonical labels
  alongside legacy; persist diagnostic to manifest, no parallel Ridge yet)
- PR 3: parallel canonical-label Ridge fit + manifest comparison
- PR 4: inference parallel path (canonical-label predictor)
- PR 5: cutover (canonical labels become production)
- PR 6: cleanup of legacy-label code path post-validation

Same observe-first pattern as Phase 3 (research-LGB) and Phase 4
(regime-conditioned ensembling).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Default volatility lookback for cohort partitioning. 20 trading days
# matches the existing META_FEATURES `macro_spy_20d_vol` window so the
# ticker's "current vol regime" is computed on the same horizon the
# meta-Ridge is consuming.
DEFAULT_VOL_LOOKBACK_DAYS = 20

# Default number of vol cohorts. Deciles (10) is conventional for
# cross-sectional cohorting and gives ~90 tickers per cohort on a
# ~900-ticker universe — enough for stable EW basket returns. Halve
# to quintiles for narrower universes.
DEFAULT_N_COHORTS = 10


def compute_realized_volatility(
    close_series: pd.Series,
    lookback_days: int = DEFAULT_VOL_LOOKBACK_DAYS,
) -> pd.Series:
    """Annualized rolling realized volatility from a close-price series.

    Uses log returns for stability at extremes (matches the canonical
    alpha definition's log-return convention). Returns a Series indexed
    by date — NaN for the first ``lookback_days`` rows where insufficient
    history exists for the rolling window.
    """
    if len(close_series) < lookback_days + 1:
        return pd.Series(index=close_series.index, dtype=float)
    log_returns = np.log(close_series / close_series.shift(1))
    realized_vol = log_returns.rolling(lookback_days).std() * np.sqrt(252)
    return realized_vol


def assign_vol_cohorts(
    realized_vol_by_ticker: dict[str, pd.Series],
    n_cohorts: int = DEFAULT_N_COHORTS,
) -> pd.DataFrame:
    """Cross-sectional vol-cohort assignment per (date, ticker).

    For each date, ranks all tickers by realized volatility and partitions
    into ``n_cohorts`` equal-size buckets (cohort 0 = lowest vol, cohort
    n_cohorts-1 = highest vol).

    Returns a DataFrame with columns = ticker symbols, rows = dates,
    integer cohort assignment (0..n_cohorts-1) per cell. NaN where the
    ticker has insufficient history at that date.

    Implementation note: the cross-sectional rank happens at each date
    independently — a ticker's cohort can shift over time as its vol
    profile changes relative to peers. That's the audit's intent: the
    benchmark is "tickers with similar vol RIGHT NOW", not a static
    classification.
    """
    if not realized_vol_by_ticker:
        return pd.DataFrame()

    # Build wide DataFrame: rows=dates, cols=tickers, cells=realized_vol.
    vol_wide = pd.DataFrame(realized_vol_by_ticker)

    # Cross-sectional rank by row (date), then bucket into n_cohorts.
    # qcut returns NaN for rows where rank is undefined (all-NaN rows
    # or rows with fewer unique values than n_cohorts).
    cohorts = vol_wide.apply(
        lambda row: pd.qcut(
            row.dropna().rank(method="first"),
            q=n_cohorts,
            labels=False,
            duplicates="drop",
        ).reindex(row.index),
        axis=1,
    )
    return cohorts


def build_cohort_basket_returns(
    close_by_ticker: dict[str, pd.Series],
    cohorts: pd.DataFrame,
    horizon: int = 5,
    n_cohorts: int = DEFAULT_N_COHORTS,
) -> dict[int, pd.Series]:
    """For each cohort, compute the equal-weight forward log return at horizon h.

    For cohort c at date t with members M_c(t), the basket forward return
    is the mean of ``log(close[t+h] / close[t])`` across members, where
    membership is taken as of date t. Returns a dict ``{cohort_idx: series_of_basket_log_returns}``.

    NaN for dates where the cohort has fewer than 2 members or where the
    forward window extends past data end.

    Note: the basket return is forward-looking at time t — uses members
    classified at t but returns measured over (t, t+h). This is the
    standard event-study convention and matches the existing legacy-label
    construction in meta_trainer.py (``forward_return_{h}d`` columns).
    """
    if cohorts.empty:
        return {}

    # Build wide forward-log-return matrix (rows=dates, cols=tickers).
    forward_log_returns = {}
    for ticker, close in close_by_ticker.items():
        if len(close) < horizon + 1:
            continue
        forward_log_returns[ticker] = np.log(close.shift(-horizon) / close)
    forward_wide = pd.DataFrame(forward_log_returns)

    # Align cohorts and forward returns on common date index.
    common_dates = cohorts.index.intersection(forward_wide.index)
    cohorts_aligned = cohorts.loc[common_dates]
    forward_aligned = forward_wide.loc[common_dates]
    common_tickers = cohorts.columns.intersection(forward_wide.columns)
    cohorts_aligned = cohorts_aligned[common_tickers]
    forward_aligned = forward_aligned[common_tickers]

    cohort_baskets: dict[int, pd.Series] = {}
    for c in range(n_cohorts):
        # Mask: True where a ticker is in cohort c at that date.
        in_cohort = (cohorts_aligned == c)
        # EW return = mean of forward log returns of cohort members.
        # NaN where < 2 members in cohort at that date.
        member_returns = forward_aligned.where(in_cohort)
        n_members = in_cohort.sum(axis=1)
        basket_return = member_returns.mean(axis=1, skipna=True)
        # Replace dates with < 2 members with NaN (single-member basket
        # is meaningless as a peer-group benchmark).
        basket_return[n_members < 2] = np.nan
        cohort_baskets[c] = basket_return

    return cohort_baskets


def compute_canonical_alpha(
    close_by_ticker: dict[str, pd.Series],
    horizon: int = 5,
    vol_lookback_days: int = DEFAULT_VOL_LOOKBACK_DAYS,
    n_cohorts: int = DEFAULT_N_COHORTS,
) -> dict[str, pd.Series]:
    """End-to-end: close prices → canonical alpha labels per (date, ticker).

    Pipeline:
      1. Compute trailing realized volatility per ticker
      2. Cross-sectional cohort assignment per date
      3. EW cohort basket forward log returns per (cohort, date)
      4. Canonical alpha per (ticker, date) = ticker_log_return -
         cohort_basket_return for the cohort the ticker was in at date t

    Returns ``{ticker: series_of_canonical_alphas}``. NaN for dates
    where the ticker has insufficient history, can't be classified into
    a cohort, or the cohort has fewer than 2 members.

    This is the canonical replacement for the legacy
    ``stock_5d_return - sector_etf_5d_return`` arithmetic label that
    gets built per-ticker in meta_trainer.py.
    """
    if not close_by_ticker:
        return {}

    # Step 1: realized vol per ticker
    vol_by_ticker = {
        ticker: compute_realized_volatility(close, vol_lookback_days)
        for ticker, close in close_by_ticker.items()
    }
    # Step 2: cohort assignment
    cohorts = assign_vol_cohorts(vol_by_ticker, n_cohorts)
    if cohorts.empty:
        return {ticker: pd.Series(index=close.index, dtype=float)
                for ticker, close in close_by_ticker.items()}
    # Step 3: cohort basket forward returns
    cohort_baskets = build_cohort_basket_returns(
        close_by_ticker, cohorts, horizon, n_cohorts,
    )

    # Step 4: per-ticker canonical alpha = ticker_log_return - cohort_basket_return
    canonical_alpha: dict[str, pd.Series] = {}
    for ticker, close in close_by_ticker.items():
        if len(close) < horizon + 1:
            canonical_alpha[ticker] = pd.Series(index=close.index, dtype=float)
            continue
        # Ticker's forward log return
        ticker_fwd_log_ret = np.log(close.shift(-horizon) / close)
        # Cohort assignment per date for this ticker
        if ticker not in cohorts.columns:
            canonical_alpha[ticker] = pd.Series(index=close.index, dtype=float)
            continue
        ticker_cohort_per_date = cohorts[ticker]
        # Build the basket-return series aligned to ticker's index by
        # looking up the cohort the ticker was in at each date.
        basket_return_series = pd.Series(index=close.index, dtype=float)
        for c, basket_series in cohort_baskets.items():
            mask = (ticker_cohort_per_date == c)
            basket_return_series[mask] = basket_series.reindex(
                basket_return_series.index, method=None
            )[mask]
        canonical_alpha[ticker] = ticker_fwd_log_ret - basket_return_series

    return canonical_alpha
