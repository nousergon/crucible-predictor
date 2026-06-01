"""Residual / idiosyncratic-momentum feature construction (W2, predictor-local).

Computes the candidate feature columns consumed by
``model/residual_momentum_scorer.py`` from raw per-ticker close series + a
benchmark (sector ETF, SPY fallback) that ``training/meta_trainer.py`` already
loads in-process. These are NET-NEW, single-source RESEARCH features — they do
NOT recompute any column owned by the alpha-engine-data feature store, so the
"data module owns features" drift invariant is preserved. The durable
feature-store home is a deferred follow-up gated on the signal validating
(mirrors the ``_FUNDAMENTAL_EXCLUDE`` precedent in config.py).

W2 scope:
  - W2.1 residual momentum + vol scaling  → ``resid_mom_vol_scaled``  (IN)
  - W2.2 price-trend decomposition         → ``mom_12_1`` / ``mom_1m`` /
                                             ``mom_change`` / ``sector_mom``  (IN)
  - W2.3 factor momentum (autocorrelation of factor returns)            (DEFERRED —
    a factor-portfolio-level signal needing a cross-sectional factor-return
    panel that does not exist in-process; file as a follow-up once W2.1/2.2
    validate.)

NO-LOOK-AHEAD (load-bearing): every column is built from backward-only windows
on the full close index and is point-in-time at each date t (uses only data
through t). The rolling beta is ``.shift(1)`` so the beta used to residualize
the return at t is estimated only on data through t-1. The caller reindexes the
result to the tail-dropped ``labeled.index`` so rows align row-for-row with the
existing momentum/volatility feature matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.residual_momentum_scorer import RESIDUAL_MOMENTUM_FEATURES

_EPS = 1e-8


def compute_residual_momentum_features(
    close: pd.Series,
    benchmark_close: pd.Series | None,
    spy_close: pd.Series | None,
    *,
    beta_window: int = 60,
    mom_window: int = 252,
    skip_window: int = 21,
    vol_window: int = 20,
    mom_change_window: int = 21,
) -> pd.DataFrame:
    """Return a DataFrame indexed identically to ``close`` with exactly the
    ``RESIDUAL_MOMENTUM_FEATURES`` columns.

    Parameters
    ----------
    close : ascending-date close-price series for one ticker (full history).
    benchmark_close : the residualization / sector-momentum benchmark — the
        ticker's sector-ETF close, or ``None`` to fall back to ``spy_close``
        (matches label_generator's ``sector_etf if present else spy`` rule).
    spy_close : SPY close series (the market-leg fallback benchmark).
    beta_window : trailing window for the point-in-time market beta.
    mom_window : total momentum lookback (e.g. 252 ≈ 12 months).
    skip_window : most-recent days skipped (e.g. 21 ≈ 1 month) — implements the
        12-1 skip-month convention and isolates the 1-month reversal leg.
    vol_window : trailing window for residual-return volatility (vol scaling).
    mom_change_window : window for the momentum-change (acceleration) feature.

    Front-of-history rows where a window has not warmed up stay NaN (NOT filled)
    — exactly like ``momentum_5d`` / ``return_120d``; the scorer + cross-sectional
    rank-norm neutralize them downstream.
    """
    idx = close.index
    close = close.astype(float)

    bench_src = benchmark_close if benchmark_close is not None else spy_close
    if bench_src is None:
        # No benchmark available at all → every column is undefined.
        return pd.DataFrame(
            {c: np.full(len(idx), np.nan, dtype=np.float64) for c in RESIDUAL_MOMENTUM_FEATURES},
            index=idx,
        )

    # ``method="ffill"`` carries only PAST benchmark observations forward, never
    # future — matches the horizon-return reindex in meta_trainer.py. Never use
    # bfill / nearest here (they would leak the future).
    bench = bench_src.reindex(idx, method="ffill").astype(float)

    r = close.pct_change()
    r_bench = bench.pct_change()

    # ── Point-in-time market beta (estimated through t-1) ──────────────────
    cov = r.rolling(beta_window, min_periods=beta_window).cov(r_bench)
    var = r_bench.rolling(beta_window, min_periods=beta_window).var()
    beta = (cov / var.replace(0.0, np.nan)).shift(1)

    # ── W2.1: residual return → vol-scaled cumulative residual momentum ────
    resid_r = r - beta * r_bench
    cum_window = max(int(mom_window) - int(skip_window), 1)
    cum_resid = resid_r.rolling(cum_window, min_periods=cum_window).sum().shift(skip_window)
    resid_vol = resid_r.rolling(vol_window, min_periods=vol_window).std()
    # Information-ratio scaling: cumulative residual return over its own
    # window-level volatility (vol × sqrt(window)). Keeps the headline signal
    # on a ~Sharpe-like scale rather than a raw cumulative-return magnitude.
    resid_mom_vol_scaled = cum_resid / (resid_vol * np.sqrt(cum_window) + _EPS)

    # ── W2.2: transparent price-trend decomposition ────────────────────────
    # 12-1: return from mom_window ago to skip_window ago (skips the recent month).
    mom_12_1 = (close.shift(skip_window) / close.shift(mom_window)) - 1.0
    # 1-month return (short-term reversal input; the scorer weights it negatively).
    mom_1m = (close / close.shift(skip_window)) - 1.0
    # Momentum change (acceleration): recent-window momentum minus the prior window's.
    recent_mom = r.rolling(mom_change_window, min_periods=mom_change_window).sum()
    mom_change = recent_mom - recent_mom.shift(mom_change_window)
    # Sector momentum: the benchmark ETF's own skip-month momentum, broadcast.
    sector_mom = (bench.shift(skip_window) / bench.shift(mom_window)) - 1.0

    out = pd.DataFrame(
        {
            "resid_mom_vol_scaled": resid_mom_vol_scaled,
            "mom_12_1": mom_12_1,
            "mom_1m": mom_1m,
            "mom_change": mom_change,
            "sector_mom": sector_mom,
        },
        index=idx,
    )
    # Replace any ±inf (e.g. division artifacts) with NaN so downstream
    # neutralization is uniform.
    return out[RESIDUAL_MOMENTUM_FEATURES].replace([np.inf, -np.inf], np.nan)
