"""Deterministic residual/idiosyncratic-momentum scorer — Layer-1 component (W2).

The legacy raw-return momentum L1 is dead: its median OOS IC sits at ~-0.001
(confirmed in the live 5/30 manifest). W2 of the predictor-improvement arc
(L4469) revives the momentum leg with the strongest single finding in the canon
— **residual/idiosyncratic momentum + volatility scaling** (Blitz/Hanauer:
~half the volatility, ~2x risk-adjusted, wins horse races vs raw price
momentum) — plus a transparent price-trend decomposition.

This component is a **deterministic transparent blend** (not a GBM):
``model/momentum_scorer.py`` documents the lesson the repo already paid for — a
low-capacity momentum GBM was a "calibration tax" on the meta-Ridge stack. The
alpha lives in the *construction* of the residual factor, not in a learned
nonlinear interaction; the L2 BayesianRidge is the only learner that should
weight these sub-signals.

FEATURE SOURCING: the feature columns this scorer consumes are computed in the
**data module** (`alpha-engine-data/features/feature_engineer.py`) and read from
ArcticDB like every other feature — the predictor does NOT compute them. The
"data module owns features" invariant is intact. (An earlier draft computed them
predictor-side and re-derived a rolling beta the feature store already owns;
that was corrected.) The columns:
  - ``residual_momentum_ratio`` — vol-scaled cumulative residual log-return,
    12-1 skip-month (the headline W2.1 signal; reuses the feature store's
    beta-residualized return series).
  - ``mom_12_1_pct`` — 12-1 skip-month raw price momentum.
  - ``sector_mom_pct`` — the sector ETF's own 12-1 momentum (industry momentum).

OBSERVE-MODE (W2, not auto-promoted): trained + measured by the leak-free
machinery and emitted to the manifest, but its ``residual_momentum_score``
column is gated OUT of ``META_FEATURES`` behind ``cfg.RESIDUAL_MOMENTUM_ENABLED``
(default false). ``predict_dict`` is present for parity / the future promotion
PR but is NOT wired into ``inference/stages/run_inference.py`` while the gate is
closed.
"""

from __future__ import annotations

import numpy as np

# The feature-store columns this L1 consumes, in canonical order. These are
# produced by alpha-engine-data/features/feature_engineer.py and read from
# ArcticDB — single source of truth lives in the data module.
RESIDUAL_MOMENTUM_FEATURES: list[str] = [
    "residual_momentum_ratio",  # headline (W2.1): vol-scaled residual momentum (IR)
    "mom_12_1_pct",             # W2.2: 12-1 skip-month raw price momentum
    "sector_mom_pct",           # W2.2: sector-ETF 12-1 (industry) momentum
]

# Blend weights — module constants so any future tuning is one edit (mirrors
# momentum_scorer._W_*). Residual-dominant by design: ``residual_momentum_ratio``
# is an information ratio (~+/-3 scale) and is the highest-EV signal, so the
# standalone leak-free read primarily reflects residual-momentum skill; the
# raw decomposition terms are positive secondary tilts. The L2 is the real
# weighter once the gate is flipped.
_W_RESID = 1.0
_W_MOM12_1 = 0.5
_W_SECTOR = 0.25


def predict_array(
    X_resid_raw: np.ndarray, feature_names: list[str],
) -> np.ndarray:
    """Vectorized residual-momentum score for a (n_rows, n_features) raw-feature
    array. Used by ``training/meta_trainer.py`` for the walk-forward + final
    fit — produces one ``residual_momentum_score`` per training row.

    Uses a neutral default of 0.0 for every sub-signal (all center at zero), so
    short-history NaN rows produce a real-valued score the Ridge can weight
    rather than poisoning the meta input — same posture as momentum_scorer.
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    def _get(name: str, default: float) -> np.ndarray:
        idx = name_to_idx.get(name)
        if idx is None:
            return np.full(X_resid_raw.shape[0], default, dtype=np.float64)
        col = X_resid_raw[:, idx].astype(np.float64)
        return np.where(np.isfinite(col), col, default)

    resid = _get("residual_momentum_ratio", 0.0)
    mom12_1 = _get("mom_12_1_pct", 0.0)
    sector = _get("sector_mom_pct", 0.0)
    return _W_RESID * resid + _W_MOM12_1 * mom12_1 + _W_SECTOR * sector


def predict_dict(features: dict) -> float:
    """Single-ticker residual-momentum score for an inference-time feature dict.

    Mirrors ``predict_array`` but reads from a dict (one row's worth of raw
    features). ``None`` and non-finite both degrade to the neutral default.
    Present for parity / the future promotion PR — dormant while the
    ``RESIDUAL_MOMENTUM_ENABLED`` gate is closed.
    """
    def _safe(name: str, default: float) -> float:
        v = features.get(name)
        if v is None:
            return default
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return default if not np.isfinite(f) else f

    resid = _safe("residual_momentum_ratio", 0.0)
    mom12_1 = _safe("mom_12_1_pct", 0.0)
    sector = _safe("sector_mom_pct", 0.0)
    return _W_RESID * resid + _W_MOM12_1 * mom12_1 + _W_SECTOR * sector
