"""Deterministic residual/idiosyncratic-momentum scorer — Layer-1 component (W2).

The legacy raw-return momentum L1 is dead: across walk-forward validation its
median OOS IC sits at ~-0.001 (confirmed in the live 5/30 manifest). W2 of the
predictor-improvement arc (L4469) revives the momentum leg with the strongest
single finding in the canon — **residual/idiosyncratic momentum + volatility
scaling** (Blitz/Hanauer: ~half the volatility, ~2x risk-adjusted, wins horse
races vs raw price momentum) — plus a transparent price-trend decomposition
(12-1 skip-month, short-term reversal, momentum-change, sector momentum).

Why DETERMINISTIC (not a GBM): ``model/momentum_scorer.py`` documents the
lesson the repo already paid for — a low-capacity momentum GBM was a
"calibration tax" on the meta-Ridge stack (val_IC ~0.01 vs baseline ~0.31).
Residual momentum is a *constructed factor*: the alpha lives in the
construction (residualize, vol-scale, skip-month), not in a learned nonlinear
interaction. The L2 BayesianRidge is the only learner that should weight these
sub-signals. So this component is a transparent named baseline, single source
of truth for the formula across the training-vectorized path
(``predict_array``) and the inference per-row path (``predict_dict``).

OBSERVE-MODE (W2, not auto-promoted): this scorer is trained + measured by the
leak-free machinery and emitted to the manifest, but its ``residual_momentum_score``
column is gated OUT of ``META_FEATURES`` behind ``cfg.RESIDUAL_MOMENTUM_ENABLED``
(default false). ``predict_dict`` is present for parity / the future promotion
PR but is NOT wired into ``inference/stages/run_inference.py`` while the gate is
closed.

The feature columns this scorer consumes are produced by
``data/residual_momentum_features.py``.
"""

from __future__ import annotations

import numpy as np

# The candidate feature columns, in canonical order. Single source of truth —
# ``data/residual_momentum_features.py`` emits exactly these, and
# ``training/meta_trainer.py`` builds ``X_resid_mom`` from them.
RESIDUAL_MOMENTUM_FEATURES: list[str] = [
    "resid_mom_vol_scaled",  # headline (W2.1): vol-scaled cumulative residual return
    "mom_12_1",              # W2.2: 12-month return skipping the most recent month
    "mom_1m",                # W2.2: 1-month return (short-term reversal → negative weight)
    "mom_change",            # W2.2: change in momentum (recent vs prior window)
    "sector_mom",            # W2.2: the ticker's sector-ETF skip-month momentum
]

# Blend weights — module constants so any future tuning is one edit (mirrors
# momentum_scorer._W_*). Residual-dominant by design: ``resid_mom_vol_scaled``
# is an information-ratio (~+/-3 scale) and is the highest-EV signal, so the
# standalone leak-free read primarily reflects residual-momentum skill; the
# decomposition terms are secondary tilts. ``mom_1m`` carries a NEGATIVE weight
# (short-term reversal). The L2 is the real weighter once the gate is flipped.
_W_RESID = 1.0
_W_MOM12_1 = 0.5
_W_MOM1M = -0.5
_W_MOM_CHG = 0.5
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

    resid = _get("resid_mom_vol_scaled", 0.0)
    mom12_1 = _get("mom_12_1", 0.0)
    mom1m = _get("mom_1m", 0.0)
    mom_chg = _get("mom_change", 0.0)
    sector = _get("sector_mom", 0.0)
    return (
        _W_RESID * resid
        + _W_MOM12_1 * mom12_1
        + _W_MOM1M * mom1m
        + _W_MOM_CHG * mom_chg
        + _W_SECTOR * sector
    )


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

    resid = _safe("resid_mom_vol_scaled", 0.0)
    mom12_1 = _safe("mom_12_1", 0.0)
    mom1m = _safe("mom_1m", 0.0)
    mom_chg = _safe("mom_change", 0.0)
    sector = _safe("sector_mom", 0.0)
    return (
        _W_RESID * resid
        + _W_MOM12_1 * mom12_1
        + _W_MOM1M * mom1m
        + _W_MOM_CHG * mom_chg
        + _W_SECTOR * sector
    )
