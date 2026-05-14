"""
regime/composite.py — AQR-style risk-on/risk-off intensity.

Pure rule-based composite z-score over macro features. Zero estimation
risk — no fit, no parameters to learn beyond the equal-weight default.
Serves as the always-available regime fallback when HMM is unfit or
disagrees catastrophically with consensus.

Output convention: ``intensity_z`` is a scalar in roughly ``[-3, +3]``
where:

- Positive  → risk-off / bearish (high VIX, wide credit spreads, etc.)
- Negative  → risk-on  / bullish (low VIX, tight spreads, etc.)
- Magnitude → standard deviations from rolling-window mean.

The sign convention is deliberately the opposite of "regime intensity"
in the downstream contract — downstream consumers expect higher values
to mean "more risk-on." We invert at the substrate-orchestrator layer
(``substrate.py``) so this module stays interpretable as "stress index"
in isolation.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


# Equal-weight default. Each feature contributes the same to the
# composite intensity. Tuning these is a deliberate research exercise —
# they live here as named constants so future revisions are explicit.
DEFAULT_WEIGHTS: dict[str, float] = {
    "vix_level": 1.0,
    "hy_oas_bps": 1.0,
    "yield_curve_slope": -1.0,  # inversion is bearish → negative slope is risk-off
    "spy_20d_return": -1.0,     # positive return is risk-on → negative weight
    "market_breadth": -1.0,     # high breadth is risk-on → negative weight
    "vix_term_slope": -1.0,     # backwardation (negative slope) is risk-off
}


def _zscore(value: float, history: pd.Series) -> float:
    """Standard z-score of ``value`` against ``history``'s mean + std.

    Returns 0.0 when ``history`` is empty or has zero std (degenerate
    case — no signal to extract).
    """
    if len(history) == 0:
        return 0.0
    mu = float(history.mean())
    sigma = float(history.std(ddof=0))
    if sigma == 0.0 or not np.isfinite(sigma):
        return 0.0
    return (float(value) - mu) / sigma


def compute_composite_intensity(
    current: Mapping[str, float],
    history: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    rolling_window_weeks: int | None = 260,
) -> dict:
    """Compute composite risk-on/risk-off intensity from macro features.

    Parameters
    ----------
    current
        Feature row at the inference timestamp — mapping of feature name
        to scalar value. Missing features are skipped (excluded from
        the weighted sum).
    history
        Historical feature DataFrame indexed by date. Used to compute
        z-score normalization. Must contain at least the features
        present in ``current`` and ``weights``.
    weights
        Per-feature weight dict. Defaults to ``DEFAULT_WEIGHTS``.
    rolling_window_weeks
        Lookback window for z-score normalization. None = use full
        history. Default 260 weeks (~5y) follows AQR convention.

    Returns
    -------
    Dict with keys:

    - ``intensity_z``: scalar composite z-score (stress orientation —
      positive = risk-off).
    - ``per_feature_z``: dict of feature name → individual z-score
      (debugging + dashboard observability).
    - ``features_used``: list of features that contributed (excludes
      missing-from-current or missing-from-history).
    """
    w = dict(weights or DEFAULT_WEIGHTS)

    # Scope history window
    if rolling_window_weeks is not None and len(history) > rolling_window_weeks:
        history_window = history.tail(rolling_window_weeks)
    else:
        history_window = history

    per_feature_z: dict[str, float] = {}
    features_used: list[str] = []
    weighted_sum = 0.0
    weight_norm = 0.0

    for feat, weight in w.items():
        if feat not in current:
            continue
        if feat not in history_window.columns:
            continue
        value = current[feat]
        if value is None or not np.isfinite(float(value)):
            continue
        feat_history = history_window[feat].dropna()
        z = _zscore(float(value), feat_history)
        per_feature_z[feat] = z
        features_used.append(feat)
        weighted_sum += weight * z
        weight_norm += abs(weight)

    intensity_z = weighted_sum / weight_norm if weight_norm > 0 else 0.0

    return {
        "intensity_z": float(intensity_z),
        "per_feature_z": per_feature_z,
        "features_used": features_used,
    }


def compute_intensity_z_series(
    feature_panel: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    rolling_window_weeks: int | None = 260,
) -> pd.Series:
    """Vectorized compute of intensity_z per row of a date-indexed
    feature panel — Stage D L2 META_FEATURES integration.

    For each row in ``feature_panel``, computes the composite z-score
    using that row as ``current`` and all prior rows (within the
    rolling window) as ``history``. The output is **inverted** to match
    the substrate Lambda's downstream convention (positive = risk-on,
    negative = risk-off) so the META_FEATURE value is consistent with
    what the substrate.json artifact's ``composite.intensity_z`` field
    carries. This is the same inversion that
    ``regime/substrate.py:build_regime_substrate`` performs.

    Used at both training time (meta_trainer.py extends
    regime_features_df with this column) and inference time
    (run_inference.py same path) so the L2 Ridge sees a consistent
    intensity_z value with the same orientation regardless of code
    path. Predictor's inference Lambda does NOT yet read substrate
    artifacts directly — Stage D recomputes intensity_z in-process
    from the same macro inputs the substrate Lambda uses. When the
    substrate Lambda's artifact becomes the authoritative source at
    inference time (follow-up PR), the values will match by
    construction.

    The first ``min(rolling_window_weeks // 4, 4)`` rows in the panel
    have insufficient history for a meaningful z-score — the helper
    returns 0.0 for those rows (neutral feature value) rather than a
    noisy estimate from a tiny historical window. The Ridge L2 sees
    0.0 for those early rows; this matches the "missing-feature →
    0.0 fallback" convention in meta_trainer.py + run_inference.py.

    Args:
        feature_panel: Date-indexed DataFrame with columns matching
            the composite weight keys (vix_level, hy_oas_bps,
            yield_curve_slope, spy_20d_return, market_breadth,
            vix_term_slope). Missing columns are silently skipped at
            the row-level composite computation.
        weights: Per-feature weight dict; defaults to ``DEFAULT_WEIGHTS``.
        rolling_window_weeks: Lookback window for z-score normalization.

    Returns:
        pd.Series indexed by ``feature_panel.index`` with values in
        roughly ``[-3, +3]`` (z-units, positive = risk-on).
    """
    if feature_panel.empty:
        return pd.Series(dtype="float64")

    # Minimum history before computing a meaningful z-score. With fewer
    # than 4 rows of history the rolling z-score is dominated by sample-
    # size noise; default to 0.0 (neutral) for those early rows.
    min_history = 4

    out: dict = {}
    for i, (idx, row) in enumerate(feature_panel.iterrows()):
        if i < min_history:
            out[idx] = 0.0
            continue
        # History = all rows strictly before this one (no look-ahead).
        history = feature_panel.iloc[:i]
        result = compute_composite_intensity(
            current=row.to_dict(),
            history=history,
            weights=weights,
            rolling_window_weeks=rolling_window_weeks,
        )
        # Invert sign to match substrate.json convention
        # (positive = risk-on). The raw composite returns stress
        # orientation (positive = risk-off).
        out[idx] = -result["intensity_z"]

    return pd.Series(out, name="intensity_z")
