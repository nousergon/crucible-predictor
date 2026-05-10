"""López de Prado triple-barrier labels (regime-agnostic).

Two label flavors share the same path-walking core:

- ``triple_barrier_class_labels`` — 3-class direction labels (up / neutral
  / down). Reusable for SPY-based regime classification or per-ticker
  direction classifiers under any future continuous-conditioning rebuild.

- ``triple_barrier_alpha_labels`` — continuous alpha target: the realized
  cumulative log-return at first-touch (capped at the touched barrier)
  or at window-end (uncapped). Used by Stage 3 of the regime-conditioning
  rebuild to replace fixed-horizon L1 alpha targets — barriers can be
  regime-vol-scaled so the label adapts to vol regime continuously.

References:
- López de Prado, *Advances in Financial Machine Learning*, Ch. 3.4-3.5
- "Triple-barrier method": first-touch profit-take / stop-loss / time
  horizon barriers replace fixed-horizon labels with path-aware ones.
"""
from __future__ import annotations

import numpy as np


TRIPLE_BARRIER_SENTINEL: int = -1


def triple_barrier_class_labels(
    log_returns: np.ndarray,
    forward_window: int = 21,
    up_barrier_pct: float = 0.05,
    down_barrier_pct: float = 0.05,
) -> np.ndarray:
    """Generate 3-class triple-barrier classification labels.

    For each row, looks forward ``forward_window`` rows; classifies by
    which barrier hits first:

      - 2 (up-class)   : up barrier hit first, OR window-end cumulative
                         return > +up_barrier_pct / 2
      - 0 (down-class) : down barrier hit first, OR window-end cumulative
                         return < -down_barrier_pct / 2
      - 1 (neutral)    : neither barrier hit, window-end cumulative
                         return in [-down/2, +up/2]

    Tail rows where the forward window extends past data end get
    ``TRIPLE_BARRIER_SENTINEL`` (= -1).

    Args:
        log_returns: (n,) log-return series (any underlying — SPY for
            market-regime use, per-ticker for stock-direction use)
        forward_window: trading days forward to check
        up_barrier_pct: cumulative log-return threshold for up-class hit
        down_barrier_pct: cumulative log-return threshold for down-class hit
            (positive value; the function negates internally)

    Returns:
        (n,) int8 array — class label per row, sentinel (-1) for tail.
    """
    n = len(log_returns)
    labels = np.full(n, TRIPLE_BARRIER_SENTINEL, dtype=np.int8)
    for i in range(n - forward_window):
        cum = 0.0
        up_hit = False
        down_hit = False
        for j in range(forward_window):
            cum += log_returns[i + 1 + j]
            if cum >= up_barrier_pct:
                up_hit = True
                break
            if cum <= -down_barrier_pct:
                down_hit = True
                break
        if up_hit:
            labels[i] = 2
        elif down_hit:
            labels[i] = 0
        else:
            if cum > up_barrier_pct / 2:
                labels[i] = 2
            elif cum < -down_barrier_pct / 2:
                labels[i] = 0
            else:
                labels[i] = 1
    return labels


def triple_barrier_alpha_labels(
    log_returns: np.ndarray,
    forward_window: int = 21,
    up_barrier_pct: float = 0.05,
    down_barrier_pct: float = 0.05,
) -> np.ndarray:
    """Generate continuous triple-barrier alpha labels.

    For each row, looks forward ``forward_window`` rows; returns the
    realized cumulative log-return at the moment the path:

      - first touches the up barrier → label = +up_barrier_pct (capped)
      - first touches the down barrier → label = -down_barrier_pct (capped)
      - reaches the time-out window → label = window-end cumulative return

    Tail rows where the forward window extends past data end get NaN.

    The capping at touched barriers reflects the realistic execution
    model — a triple-barrier strategy exits at the touched barrier, so
    the realized return is the barrier value, not the post-touch path.
    Time-out rows return uncapped because no exit triggered.

    For Stage 3 of the regime-conditioning rebuild, callers pass
    regime-vol-scaled barrier widths (e.g., barriers as multiples of
    realized vol) so the label adapts to vol regime continuously.

    Args:
        log_returns: (n,) log-return series
        forward_window: time-out barrier in trading days
        up_barrier_pct: profit-take barrier (cumulative log return)
        down_barrier_pct: stop-loss barrier as positive number; the
            function negates internally

    Returns:
        (n,) float64 array — realized return label per row, NaN for tail.
    """
    n = len(log_returns)
    labels = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - forward_window):
        cum = 0.0
        hit = False
        for j in range(forward_window):
            cum += log_returns[i + 1 + j]
            if cum >= up_barrier_pct:
                labels[i] = up_barrier_pct
                hit = True
                break
            if cum <= -down_barrier_pct:
                labels[i] = -down_barrier_pct
                hit = True
                break
        if not hit:
            labels[i] = cum
    return labels
