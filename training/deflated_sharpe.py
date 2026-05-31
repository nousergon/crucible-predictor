"""Deflated Sharpe Ratio + Probabilistic Sharpe Ratio — ROADMAP L4469 W1.3.

Bailey & López de Prado (2012/2014). The predictor's per-date cross-sectional
IC series is treated as the "returns": the IC information ratio
``IC-IR = mean(IC) / std(IC)`` is the Sharpe analog. The **Probabilistic Sharpe
Ratio** PSR(SR*) is the probability the TRUE IC-IR exceeds a benchmark SR* given
sample length, skewness and kurtosis. The **Deflated Sharpe Ratio** sets SR* to
the EXPECTED MAXIMUM IC-IR under the null (true skill = 0) across ``n_trials``
independent trials — so a strategy cherry-picked from many trials must clear a
HIGHER bar. This is the anti-false-discovery deflation: ~20 repeated backtests
manufacture a spurious 5%-significant strategy (López de Prado), and our weekly
retrain + design-choice search is exactly that kind of multiple testing.

OBSERVE ONLY in W1.3 — computed and reported, NOT used to gate promotion. W1.4
flips the gate to require ``dsr >= threshold``.

PBO via CSCV is deliberately DEFERRED to W1.3b: the Probability of Backtest
Overfitting measures overfitting of model *selection* and needs a grid of
candidate configs to select among. The predictor promotes ONE config per run
today, so PBO is degenerate; it becomes meaningful once W2/W4 introduce L1 /
architecture variants. Shipping a degenerate PBO now would be a bandaid.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

_EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(
    sr: float, n: int, skew: float = 0.0, kurt: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """PSR(SR*) — P(true Sharpe > ``sr_benchmark``) given an observed Sharpe
    ``sr`` over ``n`` observations with ``skew`` and (non-excess) ``kurt``.
    Bailey-LdP: ``Φ( (SR - SR*)·sqrt(n-1) / sqrt(1 - skew·SR + (kurt-1)/4·SR²) )``.
    """
    if n < 2 or not np.isfinite(sr):
        return float("nan")
    var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    denom = math.sqrt(var) if var > 1e-12 else 1e-6
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected MAXIMUM Sharpe across ``n_trials`` independent strategies under
    the null (true Sharpe = 0), each drawn with cross-trial Sharpe std
    ``sr_std``. Bailey-LdP closed form using the Euler-Mascheroni constant:
    ``sr_std · [ (1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e)) ]``.
    """
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return float(sr_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    ic_series,
    *,
    n_trials: int,
    sr_trials_std: float | None = None,
    threshold: float = 0.95,
) -> dict:
    """IC-IR + its Deflated Sharpe Ratio over the per-date cross-sectional IC
    series (OBSERVE).

    ``n_trials`` is the number of effective configurations searched to obtain
    this model (a conservative placeholder until cumulative trial-count tracking
    lands — W1.3b). ``sr_trials_std`` is the std of trial Sharpes; when not
    supplied it falls back to ``1/sqrt(n)`` (the single-path sampling std — a
    conservative deflation). Returns observe stats incl. ``psr_vs_zero``
    (trial-count-independent significance), ``dsr`` (deflated), and a
    ``would_promote`` flag at ``threshold``.
    """
    ic = np.asarray([x for x in np.asarray(ic_series, dtype=float)
                     if np.isfinite(x)], dtype=float)
    n = int(ic.size)
    if n < 5 or ic.std(ddof=1) < 1e-12:
        return {
            "status": "insufficient", "n": n,
            "ic_ir": float("nan"), "psr_vs_zero": float("nan"),
            "dsr": float("nan"), "would_promote": False,
        }
    sr = float(ic.mean() / ic.std(ddof=1))
    sk = float(_skew(ic))
    ku = float(_kurtosis(ic, fisher=False))  # non-excess (normal == 3)
    if sr_trials_std is None:
        sr_trials_std = 1.0 / math.sqrt(n)
    sr_star = expected_max_sharpe(n_trials, sr_trials_std)
    psr0 = probabilistic_sharpe_ratio(sr, n, sk, ku, 0.0)
    dsr = probabilistic_sharpe_ratio(sr, n, sk, ku, sr_star)

    def _r(v, p=6):
        return round(float(v), p) if np.isfinite(v) else float("nan")

    return {
        "status": "ok",
        "n": n,
        "ic_ir": _r(sr),
        "ic_mean": _r(float(ic.mean())),
        "ic_std": _r(float(ic.std(ddof=1))),
        "skew": _r(sk, 4),
        "kurtosis": _r(ku, 4),
        "n_trials": int(n_trials),
        "sr_benchmark_maxnull": _r(sr_star),
        "psr_vs_zero": _r(psr0),
        "dsr": _r(dsr),
        "threshold": threshold,
        "would_promote": bool(np.isfinite(dsr) and dsr >= threshold),
    }
