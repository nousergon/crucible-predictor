"""Tests for the Deflated Sharpe Ratio battery (ROADMAP L4469 W1.3).

Pins the Bailey-LdP properties: PSR monotonic in the observed Sharpe; the
expected-max-Sharpe deflation grows with the trial count; the DSR is ≤ PSR(0)
(deflation never inflates significance); and a strong IC series promotes while a
noise series does not.
"""
from __future__ import annotations

import numpy as np

from training.deflated_sharpe import (
    deflated_sharpe_ratio,
    downside_ic_stats,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


# ── probabilistic_sharpe_ratio ────────────────────────────────────────────


def test_psr_monotonic_in_sharpe():
    lo = probabilistic_sharpe_ratio(0.05, 200, 0.0, 3.0, 0.0)
    hi = probabilistic_sharpe_ratio(0.30, 200, 0.0, 3.0, 0.0)
    assert 0.0 <= lo <= hi <= 1.0


def test_psr_zero_sharpe_is_half():
    # observed SR == benchmark → PSR = 0.5 (Φ(0))
    p = probabilistic_sharpe_ratio(0.0, 200, 0.0, 3.0, 0.0)
    assert abs(p - 0.5) < 1e-6


def test_psr_grows_with_sample_size():
    small = probabilistic_sharpe_ratio(0.1, 30, 0.0, 3.0, 0.0)
    large = probabilistic_sharpe_ratio(0.1, 2000, 0.0, 3.0, 0.0)
    assert large > small  # more data → more confident the SR>0


# ── expected_max_sharpe ───────────────────────────────────────────────────


def test_expected_max_sharpe_grows_with_trials():
    s5 = expected_max_sharpe(5, 0.1)
    s50 = expected_max_sharpe(50, 0.1)
    s500 = expected_max_sharpe(500, 0.1)
    assert 0 < s5 < s50 < s500  # more trials → higher expected max under null


def test_expected_max_sharpe_degenerate():
    assert expected_max_sharpe(1, 0.1) == 0.0
    assert expected_max_sharpe(50, 0.0) == 0.0


# ── deflated_sharpe_ratio ─────────────────────────────────────────────────


def _ic_series(mean, std, n, seed):
    rng = np.random.default_rng(seed)
    return (mean + std * rng.standard_normal(n)).tolist()


def test_dsr_never_exceeds_psr_zero():
    """Deflation can only LOWER significance: DSR (vs the expected-max-null
    benchmark) ≤ PSR(0)."""
    ics = _ic_series(0.04, 0.10, 250, seed=1)
    out = deflated_sharpe_ratio(ics, n_trials=20)
    assert out["status"] == "ok"
    assert out["dsr"] <= out["psr_vs_zero"] + 1e-9


def test_strong_ic_series_promotes():
    # high mean IC, low noise, many dates → strong IC-IR → DSR clears 0.95
    ics = _ic_series(0.08, 0.06, 400, seed=2)
    out = deflated_sharpe_ratio(ics, n_trials=15, threshold=0.95)
    assert out["ic_ir"] > 0.5
    assert out["passes_overfit_gate"] is True


def test_noise_ic_series_does_not_promote():
    # zero-mean noise → IC-IR ~ 0 → PSR(0) ~ 0.5, DSR well below 0.95
    ics = _ic_series(0.0, 0.10, 400, seed=3)
    out = deflated_sharpe_ratio(ics, n_trials=15, threshold=0.95)
    assert abs(out["ic_ir"]) < 0.2
    assert out["passes_overfit_gate"] is False


def test_more_trials_lowers_dsr():
    ics = _ic_series(0.05, 0.10, 300, seed=4)
    few = deflated_sharpe_ratio(ics, n_trials=5)
    many = deflated_sharpe_ratio(ics, n_trials=500)
    assert many["dsr"] <= few["dsr"]  # deflation strengthens with trials


def test_insufficient_series_status():
    out = deflated_sharpe_ratio([0.1, 0.2], n_trials=10)
    assert out["status"] == "insufficient"
    assert out["passes_overfit_gate"] is False


# ── downside_ic_stats (skilled-risk basket: Sortino-of-IC + CVaR) ─────────


def test_downside_penalizes_negative_tail():
    """Sortino/CVaR are downside-aware: a series with its dispersion on the
    DOWNSIDE has a worse CVaR-of-IC and more negative-IC days than an
    upside-skewed series — where symmetric IC-IR would miss the asymmetry."""
    rng = np.random.default_rng(7)
    up = np.where(rng.random(400) < 0.1, 0.5, 0.02)      # occasional big +
    down = np.where(rng.random(400) < 0.1, -0.3, 0.07)   # occasional big -
    s_up = downside_ic_stats(up.tolist())
    s_down = downside_ic_stats(down.tolist())
    assert s_down["frac_negative_ic"] > s_up["frac_negative_ic"]
    assert s_down["cvar_of_ic"] < s_up["cvar_of_ic"]  # worse bad-day tail
    assert s_down["sortino_of_ic"] is not None         # finite (penalized)


def test_downside_no_negative_days_sortino_none_but_passes():
    out = downside_ic_stats([0.05] * 50)  # all positive → no downside
    assert out["n_downside_days"] == 0
    assert out["sortino_of_ic"] is None   # JSON-safe "infinitely good"
    assert out["passes_downside_gate"] is True


def test_downside_cvar_is_worst_tail_mean():
    ics = [-0.5, -0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    out = downside_ic_stats(ics, cvar_pct=20.0)  # worst 20% of 10 = 2 days
    assert abs(out["cvar_of_ic"] - (-0.45)) < 1e-6
    assert out["worst_ic"] == -0.5


def test_downside_insufficient_status():
    out = downside_ic_stats([0.1, 0.2])
    assert out["status"] == "insufficient"
    assert out["passes_downside_gate"] is False


# ── downside-deviation denominator convention (alpha-engine-config-I7271) ──
#
# Brian ruled 2026-08-13 "use sota": downside deviation divides the sum of
# squared downside shortfalls by N (all observations), not by n_downside
# (the count of shortfall days) — the standard Sortino/Satchell definition,
# and the convention nousergon_lib.quant.riskstats.sortino_ratio already uses.

#: 10 observations, 3 negative — same fixture as
#: training/self_test_cases.py::DOWNSIDE_IC, asymmetric so a sign error can't
#: cancel.
_IC_SERIES = (0.05, 0.03, -0.02, 0.07, -0.04, 0.01, 0.06, -0.01, 0.02, 0.08)


def _dd_divide_by_n(ic_series, target=0.0):
    """The FIXED (post-I7271) convention: RMS of the downside shortfall over
    the full sample N."""
    n = len(ic_series)
    downside = [min(0.0, v - target) for v in ic_series]
    return (sum(d * d for d in downside) / n) ** 0.5


def _dd_divide_by_n_downside(ic_series, target=0.0):
    """The PRE-FIX (retired) convention: RMS over the downside COUNT only —
    what ``downside_ic_stats`` returned before this fix."""
    downside = [v - target for v in ic_series if v < target]
    return (sum(d * d for d in downside) / len(downside)) ** 0.5


def test_downside_deviation_divides_by_n_not_n_downside():
    """The new denominator, asserted directly against the production output.

    Verified 2026-08-13 to FAIL against the pre-fix implementation: pre-fix,
    ``downside_deviation`` returned 0.026458 (RMS over n_downside=3) against
    this test's expected 0.014491 (RMS over n=10) — off by exactly the
    sqrt(10/3)=1.8257x factor the issue measured. This test is the guard on
    that regression (champion-challenger policy §7.4 — a guard must be
    verified to fail without the fix).
    """
    out = downside_ic_stats(list(_IC_SERIES))
    expected_dd = _dd_divide_by_n(_IC_SERIES)
    assert abs(out["downside_deviation"] - expected_dd) < 1e-6
    # Directly rules out the retired convention re-appearing under a refactor.
    pre_fix_dd = _dd_divide_by_n_downside(_IC_SERIES)
    assert abs(out["downside_deviation"] - pre_fix_dd) > 1e-3


def test_downside_deviation_agrees_with_nousergon_lib_riskstats():
    """Cross-implementation agreement to 1e-9 with the fleet reference
    (nousergon_lib.quant.riskstats.sortino_ratio), unrounded on both sides.
    ``periods_per_year=1`` disables the lib's annualization (meaningless for an
    IC series) and ``risk_free_rate=0.0`` matches ``downside_ic_stats``'s
    default ``target=0.0``.
    """
    from nousergon_lib.quant.riskstats import sortino_ratio

    n = len(_IC_SERIES)
    mean_ic = sum(_IC_SERIES) / n
    local_dd = _dd_divide_by_n(_IC_SERIES)
    local_sortino = mean_ic / local_dd

    lib_sortino = sortino_ratio(
        list(_IC_SERIES), risk_free_rate=0.0, periods_per_year=1,
    )
    assert abs(local_sortino - lib_sortino) < 1e-9

    # And the production entrypoint (6dp-rounded) agrees to the rounding floor.
    out = downside_ic_stats(list(_IC_SERIES))
    assert abs(out["sortino_of_ic"] - lib_sortino) < 1e-5
