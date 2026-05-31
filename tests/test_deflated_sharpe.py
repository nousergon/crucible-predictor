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
    assert out["would_promote"] is True


def test_noise_ic_series_does_not_promote():
    # zero-mean noise → IC-IR ~ 0 → PSR(0) ~ 0.5, DSR well below 0.95
    ics = _ic_series(0.0, 0.10, 400, seed=3)
    out = deflated_sharpe_ratio(ics, n_trials=15, threshold=0.95)
    assert abs(out["ic_ir"]) < 0.2
    assert out["would_promote"] is False


def test_more_trials_lowers_dsr():
    ics = _ic_series(0.05, 0.10, 300, seed=4)
    few = deflated_sharpe_ratio(ics, n_trials=5)
    many = deflated_sharpe_ratio(ics, n_trials=500)
    assert many["dsr"] <= few["dsr"]  # deflation strengthens with trials


def test_insufficient_series_status():
    out = deflated_sharpe_ratio([0.1, 0.2], n_trials=10)
    assert out["status"] == "insufficient"
    assert out["would_promote"] is False
