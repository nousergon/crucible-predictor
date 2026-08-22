"""Tests for the config#1524 low-n selection-bias estimator prototype.

Covers the three properties the kill-gate rests on:
1. the e-process is a valid e-value (Ville / ``E[W_T] <= 1`` under H0),
2. e-BH controls FDR under dependence on an all-null family,
3. the estimator SEPARATES overfit from real better than fresh-best-wins at
   low-n (the config#1524 kill-gate criterion), and blocks when nothing defends.
"""

from __future__ import annotations

import numpy as np

from analysis.low_n_selection_estimator import (
    ChallengerCohorts,
    cohorts_from_realized_records,
    e_bh,
    e_process,
    estimator_promote,
    fresh_best_wins_promote,
)
from scripts.low_n_estimator_kill_gate import run_synthetic


def test_e_process_is_valid_e_value_under_null():
    """Under H0 (E[D]=0) the terminal wealth is an e-value: E[W_T] <= 1 and
    P(sup W_t >= 1/alpha) <= alpha (Ville)."""
    rng = np.random.default_rng(0)
    alpha = 0.10
    terminals = []
    exceed = 0
    trials = 4000
    for _ in range(trials):
        d = rng.normal(0.0, 1.0, size=8)
        res = e_process(d, bound=3.0)
        terminals.append(res.e_value)
        if res.running_max_e >= 1.0 / alpha:
            exceed += 1
    assert np.mean(terminals) <= 1.10  # ~1 up to MC noise; never blows past 1
    assert exceed / trials <= alpha + 0.02  # Ville bound (+MC slack)


def test_e_process_grows_under_real_edge():
    """A genuine positive edge accumulates wealth (power exists)."""
    rng = np.random.default_rng(1)
    strong = [e_process(rng.normal(0.8, 1.0, size=10), bound=3.0).running_max_e
              for _ in range(200)]
    null = [e_process(rng.normal(0.0, 1.0, size=10), bound=3.0).running_max_e
            for _ in range(200)]
    assert np.median(strong) > np.median(null)


def test_e_bh_controls_fdr_on_all_null_family():
    """All-null challenger family: e-BH false-rejection rate stays <= alpha."""
    rng = np.random.default_rng(2)
    alpha = 0.10
    false_rejections = 0
    families = 2000
    for _ in range(families):
        e_values = [
            e_process(rng.normal(0.0, 1.0, size=6), bound=3.0).running_max_e
            for _ in range(8)
        ]
        if any(e_bh(e_values, alpha=alpha)):
            false_rejections += 1
    # With all nulls, FDR == P(any rejection); e-BH bounds it by alpha.
    assert false_rejections / families <= alpha + 0.02


def test_e_bh_empty_and_singleton():
    assert e_bh([], alpha=0.1) == []
    assert e_bh([100.0], alpha=0.1) == [True]  # 100 >= 1/(0.1*1)=10
    assert e_bh([1.0], alpha=0.1) == [False]  # 1 < 10


def test_fresh_best_wins_never_blocks_and_picks_max_mean():
    cohorts = [
        ChallengerCohorts("a", np.array([0.1, 0.2, 0.0])),
        ChallengerCohorts("b", np.array([0.9, 0.8, 1.0])),
    ]
    v = fresh_best_wins_promote(cohorts)
    assert v.promoted_id == "b"
    assert v.blocked is False


def test_estimator_blocks_when_nothing_defended():
    """A family of pure noise should mostly BLOCK (fail-loud), not promote."""
    rng = np.random.default_rng(3)
    blocks = 0
    for _ in range(300):
        fam = [
            ChallengerCohorts(f"n{j}", rng.normal(0.0, 1.0, size=5))
            for j in range(6)
        ]
        if estimator_promote(fam, alpha=0.10).blocked:
            blocks += 1
    assert blocks / 300 >= 0.85  # overwhelmingly blocks on noise


def test_kill_gate_passes_synthetic():
    """The headline config#1524 kill-gate: across a cohort-depth sweep the
    estimator (i) keeps overfit promotion FDR-controlled at every depth, (ii)
    beats fresh-best-wins where the low-n selection bias bites, and (iii) grows
    power as cohorts accrue."""
    result = run_synthetic(trials=1500)
    assert result["kill_gate"] == "PASS", result
    assert result["checks"]["fdr_controlled_all_depths"]
    assert result["checks"]["separates_where_low_n_bias_bites"]
    assert result["checks"]["power_emerges_by_max_depth"]
    # A concrete, decision-useful output: cohorts needed before the estimator
    # can defensibly promote.
    assert result["d_star_cohorts_for_power"] is not None
    # Estimator never chases the overfit more than the baseline does.
    for row in result["sweep"]:
        assert row["estimator_overfit_rate"] <= row["baseline_overfit_rate"] + 0.02


def test_cohorts_from_realized_records_adapter():
    records = {
        "c1": [
            {"realized_alpha": 0.03, "champion_realized_alpha": 0.01},
            {"realized_alpha": 0.02, "champion_realized_alpha": 0.00},
        ]
    }
    cohorts = cohorts_from_realized_records(records)
    assert len(cohorts) == 1
    assert cohorts[0].challenger_id == "c1"
    np.testing.assert_allclose(cohorts[0].diffs, [0.02, 0.02])
