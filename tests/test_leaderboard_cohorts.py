"""Tests for the alpha-engine-config-I12662 independent-cohort data gate and the
pre-registered planted-overfit verdict of the config#1524 real run."""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest

from analysis.leaderboard_cohorts import (
    MIN_INDEPENDENT_COHORTS,
    independent_depth_gate,
    non_overlapping_dates,
    records_from_leaderboard,
)
from analysis.low_n_selection_estimator import ChallengerCohorts
from scripts import low_n_estimator_kill_gate as kg


def _weekly(start: date, n: int) -> list[str]:
    return [(start + timedelta(weeks=i)).isoformat() for i in range(n)]


def _fixture_leaderboard() -> dict:
    """Synthetic board in the scanner/producer shape (weekly dates, 21d)."""
    dates = _weekly(date(2026, 6, 5), 20)  # Fridays
    champ = {d: 0.01 for d in dates}
    return {
        "leaderboard_id": "fixture",
        "champion": "champ",
        "horizon_days": 21,
        "specs": [
            {"name": "champ", "kind": "champion",
             "topn_alpha_vs_population_by_date": champ},
            {"name": "deep", "kind": "challenger",
             "topn_alpha_vs_population_by_date": {d: 0.03 for d in dates}},
            {"name": "shallow", "kind": "challenger",
             "topn_alpha_vs_population_by_date": {d: 0.02 for d in dates[:6]}},
            {"name": "orphan", "kind": "challenger",
             "topn_alpha_vs_population_by_date": {"2025-01-03": 0.5}},
            {"name": "gone", "kind": "retired",
             "topn_alpha_vs_population_by_date": {d: 0.9 for d in dates}},
        ],
        "horizons": [{"horizon_days": 126, "specs": []}],
    }


def test_weekly_dates_overlap_on_a_21d_horizon():
    """20 weekly dates are ~4 independent 21d cohorts, not 20."""
    kept = non_overlapping_dates(_weekly(date(2026, 6, 5), 20), horizon_days=21)
    # 2026-06-05 → next start must be >= 21 NYSE sessions later (Juneteenth and
    # July 4th observed fall inside the first window, so it is 2026-07-10).
    assert kept[:2] == ["2026-06-05", "2026-07-10"]
    assert len(kept) == 4


def test_non_overlap_counts_trading_days_not_calendar_days():
    # 2026-06-05 + 21 sessions = 2026-07-08 (holidays 06-19 and 07-03 skipped).
    assert non_overlapping_dates(["2026-06-05", "2026-07-07"], horizon_days=21) == [
        "2026-06-05"
    ]
    assert non_overlapping_dates(["2026-07-08", "2026-06-05"], horizon_days=21) == [
        "2026-06-05", "2026-07-08"
    ]


def test_records_pair_with_champion_and_skip_champion_and_retired():
    records, depths = records_from_leaderboard(_fixture_leaderboard())
    assert set(records) == {"deep", "shallow", "orphan"}
    assert depths == {"deep": 4, "shallow": 2, "orphan": 0}
    row = records["deep"][0]
    assert row == {
        "date": "2026-06-05",
        "realized_alpha": 0.03,
        "champion_realized_alpha": 0.01,
    }


def test_records_refuse_a_missing_horizon_or_champion():
    lb = _fixture_leaderboard()
    with pytest.raises(ValueError, match="no 63-day block"):
        records_from_leaderboard(lb, horizon_days=63)
    lb["champion"] = "nobody"
    with pytest.raises(ValueError, match="no spec row"):
        records_from_leaderboard(lb)


def test_gate_requires_four_independent_cohorts_per_challenger():
    assert MIN_INDEPENDENT_COHORTS == 4
    records, _ = records_from_leaderboard(_fixture_leaderboard())
    gate = independent_depth_gate(records)
    assert not gate.passed
    assert any("shallow: 2" in r for r in gate.reasons)
    assert any("orphan: 0" in r for r in gate.reasons)
    assert independent_depth_gate({"deep": records["deep"]}).passed


def test_gate_refuses_overlapping_or_undated_records():
    dates = _weekly(date(2026, 6, 5), 4)  # 4 rows, but weekly → overlapping
    rows = [{"date": d, "realized_alpha": 0.0, "champion_realized_alpha": 0.0}
            for d in dates]
    gate = independent_depth_gate({"x": rows})
    assert not gate.passed and any("overlap" in r for r in gate.reasons)
    undated = [{"realized_alpha": 0.0, "champion_realized_alpha": 0.0}] * 5
    gate = independent_depth_gate({"y": undated})
    assert not gate.passed and any("no 'date'" in r for r in gate.reasons)


def _spaced_records(diffs_by_id: dict[str, list[float]]) -> dict[str, list[dict]]:
    out = {}
    for cid, diffs in diffs_by_id.items():
        out[cid] = [
            {"date": (date(2025, 1, 6) + timedelta(days=42 * i)).isoformat(),
             "realized_alpha": float(x), "champion_realized_alpha": 0.0}
            for i, x in enumerate(diffs)
        ]
    return out


def test_run_real_refuses_below_the_gate(tmp_path, monkeypatch):
    records, _ = records_from_leaderboard(_fixture_leaderboard())
    result = kg.run_real(records, source="fixture")
    assert result["refused"] is True and result["kill_gate"] is None
    lb_path = tmp_path / "lb.json"
    lb_path.write_text(json.dumps(_fixture_leaderboard()))
    logged = []
    monkeypatch.setattr(kg, "_append_experiments_log", lambda *a: logged.append(a))
    assert kg.main(["--leaderboard", str(lb_path), "--log"]) == kg.EXIT_REFUSED
    assert logged == []  # a refused run never writes a verdict


def test_planted_control_passes_when_fresh_best_wins_is_fooled():
    """A real family with no edge: the planted best-of-K null tops the
    fresh-best-wins ranking, and the estimator must not promote it."""
    null = [1.0, -1.0, 1.0, -1.0, 0.5, -0.5]
    result = kg.run_real(
        _spaced_records({"a": null[:4], "b": null[:6]}), source="t"
    )
    ctl = result["control"]
    assert [r["n_cohorts"] for r in ctl["sweep"]] == [4, 6]
    assert ctl["sweep"][0]["baseline_planted_rate"] >= 0.5
    assert all(r["estimator_planted_rate"] <= kg.CONTROL_ALPHA for r in ctl["sweep"])
    assert result["kill_gate"] == "PASS"


def test_planted_control_is_inconclusive_when_the_baseline_is_not_fooled():
    """Ruling 1b (2026-10-28): if fresh-best-wins never promotes the planted
    overfit at the shallowest depth, the control has no contrast and cannot
    falsify the estimator → INCONCLUSIVE, which never exits like a PASS."""
    strong = [6.0, 6.2, 5.8, 6.1]
    records = _spaced_records({"a": [1.0, -1.0, 1.0, -1.0], "s": strong})
    result = kg.run_real(records, source="t")
    assert result["control"]["checks"]["fdr_controlled_all_measured_depths"]
    assert not result["control"]["checks"]["contrast_at_shallowest_measured_depth"]
    assert not result["control"]["checks"]["separates_at_shallowest_measured_depth"]
    assert result["kill_gate"] == "INCONCLUSIVE"
    assert result["kill_gate_label"] == "INCONCLUSIVE"


def test_an_fdr_failure_falsifies_even_without_contrast(monkeypatch):
    """INCONCLUSIVE covers only a control with no contrast. An estimator that
    promotes the planted overfit above alpha has failed the safety leg, and
    that is FALSIFIED whatever the baseline did."""
    from analysis.low_n_selection_estimator import fresh_best_wins_promote

    class _Promotes:
        def __init__(self, cid):
            self.blocked, self.promoted_id = False, cid
            self.rejected_ids, self.e_values = (), {}

    def always_planted(fam, *, alpha):
        ids = [c.challenger_id for c in fam]
        return _Promotes(kg.PLANTED_ID if kg.PLANTED_ID in ids else ids[0])

    def never_planted(fam):
        res = fresh_best_wins_promote([c for c in fam if c.challenger_id != kg.PLANTED_ID])
        return res

    monkeypatch.setattr(kg, "estimator_promote", always_planted)
    monkeypatch.setattr(kg, "fresh_best_wins_promote", never_planted)
    result = kg.run_real(
        _spaced_records({"a": [1.0, -1.0, 1.0, -1.0]}), source="t"
    )
    assert not result["control"]["checks"]["contrast_at_shallowest_measured_depth"]
    assert not result["control"]["checks"]["fdr_controlled_all_measured_depths"]
    assert result["kill_gate"] == "FALSIFIED"


def test_inconclusive_exits_non_zero_and_distinct(tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(
        _spaced_records({"a": [1.0, -1.0, 1.0, -1.0], "s": [6.0, 6.2, 5.8, 6.1]})
    ))
    logged = []
    monkeypatch.setattr(kg, "_append_experiments_log", lambda *a: logged.append(a))
    code = kg.main(["--cohorts", str(path), "--log"])
    assert code == kg.EXIT_INCONCLUSIVE
    assert code not in (kg.EXIT_PASS, kg.EXIT_FALSIFIED, kg.EXIT_REFUSED)
    assert logged and logged[0][0]["kill_gate"] == "INCONCLUSIVE"


def test_power_leg_is_reported_and_never_gates(monkeypatch):
    """Ruling 3b: power is reported at every measured depth; a PASS with power
    under the floor is labelled 'PASS (safety only)' and still exits 0."""
    null = [1.0, -1.0, 1.0, -1.0, 0.5, -0.5]
    records = _spaced_records({"a": null[:4], "b": null[:6]})
    result = kg.run_real(records, source="t")
    power = result["control"]["power"]
    assert power["gates_verdict"] is False
    assert power["effect_sd"] == kg.CONTROL_POWER_EFFECT_SD
    assert [r["n_cohorts"] for r in power["sweep"]] == [4, 6]
    assert result["kill_gate"] == "PASS"

    def no_power(cohorts, *, depths, **kw):
        return {
            "effect_sd": 1.0, "report_floor": 0.2, "gates_verdict": False,
            "sweep": [{"n_cohorts": n, "estimator_true_edge_rate": 0.0} for n in depths],
            "meets_report_floor_all_measured_depths": False,
        }

    monkeypatch.setattr(kg, "power_leg", no_power)
    weak = kg.run_real(records, source="t")
    assert weak["kill_gate"] == "PASS"
    assert weak["kill_gate_label"] == kg.SAFETY_ONLY_LABEL
    # The safety legs are untouched by the power leg's presence.
    assert weak["control"]["sweep"] == result["control"]["sweep"]


def test_power_leg_does_not_disturb_the_safety_draws(monkeypatch):
    """The power leg has its own seed stream: removing it leaves the planted
    overfit sweep byte-identical, so the 10-26 safety rule draws what it drew."""
    records = _spaced_records({"a": [0.3, -0.8, 1.1, -0.2], "b": [0.5, -1.2, 0.9, 0.4, -0.6]})
    with_power = kg.run_real(records, source="t")["control"]["sweep"]
    monkeypatch.setattr(
        kg, "power_leg",
        lambda cohorts, *, depths, **kw: {"sweep": [], "gates_verdict": False,
                                          "meets_report_floor_all_measured_depths": True},
    )
    without = kg.run_real(records, source="t")["control"]["sweep"]
    assert with_power == without


def test_gate_covers_the_whole_leaderboard():
    """Ruling 2a: an arm with no common dates stays in the record set at depth
    0 and refuses the whole board; it never leaves the e-BH family."""
    records, depths = records_from_leaderboard(_fixture_leaderboard())
    assert depths["orphan"] == 0 and "orphan" in records
    result = kg.run_real(records, source="fixture")
    assert result["refused"] is True
    assert any(r.startswith("orphan:") for r in result["gate"]["reasons"])
    assert kg.CONTROL_GATE_SCOPE == "whole_leaderboard"


def test_verdict_is_scale_invariant_and_deterministic():
    """Diffs are standardised by the pooled within-challenger SD, so raw alpha
    units (~1e-2) give the same verdict as unit-scale diffs."""
    diffs = {"a": [0.3, -0.8, 1.1, -0.2], "b": [0.5, -1.2, 0.9, 0.4, -0.6]}
    unit = kg.run_real(_spaced_records(diffs), source="t")
    small = kg.run_real(
        _spaced_records({k: [x / 100 for x in v] for k, v in diffs.items()}),
        source="t",
    )
    assert unit["control"]["sweep"] == small["control"]["sweep"]
    assert unit["kill_gate"] == small["kill_gate"]


def test_pooled_within_sd_ignores_between_challenger_edge():
    a = ChallengerCohorts("a", np.array([1.0, -1.0, 1.0, -1.0]))
    b = ChallengerCohorts("b", np.array([11.0, 9.0, 11.0, 9.0]))
    assert kg.pooled_within_sd([a, b]) == pytest.approx(np.sqrt(8 / 6))


def test_control_parameters_are_the_preregistered_ones():
    assert (kg.CONTROL_ALPHA, kg.CONTROL_TRIALS, kg.CONTROL_K_FORKS,
            kg.CONTROL_SHALLOW_BASELINE_FLOOR, kg.CONTROL_SEED) == (
        0.10, 2000, 8, 0.15, 12662)


def test_amended_parameters_are_the_preregistered_ones():
    """The 2026-10-28 amendment (rulings 1b/2a/3b), pinned like the original."""
    assert (kg.CONTROL_NO_CONTRAST_VERDICT, kg.CONTROL_GATE_SCOPE,
            kg.CONTROL_POWER_EFFECT_SD, kg.CONTROL_POWER_REPORT_FLOOR,
            kg.CONTROL_POWER_GATES_VERDICT, kg.CONTROL_POWER_SEED_OFFSET,
            kg.SAFETY_ONLY_LABEL) == (
        "INCONCLUSIVE", "whole_leaderboard", 1.0, 0.20, False, 100_000,
        "PASS (safety only)")
    assert (kg.EXIT_PASS, kg.EXIT_FALSIFIED, kg.EXIT_REFUSED,
            kg.EXIT_INCONCLUSIVE) == (0, 1, 2, 3)
