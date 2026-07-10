"""Unit tests for the W4.2 dead-L1 pruning monitor (config#1994, OBSERVE-ONLY).

Pins the pure classification contract: a dead-L1 CANDIDATE must carry ~0 leak-free
meta ΔIC (dropping it does not degrade IC) AND ~0 standardized Ridge weight —
BOTH lenses — and the monitor must never claim more than a candidate (no
de-weight/delete; that is a downstream, persistence-gated step).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dead_l1_monitor import (
    DEAD_L1_COEF_MAG_EPS,
    DEAD_L1_LOMO_DELTA_EPS,
    summarize_dead_l1,
)


def _lomo(per_feature: dict, full_ic: float = 0.05) -> dict:
    return {"status": "ok", "full_xsec_ic": full_ic, "per_feature": per_feature}


def test_flags_only_when_both_lenses_agree():
    lomo = _lomo(
        {
            # dead: negative ΔIC (dropping it improves IC) AND ~0 weight
            "dead_l1": {"delta_vs_full": -0.002},
            # load-bearing: big positive ΔIC AND large weight
            "live_l1": {"delta_vs_full": 0.03},
            # carries no IC but STILL weighted → NOT a candidate (diversity premium)
            "weighted_but_flat": {"delta_vs_full": -0.001},
            # near-zero weight but positive ΔIC → NOT a candidate
            "zero_weight_but_useful": {"delta_vs_full": 0.02},
        }
    )
    importance = {
        "standardized_coef": {
            "dead_l1": 0.0002,
            "live_l1": 0.8,
            "weighted_but_flat": 0.5,
            "zero_weight_but_useful": 0.0001,
        }
    }
    out = summarize_dead_l1(lomo, importance, coefficients={"dead_l1": -1e-5})

    assert out["status"] == "ok"
    assert out["dead_candidates"] == ["dead_l1"]
    assert out["n_dead_candidates"] == 1
    assert out["per_feature"]["dead_l1"]["is_dead_candidate"] is True
    assert out["per_feature"]["live_l1"]["is_dead_candidate"] is False
    # diversity-premium guard: flat IC but still weighted is NOT dead
    assert out["per_feature"]["weighted_but_flat"]["carries_no_leakfree_ic"] is True
    assert out["per_feature"]["weighted_but_flat"]["near_zero_weight"] is False
    assert out["per_feature"]["weighted_but_flat"]["is_dead_candidate"] is False
    assert out["per_feature"]["zero_weight_but_useful"]["is_dead_candidate"] is False


def test_threshold_boundaries():
    # exactly at the eps boundaries: delta == +eps (<=, counts as no-IC) and
    # coef_mag == eps (<=, counts as near-zero) → dead.
    lomo = _lomo({"edge": {"delta_vs_full": DEAD_L1_LOMO_DELTA_EPS}})
    importance = {"standardized_coef": {"edge": DEAD_L1_COEF_MAG_EPS}}
    out = summarize_dead_l1(lomo, importance)
    assert out["per_feature"]["edge"]["is_dead_candidate"] is True

    # just past either boundary → not dead
    lomo2 = _lomo({"edge": {"delta_vs_full": DEAD_L1_LOMO_DELTA_EPS * 2}})
    out2 = summarize_dead_l1(lomo2, importance)
    assert out2["per_feature"]["edge"]["is_dead_candidate"] is False


def test_missing_coefficient_never_flags():
    # coefficient magnitude unavailable → cannot confirm near-zero weight → no flag.
    lomo = _lomo({"orphan": {"delta_vs_full": -0.01}})
    out = summarize_dead_l1(lomo, importance={"standardized_coef": {}})
    assert out["per_feature"]["orphan"]["coef_magnitude"] is None
    assert out["per_feature"]["orphan"]["near_zero_weight"] is False
    assert out["per_feature"]["orphan"]["is_dead_candidate"] is False


def test_upstream_status_short_circuits():
    for bad in ({"status": "not_run"}, {"status": "error", "error": "x"}, None, "nope", {}):
        out = summarize_dead_l1(bad, importance={"standardized_coef": {}})
        assert out["status"] != "ok"


def test_is_pure_observe_only_contract():
    lomo = _lomo({"dead_l1": {"delta_vs_full": -0.01}})
    importance = {"standardized_coef": {"dead_l1": 0.0}}
    before = dict(lomo)
    out = summarize_dead_l1(lomo, importance)
    # input dicts are not mutated (pure)
    assert lomo == before
    # the observe-only, no-action contract is stated in the output
    assert "observe-only" in out["contract"].lower()
    assert "no de-weight" in out["contract"].lower()
    assert "config#624" in out["contract"]
