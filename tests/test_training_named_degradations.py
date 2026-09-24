"""Silent training degradations are named on the result.

alpha-engine-config-I11481 (research calibrator leakage fallback) and
-I11482 (RESEARCH_META_FEATURES absent from every row). On rehearsal
2026-09-23 both conditions held in all three training runs: 16 of 16 folds
fell back to the leaky prod_calibrator, and three research meta features were
zero on 25639/25639 rows. Neither reached ``data_completeness``, so the manifest
the model zoo reads described those runs as trained on complete inputs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import meta_trainer as mt
from training.data_completeness import assert_trainable, record_degradation

_REPO = Path(__file__).resolve().parent.parent


def _ok_block() -> dict:
    return {"status": "ok", "failures": [], "degradations": []}


def test_all_folds_falling_back_is_one_named_degradation():
    block = _ok_block()
    dates = np.array(["2026-03-04", "2026-07-10"], dtype="datetime64[D]")
    entry = mt._record_calibrator_leakage(
        block, list(range(1, 17)), n_folds=16, research_score_dates=dates,
    )
    assert block["status"] == "degraded"
    assert block["degradations"] == [entry]
    assert entry["input"] == "research_calibrator.per_fold"
    assert entry["status"] == "leakage_fallback"
    assert entry["n_folds_fallback"] == 16 and entry["n_folds"] == 16
    assert "2026-03-04..2026-07-10" in entry["reason"]


def test_no_fallback_records_nothing():
    block = _ok_block()
    assert mt._record_calibrator_leakage(block, [], n_folds=16) is None
    assert block == _ok_block()


def test_fully_absent_research_features_are_one_named_degradation():
    block = _ok_block()
    absent = ["guidance_direction", "risk_factor_count_delta_raw", "management_tone_zscore"]
    entry = mt._record_declared_absent(block, absent, n_rows=25639)
    assert block["status"] == "degraded"
    assert entry["status"] == "producer_absent"
    assert entry["features_affected"] == absent
    assert "25639" in entry["reason"]


def test_declared_absent_feeds_the_recorder_from_the_real_rows():
    from model.meta_model import RESEARCH_META_FEATURES

    feat = RESEARCH_META_FEATURES[0]
    rows = [{"other": 1.0}, {"other": 2.0}]
    absent = mt.declared_absent_features(rows, [feat, "other"])
    assert absent == [feat]
    block = _ok_block()
    mt._record_declared_absent(block, absent, n_rows=len(rows))
    assert block["degradations"][0]["features_affected"] == [feat]


def test_a_degradation_never_downgrades_a_failure_and_never_gates():
    block = {"status": "fail", "failures": [], "degradations": []}
    record_degradation(block, input="x", status="y", reason="z")
    assert block["status"] == "fail"
    ok = _ok_block()
    record_degradation(ok, input="x", status="y", reason="z")
    assert_trainable(ok)  # degraded is recorded, not raised


def test_run_meta_training_records_both():
    src = (_REPO / "training/meta_trainer.py").read_text()
    assert "_record_calibrator_leakage(\n        data_completeness, calibrator_leakage_folds" in src
    assert "_record_declared_absent(data_completeness, _declared_absent" in src
    # Both fallback branches are counted, including "no score_performance at all".
    loop = src[src.index("calibrator_leakage_folds: list[int] = []"):]
    loop = loop[: loop.index("_record_calibrator_leakage(")]
    assert loop.count("calibrator_leakage_folds.append(i + 1)") == 2


# ── alpha-engine-config-I11489: `absent` means a real Map failure ───────────


def test_select_absent_excludes_arms_the_register_never_trains():
    from training import model_zoo as mz

    specs = [
        {"id": "residual-momentum", "status": "active"},
        {"id": "sota-directional-combine", "status": "active"},
        {"id": "horizon-60d", "status": "active", "overrides": {"FORWARD_DAYS": 60}},
        {"id": "horizon-90d", "status": "active", "overrides": {"FORWARD_DAYS": 90}},
    ]
    applicable = {a.spec_id for a in mz.resolve_arms(specs) if a.trainable}
    assert "horizon-60d" not in applicable, "fixture must match the register"
    ids = [s["id"] for s in specs]

    absent, not_trained = mz._absent_and_untrainable(
        specs, ids, present=["residual-momentum", "sota-directional-combine"])
    assert absent == []  # a clean rotation
    assert not_trained == ["horizon-60d", "horizon-90d"]

    absent, _ = mz._absent_and_untrainable(specs, ids, present=["residual-momentum"])
    assert absent == ["sota-directional-combine"]  # a real failure stands alone
