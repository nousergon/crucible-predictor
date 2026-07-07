"""tests/test_ic_reliability.py — alpha-engine-config#969.

Behavioral unit tests for the pure ``classify_ic_reliability`` helper (the
producer-side IC reliability taxonomy) + a source-text invariant that the
meta-training manifest actually emits the ``ic_reliability`` map.

The classification rule: leak-free / CPCV / walk-forward-OOS methodologies →
``"high"``; in-sample / overlapping-pooled / split-only / unknown → ``"low"``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from training.ic_reliability import (
    build_ic_reliability_map,
    classify_ic_reliability,
)


# ── HIGH: leak-free methodologies ──────────────────────────────────────────
@pytest.mark.parametrize(
    "field",
    [
        "meta_model_oos_ic_cpcv",
        "meta_model_oos_ic_leakfree",
        "meta_oos_ic_leakfree_no_expected_move",
        "meta_oos_ic_leakfree_post2020",
        "meta_oos_ic_leakfree_nonlinear",
        "meta_oos_ic_cpcv_nonlinear",
        "momentum_median_ic",
        "volatility_median_ic",
        "residual_momentum_median_ic",
        "residual_momentum_leakfree_oos_ic",
        "meta_l1_standalone_alpha_ic",
    ],
)
def test_leakfree_fields_are_high(field):
    assert classify_ic_reliability(field) == "high"


# ── LOW: in-sample / overlapping-pooled / split-only methodologies ─────────
@pytest.mark.parametrize(
    "field",
    [
        "meta_model_in_sample_ic",
        "meta_model_ic",            # unsuffixed in-sample alias
        "meta_model_oos_ic",        # OOS at L1 but in-sample + overlapping-pooled at meta
        "momentum_test_ic",
        "volatility_test_ic",
        "research_gbm_train_ic",
        "research_gbm_val_ic",
    ],
)
def test_in_sample_fields_are_low(field):
    assert classify_ic_reliability(field) == "low"


def test_representative_in_sample_vs_leakfree_pair():
    # The load-bearing distinction: same base metric, in-sample vs leak-free.
    assert classify_ic_reliability("meta_model_ic") == "low"
    assert classify_ic_reliability("meta_model_oos_ic_leakfree") == "high"


def test_unknown_field_fails_safe_to_low():
    # A brand-new IC field whose methodology hasn't been audited must NOT be
    # silently trusted as high-reliability.
    assert classify_ic_reliability("some_new_experimental_ic") == "low"


def test_case_insensitive():
    assert classify_ic_reliability("META_MODEL_OOS_IC_CPCV") == "high"
    assert classify_ic_reliability("Meta_Model_In_Sample_IC") == "low"


def test_explicit_low_beats_generic_high_marker():
    # meta_model_oos_ic contains "oos" but is explicitly LOW; it must not be
    # accidentally promoted to high by any generic marker.
    assert classify_ic_reliability("meta_model_oos_ic") == "low"


def test_build_map_covers_every_input_and_classifies():
    fields = ["meta_model_oos_ic_cpcv", "meta_model_ic", "momentum_median_ic"]
    m = build_ic_reliability_map(fields)
    assert set(m) == set(fields)
    assert m["meta_model_oos_ic_cpcv"] == "high"
    assert m["meta_model_ic"] == "low"
    assert m["momentum_median_ic"] == "high"
    # every value is a valid literal
    assert all(v in ("high", "low") for v in m.values())


# ── Manifest emits the map (source-text invariant, mirrors the repo's other
#    meta_trainer manifest tests which are source-invariant by design) ──────
_META_TRAINER = (
    Path(__file__).resolve().parent.parent / "training" / "meta_trainer.py"
)


@pytest.fixture(scope="module")
def meta_trainer_source() -> str:
    return _META_TRAINER.read_text()


def test_manifest_emits_ic_reliability_key(meta_trainer_source):
    assert 'manifest["ic_reliability"]' in meta_trainer_source, (
        "meta_trainer no longer assigns the ic_reliability map into the "
        "manifest — the evaluator (crucible-evaluator#969) reads this to tag "
        "predictor ICs with reliability. Additive S3 contract field."
    )


def test_manifest_uses_the_shared_classifier(meta_trainer_source):
    assert "build_ic_reliability_map" in meta_trainer_source, (
        "meta_trainer must build ic_reliability via the shared "
        "training.ic_reliability helper (single source of the taxonomy)."
    )


def test_manifest_covers_representative_leakfree_and_in_sample_fields(meta_trainer_source):
    # The map's field list must include both a representative leak-free and a
    # representative in-sample field so the emitted map is meaningfully mixed.
    assert '"meta_model_oos_ic_cpcv"' in meta_trainer_source
    assert '"meta_model_in_sample_ic"' in meta_trainer_source
