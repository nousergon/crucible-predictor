"""alpha-engine-config-I11522: a feature block that is 0% finite must not let
its arm grade ``valid``.

In rehearsal-2026-09-23-2 the Stage-1b macro array logged ``finite_pct=0.00``
(and did so on every weekly run from 2026-07-18), yet ``volatility_macro_aug``
graded ``valid``: best_iteration 49, val_ic 0.314. It carried the fit on its
six volatility columns alone. These tests feed that exact shape.
"""
from __future__ import annotations

import numpy as np
import pytest

from training.l1_fit_validity import (
    L1_FIT_REGISTER,
    L1FitValidityError,
    assert_l1_fits_valid,
    evaluate_l1_fits,
    measure_feature_block,
)

MACRO_COLS = [
    "spy_20d_return", "spy_20d_vol", "vix_level", "vix_term_slope",
    "yield_curve_slope", "market_breadth", "vix_vix3m_ratio",
    "market_breadth_200d", "yield_curve_10y_2y", "hy_oas_level",
    "hy_oas_change_21d", "credit_spread", "credit_spread_change_21d",
]

# The rehearsal's fit record, verbatim except for the block report.
REHEARSAL_MACRO_AUG = {
    "fitted": True, "best_iteration": 49, "val_ic": 0.314331,
    "train_ic": None, "n_estimators": 2000, "n_samples": 1_700_000,
    "output_dispersion": None,
}
OTHER_ARMS = {
    "research_gbm": {"fitted": True, "best_iteration": 500, "val_ic": 0.2137,
                     "train_ic": 0.41, "n_estimators": 500, "n_samples": 4804},
    "volatility": {"fitted": True, "best_iteration": 97, "val_ic": 0.335,
                   "train_ic": None, "n_estimators": 2000, "n_samples": 1_700_000},
}


def _zero_finite_block(n_rows=500):
    """Every row has at least one NaN column: two columns entirely empty,
    the shape of an absent ArcticDB macro symbol."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_rows, len(MACRO_COLS))).astype(np.float32)
    X[:, MACRO_COLS.index("yield_curve_10y_2y")] = np.nan
    X[:, MACRO_COLS.index("credit_spread")] = np.nan
    return X


def _grade(block_report):
    fit = dict(REHEARSAL_MACRO_AUG, feature_blocks={"macro": block_report})
    return evaluate_l1_fits({**OTHER_ARMS, "volatility_macro_aug": fit})


def test_zero_finite_macro_block_is_not_valid_and_names_the_block():
    report = measure_feature_block(_zero_finite_block(), MACRO_COLS)
    assert report["finite_pct"] == 0.0
    block = _grade(report)
    arm = block["arms"]["volatility_macro_aug"]
    assert arm["status"] == "starved_input"
    assert arm["status"] != "valid"
    assert "'macro'" in arm["reason"]
    # The reason names the columns that starved it.
    assert "yield_curve_10y_2y=0.00" in arm["reason"]
    assert "credit_spread=0.00" in arm["reason"]


def test_starved_observe_only_arm_degrades_the_run_by_name_not_fails_it():
    block = _grade(measure_feature_block(_zero_finite_block(), MACRO_COLS))
    assert block["failures"] == []
    assert block["status"] == "degraded"
    assert [d["arm"] for d in block["degradations"]] == ["volatility_macro_aug"]
    assert block["degradations"][0]["status"] == "starved_input"
    assert_l1_fits_valid(block)  # observe-only: does not stop the run


def test_a_declared_block_that_is_not_reported_is_starved_not_valid():
    block = evaluate_l1_fits({**OTHER_ARMS, "volatility_macro_aug": dict(REHEARSAL_MACRO_AUG)})
    arm = block["arms"]["volatility_macro_aug"]
    assert arm["status"] == "starved_input"
    assert "not reported" in arm["reason"]


def test_a_fed_block_leaves_the_arm_valid():
    X = np.random.default_rng(1).normal(size=(500, len(MACRO_COLS)))
    X[:40] = np.nan  # z-score warm-up rows
    report = measure_feature_block(X, MACRO_COLS)
    assert report["finite_pct"] == pytest.approx(0.92)
    block = _grade(report)
    assert block["arms"]["volatility_macro_aug"]["status"] == "valid"
    assert block["status"] == "ok"


def test_the_floor_is_declared_on_the_register_not_inferred():
    spec = next(s for s in L1_FIT_REGISTER if s.name == "volatility_macro_aug")
    assert dict(spec.feature_block_floors) == {"macro": 0.50}
    assert spec.starved_input_severity == "optional"


def test_a_required_starved_arm_fails_the_run():
    from dataclasses import replace
    reg = tuple(
        replace(s, starved_input_severity="required") if s.name == "volatility_macro_aug" else s
        for s in L1_FIT_REGISTER
    )
    fit = dict(REHEARSAL_MACRO_AUG, feature_blocks={
        "macro": measure_feature_block(_zero_finite_block(), MACRO_COLS)})
    block = evaluate_l1_fits({**OTHER_ARMS, "volatility_macro_aug": fit}, register=reg)
    assert block["status"] == "failed"
    with pytest.raises(L1FitValidityError, match="starved_input"):
        assert_l1_fits_valid(block)


def test_starved_input_outranks_a_fit_quality_verdict():
    fit = dict(REHEARSAL_MACRO_AUG, best_iteration=3, feature_blocks={
        "macro": measure_feature_block(_zero_finite_block(), MACRO_COLS)})
    block = evaluate_l1_fits({**OTHER_ARMS, "volatility_macro_aug": fit})
    arm = block["arms"]["volatility_macro_aug"]
    assert arm["status"] == "starved_input"
    assert {i["status"] for i in arm["issues"]} >= {"starved_input", "underfit_early_stop"}
    # ...but the required fit failure is NOT demoted to a degradation.
    assert block["status"] == "failed"
    assert [f["arm"] for f in block["failures"]] == ["volatility_macro_aug"]
