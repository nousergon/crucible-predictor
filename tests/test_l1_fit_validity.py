"""Regression tests for the L1 fit-validity gate (alpha-engine-config-I9271).

Every threshold in ``L1_FIT_REGISTER`` is asserted against the three real
weekly vintages that produced the served-alpha dispersion collapse, so a future
change to a floor has to move a test that names the measured numbers.

Verified RED before the fix: with no ``training/l1_fit_validity`` module the
whole file fails to import, which is the honest pre-state — the 2026-08-28
manifest carried ``best_iteration: 2``, ``val_ic: 0.01403``,
``train_val_ic_ratio: 8.987169`` and ``overfit_warn: true``, and no code path
in the repository read any of them.
"""
from __future__ import annotations

import math

import pytest

from training.arm_validity import (
    XSEC_FEATURES,
    coef_norm,
    evaluate_arm_validity,
    xsec_coef_norm,
)
from training.l1_fit_validity import (
    L1_FIT_REGISTER,
    L1FitValidityError,
    assert_l1_fits_valid,
    evaluate_l1_fits,
    measure_output_dispersion,
)


# The three shipped `models.research_gbm` blocks, verbatim from
# s3://alpha-engine-research/predictor/registry/<version_id>/manifest.json.
RESEARCH_GBM_VINTAGES = {
    "2026-08-14": {
        "fitted": True, "best_iteration": 3, "val_ic": 0.042958,
        "train_ic": 0.265918, "n_estimators": 500, "n_samples": 1928,
        # Replayed: the shipped booster over the 2026-08-28 signals snapshot's
        # 903 tickers produced 4 distinct values.
        "output_dispersion": 0.011149,
    },
    "2026-08-21": {
        "fitted": True, "best_iteration": 500, "val_ic": 0.213677,
        "train_ic": 0.412136, "n_estimators": 500, "n_samples": 4804,
        "output_dispersion": 0.107021,
    },
    "2026-08-28": {
        "fitted": True, "best_iteration": 2, "val_ic": 0.01403,
        "train_ic": 0.126092, "n_estimators": 500, "n_samples": 8372,
        "output_dispersion": 0.003731,
    },
}

# The volatility arms across the same three vintages — measured healthy, and
# present so the gate is shown not to fire on the arms it should leave alone.
HEALTHY_VOL_FITS = {
    "volatility": {
        "fitted": True, "best_iteration": 120, "val_ic": 0.33,
        "train_ic": None, "n_estimators": 2000, "n_samples": 100000,
        "output_dispersion": None,
    },
    "volatility_macro_aug": {
        "fitted": True, "best_iteration": 145, "val_ic": 0.331344,
        "train_ic": None, "n_estimators": 2000, "n_samples": 100000,
        "output_dispersion": None,
    },
    "volatility_risk_aug": {
        "fitted": True, "best_iteration": 150, "val_ic": 0.334690,
        "train_ic": None, "n_estimators": 2000, "n_samples": 100000,
        "output_dispersion": None,
    },
}


def _fits(vintage: str) -> dict:
    return {"research_gbm": dict(RESEARCH_GBM_VINTAGES[vintage]), **HEALTHY_VOL_FITS}


# ── The replay the issue's Closes-when requires ─────────────────────────────

def test_gate_blocks_the_2026_08_28_vintage():
    block = evaluate_l1_fits(_fits("2026-08-28"))
    assert block["status"] == "failed"
    rg = block["arms"]["research_gbm"]
    assert rg["status"] == "underfit_early_stop"
    assert rg["best_iteration"] == 2
    with pytest.raises(L1FitValidityError) as exc:
        assert_l1_fits_valid(block)
    assert "research_gbm" in str(exc.value)
    assert "iteration 2" in str(exc.value)


def test_gate_passes_the_2026_08_21_vintage():
    block = evaluate_l1_fits(_fits("2026-08-21"))
    assert block["status"] == "ok", block["failures"]
    assert_l1_fits_valid(block)  # must not raise


def test_gate_also_blocks_2026_08_14_and_that_is_the_correct_verdict():
    """Not an over-tight threshold: 08-14 shipped 4 distinct values over 903
    names (std 0.011149) against 08-21's 136 (std 0.107021)."""
    block = evaluate_l1_fits(_fits("2026-08-14"))
    assert block["status"] == "failed"
    assert block["arms"]["research_gbm"]["status"] == "underfit_early_stop"


@pytest.mark.parametrize("vintage,expected", [
    ("2026-08-14", False), ("2026-08-21", True), ("2026-08-28", False),
])
def test_each_floor_independently_reproduces_the_verdict(vintage, expected):
    """best_iteration, |val_ic| and the train/val ratio each separate the three
    vintages on their own — no floor is carrying the others."""
    spec = next(s for s in L1_FIT_REGISTER if s.name == "research_gbm")
    fit = RESEARCH_GBM_VINTAGES[vintage]
    ratio = abs(fit["train_ic"]) / abs(fit["val_ic"])
    assert (fit["best_iteration"] >= spec.min_best_iteration) is expected
    assert (abs(fit["val_ic"]) >= spec.min_abs_val_ic) is expected
    assert (ratio <= spec.max_train_val_ic_ratio) is expected


# ── The dead warning fields become load-bearing ─────────────────────────────

def test_train_val_ic_ratio_alone_fails_the_run():
    """`train_val_ic_ratio: 8.987169` and `overfit_warn: true` sat on the
    2026-08-28 manifest and gated nothing. Here the ratio alone is fatal."""
    fits = _fits("2026-08-21")
    # 500 iterations and a strong val_ic — only the ratio is bad.
    fits["research_gbm"].update(train_ic=0.9, val_ic=0.1, best_iteration=500)
    block = evaluate_l1_fits(fits)
    assert block["arms"]["research_gbm"]["status"] == "train_val_gap"
    assert block["arms"]["research_gbm"]["train_val_ic_ratio"] == 9.0
    with pytest.raises(L1FitValidityError):
        assert_l1_fits_valid(block)


def test_val_ic_alone_fails_the_run():
    fits = _fits("2026-08-21")
    fits["research_gbm"].update(val_ic=0.01, train_ic=0.02, best_iteration=500)
    block = evaluate_l1_fits(fits)
    assert block["arms"]["research_gbm"]["status"] == "no_val_signal"
    with pytest.raises(L1FitValidityError):
        assert_l1_fits_valid(block)


# ── I9376: a val_ic that PASSES the floor but is not distinguishable from
# ── noise at this sample size is reported insufficient, never a pass ───────

def test_wide_val_ic_se_downgrades_a_pass_to_insufficient():
    """The measured 08-28-era geometry: 10 validation dates against a 10-day
    label is n_eff ~ 1, so the per-date IC std (champion arch 0.0528) IS the
    val_ic standard error — larger than the 0.05 min_abs_val_ic floor it is
    being compared against. A val_ic that clears the floor here is not a
    measurement of signal; it is noise that happened to clear it."""
    fits = _fits("2026-08-21")
    fits["research_gbm"].update(
        val_ic=0.06, train_ic=0.09, best_iteration=500,
        val_ic_precision={
            "n_val_dates_scored": 10, "mean_per_date_ic": 0.06,
            "per_date_ic_std": 0.0528, "n_eff": 1.0, "val_ic_se": 0.0528,
            "label_horizon_days": 10,
        },
    )
    block = evaluate_l1_fits(fits)
    rg = block["arms"]["research_gbm"]
    assert rg["status"] == "insufficient"
    assert block["status"] == "degraded"
    # Never blocking — a required arm's insufficiency does not fail the run.
    assert_l1_fits_valid(block)  # must not raise
    assert block["degradations"]
    assert not block["failures"]


def test_narrow_val_ic_se_is_still_a_plain_pass():
    """The 2026-08-21 vintage's own precision (500 dates worth of n_eff via a
    long panel) does not trip the new check — same fixture as
    ``test_gate_passes_the_2026_08_21_vintage``, with an SE well under the
    floor made explicit."""
    fits = _fits("2026-08-21")
    fits["research_gbm"]["val_ic_precision"] = {
        "n_val_dates_scored": 500, "mean_per_date_ic": 0.213677,
        "per_date_ic_std": 0.0528, "n_eff": 50.0, "val_ic_se": 0.007465,
        "label_horizon_days": 10,
    }
    block = evaluate_l1_fits(fits)
    assert block["arms"]["research_gbm"]["status"] == "valid"
    assert block["status"] == "ok"


def test_insufficient_never_masks_a_real_failure():
    """A val_ic that FAILS the floor stays `no_val_signal` even when its SE is
    also wide — insufficient only ever downgrades a would-be pass."""
    fits = _fits("2026-08-21")
    fits["research_gbm"].update(
        val_ic=0.01, train_ic=0.02, best_iteration=500,
        val_ic_precision={
            "n_val_dates_scored": 10, "mean_per_date_ic": 0.01,
            "per_date_ic_std": 0.0528, "n_eff": 1.0, "val_ic_se": 0.0528,
            "label_horizon_days": 10,
        },
    )
    block = evaluate_l1_fits(fits)
    assert block["arms"]["research_gbm"]["status"] == "no_val_signal"
    with pytest.raises(L1FitValidityError):
        assert_l1_fits_valid(block)


# ── Absence and unmeasurability are never a pass ────────────────────────────

def test_absent_canonical_arm_is_a_failure_not_a_fallback():
    """The bucket-lookup fallback silently supplying `research_calibrator_prob`
    is the substitution the gate exists to refuse."""
    fits = dict(HEALTHY_VOL_FITS)  # research_gbm missing entirely
    block = evaluate_l1_fits(fits)
    assert block["arms"]["research_gbm"]["status"] == "absent"
    with pytest.raises(L1FitValidityError) as exc:
        assert_l1_fits_valid(block)
    assert "research_calibrator_prob" in str(exc.value)


def test_unmeasured_best_iteration_is_a_failure_not_a_pass():
    """`volatility` shipped `test_ic` and nothing else for its whole life."""
    fits = _fits("2026-08-21")
    fits["volatility"] = {"fitted": True, "best_iteration": None,
                          "val_ic": None, "n_estimators": 2000}
    block = evaluate_l1_fits(fits)
    assert block["arms"]["volatility"]["status"] == "unmeasured"
    with pytest.raises(L1FitValidityError):
        assert_l1_fits_valid(block)


def test_optional_arm_absence_is_not_evaluated_not_a_failure():
    fits = {"research_gbm": dict(RESEARCH_GBM_VINTAGES["2026-08-21"]),
            "volatility": HEALTHY_VOL_FITS["volatility"]}
    block = evaluate_l1_fits(fits)
    assert block["status"] == "ok"
    assert block["arms"]["volatility_macro_aug"]["status"] == "not_evaluated"


def test_every_declared_l1_is_graded_every_run():
    """A verdict for every registered arm, including the ones that passed — a
    findings list containing only failures cannot be audited."""
    block = evaluate_l1_fits(_fits("2026-08-21"))
    assert set(block["arms"]) == {s.name for s in L1_FIT_REGISTER}


# ── Recorded properties ─────────────────────────────────────────────────────

def test_early_stopping_never_fired_is_recorded_but_does_not_block():
    block = evaluate_l1_fits(_fits("2026-08-21"))
    notes = [n for n in block["notes"] if n["arm"] == "research_gbm"]
    assert notes and notes[0]["note"] == "early_stopping_never_fired"
    assert block["status"] == "ok"


def test_output_dispersion_is_raw_never_standardized():
    """champion-challenger §5.3: a scale-invariant form divides the collapse
    away. 0.107021 -> 0.003731 is a 28.7x drop; the standardized ratio is 1.0."""
    healthy = [0.4, 0.5, 0.6]
    collapsed = [0.4986, 0.5, 0.5014]
    d_healthy = measure_output_dispersion(healthy)
    d_collapsed = measure_output_dispersion(collapsed)
    assert d_healthy / d_collapsed > 50
    # The standardized version of each is identical, which is the trap.
    def _std_ratio(xs):
        m = sum(xs) / len(xs)
        s = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
        return s / s
    assert _std_ratio(healthy) == _std_ratio(collapsed) == 1.0


def test_dispersion_of_a_constant_arm_is_zero_not_none():
    assert measure_output_dispersion([0.5, 0.5, 0.5]) == 0.0
    assert measure_output_dispersion([0.5]) is None
    assert measure_output_dispersion([]) is None


# ── The cross-sectional coefficient norm ────────────────────────────────────

# `models.meta_model.importance.standardized_coef`, VERBATIM from the three
# shipped manifests at
# s3://alpha-engine-research/predictor/registry/<version_id>/manifest.json.
STD_COEF = {
    "2026-08-14": {
        "research_calibrator_prob": 0.22661,
        "momentum_score": -0.090155,
        "expected_move": 0.179741,
        "research_composite_score": -0.005712,
        "research_conviction": -0.040151,
        "sector_macro_modifier": -0.072618,
        "macro_spy_20d_return": 0.505912,
        "macro_spy_20d_vol": 0.115091,
        "macro_vix_level": -0.232367,
        "macro_vix_term_slope": 0.349847,
        "macro_yield_curve_slope": 0.055587,
        "macro_market_breadth": 0.324709,
        "regime_intensity_z": -0.532399,
        "guidance_direction": 0.0,
        "risk_factor_count_delta_raw": 0.0,
        "management_tone_zscore": 0.0,
    },
    "2026-08-21": {
        "research_calibrator_prob": 0.238183,
        "momentum_score": -0.044004,
        "expected_move": 0.041321,
        "research_composite_score": 0.106641,
        "research_conviction": -0.000125,
        "sector_macro_modifier": -0.034378,
        "macro_spy_20d_return": 0.0,
        "macro_spy_20d_vol": 0.0,
        "macro_vix_level": 0.0,
        "macro_vix_term_slope": 0.0,
        "macro_yield_curve_slope": 0.0,
        "macro_market_breadth": 0.0,
        "regime_intensity_z": 0.0,
        "guidance_direction": 0.0,
        "risk_factor_count_delta_raw": 0.0,
        "management_tone_zscore": 0.0,
    },
    "2026-08-28": {
        "research_calibrator_prob": 0.028427,
        "momentum_score": -0.007672,
        "expected_move": 0.06555,
        "research_composite_score": 0.081429,
        "research_conviction": -0.001404,
        "sector_macro_modifier": -0.054197,
        "macro_spy_20d_return": 0.0,
        "macro_spy_20d_vol": 0.0,
        "macro_vix_level": 0.0,
        "macro_vix_term_slope": 0.0,
        "macro_yield_curve_slope": 0.0,
        "macro_market_breadth": 0.0,
        "regime_intensity_z": 0.0,
        "guidance_direction": 0.0,
        "risk_factor_count_delta_raw": 0.0,
        "management_tone_zscore": 0.0,
    },
}


def test_xsec_norm_tracks_the_served_dispersion_and_the_full_norm_does_not():
    x = {v: xsec_coef_norm(c) for v, c in STD_COEF.items()}
    f = {v: coef_norm(c) for v, c in STD_COEF.items()}
    # The measured series in alpha-engine-config-I9271.
    assert x["2026-08-14"] == pytest.approx(0.3142, abs=5e-3)
    assert x["2026-08-21"] == pytest.approx(0.2701, abs=5e-3)
    assert x["2026-08-28"] == pytest.approx(0.1214, abs=5e-3)
    # 0.860 then 0.449, against a served alpha_stdev ratio of 0.347.
    assert x["2026-08-21"] / x["2026-08-14"] == pytest.approx(0.860, abs=0.02)
    assert x["2026-08-28"] / x["2026-08-21"] == pytest.approx(0.449, abs=0.02)
    # The full-vector norm's first step is 0.279 — driven entirely by the
    # market-wide macro block, which costs the within-date spread nothing.
    assert f["2026-08-21"] / f["2026-08-14"] == pytest.approx(0.279, abs=0.02)


def test_collapse_check_gates_on_the_xsec_norm_not_the_full_vector():
    """Regression for the threshold/quantity mismatch: at min_norm_ratio=0.50
    the full-vector norm refuses 2026-08-21 (ratio 0.279), which is the
    opposite of the verdict the floor was sited to give."""
    prior_x = [xsec_coef_norm(STD_COEF["2026-08-14"])]
    ok = evaluate_arm_validity(
        arm="v3.0-meta", standardized_coef=STD_COEF["2026-08-21"],
        meta_X=[[1.0, 2.0], [3.0, 4.0]], feature_names=["a", "b"],
        prior_standardized_coef=STD_COEF["2026-08-14"],
        prior_coef_norms=prior_x, min_norm_ratio=0.50,
    )
    collapse = next(c for c in ok["checks"] if c["check"] == "coef_norm_collapse")
    assert collapse["status"] == "pass", collapse["reason"]

    prior_x2 = sorted([xsec_coef_norm(STD_COEF["2026-08-14"]),
                       xsec_coef_norm(STD_COEF["2026-08-21"])])
    bad = evaluate_arm_validity(
        arm="v3.0-meta", standardized_coef=STD_COEF["2026-08-28"],
        meta_X=[[1.0, 2.0], [3.0, 4.0]], feature_names=["a", "b"],
        prior_standardized_coef=STD_COEF["2026-08-21"],
        prior_coef_norms=prior_x2, min_norm_ratio=0.50,
    )
    collapse = next(c for c in bad["checks"] if c["check"] == "coef_norm_collapse")
    assert collapse["status"] == "fail"
    assert "CROSS-SECTIONAL" in collapse["reason"]


def test_both_norms_are_recorded_on_the_block():
    block = evaluate_arm_validity(
        arm="v3.0-meta", standardized_coef=STD_COEF["2026-08-14"],
        meta_X=[[1.0, 2.0], [3.0, 4.0]], feature_names=["a", "b"],
    )
    assert block["xsec_coef_norm"] == pytest.approx(0.3142, abs=5e-3)
    assert block["full_coef_norm"] == pytest.approx(0.9676, abs=5e-3)
    # The historical key carries the GATED quantity so the series stays
    # comparable with what prior vintages are re-measured to.
    assert block["coef_norm"] == block["xsec_coef_norm"]


def test_market_wide_features_are_excluded_by_name():
    assert "macro_vix_level" not in XSEC_FEATURES
    assert "regime_intensity_z" not in XSEC_FEATURES
    assert "research_calibrator_prob" in XSEC_FEATURES


# ── 2026-09-05: a split that could not be BUILT is `insufficient`, not a failure ──
#
# Measured on weekly SF watch-rerun-2026-09-04-2 (after the arm-validity gate
# had passed): `build_purged_split` reported INSUFFICIENT for research_gbm —
# "the validation block had to take 90.7% of the panel's rows to span 50
# dates; measured 0.9068, required <= 0.4 … rows-per-date range 1..895 over
# 107 dates". The arm was registered fitted=False, and this gate graded that
# `not_fitted` → required → the task failed. Both modules' rule 1 says the
# opposite: "a check that cannot be computed is reported insufficient and does
# not block" (champion-challenger-policy §5.1) — the same rule this file
# already applies to a val_ic whose standard error swamps its floor (I9376).
# A split whose floors the PANEL cannot meet is that case one step earlier.
# It is loud (status on the manifest, a degradation finding, a WARNING), the
# bucket-lookup fallback is declared, and any OTHER reason for fitted=False —
# a degenerate design matrix, a booster that never built — still fails.

_INSUFFICIENT_SPLIT_BLOCK = {
    "status": "insufficient",
    "reason": (
        "the validation block had to take 90.7% of the panel's rows to span 50 "
        "dates; measured 0.9068, required <= 0.4."
    ),
    "min_val_dates": 50, "max_val_row_fraction": 0.4,
}


def test_a_split_the_panel_cannot_support_is_insufficient_not_a_failure():
    fits = dict(HEALTHY_VOL_FITS)
    fits["research_gbm"] = {
        "fitted": False,
        "reason": "UnmeasurableSplitError: research_gbm: refusing to fit on an unmeasurable split — " + _INSUFFICIENT_SPLIT_BLOCK["reason"],
        "not_fitted_kind": "split_insufficient",
        "split": dict(_INSUFFICIENT_SPLIT_BLOCK),
        "design_matrix": {"n_rows": 14940, "n_varying_features": 6},
    }
    block = evaluate_l1_fits(fits)
    rg = block["arms"]["research_gbm"]
    assert rg["status"] == "insufficient", rg
    assert "90.7%" in rg["reason"] and "research_calibrator_prob" in rg["reason"]
    assert block["status"] == "degraded"
    assert block["failures"] == []
    assert [d["arm"] for d in block["degradations"]] == ["research_gbm"]
    assert_l1_fits_valid(block)  # must not raise


def test_fitted_false_for_any_other_reason_is_still_a_failure():
    fits = dict(HEALTHY_VOL_FITS)
    fits["research_gbm"] = {
        "fitted": False,
        "reason": "DegenerateDesignMatrixError: 5 of 6 research columns are constant",
        "split": {"status": "ok", "reason": None},
        "design_matrix": {"n_rows": 14940, "n_varying_features": 1},
    }
    block = evaluate_l1_fits(fits)
    assert block["arms"]["research_gbm"]["status"] == "not_fitted"
    with pytest.raises(L1FitValidityError):
        assert_l1_fits_valid(block)


def test_fitted_false_with_no_split_record_is_still_a_failure():
    fits = dict(HEALTHY_VOL_FITS)
    fits["research_gbm"] = {"fitted": False, "reason": "booster never built"}
    block = evaluate_l1_fits(fits)
    assert block["arms"]["research_gbm"]["status"] == "not_fitted"
    with pytest.raises(L1FitValidityError):
        assert_l1_fits_valid(block)


def test_an_insufficient_split_block_without_the_typed_refusal_is_still_a_failure():
    """A degenerate design matrix caught in the same try/except carries an
    `insufficient` split block too (tests/test_purged_split.py pins that
    shape). Only the TYPED split refusal downgrades; the split block alone
    must not."""
    fits = dict(HEALTHY_VOL_FITS)
    fits["research_gbm"] = {
        "fitted": False,
        "reason": "DegenerateDesignMatrixError: 1 varying column(s) of 9",
        "not_fitted_kind": "error",
        "split": dict(_INSUFFICIENT_SPLIT_BLOCK),
    }
    block = evaluate_l1_fits(fits)
    assert block["arms"]["research_gbm"]["status"] == "not_fitted"
    with pytest.raises(L1FitValidityError):
        assert_l1_fits_valid(block)
