"""Fit-time detection of a cross-sectionally degenerate meta-model.

alpha-engine-config: the 2026-09-08 `predictor-inference` page ("VARIANCE
FALLBACK ENGAGED: calibrator outputs collapsed to 2 unique p_up bins across 29
tickers") was NOT a calibrator defect. It fired on three shadow arms whose L2
emits ONE alpha value for 24 of 29 tickers.

Measured on `predictor/registry/spec-sota-combine-2026-07-24-8578f8ae/
manifest.json` — every coefficient of appreciable size sits on a feature that
is CONSTANT within a date:

    macro_market_breadth      +0.188545   <- date-constant
    macro_spy_20d_vol         +0.164526   <- date-constant
    macro_yield_curve_slope   -0.119762   <- date-constant
    macro_spy_20d_return      +0.120213   <- date-constant
    macro_vix_term_slope      +0.103617   <- date-constant
    regime_intensity_z        -0.035828   <- date-constant
    residual_momentum_score   +0.013349   <- cross-sectional
    research_calibrator_prob  +0.009638   <- cross-sectional
    research_composite_score  +0.002761   <- cross-sectional
    research_conviction       -0.005465   <- cross-sectional
    sector_macro_modifier     -0.010126   <- cross-sectional

Such a model has a healthy-looking time-series variance and essentially no
cross-sectional variance. Its served alphas span 1.7% of the domain its own
calibrator was fitted on, so ANY calibrator plateaus on them. The defect is in
the FIT, and this is the fit-time test for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.xsec_variance_share import (
    CrossSectionDegenerate,
    assert_cross_sectionally_live,
    cross_sectional_variance_share,
)


def _panel(n_dates: int = 40, n_per_date: int = 30, seed: int = 7):
    """(meta_X, dates, feature_names) with one date-constant and one
    cross-sectionally varying feature."""
    rng = np.random.default_rng(seed)
    rows, dates = [], []
    for d in range(n_dates):
        macro = float(rng.normal())          # one value for the whole date
        for _ in range(n_per_date):
            rows.append([macro, float(rng.normal())])
            dates.append(f"2026-01-{d + 1:02d}")
    return np.asarray(rows, dtype=float), dates, ["macro_const", "xsec_feature"]


def test_date_constant_feature_reports_zero_cross_sectional_sd():
    X, dates, names = _panel()
    out = cross_sectional_variance_share(X, dates, names, {"macro_const": 1.0, "xsec_feature": 1.0})
    assert out["status"] == "ok"
    per = out["per_feature"]
    assert per["macro_const"]["xsec_sd"] == pytest.approx(0.0, abs=1e-12)
    assert per["macro_const"]["is_xsec_dead"] is True
    assert per["xsec_feature"]["xsec_sd"] > 0.5
    assert per["xsec_feature"]["is_xsec_dead"] is False
    assert out["n_xsec_dead_features"] == 1


def test_the_sota_combine_shape_is_flagged_degenerate():
    """The measured 2026-07-24 shape: all the coefficient mass on the
    date-constant feature, a rounding-error coefficient on the only feature
    that varies within a date."""
    X, dates, names = _panel()
    out = cross_sectional_variance_share(
        X, dates, names, {"macro_const": 0.1885, "xsec_feature": 0.0133},
    )
    assert out["xsec_variance_share"] < 0.10
    assert out["verdict"] == "degenerate_cross_section"
    with pytest.raises(CrossSectionDegenerate) as e:
        assert_cross_sectionally_live(out)
    # The raise must carry the numbers, not just a verdict.
    assert "xsec_variance_share" in str(e.value)


def test_a_cross_sectionally_live_model_passes():
    X, dates, names = _panel()
    out = cross_sectional_variance_share(
        X, dates, names, {"macro_const": 0.2, "xsec_feature": 1.75},
    )
    assert out["xsec_variance_share"] > 0.10
    assert out["verdict"] == "ok"
    assert_cross_sectionally_live(out)  # must not raise


def test_share_is_scale_free_in_the_target_units():
    """Multiplying every coefficient by a constant rescales the model but does
    NOT change how much of its variance is cross-sectional — the metric must
    be invariant, or a scale collapse would be read as a cross-section fix."""
    X, dates, names = _panel()
    coefs = {"macro_const": 0.1885, "xsec_feature": 0.0133}
    a = cross_sectional_variance_share(X, dates, names, coefs)
    b = cross_sectional_variance_share(
        X, dates, names, {k: v * 0.01 for k, v in coefs.items()},
    )
    assert a["xsec_variance_share"] == pytest.approx(b["xsec_variance_share"], rel=1e-9)
    assert b["xsec_sd"] == pytest.approx(a["xsec_sd"] * 0.01, rel=1e-9)


def test_intercept_does_not_move_the_share():
    X, dates, names = _panel()
    coefs = {"macro_const": 0.2, "xsec_feature": 1.75}
    a = cross_sectional_variance_share(X, dates, names, coefs, intercept=0.0)
    b = cross_sectional_variance_share(X, dates, names, coefs, intercept=-13.5)
    assert a["xsec_variance_share"] == pytest.approx(b["xsec_variance_share"], rel=1e-9)


def test_single_date_panel_refuses_rather_than_reporting_a_number():
    """With one date there is no time-series component, so the share is 1.0 by
    construction and means nothing. Reporting 1.0 would read as PASS — the
    exact `no data rendered as green` failure. It must say insufficient."""
    X, dates, names = _panel(n_dates=1)
    out = cross_sectional_variance_share(X, dates, names, {"macro_const": 1.0, "xsec_feature": 1.0})
    assert out["status"] == "insufficient_data"
    assert out["xsec_variance_share"] is None
    assert out["verdict"] == "unmeasurable"
    assert "n_dates" in out["reason"]
    with pytest.raises(CrossSectionDegenerate):
        assert_cross_sectionally_live(out)


def test_an_unmeasurable_summary_never_passes_the_gate():
    """`unmeasurable` is not `ok`. A component emitting nothing is unobserved,
    never healthy."""
    for status in ("insufficient_data", "error", "not_run"):
        with pytest.raises(CrossSectionDegenerate):
            assert_cross_sectionally_live({"status": status, "verdict": "unmeasurable"})


def test_all_zero_coefficients_are_degenerate_not_a_divide_by_zero():
    X, dates, names = _panel()
    out = cross_sectional_variance_share(X, dates, names, {"macro_const": 0.0, "xsec_feature": 0.0})
    assert out["verdict"] == "degenerate_cross_section"
    assert out["total_sd"] == pytest.approx(0.0, abs=1e-12)
    assert out["reason"]


def test_missing_coefficient_for_a_named_feature_raises():
    """Silently treating an absent coefficient as 0.0 would understate the
    model's real cross-sectional variance and turn a wiring bug into a PASS."""
    X, dates, names = _panel()
    with pytest.raises(ValueError, match="xsec_feature"):
        cross_sectional_variance_share(X, dates, names, {"macro_const": 1.0})


def test_row_count_mismatch_raises():
    X, dates, names = _panel()
    with pytest.raises(ValueError, match="dates"):
        cross_sectional_variance_share(X, dates[:-5], names, {"macro_const": 1.0, "xsec_feature": 1.0})


def test_non_finite_rows_are_refused_not_dropped():
    X, dates, names = _panel()
    X[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cross_sectional_variance_share(X, dates, names, {"macro_const": 1.0, "xsec_feature": 1.0})


def test_dead_coefficient_mass_share_is_reported():
    X, dates, names = _panel()
    out = cross_sectional_variance_share(
        X, dates, names, {"macro_const": 0.1885, "xsec_feature": 0.0133},
    )
    # |0.1885| * sd(macro) vs |0.0133| * sd(xsec); both features are ~unit sd.
    assert out["coef_mass_on_xsec_dead"] > 0.9
