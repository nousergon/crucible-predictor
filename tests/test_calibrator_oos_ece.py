"""The calibrator's quality number must be able to go bad
(alpha-engine-config-I10181).

``PlattCalibrator.fit`` computed ``_ece_after`` by calling ``predict_proba`` on
THE SAME ``raw_alphas`` it had just fitted on. Under ``method="isotonic"`` that
is near-zero by construction: PAVA reproduces the pooled empirical frequency of
each block, so in-sample ECE measures the fit's own arithmetic.

Measured on the live artifact
``predictor/weights/meta/isotonic_calibrator.pkl.meta.json`` (deployed
2026-09-05)::

    {"method": "isotonic", "n_samples": 14940,
     "ece_before": 0.017291, "ece_after": 3e-06}

3e-06 on 14,940 samples is not evidence of calibration — and it was logged as a
99.98% reduction, persisted, reloaded, and printed on every healthy inference
pass. The one number the system reported about calibrator quality could not go
bad, and a reader who checked it was reassured by an identity.
"""

from __future__ import annotations

import numpy as np
import pytest

from model.calibrator import PlattCalibrator


def _dates(n, per_day=30, start=1):
    """Time-ordered day labels, ``per_day`` rows each."""
    return [f"2026-{1 + (start + i // per_day) // 28:02d}-"
            f"{1 + (start + i // per_day) % 28:02d}" for i in range(n)]


def _miscalibrated(n=3000, seed=5):
    """A fit whose in-sample ECE is ~0 and whose held-out ECE is large: the
    relationship between alpha and P(UP) REVERSES in the last fifth of the
    sample. An isotonic fit interpolates the whole thing in-sample; on data it
    did not see, it is confidently wrong."""
    rng = np.random.default_rng(seed)
    alphas = rng.normal(0, 0.02, n)
    up = (alphas > 0).astype(np.int32)
    cut = int(n * 0.8)
    up[cut:] = 1 - up[cut:]          # the regime flips out of sample
    return alphas, up, _dates(n)


def _well_calibrated(n=3000, seed=5):
    rng = np.random.default_rng(seed)
    alphas = rng.normal(0, 0.02, n)
    p = 1.0 / (1.0 + np.exp(-alphas * 60.0))
    up = (rng.random(n) < p).astype(np.int32)
    return alphas, up, _dates(n)


# ── the defect ───────────────────────────────────────────────────────────────

def test_a_miscalibrated_isotonic_fit_reports_a_large_oos_ece(caplog):
    """RED against the pre-I10181 tree: there was no OOS number at all, and the
    in-sample one is ~0 for this exact fit."""
    a, up, d = _miscalibrated()
    cal = PlattCalibrator(method="isotonic").fit(a, up, dates=d)
    m = cal.metrics()
    assert m["ece_after_in_sample"] < 0.01, (
        "the in-sample number is near zero by construction — that is the point"
    )
    assert m["ece_after_oos"] is not None
    assert m["ece_after_oos"] > 0.10
    assert m["n_oos_samples"] > 0


def test_a_well_calibrated_isotonic_fit_reports_a_small_oos_ece():
    a, up, d = _well_calibrated()
    cal = PlattCalibrator(method="isotonic").fit(a, up, dates=d)
    assert cal.metrics()["ece_after_oos"] < 0.10


# ── the split is time-ordered and embargoed ──────────────────────────────────

def test_the_split_is_by_DATE_not_by_row():
    """A row-index split would put the same day on both sides, which leaks: the
    meta predictions for one date share the market's move."""
    a, up, d = _well_calibrated()
    cal = PlattCalibrator(method="isotonic").fit(a, up, dates=d)
    m = cal.metrics()
    assert m["ece_oos_split"]["train_dates"]
    assert m["ece_oos_split"]["oos_dates"]
    assert not (set(m["ece_oos_split"]["train_dates"])
                & set(m["ece_oos_split"]["oos_dates"]))
    assert max(m["ece_oos_split"]["train_dates"]) < min(m["ece_oos_split"]["oos_dates"])


def test_the_embargo_removes_dates_between_train_and_oos():
    a, up, d = _well_calibrated()
    plain = PlattCalibrator(method="isotonic").fit(a, up, dates=d)
    embargoed = PlattCalibrator(method="isotonic").fit(
        a, up, dates=d, embargo_days=5,
    )
    assert (embargoed.metrics()["ece_oos_split"]["n_embargoed_dates"] == 5)
    assert (len(embargoed.metrics()["ece_oos_split"]["train_dates"])
            < len(plain.metrics()["ece_oos_split"]["train_dates"]))


# ── absence is never the in-sample number ────────────────────────────────────

def test_without_dates_the_oos_number_is_null_with_a_reason():
    """A random split would leak across dates, so no number is produced at all.
    It must NOT silently be the in-sample one."""
    a, up, _ = _well_calibrated()
    m = PlattCalibrator(method="isotonic").fit(a, up).metrics()
    assert m["ece_after_oos"] is None
    assert m["n_oos_samples"] == 0
    assert "date" in m["ece_oos_reason"].lower()
    assert m["ece_after_in_sample"] is not None


def test_too_few_holdout_samples_yields_null_with_a_reason():
    a, up, d = _well_calibrated(n=260)
    m = PlattCalibrator(method="isotonic").fit(a, up, dates=d).metrics()
    assert m["ece_after_oos"] is None
    assert m["ece_oos_reason"]
    assert "100" in m["ece_oos_reason"]


def test_a_single_date_cannot_be_split_and_says_so():
    a, up, _ = _well_calibrated()
    d = ["2026-09-04"] * len(a)
    m = PlattCalibrator(method="isotonic").fit(a, up, dates=d).metrics()
    assert m["ece_after_oos"] is None
    assert "date" in m["ece_oos_reason"].lower()


# ── the served model is unchanged ────────────────────────────────────────────

def test_the_fitted_model_is_still_the_one_fitted_on_ALL_rows():
    """The OOS number is measured on a split fit and then discarded. Serving on
    a model fitted to 80% of the data would be a silent capability regression
    smuggled in with a metric."""
    a, up, d = _well_calibrated()
    with_dates = PlattCalibrator(method="isotonic").fit(a, up, dates=d)
    without = PlattCalibrator(method="isotonic").fit(a, up)
    probe = np.linspace(-0.05, 0.05, 41)
    np.testing.assert_allclose(
        with_dates.predict_proba(probe), without.predict_proba(probe),
    )
    assert with_dates.metrics()["n_samples"] == len(a)


# ── naming, and the readers ──────────────────────────────────────────────────

def test_the_in_sample_number_keeps_its_old_key_and_gains_an_explicit_one():
    """`ece_after` is read by training/train_handler.py (the training email) and
    by the backtester's sidecar reader in another repo. It keeps its meaning —
    in-sample — and `ece_after_in_sample` is the name that cannot be
    misread. Breaking the old key in this PR would break a cross-repo reader
    for a rename."""
    a, up, d = _well_calibrated()
    m = PlattCalibrator(method="isotonic").fit(a, up, dates=d).metrics()
    assert m["ece_after"] == m["ece_after_in_sample"]


def test_platt_is_not_reported_as_a_regression_against_isotonic():
    """A 1-parameter logistic cannot interpolate 14,940 points, so its
    in-sample number was never degenerate. The OOS number must be computed for
    it by the SAME code — the shadow-Platt-vs-live-isotonic comparison must not
    become a comparison of two different quantities."""
    a, up, d = _well_calibrated()
    platt = PlattCalibrator(method="platt").fit(a, up, dates=d)
    iso = PlattCalibrator(method="isotonic").fit(a, up, dates=d)
    for m in (platt.metrics(), iso.metrics()):
        assert m["ece_after_oos"] is not None
        assert m["ece_oos_split"]["oos_dates"]
    assert (platt.metrics()["ece_oos_split"]["oos_dates"]
            == iso.metrics()["ece_oos_split"]["oos_dates"])


def test_the_sidecar_round_trips_the_new_fields(tmp_path):
    a, up, d = _well_calibrated()
    cal = PlattCalibrator(method="isotonic").fit(a, up, dates=d)
    path = tmp_path / "cal.pkl"
    cal.save(path)
    back = PlattCalibrator.load(path)
    assert back.metrics()["ece_after_oos"] == cal.metrics()["ece_after_oos"]
    assert back.metrics()["ece_after_in_sample"] == cal.metrics()["ece_after_in_sample"]
    assert back.metrics()["n_oos_samples"] == cal.metrics()["n_oos_samples"]


def test_an_unfitted_calibrator_reports_nulls_not_zeros():
    cal = PlattCalibrator(method="isotonic")
    m = cal.metrics()
    assert m["ece_after_oos"] is None
    assert m["ece_after_in_sample"] is None
