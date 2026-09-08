"""The incumbent's cross-sectional dispersion, recomputed on the CANDIDATE's
own panel (alpha-engine-config-I10185).

The comparison that decides a promotion must hold the panel constant, or it
measures the data rather than the model. The incumbent manifest's own
``xsec_sd`` was measured weeks ago on a different vintage's design matrix and
is the wrong number; the right one is the incumbent's FROZEN coefficients
scored on the rows the candidate was just fitted on.

This mirrors ``training/incumbent_rescore.py`` exactly — same bundle, same ETag
verification, same feature-contract rules, same never-raises posture — because
it answers the same shape of question about the same model, and a second set of
rules for loading a frozen incumbent is a second thing to keep true.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.xsec_magnitude import incumbent_xsec_sd_on_candidate_panel


def _panel(n_dates=40, n_per_date=30, seed=11):
    rng = np.random.default_rng(seed)
    rows, dates = [], []
    for d in range(n_dates):
        macro = float(rng.normal())
        for _ in range(n_per_date):
            rows.append([macro, float(rng.normal())])
            dates.append(f"2026-01-{d + 1:02d}")
    return np.asarray(rows, dtype=float), dates, ["macro_const", "xsec_feature"]


class _Model:
    def __init__(self, coefs, names, intercept=0.0):
        self._coefficients = dict(coefs, intercept=intercept)
        self._feature_names = list(names)


@pytest.fixture
def patched(monkeypatch):
    """Wire the loader/verifier seams the module borrows from incumbent_rescore."""
    state = {"model": None, "verify_raises": None, "served": "inc-v1"}

    import training.xsec_magnitude as mod

    monkeypatch.setattr(mod, "_resolve_served_version", lambda s3, b: state["served"])

    def _verify(s3, bucket, vid):
        if state["verify_raises"]:
            raise state["verify_raises"]

    monkeypatch.setattr(mod, "_verify_bundle", _verify)
    monkeypatch.setattr(mod, "_load_meta_model", lambda s3, b, v: state["model"])
    return state


def test_the_incumbent_is_scored_on_the_candidates_rows(patched):
    X, dates, names = _panel()
    patched["model"] = _Model({"macro_const": 0.2, "xsec_feature": 1.75}, names)
    out = incumbent_xsec_sd_on_candidate_panel(
        object(), "b", meta_X=X, dates=dates, train_meta_features=names,
    )
    assert out["status"] == "ok"
    assert out["incumbent_version_id"] == "inc-v1"
    assert out["basis"] == "incumbent_coefficients_on_candidate_panel"
    # 1.75 * sd(xsec_feature) ~= 1.75 on a unit-normal column.
    assert out["xsec_sd"] == pytest.approx(1.75, rel=0.05)
    assert out["xsec_variance_share"] > 0.10


def test_a_column_reorder_is_not_a_contract_mismatch(patched):
    X, dates, names = _panel()
    patched["model"] = _Model(
        {"xsec_feature": 1.75, "macro_const": 0.2}, list(reversed(names)),
    )
    out = incumbent_xsec_sd_on_candidate_panel(
        object(), "b", meta_X=X, dates=dates, train_meta_features=names,
    )
    assert out["status"] == "ok"
    assert out["xsec_sd"] == pytest.approx(1.75, rel=0.05)


def test_a_genuine_feature_mismatch_reports_rather_than_zero_filling(patched):
    """Zero-filling the difference is alpha-engine-config-I5949 — a model
    silently scored on zeros. It would UNDERSTATE the incumbent's dispersion,
    which makes the candidate's ratio look better, which turns a defect into a
    pass. Refuse to produce a number."""
    X, dates, names = _panel()
    patched["model"] = _Model(
        {"macro_const": 0.2, "something_else": 1.0}, ["macro_const", "something_else"],
    )
    out = incumbent_xsec_sd_on_candidate_panel(
        object(), "b", meta_X=X, dates=dates, train_meta_features=names,
    )
    assert out["status"] == "feature_contract_mismatch"
    assert out["xsec_sd"] is None
    assert "something_else" in out["reason"]


def test_an_unverifiable_bundle_produces_no_number(patched):
    X, dates, names = _panel()
    patched["model"] = _Model({"macro_const": 0.2, "xsec_feature": 1.75}, names)
    patched["verify_raises"] = RuntimeError("etag drift")
    out = incumbent_xsec_sd_on_candidate_panel(
        object(), "b", meta_X=X, dates=dates, train_meta_features=names,
    )
    assert out["status"] == "bundle_unverifiable"
    assert out["xsec_sd"] is None


def test_no_incumbent_is_a_named_status_not_an_error(patched):
    X, dates, names = _panel()
    patched["served"] = None
    out = incumbent_xsec_sd_on_candidate_panel(
        object(), "b", meta_X=X, dates=dates, train_meta_features=names,
    )
    assert out["status"] == "no_incumbent"
    assert out["xsec_sd"] is None


def test_it_never_raises(patched):
    """A failure here must not take a weekly rotation down: the veto reads the
    absent number as uncomputable and refuses a below-floor candidate anyway."""
    X, dates, names = _panel()

    def _boom(*a, **k):
        raise ValueError("boom")

    import training.xsec_magnitude as mod
    mod._load_meta_model = _boom
    out = incumbent_xsec_sd_on_candidate_panel(
        object(), "b", meta_X=X, dates=dates, train_meta_features=names,
    )
    assert out["status"] == "error"
    assert out["xsec_sd"] is None
    assert "boom" in out["reason"]


def test_a_model_without_embedded_feature_names_is_refused(patched):
    X, dates, names = _panel()
    patched["model"] = _Model({"macro_const": 0.2, "xsec_feature": 1.75}, [])
    out = incumbent_xsec_sd_on_candidate_panel(
        object(), "b", meta_X=X, dates=dates, train_meta_features=names,
    )
    assert out["status"] == "feature_contract_mismatch"
    assert out["xsec_sd"] is None


def test_a_model_without_coefficients_is_refused(patched):
    """xsec_sd is computed from the COEFFICIENT vector, by the same function
    that measures the candidate, so both sides are the identical arithmetic on
    the identical rows. A model that exposes no coefficients cannot be put
    through it, and a predict()-based substitute would not be the same
    quantity."""
    class _NoCoefs:
        _feature_names = ["macro_const", "xsec_feature"]

    X, dates, names = _panel()
    patched["model"] = _NoCoefs()
    out = incumbent_xsec_sd_on_candidate_panel(
        object(), "b", meta_X=X, dates=dates, train_meta_features=names,
    )
    assert out["status"] == "coefficients_unavailable"
    assert out["xsec_sd"] is None
