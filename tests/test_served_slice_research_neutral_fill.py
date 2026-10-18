"""The served slice is measured while a declared research producer is absent.

alpha-engine-config-I11477. Every model-zoo select since 2026-09-25 recorded
``served_slice_dispersion`` as uncomputable for EVERY version:

    OOS panel is missing 3 of the model's features ['guidance_direction',
    'risk_factor_count_delta_raw', 'management_tone_zscore'] — refusing to
    zero-fill (alpha-engine-config-I5949)

Those three are ``RESEARCH_META_FEATURES`` with no producer
(alpha-engine-config-I11482). Serving feeds them ``RESEARCH_META_NEUTRAL`` and
training fit on the same value, so refusing them blocked a measurement of
exactly what would serve. The I5949 refusal still holds for every CORE feature.

Pre-fix, ``test_missing_research_features_are_scored_at_the_serving_neutral``
and ``test_served_slice_is_measured_and_names_the_filled_columns`` raised /
reported uncomputable.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.meta_model import RESEARCH_META_FEATURES, RESEARCH_META_NEUTRAL
from training import served_slice_dispersion as ssd

ABSENT = ["guidance_direction", "risk_factor_count_delta_raw",
          "management_tone_zscore"]
CORE = ["momentum_score", "expected_move"]


class _Ridge:
    """A linear head with a non-zero weight on every feature, so a fill value
    other than the neutral would change the score."""

    def __init__(self, names):
        self._feature_names = list(names)
        self.coef = np.linspace(0.5, 1.5, len(names))

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef + 0.01


class _Calibrator:
    def predict(self, alpha):
        return 1.0 / (1.0 + np.exp(-np.asarray(alpha, dtype=float) * 10.0))


def _panel(n_dates=25, n_names=40, seed=0, drop=()):
    rng = np.random.default_rng(seed)
    dates = np.repeat(
        pd.date_range("2026-01-01", periods=n_dates).strftime("%Y-%m-%d"), n_names)
    cols = {"date": dates}
    for f in CORE + ABSENT:
        cols[f] = rng.normal(size=len(dates))
    return pd.DataFrame(cols).drop(columns=list(drop))


def test_the_absent_features_are_the_declared_soft_set():
    assert set(ABSENT) <= set(RESEARCH_META_FEATURES)
    assert not set(CORE) & set(RESEARCH_META_FEATURES)


def test_missing_research_features_are_scored_at_the_serving_neutral():
    model = _Ridge(CORE + ABSENT)
    lacking = _panel(drop=ABSENT)
    carrying = lacking.assign(**{f: RESEARCH_META_NEUTRAL for f in ABSENT})

    alpha, p_up = ssd.score_panel(model, _Calibrator(), lacking)
    ref_alpha, ref_p_up = ssd.score_panel(model, _Calibrator(), carrying)

    np.testing.assert_allclose(alpha, ref_alpha)
    np.testing.assert_allclose(p_up, ref_p_up)
    assert ssd.research_neutral_fills(model, lacking) == ABSENT
    assert ssd.research_neutral_fills(model, carrying) == []


def test_the_callers_panel_is_not_mutated():
    lacking = _panel(drop=ABSENT)
    before = list(lacking.columns)
    ssd.score_panel(_Ridge(CORE + ABSENT), _Calibrator(), lacking)
    assert list(lacking.columns) == before


def test_a_missing_core_feature_is_still_refused_alongside_research_ones():
    """I5949 is not widened: only the RESEARCH_META_FEATURES list is soft."""
    with pytest.raises(RuntimeError, match="refusing to zero-fill") as exc:
        ssd.score_panel(_Ridge(CORE + ABSENT), _Calibrator(),
                        _panel(drop=ABSENT + ["momentum_score"]))
    assert "momentum_score" in str(exc.value)
    assert not any(f in str(exc.value) for f in ABSENT)


def test_served_slice_is_measured_and_names_the_filled_columns(monkeypatch):
    model = _Ridge(CORE + ABSENT)
    monkeypatch.setattr(
        ssd, "load_scoring_head", lambda s3, bucket, vid: (model, _Calibrator()))

    out = ssd.served_slice_metrics(
        None, "bkt", ["v3.0-meta-a", "v3.0-meta-b"],
        panel=_panel(drop=ABSENT), panel_key="oos_rows/v3.0-meta/x.parquet",
        reference_version_id="v3.0-meta-a",
    )

    assert out["status"] == "measured", out
    assert out["errors"] == {}
    for vid in ("v3.0-meta-a", "v3.0-meta-b"):
        assert out["metrics"][vid]["research_neutral_filled"] == ABSENT


def test_a_complete_panel_records_no_fill(monkeypatch):
    model = _Ridge(CORE + ABSENT)
    monkeypatch.setattr(
        ssd, "load_scoring_head", lambda s3, bucket, vid: (model, _Calibrator()))
    out = ssd.served_slice_metrics(
        None, "bkt", ["v3.0-meta-a"], panel=_panel(), panel_key="k")
    assert out["status"] == "measured", out
    assert out["metrics"]["v3.0-meta-a"]["research_neutral_filled"] == []
