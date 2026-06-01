"""Tests for the residual-momentum L1 scorer (W2, L4469).

Covers the interface contract meta_trainer + (future) inference depend on:
vectorized predict_array, NaN→neutral degradation, determinism, and
predict_array≡predict_dict parity (the two-path drift the momentum module
warns about). The scorer is deterministic/stateless — no fit/save/load.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import residual_momentum_scorer as rms
from model.residual_momentum_scorer import RESIDUAL_MOMENTUM_FEATURES


def _synthetic_X(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, len(RESIDUAL_MOMENTUM_FEATURES))).astype(np.float64)


class TestContract:
    def test_feature_list_is_canonical(self):
        # These are FEATURE-STORE column names (computed in alpha-engine-data).
        assert RESIDUAL_MOMENTUM_FEATURES == [
            "residual_momentum_ratio", "mom_12_1_pct", "sector_mom_pct",
        ]

    def test_predict_array_shape(self):
        X = _synthetic_X(150)
        out = rms.predict_array(X, RESIDUAL_MOMENTUM_FEATURES)
        assert out.shape == (150,)
        assert np.all(np.isfinite(out))

    def test_missing_feature_uses_neutral_default(self):
        # A feature_names list missing a column → that sub-signal is neutral 0,
        # not a crash. Score must still be finite and equal the blend of present
        # columns.
        X = _synthetic_X(50)
        names = ["residual_momentum_ratio", "mom_12_1_pct"]  # drop sector_mom_pct
        out = rms.predict_array(X[:, : len(names)], names)
        assert out.shape == (50,)
        assert np.all(np.isfinite(out))


class TestNaNHandling:
    def test_all_nan_row_degrades_to_zero(self):
        X = np.full((3, len(RESIDUAL_MOMENTUM_FEATURES)), np.nan, dtype=np.float64)
        out = rms.predict_array(X, RESIDUAL_MOMENTUM_FEATURES)
        # Every sub-signal defaults to 0 → blend is exactly 0.
        np.testing.assert_allclose(out, np.zeros(3), atol=1e-12)

    def test_partial_nan_neutralized_not_propagated(self):
        X = _synthetic_X(10)
        X[0, 0] = np.nan  # NaN in resid_mom_vol_scaled of row 0
        out = rms.predict_array(X, RESIDUAL_MOMENTUM_FEATURES)
        assert np.all(np.isfinite(out))


class TestDeterminism:
    def test_called_twice_identical(self):
        X = _synthetic_X(120, seed=3)
        a = rms.predict_array(X, RESIDUAL_MOMENTUM_FEATURES)
        b = rms.predict_array(X, RESIDUAL_MOMENTUM_FEATURES)
        np.testing.assert_array_equal(a, b)


class TestDictParity:
    def test_predict_dict_matches_predict_array_row(self):
        X = _synthetic_X(40, seed=7)
        arr = rms.predict_array(X, RESIDUAL_MOMENTUM_FEATURES)
        for i in (0, 13, 39):
            row = {name: float(X[i, j]) for j, name in enumerate(RESIDUAL_MOMENTUM_FEATURES)}
            assert abs(rms.predict_dict(row) - arr[i]) < 1e-9

    def test_predict_dict_none_and_missing_degrade(self):
        # Missing keys + None both → neutral; with all neutral the score is 0.
        assert abs(rms.predict_dict({})) < 1e-12
        assert abs(rms.predict_dict({"residual_momentum_ratio": None})) < 1e-12

    def test_known_weighting_sign(self):
        # All three sub-signals carry POSITIVE weights (momentum); each lifts
        # the score, and the residual headline dominates.
        base = rms.predict_dict({})
        assert rms.predict_dict({"residual_momentum_ratio": 1.0}) > base
        assert rms.predict_dict({"mom_12_1_pct": 1.0}) > base
        assert rms.predict_dict({"sector_mom_pct": 1.0}) > base
        # Headline residual term dominates the raw decomposition terms.
        assert (rms.predict_dict({"residual_momentum_ratio": 1.0})
                > rms.predict_dict({"mom_12_1_pct": 1.0}))
