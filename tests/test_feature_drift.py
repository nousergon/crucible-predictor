"""Tests for monitoring/feature_drift.py (config#859)."""

from __future__ import annotations

import numpy as np
import pytest

from monitoring.feature_drift import (
    CROSS_SECTIONAL_DRIFT_FEATURES,
    build_training_reference,
    compute_feature_drift_ks,
)

# A feature-name order mixing cross-sectional + market-wide columns.
NAMES = [
    "research_calibrator_prob", "momentum_score", "expected_move",
    "research_composite_score", "research_conviction", "sector_macro_modifier",
    "macro_spy_20d_return", "regime_intensity_z",  # market-wide → excluded
]


def _matrix(n, *, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    cols = []
    for j in range(len(NAMES)):
        cols.append(rng.normal(shift, 1.0, n))
    return np.column_stack(cols)


class TestReference:
    def test_only_cross_sectional_features_kept(self):
        ref = build_training_reference(_matrix(200), NAMES, trained_date="2026-06-13")
        assert set(ref["features"]) == set(CROSS_SECTIONAL_DRIFT_FEATURES)
        assert "macro_spy_20d_return" not in ref["samples"]
        assert "regime_intensity_z" not in ref["samples"]
        assert ref["trained_date"] == "2026-06-13"

    def test_subsample_caps_size(self):
        ref = build_training_reference(_matrix(5000), NAMES)
        # capped at 2000 per feature.
        assert all(len(s) <= 2000 for s in ref["samples"].values())

    def test_nan_dropped(self):
        m = _matrix(100)
        m[0, 0] = np.nan
        ref = build_training_reference(m, NAMES)
        assert all(np.isfinite(ref["samples"]["research_calibrator_prob"]))


class TestComputeKS:
    def test_no_drift_low_ks(self):
        ref = build_training_reference(_matrix(1000, seed=1), NAMES)
        # Same distribution, different draw → KS should be small.
        out = compute_feature_drift_ks(_matrix(500, seed=2), NAMES, ref)
        assert out is not None
        assert out["max_ks"] < 0.15
        assert out["n_features"] == len(CROSS_SECTIONAL_DRIFT_FEATURES)
        assert out["n_samples"] >= 30

    def test_strong_drift_high_ks(self):
        ref = build_training_reference(_matrix(1000, seed=1), NAMES)
        # Shift the inference distribution hard → KS near 1.
        out = compute_feature_drift_ks(_matrix(500, shift=5.0, seed=3), NAMES, ref)
        assert out["max_ks"] > 0.9
        # per_feature sorted worst-first.
        vals = list(out["per_feature"].values())
        assert vals == sorted(vals, reverse=True)

    def test_absent_reference_returns_none(self):
        assert compute_feature_drift_ks(_matrix(100), NAMES, {}) is None
        assert compute_feature_drift_ks(_matrix(100), NAMES, {"samples": {}}) is None

    def test_too_few_samples_returns_none(self):
        ref = build_training_reference(_matrix(1000), NAMES)
        # 10 inference rows < _MIN_KS_SAMPLES (30) → nothing graded.
        assert compute_feature_drift_ks(_matrix(10), NAMES, ref) is None

    def test_reference_trained_date_propagates(self):
        ref = build_training_reference(_matrix(1000), NAMES, trained_date="2026-06-13")
        out = compute_feature_drift_ks(_matrix(500), NAMES, ref)
        assert out["reference_trained_date"] == "2026-06-13"


class TestRoundTrip:
    def test_save_load(self):
        from unittest.mock import MagicMock

        from monitoring.feature_drift import (
            FEATURE_DRIFT_REFERENCE_KEY, load_training_reference, save_training_reference,
        )
        ref = build_training_reference(_matrix(100), NAMES, trained_date="2026-06-13")
        stub = MagicMock()
        key = save_training_reference(ref, bucket="b", s3_client=stub)
        assert key == FEATURE_DRIFT_REFERENCE_KEY
        body = stub.put_object.call_args.kwargs["Body"]
        import json
        stub.get_object.return_value = {"Body": type("B", (), {"read": lambda self: body})()}
        loaded = load_training_reference(bucket="b", s3_client=stub)
        assert loaded["features"] == ref["features"]
        assert json.loads(body.decode())["trained_date"] == "2026-06-13"
