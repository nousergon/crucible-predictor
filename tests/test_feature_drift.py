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


# ── alpha-engine-config-I10063: producer provenance ──────────────────────────
#
# research_calibrator_prob has two producers (ResearchGBMScorer, bucket-lookup
# ResearchCalibrator). A training run whose GBM is not fitted builds its
# reference from bucket-lookup values while serving keeps the GBM — and the KS
# between them measured 0.986 on 2026-10-05 with no drift involved. These
# tests pin that both sides record their producer and that the block says
# when they differ.


def _bucket_like_matrix(n, seed=0):
    """research_calibrator_prob as a bucket lookup emits it: a few discrete
    hit rates. Other columns are ordinary noise."""
    m = _matrix(n, seed=seed)
    rng = np.random.default_rng(seed + 100)
    # Weighted like the 2026-10-02 reference: median 0.5714, p5 0.5.
    m[:, 0] = rng.choice(
        [0.4172, 0.5, 0.5714, 0.625], size=n, p=[0.04, 0.08, 0.63, 0.25],
    )
    return m


def _gbm_like_matrix(n, seed=0):
    """research_calibrator_prob as a shallow GBM emits it at serving."""
    m = _matrix(n, seed=seed)
    rng = np.random.default_rng(seed + 200)
    m[:, 0] = rng.choice([0.4639, 0.496], size=n)
    return m


class TestProducerProvenance:
    def test_reference_records_producers(self):
        from monitoring.feature_drift import research_calibrator_producer

        ref = build_training_reference(
            _matrix(200), NAMES, producers=research_calibrator_producer(False),
        )
        assert ref["producers"] == {"research_calibrator_prob": "bucket_lookup"}

    def test_reference_without_producers_has_no_key(self):
        # Schema stays additive: a caller that passes nothing writes what it
        # wrote before.
        assert "producers" not in build_training_reference(_matrix(200), NAMES)

    def test_producer_helper(self):
        from monitoring.feature_drift import research_calibrator_producer

        assert research_calibrator_producer(True) == {
            "research_calibrator_prob": "research_gbm"}
        assert research_calibrator_producer(False) == {
            "research_calibrator_prob": "bucket_lookup"}

    def test_mismatch_is_flagged_and_ks_unchanged(self):
        """The 2026-10-05 shape: bucket-lookup reference, GBM at serving."""
        from monitoring.feature_drift import research_calibrator_producer

        m_ref = _bucket_like_matrix(1000, seed=1)
        m_inf = _gbm_like_matrix(35, seed=2)
        ref = build_training_reference(
            m_ref, NAMES, trained_date="2026-10-02",
            producers=research_calibrator_producer(False),
        )
        plain = compute_feature_drift_ks(m_inf, NAMES, dict(ref, producers={}))
        out = compute_feature_drift_ks(
            m_inf, NAMES, ref, serving_producers=research_calibrator_producer(True),
        )
        assert out["producer_parity"] == "mismatch"
        assert out["producer_mismatch"] == ["research_calibrator_prob"]
        assert out["producers"] == {
            "reference": {"research_calibrator_prob": "bucket_lookup"},
            "serving": {"research_calibrator_prob": "research_gbm"},
        }
        # The KS itself is reported exactly as before — only labelled.
        assert out["per_feature"] == plain["per_feature"]
        assert out["max_ks"] == plain["max_ks"]
        assert out["per_feature"]["research_calibrator_prob"] > 0.9

    def test_match(self):
        from monitoring.feature_drift import research_calibrator_producer

        ref = build_training_reference(
            _matrix(1000, seed=1), NAMES,
            producers=research_calibrator_producer(True),
        )
        out = compute_feature_drift_ks(
            _matrix(500, seed=2), NAMES, ref,
            serving_producers=research_calibrator_producer(True),
        )
        assert out["producer_parity"] == "match"
        assert out["producer_mismatch"] == []

    @pytest.mark.parametrize("ref_producers,serving", [
        (None, True),     # a reference written before I10063
        (False, None),    # a caller that did not say what served
        (None, None),
    ])
    def test_absence_is_undeclared_never_match(self, ref_producers, serving):
        from monitoring.feature_drift import research_calibrator_producer

        ref = build_training_reference(
            _matrix(1000, seed=1), NAMES,
            producers=(None if ref_producers is None
                       else research_calibrator_producer(ref_producers)),
        )
        out = compute_feature_drift_ks(
            _matrix(500, seed=2), NAMES, ref,
            serving_producers=(None if serving is None
                               else research_calibrator_producer(serving)),
        )
        assert out["producer_parity"] == "undeclared"
        assert out["producer_mismatch"] == []


class TestProducerCallSites:
    """The two call sites must describe the producer the model actually used.

    Inference: ``_set_feature_drift_ks`` derives the serving producer from
    ``ctx.meta_models['research_gbm']``; it must agree with the predicate the
    per-ticker loop uses to let the GBM override the bucket lookup.
    Training: the reference must be stamped from whether Step 6c recomputed
    research_calibrator_prob through the GBM.
    """

    def _run(self, research_gbm, monkeypatch):
        import monitoring.feature_drift as fd
        from inference.stages import run_inference as ri
        from model.meta_model import META_FEATURES

        ref = build_training_reference(
            _bucket_like_matrix(1000, seed=1), NAMES,
            producers=fd.research_calibrator_producer(False),
        )
        monkeypatch.setattr(fd, "load_training_reference", lambda **_: ref)
        rows = [
            {f: float(v) for f, v in zip(META_FEATURES, r)}
            for r in np.random.default_rng(3).normal(0.5, 0.1, (40, len(META_FEATURES)))
        ]

        class Ctx:
            bucket = "b"
            meta_models = {} if research_gbm is None else {"research_gbm": research_gbm}

        ctx = Ctx()
        ri._set_feature_drift_ks(ctx, rows)
        return ctx.feature_drift_ks

    def test_inference_fitted_gbm_serves_gbm(self, monkeypatch):
        gbm = type("G", (), {"fitted": True})()
        out = self._run(gbm, monkeypatch)
        assert out["producers"]["serving"] == {"research_calibrator_prob": "research_gbm"}
        assert out["producer_parity"] == "mismatch"

    @pytest.mark.parametrize("gbm", [None, type("G", (), {"fitted": False})()])
    def test_inference_no_gbm_serves_bucket_lookup(self, gbm, monkeypatch):
        out = self._run(gbm, monkeypatch)
        assert out["producers"]["serving"] == {"research_calibrator_prob": "bucket_lookup"}
        assert out["producer_parity"] == "match"

    def test_inference_override_predicate_unchanged(self):
        """If the loop's GBM-override predicate changes, the serving producer
        derivation in ``_set_feature_drift_ks`` must change with it."""
        import inspect

        from inference.stages import run_inference as ri

        loop = inspect.getsource(ri._run_meta_inference)
        assert 'if research_gbm is not None and getattr(research_gbm, "fitted", False):' in loop
        assert '"research_calibrator_prob": research_cal_prob,' in loop
        drift = inspect.getsource(ri._set_feature_drift_ks)
        assert 'getattr(_research_gbm, "fitted", False)' in drift

    def test_training_stamps_producer_after_gbm_recompute(self):
        import inspect

        from training import meta_trainer

        src = inspect.getsource(meta_trainer)
        recompute = src.index('row["research_calibrator_prob"] = float(')
        flip = src.index("research_calibrator_prob_from_gbm = True")
        build = src.index("_drift_ref = build_training_reference(")
        # Flipped only after the recompute loop, and read when the reference
        # is built from meta_X.
        assert recompute < flip < build
        call = src[build:src.index(")", src.index("trained_date=", build)) + 200]
        assert "producers=research_calibrator_producer(research_calibrator_prob_from_gbm)" in call
