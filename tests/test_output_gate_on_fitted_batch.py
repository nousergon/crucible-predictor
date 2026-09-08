"""The promotion output-distribution gate must also see the candidate's OWN
output scale (alpha-engine-config-I10185).

The measured hole: ``v3.0-meta-2026-09-04-cc3271ea``'s promotion-time
``output_distribution_gate`` PASSED — ``n_unique_p_up`` 15, ``stdev_p_up``
0.191, ``modal_fraction`` 0.16 — on ``n_synthetic = 25`` alphas spanned over a
FIXED ``alpha_range = [-0.05, 0.05]`` (both read from that version's
``manifest.json``). The model's own fitted output spanned roughly a fifth of
that range: its training-panel cross-sectional sd was 0.010350, and its first
served batch produced ``alpha_stdev`` 0.005819.

A gate fed a synthetic input range structurally cannot observe the candidate's
own output scale — it measures the CALIBRATOR's shape over an interval chosen by
the gate, which is a different and also-useful question. Both are needed; only
one existed.
"""

from __future__ import annotations

import numpy as np
import pytest

from model.output_distribution_gate import (
    validate_calibrator_distribution,
    validate_fitted_batch_distribution,
)


class _Isotonic:
    """A calibrator with real resolution across [-0.05, 0.05] and a flat region
    in the middle — the measured shape."""

    _fitted = True
    method = "isotonic"

    def calibrate_prediction(self, raw_alpha, label_clip=0.15):
        a = float(raw_alpha)
        if abs(a) < 0.012:
            p_up = 0.4940          # the plateau the collapsed batch lands on
        else:
            p_up = float(np.clip(0.5 + a * 6.0, 0.01, 0.99))
        return {
            "p_up": round(p_up, 4),
            "predicted_direction": "UP" if a > 0 else "DOWN",
        }


def _collapsed_y_hat(n=200, sd=0.010350, seed=3):
    """The champion's fitted scale."""
    return np.random.default_rng(seed).normal(0.0, sd, n)


def _healthy_y_hat(n=200, sd=0.048588, seed=3):
    """The incumbent's, on the identical panel."""
    return np.random.default_rng(seed).normal(0.0, sd, n)


def test_the_synthetic_sweep_passes_the_collapsed_candidate():
    """The control: this is exactly why the synthetic path could not be the
    only check. Nothing about it is wrong — it just answers another question."""
    assert validate_calibrator_distribution(_Isotonic()).passed


def _plateaued_y_hat(n=29):
    """The measured `spec-sota-combine` shape: 24 of 29 names on one alpha,
    every value inside the calibrator's flat region."""
    return [0.000402] * (n - 5) + [
        -0.000625, -0.001788, -0.002177, -0.002400, -0.002775,
    ]


def test_the_fitted_batch_gate_refuses_a_batch_that_lands_in_the_plateau():
    """The failure the synthetic sweep structurally cannot see: the calibrator
    is healthy across the sweep's range and the candidate's own output lands
    entirely inside one flat region of it. Measured 2026-09-08 on
    `spec-sota-combine-2026-07-{24,29,30}` — 6 distinct predicted_alpha across
    29 tickers — and the 2026-05-07 incident before it (16 of 27 tickers at
    p_up=0.458) is the same class on the live champion."""
    res = validate_fitted_batch_distribution(_Isotonic(), _plateaued_y_hat())
    assert res.passed is False
    assert res.failed_check
    assert res.metrics["input"] == "candidate_fitted_y_hat"
    assert res.metrics["modal_fraction"] >= 0.5


def test_the_synthetic_sweep_passes_that_same_batch_shape():
    """The control for the test above, and the whole argument for the second
    evaluation: the calibrator under test is the SAME object in both."""
    assert validate_calibrator_distribution(_Isotonic()).passed


def test_the_fitted_batch_gate_admits_the_healthy_candidate():
    res = validate_fitted_batch_distribution(_Isotonic(), _healthy_y_hat())
    assert res.passed is True
    assert res.metrics["input"] == "candidate_fitted_y_hat"


def test_the_fitted_alpha_stdev_is_recorded_whatever_the_verdict():
    """The number that collapsed, on the axis it collapsed on. Persisted on a
    pass too — a field present only on the failure path cannot be trended and
    its absence reads as healthy."""
    for y_hat in (_collapsed_y_hat(), _healthy_y_hat()):
        res = validate_fitted_batch_distribution(_Isotonic(), y_hat)
        assert res.metrics["fitted_alpha_stdev"] == pytest.approx(
            float(np.std(y_hat)), rel=1e-9,
        )
        assert res.metrics["n_fitted"] == len(y_hat)


def test_the_synthetic_result_says_which_input_it_used():
    """Both evaluations land in the same manifest block, so each must name its
    own input or a reader cannot tell them apart."""
    m = validate_calibrator_distribution(_Isotonic()).metrics
    assert m["input"] == "synthetic_alpha_sweep"


def test_an_empty_fitted_batch_is_unmeasurable_not_a_pass():
    res = validate_fitted_batch_distribution(_Isotonic(), [])
    assert res.passed is False
    assert res.failed_check == "unmeasurable"
    assert "no fitted" in res.reason.lower()


def test_an_unfitted_calibrator_is_reported_not_silently_passed():
    class _Unfitted:
        _fitted = False

        def calibrate_prediction(self, raw_alpha, label_clip=0.15):
            return {"p_up": 0.5, "predicted_direction": "DOWN"}

    res = validate_fitted_batch_distribution(_Unfitted(), _healthy_y_hat())
    assert res.metrics["calibrator_fitted"] is False
    assert res.passed is True
    assert "fallback" in res.reason.lower()
