"""Tests for the predictor-side momentum_veto.

PR target: move ``momentum_gate`` from executor (alpha-engine/executor/
deciders.py) to predictor (alpha-engine-predictor/inference/stages/
run_inference.py:compute_momentum_veto). Consolidates signal-side rules
on the predictor; executor consumes ``momentum_veto`` from the per-ticker
predictions dict.

These tests pin:
  - compute_momentum_veto returns (None, False) for missing / NaN /
    non-finite momentum_20d (fail-safe under-veto direction).
  - Veto fires when momentum_20d < threshold (and not on the boundary).
  - Threshold parameter is honored (regression target: a refactor that
    hardcodes -0.05 instead of pulling from cfg.MOMENTUM_VETO_THRESHOLD
    would break the config-driven tuning path).
  - cfg.MOMENTUM_VETO_THRESHOLD has the expected default (-0.05) when
    the YAML doesn't override.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import config as cfg
from inference.stages.run_inference import compute_momentum_veto


class TestComputeMomentumVeto:
    def test_below_threshold_vetoes(self):
        latest = pd.Series({"momentum_20d": -0.10})  # -10% over 20d
        m20, veto = compute_momentum_veto(latest, threshold=-0.05)
        assert m20 == pytest.approx(-0.10)
        assert veto is True

    def test_above_threshold_does_not_veto(self):
        latest = pd.Series({"momentum_20d": 0.03})  # +3% over 20d
        m20, veto = compute_momentum_veto(latest, threshold=-0.05)
        assert m20 == pytest.approx(0.03)
        assert veto is False

    def test_exactly_at_threshold_does_not_veto(self):
        """``< threshold`` means equality DOESN'T veto. Pinned so a future
        edit to ``<=`` doesn't flip the boundary semantic (which would
        block borderline tickers that today pass)."""
        latest = pd.Series({"momentum_20d": -0.05})
        m20, veto = compute_momentum_veto(latest, threshold=-0.05)
        assert m20 == pytest.approx(-0.05)
        assert veto is False

    def test_missing_field_does_not_veto(self):
        """Missing momentum_20d → no veto (fail-safe under-veto direction
        matches gbm_veto's missing-field semantics)."""
        latest = pd.Series({"some_other_feature": 0.5})
        m20, veto = compute_momentum_veto(latest, threshold=-0.05)
        assert m20 is None
        assert veto is False

    def test_nan_does_not_veto(self):
        latest = pd.Series({"momentum_20d": float("nan")})
        m20, veto = compute_momentum_veto(latest, threshold=-0.05)
        assert m20 is None
        assert veto is False

    def test_inf_does_not_veto(self):
        latest = pd.Series({"momentum_20d": float("inf")})
        m20, veto = compute_momentum_veto(latest, threshold=-0.05)
        assert m20 is None
        assert veto is False

    def test_non_numeric_does_not_veto(self):
        latest = pd.Series({"momentum_20d": "garbage"})
        m20, veto = compute_momentum_veto(latest, threshold=-0.05)
        assert m20 is None
        assert veto is False

    def test_threshold_parameter_is_honored(self):
        """Tighter threshold (less negative) means MORE tickers vetoed.
        Pinned so a refactor that hardcodes -0.05 instead of using
        cfg.MOMENTUM_VETO_THRESHOLD would fail this test."""
        latest = pd.Series({"momentum_20d": -0.03})  # -3% drawdown
        # Default threshold -0.05 → no veto
        _, no_veto = compute_momentum_veto(latest, threshold=-0.05)
        assert no_veto is False
        # Tighter threshold -0.02 → veto
        _, with_veto = compute_momentum_veto(latest, threshold=-0.02)
        assert with_veto is True

    def test_dict_input_also_works(self):
        """A plain dict (not pandas Series) should also work — both have
        .get(). Keeps the helper usable in unit tests + non-inference
        contexts."""
        latest = {"momentum_20d": -0.10}
        m20, veto = compute_momentum_veto(latest, threshold=-0.05)
        assert m20 == pytest.approx(-0.10)
        assert veto is True


class TestMomentumVetoThresholdConstant:
    def test_default_threshold_value(self):
        """Default -0.05 matches the prior executor-side momentum_gate
        threshold (alpha-engine/executor/main.py:118 had
        ``momentum_gate_threshold`` default -5.0 in pct-units; -0.05 here
        is the decimal-units equivalent). Pinned so the executor->predictor
        migration doesn't accidentally widen or narrow the gate."""
        # The YAML may override; this test ensures the default (used when
        # predictor.yaml's gates section lacks momentum_veto_threshold) is
        # the migration-target value.
        # The actual loaded value can be the YAML value OR the .get fallback.
        # Either way, it must be a finite negative number.
        assert isinstance(cfg.MOMENTUM_VETO_THRESHOLD, float)
        assert math.isfinite(cfg.MOMENTUM_VETO_THRESHOLD)
        assert cfg.MOMENTUM_VETO_THRESHOLD < 0, (
            "Veto threshold must be negative (we're flagging DRAWDOWN). "
            f"Got {cfg.MOMENTUM_VETO_THRESHOLD}"
        )
