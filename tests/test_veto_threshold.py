"""Tests for inference/stages/write_output.py — veto threshold logic."""

from unittest.mock import patch

import pytest

from inference.stages.write_output import get_veto_threshold, _load_predictor_params_from_s3
import inference.stages.write_output as wo


class TestGetVetoThreshold:
    def setup_method(self):
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_neutral_regime(self, _mock):
        result = get_veto_threshold("bucket", "neutral")
        assert result == pytest.approx(0.65)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.30})
    def test_bear_regime_lowers(self, _mock):
        # Post-2026-05-12 confidence semantics: |p_up - 0.5| * 2 ∈ [0, 1].
        result = get_veto_threshold("bucket", "bear")
        assert result == pytest.approx(0.10)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.30})
    def test_caution_regime(self, _mock):
        result = get_veto_threshold("bucket", "caution")
        assert result == pytest.approx(0.20)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.30})
    def test_bull_regime_raises(self, _mock):
        result = get_veto_threshold("bucket", "bull")
        assert result == pytest.approx(0.40)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.30})
    def test_bullish_alias(self, _mock):
        result = get_veto_threshold("bucket", "bullish")
        assert result == pytest.approx(0.40)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.30})
    def test_bearish_alias(self, _mock):
        result = get_veto_threshold("bucket", "bearish")
        assert result == pytest.approx(0.10)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_unknown_regime_no_adjustment(self, _mock):
        result = get_veto_threshold("bucket", "unknown")
        assert result == pytest.approx(0.65)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_empty_regime(self, _mock):
        result = get_veto_threshold("bucket", "")
        assert result == pytest.approx(0.65)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.10})
    def test_bear_clamped_at_zero(self, _mock):
        result = get_veto_threshold("bucket", "bear")
        assert result == pytest.approx(0.0)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.75})
    def test_bull_clamped_at_080(self, _mock):
        result = get_veto_threshold("bucket", "bull")
        assert result == pytest.approx(0.80)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value=None)
    def test_no_s3_uses_config_default(self, _mock):
        result = get_veto_threshold("bucket", "neutral")
        assert 0.0 <= result <= 0.80


class TestLoadPredictorParams:
    def setup_method(self):
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False

    def test_no_s3_returns_none(self):
        result = _load_predictor_params_from_s3("nonexistent-bucket")
        assert result is None

    def test_caches_after_first_call(self):
        wo._predictor_params_loaded = True
        wo._predictor_params_cache = {"veto_confidence": 0.70}
        result = _load_predictor_params_from_s3("bucket")
        assert result == {"veto_confidence": 0.70}


class TestDrawdownVetoClamp:
    """PR 3: drawdown effective-regime veto clamp (gated default-off,
    most-protective composition with the forced-bear clamp)."""

    # base=0.30, cap=0.20 ⇒ bear_floor=0.10, caution_floor=0.20.
    _OFF = {"veto_confidence": 0.30}
    _ON = {"veto_confidence": 0.30, "drawdown_regime_enabled": True}

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_flag_off_is_observe_only_no_behavior_change(self, m):
        m.return_value = self._OFF
        # dd=bear but flag off ⇒ threshold unchanged (0.30); counterfactual
        # only logged.
        assert get_veto_threshold(
            "b", drawdown_effective_regime="bear"
        ) == pytest.approx(0.30)

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_flag_on_bear_clamps_to_bear_floor(self, m):
        m.return_value = self._ON
        assert get_veto_threshold(
            "b", drawdown_effective_regime="bear"
        ) == pytest.approx(0.10)

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_flag_on_caution_clamps_to_milder_floor(self, m):
        m.return_value = self._ON
        assert get_veto_threshold(
            "b", drawdown_effective_regime="caution"
        ) == pytest.approx(0.20)

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_flag_on_benign_regimes_no_clamp(self, m):
        m.return_value = self._ON
        for r in ("bull", "neutral", None, ""):
            assert get_veto_threshold(
                "b", drawdown_effective_regime=r
            ) == pytest.approx(0.30)

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_most_protective_min_with_existing_threshold(self, m):
        # bull market_regime would raise threshold to 0.40, but dd=bear
        # clamp wins (min ⇒ 0.10).
        m.return_value = self._ON
        assert get_veto_threshold(
            "b", market_regime="bull", drawdown_effective_regime="bear"
        ) == pytest.approx(0.10)

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_composition_with_forced_bear_is_most_protective(self, m):
        # forced_bear (enabled) → bear_floor 0.10; dd=caution → 0.20.
        # Most-protective = min = 0.10.
        m.return_value = {
            "veto_confidence": 0.30,
            "drawdown_regime_enabled": True,
            "regime_forced_bear_enabled": True,
        }
        assert get_veto_threshold(
            "b", forced_bear=True, drawdown_effective_regime="caution"
        ) == pytest.approx(0.10)

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_both_flags_off_default_path_unchanged(self, m):
        # Regression: the accumulation refactor must not change the
        # all-off path (forced_bear False, no dd) — plain base.
        m.return_value = {"veto_confidence": 0.30}
        assert get_veto_threshold("b", market_regime="neutral") == pytest.approx(
            0.30
        )
