"""
tests/test_daily_predict.py — Unit tests for inference/daily_predict.py.

Tests standalone utility functions: email builder, veto threshold logic.
Does NOT test the full prediction pipeline (which requires S3 + model weights).
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Tests: _build_predictor_email
# ---------------------------------------------------------------------------

class TestBuildPredictorEmail:
    """Tests for _build_predictor_email() — email formatting helper."""

    def _make_predictions(self, n: int = 5) -> list[dict]:
        """Create synthetic prediction dicts."""
        # Binary direction post-2026-05-12 (ROADMAP L1615).
        directions = ["UP", "DOWN"]
        preds = []
        for i in range(n):
            preds.append({
                "ticker": f"TICK{i}",
                "predicted_alpha": 0.05 - i * 0.025,
                "predicted_direction": directions[i % 2],
                "prediction_confidence": 0.7 + i * 0.05,
                "combined_rank": float(i + 1),
                "watchlist_source": "tracked" if i % 2 == 0 else "buy_candidate",
            })
        return preds

    def _make_metrics(self) -> dict:
        return {
            "model_version": "gbm_v1.0",
            "ic_30d": 0.0812,
            "inference_mode": "mse",
        }

    def test_returns_three_strings(self):
        """_build_predictor_email should return (subject, html, plain)."""
        from inference.daily_predict import _build_predictor_email

        preds = self._make_predictions()
        metrics = self._make_metrics()

        result = _build_predictor_email(preds, metrics, "2024-06-01")
        assert isinstance(result, tuple)
        assert len(result) == 3

        subject, html_body, plain_body = result
        assert isinstance(subject, str)
        assert isinstance(html_body, str)
        assert isinstance(plain_body, str)

    def test_subject_contains_date(self):
        """Subject line should include the prediction date."""
        from inference.daily_predict import _build_predictor_email

        preds = self._make_predictions()
        metrics = self._make_metrics()

        subject, _, _ = _build_predictor_email(preds, metrics, "2024-06-01")
        assert "2024-06-01" in subject

    def test_subject_contains_direction_counts(self):
        """Subject line should include UP and DOWN counts."""
        from inference.daily_predict import _build_predictor_email

        preds = self._make_predictions(6)  # 2 UP, 2 FLAT, 2 DOWN
        metrics = self._make_metrics()

        subject, _, _ = _build_predictor_email(preds, metrics, "2024-06-01")
        assert "UP" in subject
        assert "DOWN" in subject

    def test_html_omits_per_ticker_rows_and_links_to_console(self):
        """config#856: the per-ticker table moved to the console Predictor
        page — the slim email must NOT inline ticker rows, and must carry the
        console deep-link that renders them."""
        from inference.daily_predict import _build_predictor_email
        from inference.stages.write_output import predictor_report_url

        preds = self._make_predictions(3)
        metrics = self._make_metrics()

        _, html, _ = _build_predictor_email(preds, metrics, "2024-06-01")
        for p in preds:
            assert p["ticker"] not in html
        assert predictor_report_url("2024-06-01") in html

    def test_plain_omits_per_ticker_rows_and_links_to_console(self):
        """Plain-text counterpart of the slim contract (config#856)."""
        from inference.daily_predict import _build_predictor_email
        from inference.stages.write_output import predictor_report_url

        preds = self._make_predictions(3)
        metrics = self._make_metrics()

        _, _, plain = _build_predictor_email(preds, metrics, "2024-06-01")
        for p in preds:
            assert p["ticker"] not in plain
        assert predictor_report_url("2024-06-01") in plain

    def test_veto_section_appears_for_high_confidence_down(self):
        """Vetoes render from the authoritative gbm_veto boolean (config#1815).

        The fixture stamps gbm_veto exactly as run() does before the email
        builds (VETOME: negative α + bottom-half rank + confidence 0.95 ≥
        the 0.65 threshold → True). The email must not re-derive the rule.
        """
        from inference.daily_predict import _build_predictor_email

        preds = [
            {
                "ticker": "GOOD",
                "predicted_alpha": 0.05,
                "predicted_direction": "UP",
                "prediction_confidence": 0.70,
                "combined_rank": 1.0,
                "watchlist_source": "tracked",
                "gbm_veto": False,
            },
            {
                "ticker": "VETOME",
                "predicted_alpha": -0.05,
                "predicted_direction": "DOWN",
                "prediction_confidence": 0.95,
                "combined_rank": 2.0,  # > n_preds/2 (2 > 1)
                "watchlist_source": "tracked",
                "gbm_veto": True,
            },
        ]
        metrics = self._make_metrics()

        subject, html, plain = _build_predictor_email(
            preds, metrics, "2024-06-01", veto_threshold=0.65
        )
        # The email surfaces only the aggregate veto COUNT (single-sourced from
        # gbm_veto); the per-ticker veto badge moved to the console page
        # (config#856). Exactly one name (VETOME) has gbm_veto=True.
        assert "1 veto" in subject
        assert "Vetoes" in html
        assert "Vetoes: 1" in plain
        assert "VETOME" not in html  # per-ticker rows are on the console now

    def test_empty_predictions(self):
        """Should handle empty prediction list gracefully."""
        from inference.daily_predict import _build_predictor_email

        subject, html, plain = _build_predictor_email([], self._make_metrics(), "2024-06-01")
        assert isinstance(subject, str)
        assert "0 UP" in subject
        assert "No predictions" in html or "none" in html.lower() or len(html) > 0

    def test_with_signals_data(self):
        """The market regime is still surfaced (subject + summary) when
        signals_data is provided; the full research brief / per-ticker
        universe moved to the console page (config#856)."""
        from inference.daily_predict import _build_predictor_email

        signals = {
            "market_regime": "bullish",
            "universe": [
                {"ticker": "AAPL", "score": 85.5, "conviction": "HIGH",
                 "signal": "BUY", "sector": "Technology"},
            ],
            "buy_candidates": [],
            "sector_ratings": {
                "Technology": {"rating": 80, "modifier": 1.15},
                "Healthcare": {"rating": 65, "modifier": 1.0},
            },
        }
        preds = self._make_predictions(2)
        metrics = self._make_metrics()

        subject, html, plain = _build_predictor_email(
            preds, metrics, "2024-06-01", signals_data=signals
        )
        assert "BULLISH" in subject
        assert "BULLISH" in html  # regime pill still in the summary
        # The per-ticker universe (AAPL) is on the console page now, not inlined.
        assert "AAPL" not in html

    def test_unscored_buy_candidates_surface_in_data_warning(self):
        """config#856 slimmed this email: the Buy Candidates / research-brief
        table moved to the console Predictor page. The no-silent-fails
        invariant this test protects — actionable buy candidates the GBM did
        NOT score must still reach the inbox, not hide behind the console
        click — is preserved via the DATA WARNING block (mirroring the EOD
        email), so TICK2/TICK3 still surface even though the full brief no
        longer inlines."""
        from inference.daily_predict import _build_predictor_email

        # Predictions cover only TICK1 — TICK2/TICK3 are in buy_candidates but unscored
        preds = [{
            "ticker": "TICK1",
            "predicted_alpha": 0.02,
            "predicted_direction": "UP",
            "prediction_confidence": 0.60,
            "combined_rank": 1.0,
            "watchlist_source": "tracked",
        }]
        signals = {
            "market_regime": "neutral",
            "universe": [
                {"ticker": "TICK1", "score": 80.0, "conviction": "rising",
                 "signal": "ENTER", "sector": "Tech"},
                {"ticker": "TICK2", "score": 72.0, "conviction": "rising",
                 "signal": "ENTER", "sector": "Tech"},
                {"ticker": "TICK3", "score": 68.0, "conviction": "stable",
                 "signal": "ENTER", "sector": "Energy"},
            ],
            "buy_candidates": [
                {"ticker": "TICK1", "score": 80.0, "conviction": "rising",
                 "signal": "ENTER", "sector": "Tech"},
                {"ticker": "TICK2", "score": 72.0, "conviction": "rising",
                 "signal": "ENTER", "sector": "Tech"},
                {"ticker": "TICK3", "score": 68.0, "conviction": "stable",
                 "signal": "ENTER", "sector": "Energy"},
            ],
        }
        subject, html, plain = _build_predictor_email(
            preds, self._make_metrics(), "2024-06-01", signals_data=signals
        )
        # Unscored buy candidates surface via the DATA WARNING (not the old
        # inline "Buy Candidates" table, which now lives on the console page).
        assert "DATA WARNING" in html
        assert "DATA WARNING" in plain
        assert "not scored by GBM" in html
        assert "not scored by GBM" in plain
        assert "TICK2" in html and "TICK3" in html
        assert "TICK2" in plain and "TICK3" in plain
        # The full per-ticker table is gone from the email — TICK1 (which WAS
        # scored) is not inlined; the console page renders the complete table.
        assert "TICK1" not in html

    def test_missing_ic_metric(self):
        """Should handle missing ic_30d metric without crashing."""
        from inference.daily_predict import _build_predictor_email

        metrics = {"model_version": "gbm_v1.0"}
        preds = self._make_predictions(1)

        subject, html, plain = _build_predictor_email(preds, metrics, "2024-06-01")
        assert isinstance(html, str)


# ---------------------------------------------------------------------------
# Tests: get_veto_threshold
# ---------------------------------------------------------------------------

class TestGetVetoThreshold:
    """Tests for get_veto_threshold() — regime-adaptive veto threshold."""

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_base_threshold_from_s3(self, mock_load):
        """Should use veto_confidence from S3 params when available."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.65}
        result = get_veto_threshold("test-bucket", "neutral")
        assert result == 0.65

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_fallback_to_config(self, mock_load):
        """Should fall back to cfg.MIN_CONFIDENCE when S3 params are None."""
        from inference.daily_predict import get_veto_threshold
        import config as cfg

        mock_load.return_value = None
        result = get_veto_threshold("test-bucket", "neutral")
        assert result == cfg.MIN_CONFIDENCE

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_bear_regime_lowers_threshold(self, mock_load):
        """Bear regime should lower the threshold by 0.20 (new convention)."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.30}
        result = get_veto_threshold("test-bucket", "bear")
        assert result == pytest.approx(0.10)

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_bullish_regime_raises_threshold(self, mock_load):
        """Bullish regime should raise the threshold by 0.10 (new convention)."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.30}
        result = get_veto_threshold("test-bucket", "bullish")
        assert result == pytest.approx(0.40)

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_caution_regime(self, mock_load):
        """Caution regime should lower threshold by 0.10 (new convention)."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.30}
        result = get_veto_threshold("test-bucket", "caution")
        assert result == pytest.approx(0.20)

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_threshold_clamped_floor(self, mock_load):
        """Threshold should not go below 0.0 even with bear adjustment."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.10}
        result = get_veto_threshold("test-bucket", "bear")
        assert result == 0.0

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_threshold_clamped_ceiling(self, mock_load):
        """Threshold should not exceed 0.80 even with bullish adjustment."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.75}
        result = get_veto_threshold("test-bucket", "bullish")
        assert result == 0.80

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_empty_regime_string(self, mock_load):
        """Empty regime string should apply no adjustment."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.65}
        result = get_veto_threshold("test-bucket", "")
        assert result == 0.65

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_unknown_regime_no_adjustment(self, mock_load):
        """Unknown regime should apply no adjustment."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.65}
        result = get_veto_threshold("test-bucket", "sideways")
        assert result == 0.65
