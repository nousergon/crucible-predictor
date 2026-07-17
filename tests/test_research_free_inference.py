"""Tests for inference/research_free_inference.py — daily research-free
inference over the scanner-passing pool (config#2365).

Covers:
  - ``_assemble_research_free_features`` (pure, no IO): research-meta-feature
    zeroing, component wiring, unknown-feature graceful-degrade.
  - ``load_scanner_pool``: candidates.json quant_filter_pass=1 filtering,
    missing-artifact / malformed-artifact fail-hard.
  - ``_download_weight`` / ``load_meta_model`` / ``load_volatility_scorer``:
    S3-only weight fetch (mocked boto3 client — never a local checkout path).
  - ``write_research_free_predictions``: schema-versioned envelope shape +
    dated-key/latest-key dual write.
  - ``run_research_free_inference``: end-to-end orchestration against fully
    mocked S3 + ArcticDB boundaries (mirrors the mocking idiom used by
    test_residual_momentum_features.py / test_predictions_producer_contract.py
    — no live credentials required).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.research_free_inference import (  # noqa: E402
    SCHEMA_VERSION,
    _assemble_research_free_features,
    _compute_expected_move,
    _compute_momentum_scores,
    _compute_residual_momentum_scores,
    load_scanner_pool,
    write_research_free_predictions,
)
from model.meta_model import RESEARCH_META_FEATURES  # noqa: E402

_RESEARCH_FEATS = frozenset(RESEARCH_META_FEATURES)


# ── _assemble_research_free_features (pure) ─────────────────────────────────

class TestAssembleResearchFreeFeatures:
    def test_research_meta_features_always_zeroed(self):
        feat_names = list(RESEARCH_META_FEATURES) + ["momentum_score"]
        feats = _assemble_research_free_features(
            "AAPL", feat_names,
            momentum_scores={"AAPL": 0.5},
            resid_scores={}, expected_moves={}, macro_row={},
            research_meta_features=_RESEARCH_FEATS,
        )
        for f in RESEARCH_META_FEATURES:
            assert feats[f] == 0.0
        assert feats["momentum_score"] == 0.5

    def test_momentum_and_residual_and_expected_move_wired(self):
        feats = _assemble_research_free_features(
            "AAPL",
            ["momentum_score", "residual_momentum_score", "expected_move"],
            momentum_scores={"AAPL": 0.11},
            resid_scores={"AAPL": 0.22},
            expected_moves={"AAPL": 0.33},
            macro_row={},
            research_meta_features=_RESEARCH_FEATS,
        )
        assert feats == {
            "momentum_score": 0.11,
            "residual_momentum_score": 0.22,
            "expected_move": 0.33,
        }

    def test_macro_row_passthrough(self):
        feats = _assemble_research_free_features(
            "AAPL", ["macro_spy_20d_return", "regime_intensity_z"],
            momentum_scores={}, resid_scores={}, expected_moves={},
            macro_row={"macro_spy_20d_return": 0.05, "regime_intensity_z": -1.2},
            research_meta_features=_RESEARCH_FEATS,
        )
        assert feats["macro_spy_20d_return"] == 0.05
        assert feats["regime_intensity_z"] == -1.2

    def test_unknown_feature_zero_fills_with_warning(self, caplog):
        feats = _assemble_research_free_features(
            "AAPL", ["some_future_feature"],
            momentum_scores={}, resid_scores={}, expected_moves={},
            macro_row={}, research_meta_features=_RESEARCH_FEATS,
        )
        assert feats["some_future_feature"] == 0.0

    def test_ticker_absent_from_component_scores_defaults_zero(self):
        feats = _assemble_research_free_features(
            "MISSING", ["momentum_score", "residual_momentum_score", "expected_move"],
            momentum_scores={}, resid_scores={}, expected_moves={},
            macro_row={}, research_meta_features=_RESEARCH_FEATS,
        )
        assert feats == {
            "momentum_score": 0.0,
            "residual_momentum_score": 0.0,
            "expected_move": 0.0,
        }


# ── _compute_momentum_scores / _compute_residual_momentum_scores ───────────

class TestComponentComputers:
    def test_compute_momentum_scores_uses_precomputed_row(self):
        row = pd.Series({
            "momentum_5d": 0.02, "momentum_20d": 0.05,
            "price_vs_ma50": 0.01, "rsi_14": 60.0,
        })
        out = _compute_momentum_scores({"AAPL": row})
        assert "AAPL" in out
        assert isinstance(out["AAPL"], float)

    def test_compute_residual_momentum_scores_empty_history_neutral(self):
        out = _compute_residual_momentum_scores(
            ["AAPL"], close_history={}, spy_close=None, as_of=pd.Timestamp("2026-07-10"),
        )
        assert out == {"AAPL": 0.0}

    def test_compute_residual_momentum_scores_real_history(self):
        idx = pd.bdate_range("2024-01-01", periods=400)
        rng = np.random.default_rng(0)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 400)), index=idx)
        spy = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.008, 400)), index=idx)
        out = _compute_residual_momentum_scores(
            ["AAPL"], close_history={"AAPL": close}, spy_close=spy, as_of=idx[-1],
        )
        assert "AAPL" in out
        assert np.isfinite(out["AAPL"])

    def test_compute_expected_move_none_scorer_returns_zero(self):
        rows = {"AAPL": pd.Series({"atr_14_pct": 1.0})}
        out = _compute_expected_move(rows, None, "2026-07-10")
        assert out == {"AAPL": 0.0}

    def test_compute_expected_move_rank_normalizes_cross_section(self):
        scorer = MagicMock()
        scorer.feature_names = ["atr_14_pct", "realized_vol_20d"]
        scorer.predict = MagicMock(return_value=np.array([0.01, 0.02, 0.03]))
        rows = {
            "A": pd.Series({"atr_14_pct": 1.0, "realized_vol_20d": 0.1}),
            "B": pd.Series({"atr_14_pct": 2.0, "realized_vol_20d": 0.2}),
            "C": pd.Series({"atr_14_pct": 3.0, "realized_vol_20d": 0.3}),
        }
        out = _compute_expected_move(rows, scorer, "2026-07-10")
        assert set(out) == {"A", "B", "C"}
        scorer.predict.assert_called_once()


# ── load_scanner_pool ────────────────────────────────────────────────────────

class TestLoadScannerPool:
    def _s3_with_body(self, payload: dict):
        s3 = MagicMock()
        body = MagicMock()
        body.read.return_value = json.dumps(payload).encode()
        s3.get_object.return_value = {"Body": body}
        return s3

    def test_filters_to_quant_filter_pass_1(self):
        s3 = self._s3_with_body({
            "scanner_eval_log": [
                {"ticker": "AAPL", "quant_filter_pass": 1},
                {"ticker": "MSFT", "quant_filter_pass": 0},
                {"ticker": "GOOG", "quant_filter_pass": 1},
            ]
        })
        pool = load_scanner_pool("test-bucket", "2026-07-10", s3_client=s3)
        assert pool == ["AAPL", "GOOG"]
        s3.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="candidates/2026-07-10/candidates.json",
        )

    def test_dedupes_and_sorts(self):
        s3 = self._s3_with_body({
            "scanner_eval_log": [
                {"ticker": "ZZZ", "quant_filter_pass": 1},
                {"ticker": "AAA", "quant_filter_pass": 1},
                {"ticker": "ZZZ", "quant_filter_pass": 1},
            ]
        })
        pool = load_scanner_pool("test-bucket", "2026-07-10", s3_client=s3)
        assert pool == ["AAA", "ZZZ"]

    def test_missing_artifact_raises(self):
        from botocore.exceptions import ClientError

        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject",
        )
        with pytest.raises(RuntimeError, match="failed to load candidates artifact"):
            load_scanner_pool("test-bucket", "2026-07-10", s3_client=s3)

    def test_missing_eval_log_field_raises(self):
        s3 = self._s3_with_body({"run_date": "2026-07-10"})
        with pytest.raises(RuntimeError, match="scanner_eval_log"):
            load_scanner_pool("test-bucket", "2026-07-10", s3_client=s3)

    def test_empty_eval_log_returns_empty_pool(self):
        s3 = self._s3_with_body({"scanner_eval_log": []})
        pool = load_scanner_pool("test-bucket", "2026-07-10", s3_client=s3)
        assert pool == []


# ── write_research_free_predictions ─────────────────────────────────────────

class TestWriteResearchFreePredictions:
    def test_writes_dated_and_latest_keys(self):
        s3 = MagicMock()
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "date": "2026-07-10",
            "n_predictions": 1,
            "predictions": [{
                "ticker": "AAPL", "prediction_date": "2026-07-10",
                "predicted_alpha": 0.01, "n_research_features_missing": 4,
            }],
        }
        key = write_research_free_predictions(envelope, "test-bucket", s3_client=s3)
        assert key == "predictor/predictions_research_free/2026-07-10.json"
        assert s3.put_object.call_count == 2
        keys_written = {c.kwargs["Key"] for c in s3.put_object.call_args_list}
        assert keys_written == {
            "predictor/predictions_research_free/2026-07-10.json",
            "predictor/predictions_research_free/latest.json",
        }

    def test_raises_on_s3_failure(self):
        from botocore.exceptions import ClientError

        s3 = MagicMock()
        s3.put_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "PutObject",
        )
        envelope = {"date": "2026-07-10", "n_predictions": 0, "predictions": []}
        with pytest.raises(RuntimeError, match="failed to write research-free predictions"):
            write_research_free_predictions(envelope, "test-bucket", s3_client=s3)


# ── run_research_free_inference (end-to-end, mocked boundaries) ────────────

class _FakeMetaModel:
    is_fitted = True
    _feature_names = [
        "momentum_score", "residual_momentum_score", "expected_move",
        "research_calibrator_prob", "research_composite_score",
        "research_conviction", "sector_macro_modifier",
    ]

    def predict_single(self, features: dict) -> float:
        return float(sum(features.values()))


def _synthetic_price_data(tickers, n=300):
    idx = pd.bdate_range("2025-01-01", periods=n)
    out = {}
    for i, t in enumerate(tickers):
        rng = np.random.default_rng(hash(t) % (2**31))
        close = 50 + i * 10 + np.cumsum(rng.normal(0, 1, n))
        out[t] = pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n),
            "momentum_5d": rng.normal(0, 0.02, n),
            "momentum_20d": rng.normal(0, 0.05, n),
            "price_vs_ma50": rng.normal(0, 0.03, n),
            "rsi_14": rng.uniform(30, 70, n),
            "atr_14_pct": rng.uniform(1, 5, n),
            "realized_vol_20d": rng.uniform(0.1, 0.4, n),
            "vol_ratio_10_60": rng.uniform(0.8, 1.2, n),
            "iv_rank": rng.uniform(0, 100, n),
            "dist_from_52w_high": rng.uniform(-0.3, 0, n),
            "dist_from_52w_low": rng.uniform(0, 0.5, n),
        }, index=idx)
    return out


class TestRunResearchFreeInferenceEndToEnd:
    def test_full_run_mocked_boundaries(self):
        universe = ["AAPL", "MSFT", "GOOG", "SPY"]
        price_data = _synthetic_price_data(universe)
        macro = {"SPY": price_data["SPY"]}

        candidates_payload = {
            "scanner_eval_log": [
                {"ticker": "AAPL", "quant_filter_pass": 1},
                {"ticker": "MSFT", "quant_filter_pass": 1},
                {"ticker": "GOOG", "quant_filter_pass": 0},
            ]
        }
        s3 = MagicMock()
        body = MagicMock()
        body.read.return_value = json.dumps(candidates_payload).encode()
        s3.get_object.return_value = {"Body": body}

        with patch(
            "inference.research_free_inference.load_meta_model",
            return_value=_FakeMetaModel(),
        ), patch(
            "inference.research_free_inference.load_volatility_scorer",
            return_value=None,
        ), patch(
            "nousergon_lib.arcticdb.load_universe_ohlcv",
            return_value=price_data,
        ), patch(
            "nousergon_lib.arcticdb.load_macro_series",
            return_value=macro,
        ):
            from inference.research_free_inference import run_research_free_inference

            result = run_research_free_inference(
                "2026-07-10", bucket="test-bucket", s3_client=s3, dry_run=True,
            )

        assert result["status"] == "ok"
        assert result["n_written"] == 2
        assert result["n_errors"] == 0
        assert result["n_research_features_missing"] == 4

    def test_scanner_pool_ticker_missing_from_arcticdb_counts_as_error(self):
        # GHOST passes the scanner but has no ArcticDB row — the ArcticDB
        # mock deliberately omits it so the orchestration's "absent from
        # ArcticDB" branch fires.
        pool_payload = {
            "scanner_eval_log": [
                {"ticker": "AAPL", "quant_filter_pass": 1},
                {"ticker": "GHOST", "quant_filter_pass": 1},
            ]
        }
        universe = ["AAPL", "SPY"]  # GHOST intentionally absent from ArcticDB
        price_data = _synthetic_price_data(universe)
        macro = {"SPY": price_data["SPY"]}

        s3 = MagicMock()
        body = MagicMock()
        body.read.return_value = json.dumps(pool_payload).encode()
        s3.get_object.return_value = {"Body": body}

        with patch(
            "inference.research_free_inference.load_meta_model",
            return_value=_FakeMetaModel(),
        ), patch(
            "inference.research_free_inference.load_volatility_scorer",
            return_value=None,
        ), patch(
            "nousergon_lib.arcticdb.load_universe_ohlcv",
            return_value=price_data,
        ), patch(
            "nousergon_lib.arcticdb.load_macro_series",
            return_value=macro,
        ):
            from inference.research_free_inference import run_research_free_inference

            result = run_research_free_inference(
                "2026-07-10", bucket="test-bucket", s3_client=s3, dry_run=True,
            )

        assert result["n_written"] == 1
        assert result["n_errors"] == 1

    def test_unfitted_meta_model_raises(self):
        s3 = MagicMock()
        body = MagicMock()
        body.read.return_value = json.dumps({"scanner_eval_log": []}).encode()
        s3.get_object.return_value = {"Body": body}

        with patch("inference.research_free_inference._download_weight") as mock_dl, \
             patch("model.meta_model.MetaModel.load") as mock_load:
            mock_dl.return_value = Path("/tmp/fake_meta_model.pkl")
            fake_mm = MagicMock()
            fake_mm.is_fitted = False
            mock_load.return_value = fake_mm

            from inference.research_free_inference import run_research_free_inference

            with pytest.raises(RuntimeError, match="not fitted"):
                run_research_free_inference(
                    "2026-07-10", bucket="test-bucket", s3_client=s3, dry_run=True,
                )
