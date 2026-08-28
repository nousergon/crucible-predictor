"""Tests for inference/stages/write_output.py — predictions writing + email builder."""

import json
from unittest.mock import MagicMock, patch

import pytest

import inference.stages.write_output as wo
from inference.stages.write_output import (
    write_predictions, get_veto_threshold,
    _merge_predictions, _read_existing_predictions,
    _relative_dispersion_check, _n_high_confidence_zero_streak,
    _load_trailing_batch_history,
)
from model.output_distribution_gate import validate_live_batch_invariant_health


class TestWritePredictionsDryRun:
    """Test write_predictions in dry-run mode (no S3)."""

    def test_dry_run_prints(self, capsys):
        predictions = [
            {"ticker": "AAPL", "predicted_direction": "UP", "prediction_confidence": 0.75},
            {"ticker": "MSFT", "predicted_direction": "DOWN", "prediction_confidence": 0.68},
        ]
        metrics = {"model_version": "test-v1", "hit_rate_30d_rolling": 0.55}

        write_predictions(predictions, "2026-04-08", "bucket", metrics, dry_run=True)

        captured = capsys.readouterr()
        assert "PREDICTIONS (dry-run)" in captured.out
        assert "AAPL" in captured.out
        assert "METRICS (dry-run)" in captured.out

    def test_dry_run_counts_high_confidence(self, capsys):
        predictions = [
            {"ticker": "A", "prediction_confidence": 0.80},
            {"ticker": "B", "prediction_confidence": 0.50},
            {"ticker": "C", "prediction_confidence": 0.70},
        ]
        write_predictions(predictions, "2026-04-08", "bucket", {}, dry_run=True, veto_threshold=0.65)

        captured = capsys.readouterr()
        output = json.loads(captured.out.split("=== PREDICTIONS (dry-run) ===\n")[1].split("\n=== METRICS")[0])
        assert output["n_high_confidence"] == 2  # A (0.80) and C (0.70) >= 0.65

    def test_dry_run_includes_date(self, capsys):
        write_predictions([], "2026-04-08", "bucket", {}, dry_run=True)
        captured = capsys.readouterr()
        assert "2026-04-08" in captured.out


class TestWritePredictionsS3:
    """Test write_predictions with mocked S3."""

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    @patch("inference.stages.write_output._s3_put_json")
    def test_writes_three_keys(self, mock_put):
        predictions = [{"ticker": "AAPL", "prediction_confidence": 0.75}]
        write_predictions(predictions, "2026-04-08", "bucket", {"model_version": "v1"})
        assert mock_put.call_count == 3  # dated, latest, metrics

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    @patch("inference.stages.write_output._s3_put_json", side_effect=Exception("S3 error"))
    def test_handles_write_failure(self, mock_put):
        # config#2333: the primary dated-key write is fail-loud — a failure
        # here must raise PipelineHardFail (superseding the pre-config#2333
        # "logged but not propagated" contract this test used to assert; see
        # TestS3WriteFailLoud below for the full fail-loud/best-effort matrix).
        from inference.pipeline import PipelineHardFail
        with pytest.raises(PipelineHardFail):
            write_predictions([{"ticker": "AAPL"}], "2026-04-08", "bucket", {})


class TestGetVetoThresholdExtended:
    """Additional veto threshold tests."""

    def setup_method(self):
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.30})
    def test_case_insensitive_regime(self, _mock):
        # Post-2026-05-12: bear adjustment is -0.20 in new confidence units.
        result = get_veto_threshold("bucket", "  BEAR  ")
        assert result == pytest.approx(0.10)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.30})
    def test_none_regime(self, _mock):
        result = get_veto_threshold("bucket", None)
        assert result == pytest.approx(0.30)


class TestForcedBearClamp:
    """Stage F2 — forced-bear veto clamp (regime-fast-signal-260515.md)."""

    def setup_method(self):
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False

    _ON = {"veto_confidence": 0.30, "regime_forced_bear_enabled": True}
    _OFF = {"veto_confidence": 0.30, "regime_forced_bear_enabled": False}

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_clamp_on_forces_bear_floor(self, m):
        m.return_value = self._ON
        # neutral regime → unclamped 0.30; clamp → base-cap = 0.30-0.20 = 0.10
        assert get_veto_threshold("b", "neutral", forced_bear=True) == pytest.approx(0.10)

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_clamp_off_is_parallel_observe_noop(self, m):
        m.return_value = self._OFF
        # flag off: behavior unchanged even though forced_bear=True
        assert get_veto_threshold("b", "neutral", forced_bear=True) == pytest.approx(0.30)

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_no_clamp_when_not_forced_bear(self, m):
        m.return_value = self._ON
        assert get_veto_threshold("b", "neutral", forced_bear=False) == pytest.approx(0.30)

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_forced_bear_overrides_risk_on_wire4(self, m):
        # Wire 4 ON + strong risk-ON intensity_z would raise the threshold
        # to 0.50 (more permissive). Forced-bear must still win — the
        # max-protection property.
        m.return_value = {
            "veto_confidence": 0.30, "regime_veto_enabled": True,
            "regime_forced_bear_enabled": True,
        }
        permissive = get_veto_threshold("b", regime_intensity_z=2.0)
        assert permissive == pytest.approx(0.50)  # sanity: risk-on raises it
        m.return_value = {
            "veto_confidence": 0.30, "regime_veto_enabled": True,
            "regime_forced_bear_enabled": True,
        }
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False
        clamped = get_veto_threshold("b", regime_intensity_z=2.0, forced_bear=True)
        assert clamped == pytest.approx(0.10)  # bear floor wins

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_clamp_never_unprotects(self, m):
        # Wire 4 risk-OFF already more aggressive than the bear floor →
        # min() keeps the more-protective value (never raises threshold).
        m.return_value = {
            "veto_confidence": 0.30, "regime_veto_enabled": True,
            "regime_veto_cap": 0.30, "regime_forced_bear_enabled": True,
        }
        # intensity_z=-2 → adj -0.30 → threshold 0.00; bear_floor = 0.30-0.30 = 0.00
        out = get_veto_threshold("b", regime_intensity_z=-2.0, forced_bear=True)
        assert out == pytest.approx(0.0)


class TestMergePredictions:
    """Supplemental-scoring merge: new + existing → union, re-ranked."""

    def _pred(self, ticker: str, alpha: float) -> dict:
        return {
            "ticker": ticker,
            "predicted_alpha": alpha,
            "predicted_direction": "UP" if alpha >= 0 else "DOWN",
            "prediction_confidence": 0.55,
            "combined_rank": None,
        }

    def test_empty_existing_returns_new_with_ranks(self):
        new = [self._pred("A", 0.02), self._pred("B", -0.01)]
        merged = _merge_predictions(new, [])
        assert len(merged) == 2
        # Sorted by alpha desc: A(+0.02) rank 1, B(-0.01) rank 2
        assert merged[0]["ticker"] == "A" and merged[0]["combined_rank"] == 1
        assert merged[1]["ticker"] == "B" and merged[1]["combined_rank"] == 2

    def test_union_preserves_existing_non_overlapping(self):
        existing = [self._pred("X", 0.05), self._pred("Y", 0.03)]
        new = [self._pred("Z", 0.04)]
        merged = _merge_predictions(new, existing)
        tickers = [p["ticker"] for p in merged]
        assert set(tickers) == {"X", "Y", "Z"}
        # Rank recomputed across union: X(0.05), Z(0.04), Y(0.03)
        rank_by_ticker = {p["ticker"]: p["combined_rank"] for p in merged}
        assert rank_by_ticker == {"X": 1, "Z": 2, "Y": 3}

    def test_new_overrides_existing_on_collision(self):
        existing = [self._pred("A", 0.01)]
        new = [self._pred("A", 0.10)]  # new wins
        merged = _merge_predictions(new, existing)
        assert len(merged) == 1
        assert merged[0]["predicted_alpha"] == 0.10
        assert merged[0]["combined_rank"] == 1

    def test_rank_recomputed_across_full_union(self):
        existing = [self._pred(f"E{i}", 0.02 - i * 0.001) for i in range(3)]  # E0 best
        new = [self._pred("N0", 0.025), self._pred("N1", 0.018)]
        merged = _merge_predictions(new, existing)
        # Expected order by alpha desc: N0(.025), E0(.02), E1(.019), N1(.018), E2(.018)
        expected_top = ["N0", "E0", "E1"]
        top3 = [p["ticker"] for p in merged[:3]]
        assert top3 == expected_top
        # Ranks are contiguous 1..N
        ranks = sorted(p["combined_rank"] for p in merged)
        assert ranks == list(range(1, len(merged) + 1))

    def test_case_insensitive_ticker_keys(self):
        existing = [self._pred("aapl", 0.01)]
        new = [self._pred("AAPL", 0.05)]  # should collide
        merged = _merge_predictions(new, existing)
        assert len(merged) == 1  # merged, not both
        assert merged[0]["predicted_alpha"] == 0.05


class TestCoverageGuard:
    """Hard-fail when buy_candidates has tickers not scored by predictor.

    Defense-in-depth: the Step Function coverage-gap Choice state should
    normally re-invoke the predictor with --tickers to fill the gap before
    this write runs. This assertion catches any regression of that wiring
    instead of letting the executor see a partial predictions.json.
    """

    def _ctx(self, predictions, signals_data, explicit=False):
        from inference.pipeline import PipelineContext
        ctx = PipelineContext(
            date_str="2026-04-20",
            bucket="bucket",
            dry_run=True,  # skip write/email so we isolate the guard
            predictions=list(predictions),
            signals_data=signals_data,
            explicit_tickers=[],  # first-run path (no merge)
        )
        if explicit:
            ctx.explicit_tickers = [p["ticker"] for p in predictions]
        return ctx

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_passes_when_all_buy_candidates_scored(self, _m1, _m2):
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
                {"ticker": "B", "predicted_alpha": -0.01, "combined_rank": 2,
                 "predicted_direction": "DOWN", "prediction_confidence": 0.55},
            ],
            signals_data={"buy_candidates": [{"ticker": "A"}, {"ticker": "B"}]},
        )
        # Should not raise
        wo.run(ctx)

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_raises_on_missing_buy_candidates(self, _m1, _m2):
        from inference.pipeline import PipelineHardFail
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={
                "buy_candidates": [
                    {"ticker": "A"},
                    {"ticker": "SNDK"},  # unscored — today's bug
                    {"ticker": "WDC"},   # unscored
                ],
            },
        )
        with pytest.raises(PipelineHardFail) as exc:
            wo.run(ctx)
        assert "SNDK" in str(exc.value)
        assert "WDC" in str(exc.value)
        assert "Coverage gap" in str(exc.value)

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_empty_buy_candidates_is_no_op(self, _m1, _m2):
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={"buy_candidates": []},
        )
        # Should not raise
        wo.run(ctx)

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_missing_signals_data_is_no_op(self, _m1, _m2):
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={},
        )
        # Should not raise
        wo.run(ctx)


class TestInferenceCoverageDenominator:
    """config#1075: run(ctx) persists the tradable-universe coverage denominator.

    n_universe = |signals.json universe| and n_universe_covered = universe
    tickers that got a prediction, so the report card's inference_coverage can
    grade covered/universe ∈ [0,1] instead of a permanent N/A.
    """

    def _ctx(self, predictions, signals_data):
        from inference.pipeline import PipelineContext
        return PipelineContext(
            date_str="2026-06-14",
            bucket="bucket",
            dry_run=True,  # metrics print to stdout; no S3 write
            predictions=list(predictions),
            signals_data=signals_data,
            explicit_tickers=[],
        )

    @staticmethod
    def _metrics(capsys):
        out = capsys.readouterr().out
        return json.loads(out.split("=== METRICS (dry-run) ===\n")[1])

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_counts_universe_and_covered(self, _m1, _m2, capsys):
        # universe of 4; only A/B/C scored → covered 3, universe 4.
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
                {"ticker": "B", "predicted_alpha": 0.01, "combined_rank": 2,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
                {"ticker": "C", "predicted_alpha": 0.01, "combined_rank": 3,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={"universe": ["A", "B", "C", "D"], "buy_candidates": []},
        )
        wo.run(ctx)
        m = self._metrics(capsys)
        assert m["n_universe"] == 4
        assert m["n_universe_covered"] == 3  # D unscored

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_universe_dict_entries_normalized(self, _m1, _m2, capsys):
        # universe entries as {ticker: …} dicts must be counted too.
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={"universe": [{"ticker": "A"}, {"ticker": "B"}], "buy_candidates": []},
        )
        wo.run(ctx)
        m = self._metrics(capsys)
        assert m["n_universe"] == 2
        assert m["n_universe_covered"] == 1

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_empty_universe_records_zero(self, _m1, _m2, capsys):
        # No universe in signals.json → 0 (evaluator keeps honest N/A, no /0).
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={"buy_candidates": []},
        )
        wo.run(ctx)
        m = self._metrics(capsys)
        assert m["n_universe"] == 0
        assert m["n_universe_covered"] == 0


class TestGbmVetoConfidenceFloor:
    """Pin the confidence floor on gbm_veto computation.

    The binary UP/DOWN output forces every prediction to one extreme even
    when the underlying alpha estimate is near zero. Without a confidence
    floor, low-confidence DOWN predictions (near coin-flip on the calibrator)
    fire ``gbm_veto`` and over-block research's ENTER candidates. 2026-05-11
    incident: 15 of 30 vetos with mean DOWN confidence ~0.75; 8 of 28
    research ENTERs blocked by predictor.

    The fix gates ``gbm_veto`` on ``prediction_confidence >= veto_thresh``
    in addition to the existing ``alpha < 0`` and ``cr > n_preds/2``
    criteria. ``veto_thresh`` is the regime-adjusted value from
    ``get_veto_threshold`` (sourced from ``config/predictor_params.json``'s
    ``veto_confidence`` with ``cfg.MIN_CONFIDENCE`` fallback).

    Confidence semantics post-2026-05-12: ``|p_up - 0.5| * 2`` ∈ [0, 1]
    (ROADMAP L1615). Threshold values below are in the new convention.
    """

    def _ctx(self, predictions, signals_data=None):
        from inference.pipeline import PipelineContext
        ctx = PipelineContext(
            date_str="2026-05-11",
            bucket="bucket",
            dry_run=True,
            predictions=list(predictions),
            signals_data=signals_data or {"buy_candidates": []},
            explicit_tickers=[],
        )
        return ctx

    @patch.object(wo, "get_veto_threshold", return_value=0.40)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_low_confidence_down_does_not_veto(self, _m1, _m2):
        """A DOWN prediction below the regime-adjusted confidence floor
        must NOT trigger ``gbm_veto`` — near coin-flip is binary noise."""
        # Two DOWN preds both in bottom half; one above floor, one below.
        # Confidence in post-2026-05-12 semantics: |p_up - 0.5| * 2.
        preds = [
            {"ticker": "A", "predicted_alpha": 0.05, "combined_rank": 1,
             "predicted_direction": "UP", "prediction_confidence": 0.80},
            {"ticker": "B", "predicted_alpha": 0.03, "combined_rank": 2,
             "predicted_direction": "UP", "prediction_confidence": 0.60},
            {"ticker": "C", "predicted_alpha": -0.02, "combined_rank": 3,
             "predicted_direction": "DOWN", "prediction_confidence": 0.20},
            {"ticker": "D", "predicted_alpha": -0.08, "combined_rank": 4,
             "predicted_direction": "DOWN", "prediction_confidence": 0.90},
        ]
        ctx = self._ctx(preds)
        wo.run(ctx)
        veto_by_ticker = {p["ticker"]: p["gbm_veto"] for p in ctx.predictions}
        # A, B: UP (alpha>0) → no veto regardless of confidence
        assert veto_by_ticker["A"] is False
        assert veto_by_ticker["B"] is False
        # C: DOWN + bottom half + conf 0.20 < threshold 0.40 → NO veto (the fix)
        assert veto_by_ticker["C"] is False, (
            "DOWN at conf=0.20 below threshold=0.40 must NOT fire gbm_veto — "
            "near coin-flip is binary noise, not a high-conviction bearish call"
        )
        # D: DOWN + bottom half + conf 0.90 >= 0.40 → veto fires
        assert veto_by_ticker["D"] is True, (
            "DOWN at conf=0.90 above threshold=0.40 must fire gbm_veto — "
            "this is the high-conviction bearish signal we still want to block on"
        )

    @patch.object(wo, "get_veto_threshold", return_value=0.40)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_top_half_rank_never_vetoes(self, _m1, _m2):
        """The cross-sectional rank gate still applies — top-half ranks
        skip the veto regardless of confidence or direction. Pinned so a
        future refactor doesn't accidentally drop the rank gate while
        adding the confidence gate."""
        preds = [
            # DOWN, top half (rank 1 of 4), high conf → still no veto
            {"ticker": "X", "predicted_alpha": -0.01, "combined_rank": 1,
             "predicted_direction": "DOWN", "prediction_confidence": 0.90},
            {"ticker": "Y", "predicted_alpha": 0.05, "combined_rank": 2,
             "predicted_direction": "UP", "prediction_confidence": 0.80},
            {"ticker": "Z", "predicted_alpha": -0.05, "combined_rank": 3,
             "predicted_direction": "DOWN", "prediction_confidence": 0.90},
            {"ticker": "W", "predicted_alpha": -0.06, "combined_rank": 4,
             "predicted_direction": "DOWN", "prediction_confidence": 0.90},
        ]
        ctx = self._ctx(preds)
        wo.run(ctx)
        veto_by_ticker = {p["ticker"]: p["gbm_veto"] for p in ctx.predictions}
        assert veto_by_ticker["X"] is False  # top half (rank 1 of 4)
        assert veto_by_ticker["Z"] is True   # bottom half, high conf, DOWN
        assert veto_by_ticker["W"] is True

    @patch.object(wo, "get_veto_threshold", return_value=0.40)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_missing_confidence_field_does_not_veto(self, _m1, _m2):
        """If ``prediction_confidence`` is absent (defensive — never
        expected post-PR-#196's coverage gates), treat as 0 → never
        veto. Fail-safe direction: under-veto rather than over-veto
        when the field shape changes."""
        preds = [
            {"ticker": "A", "predicted_alpha": -0.05, "combined_rank": 2,
             "predicted_direction": "DOWN"},  # no prediction_confidence key
        ]
        ctx = self._ctx(preds)
        wo.run(ctx)
        assert ctx.predictions[0]["gbm_veto"] is False

    @patch.object(wo, "get_veto_threshold", return_value=0.10)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_threshold_is_regime_adapted_via_get_veto_threshold(self, _m1, _m2):
        """The threshold flows from ``get_veto_threshold(market_regime=…)``
        which applies the bull/bear adjustments. Pin that the per-ticker
        veto uses the regime-adjusted value, not a hardcoded constant."""
        preds = [
            # DOWN at conf=0.20 — would NOT veto at threshold 0.40 (bull),
            # but WOULD veto at threshold 0.10 (caution, this test's regime).
            {"ticker": "A", "predicted_alpha": 0.05, "combined_rank": 1,
             "predicted_direction": "UP", "prediction_confidence": 0.80},
            {"ticker": "B", "predicted_alpha": -0.02, "combined_rank": 2,
             "predicted_direction": "DOWN", "prediction_confidence": 0.20},
        ]
        ctx = self._ctx(preds)
        wo.run(ctx)
        assert ctx.predictions[1]["gbm_veto"] is True, (
            "At threshold 0.10, conf=0.20 should fire the veto"
        )


class TestPerSectorVetoShadowSoak:
    """config#921 shadow soak (Brian's ruling 2026-07-07): attach what the
    per-sector-threshold veto decision WOULD have been, without ever
    touching the live ``gbm_veto`` value computed just above this block.

    Sector-per-ticker resolves via ``ctx.signals_data['universe']`` (GICS-
    style names, canonicalized through
    ``model.research_features.SECTOR_NAME_CANONICAL``) — NOT
    ``ctx.sector_map``, which is ticker → sector-ETF-symbol (e.g. "XLE") and
    cannot key into the backtester's ``per_sector_overrides`` map.
    """

    def setup_method(self):
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False

    def _ctx(self, predictions, universe=None):
        import copy
        from inference.pipeline import PipelineContext
        # Deep-copy: predictions/universe are class-level fixtures shared
        # across tests, and wo.run() mutates prediction dicts in place
        # (gbm_veto, gbm_veto_shadow_sector, ...) — a shallow copy would leak
        # keys set by one test into the next test's shared dict instances.
        ctx = PipelineContext(
            date_str="2026-07-08",
            bucket="bucket",
            dry_run=True,
            predictions=copy.deepcopy(list(predictions)),
            signals_data={
                "buy_candidates": [],
                "universe": copy.deepcopy(universe or []),
            },
            explicit_tickers=[],
        )
        return ctx

    _PREDS = [
        # DOWN, bottom half, high confidence — fires the live gbm_veto at the
        # global 0.40 threshold used by every test in this class.
        {"ticker": "XOM", "predicted_alpha": -0.05, "combined_rank": 3,
         "predicted_direction": "DOWN", "prediction_confidence": 0.90},
        # DOWN, bottom half, confidence between the sector override (0.85)
        # and the global threshold (0.40) — live veto still fires (>=0.40),
        # but the sector-shadow decision differs only when confidence is
        # BELOW the sector threshold; see the dedicated divergence test.
        {"ticker": "JPM", "predicted_alpha": -0.03, "combined_rank": 4,
         "predicted_direction": "DOWN", "prediction_confidence": 0.50},
    ]

    _UNIVERSE = [
        {"ticker": "XOM", "sector": "Energy"},
        {"ticker": "JPM", "sector": "Financials"},  # canonicalizes -> Financial
    ]

    @patch.object(wo, "get_veto_threshold", return_value=0.40)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_flag_off_no_shadow_fields(self, _m1, _m2):
        """Default (flag unset in predictor_params): no shadow fields
        attached at all, and live gbm_veto is unaffected."""
        with patch.object(
            wo, "_load_predictor_params_from_s3",
            return_value={"veto_confidence": 0.40},
        ):
            ctx = self._ctx(self._PREDS, self._UNIVERSE)
            wo.run(ctx)
        by_ticker = {p["ticker"]: p for p in ctx.predictions}
        assert "gbm_veto_shadow_sector" not in by_ticker["XOM"]
        assert "gbm_veto_shadow_sector" not in by_ticker["JPM"]
        assert by_ticker["XOM"]["gbm_veto"] is True
        assert by_ticker["JPM"]["gbm_veto"] is True

    @patch.object(wo, "get_veto_threshold", return_value=0.40)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_flag_on_no_overrides_no_shadow_fields(self, _m1, _m2):
        """Flag on but no per_sector_overrides present — no-op, no crash."""
        with patch.object(
            wo, "_load_predictor_params_from_s3",
            return_value={
                "veto_confidence": 0.40,
                "veto_sector_shadow_enabled": True,
            },
        ):
            ctx = self._ctx(self._PREDS, self._UNIVERSE)
            wo.run(ctx)
        by_ticker = {p["ticker"]: p for p in ctx.predictions}
        assert "gbm_veto_shadow_sector" not in by_ticker["XOM"]
        assert "gbm_veto_shadow_sector" not in by_ticker["JPM"]

    @patch.object(wo, "get_veto_threshold", return_value=0.40)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_flag_on_with_override_attaches_shadow_field(self, _m1, _m2):
        """Flag on + override present for the ticker's (canonicalized)
        sector: shadow field reflects the sector-threshold decision, and
        the threshold used is recorded. Live gbm_veto is untouched."""
        with patch.object(
            wo, "_load_predictor_params_from_s3",
            return_value={
                "veto_confidence": 0.40,
                "veto_sector_shadow_enabled": True,
                "per_sector_overrides": {"Energy": 0.60, "Financial": 0.95},
            },
        ):
            ctx = self._ctx(self._PREDS, self._UNIVERSE)
            wo.run(ctx)
        by_ticker = {p["ticker"]: p for p in ctx.predictions}
        # XOM: Energy sector threshold 0.60, confidence 0.90 >= 0.60 → shadow veto True
        assert by_ticker["XOM"]["gbm_veto_shadow_sector"] is True
        assert by_ticker["XOM"]["gbm_veto_shadow_sector_threshold"] == 0.60
        # JPM: Financials -> Financial threshold 0.95, confidence 0.50 < 0.95
        # → shadow veto False, DIVERGES from live gbm_veto=True (global 0.40).
        assert by_ticker["JPM"]["gbm_veto_shadow_sector"] is False
        assert by_ticker["JPM"]["gbm_veto_shadow_sector_threshold"] == 0.95
        # Live decision path is byte-identical to the flag-off case.
        assert by_ticker["XOM"]["gbm_veto"] is True
        assert by_ticker["JPM"]["gbm_veto"] is True

    @patch.object(wo, "get_veto_threshold", return_value=0.40)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_flag_on_ticker_sector_missing_from_overrides_no_shadow_field(self, _m1, _m2):
        """Ticker's sector has no override entry — no shadow field for that
        ticker (default no-op), matching the graceful-degrade style of the
        existing regime/drawdown clamps."""
        with patch.object(
            wo, "_load_predictor_params_from_s3",
            return_value={
                "veto_confidence": 0.40,
                "veto_sector_shadow_enabled": True,
                "per_sector_overrides": {"Healthcare": 0.55},  # neither Energy nor Financial
            },
        ):
            ctx = self._ctx(self._PREDS, self._UNIVERSE)
            wo.run(ctx)
        by_ticker = {p["ticker"]: p for p in ctx.predictions}
        assert "gbm_veto_shadow_sector" not in by_ticker["XOM"]
        assert "gbm_veto_shadow_sector" not in by_ticker["JPM"]

    @patch.object(wo, "get_veto_threshold", return_value=0.40)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_flag_on_ticker_missing_from_universe_no_shadow_field(self, _m1, _m2):
        """Ticker absent from signals_data['universe'] (no sector resolvable)
        — no-op, does not crash even with overrides present."""
        with patch.object(
            wo, "_load_predictor_params_from_s3",
            return_value={
                "veto_confidence": 0.40,
                "veto_sector_shadow_enabled": True,
                "per_sector_overrides": {"Energy": 0.60},
            },
        ):
            ctx = self._ctx(self._PREDS, universe=[])  # empty universe
            wo.run(ctx)
        by_ticker = {p["ticker"]: p for p in ctx.predictions}
        assert "gbm_veto_shadow_sector" not in by_ticker["XOM"]
        assert "gbm_veto_shadow_sector" not in by_ticker["JPM"]
        # Live veto still computed normally.
        assert by_ticker["XOM"]["gbm_veto"] is True

    @patch.object(wo, "get_veto_threshold", return_value=0.40)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_shadow_soak_never_alters_live_gbm_veto_value(self, _m1, _m2):
        """Critical regression guard: run the SAME predictions through with
        the flag off vs on (with overrides that would flip the decision
        for JPM) and assert gbm_veto is byte-identical in both runs."""
        with patch.object(
            wo, "_load_predictor_params_from_s3",
            return_value={"veto_confidence": 0.40},
        ):
            ctx_off = self._ctx(self._PREDS, self._UNIVERSE)
            wo.run(ctx_off)
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False
        with patch.object(
            wo, "_load_predictor_params_from_s3",
            return_value={
                "veto_confidence": 0.40,
                "veto_sector_shadow_enabled": True,
                "per_sector_overrides": {"Energy": 0.60, "Financial": 0.95},
            },
        ):
            ctx_on = self._ctx(self._PREDS, self._UNIVERSE)
            wo.run(ctx_on)
        live_off = {p["ticker"]: p["gbm_veto"] for p in ctx_off.predictions}
        live_on = {p["ticker"]: p["gbm_veto"] for p in ctx_on.predictions}
        assert live_off == live_on == {"XOM": True, "JPM": True}


class TestReadExistingPredictions:
    """_read_existing_predictions — S3 read with clean 404 handling."""

    def test_returns_empty_on_nosuchkey(self):
        from botocore.exceptions import ClientError
        err = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "GetObject"
        )
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = err
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = _read_existing_predictions("bucket", "2026-04-20")
        assert result == []

    def test_reraises_on_other_client_errors(self):
        from botocore.exceptions import ClientError
        err = ClientError(
            {"Error": {"Code": "InternalError", "Message": "boom"}}, "GetObject"
        )
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = err
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            with pytest.raises(ClientError):
                _read_existing_predictions("bucket", "2026-04-20")

    def test_returns_predictions_on_success(self):
        payload = json.dumps({
            "date": "2026-04-20",
            "predictions": [
                {"ticker": "AAPL", "predicted_alpha": 0.02},
                {"ticker": "MSFT", "predicted_alpha": -0.01},
            ],
        }).encode()
        mock_body = MagicMock()
        mock_body.read.return_value = payload
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": mock_body}
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = _read_existing_predictions("bucket", "2026-04-20")
        assert len(result) == 2
        assert [p["ticker"] for p in result] == ["AAPL", "MSFT"]


class TestSubstrateInventoryKeys:
    """`predictor_decisions` row in nousergon_lib.transparency_inventory
    asserts l1_ic, l2_ic, confidence_calibration keys are present in
    predictor/metrics/latest.json. These tests pin the contract."""

    def _ctx(self):
        from inference.pipeline import PipelineContext
        return PipelineContext(
            date_str="2026-05-09",
            bucket="bucket",
            dry_run=True,
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={},
            explicit_tickers=[],
            inference_mode="meta",
        )

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "_load_gbm_meta")
    def test_meta_mode_emits_inventory_keys(self, mock_meta, mock_write, _v):
        mock_meta.return_value = {
            "trained_date": "2026-05-09",
            "promoted": True,
            "meta_val_ic": 0.132,
            "momentum_test_ic": 0.071,
            "volatility_test_ic": 0.043,
            "research_calibrator_n_samples": 4200,
            "isotonic_ece_before": 0.087,
            "isotonic_ece_after": 0.021,
            "isotonic_n_samples": 1500,
        }
        wo.run(self._ctx())

        assert mock_write.called
        metrics = mock_write.call_args.args[3]
        assert "l1_ic" in metrics
        assert "l2_ic" in metrics
        assert "confidence_calibration" in metrics
        assert metrics["l1_ic"] == {
            "momentum": 0.071,
            "volatility": 0.043,
            "research_calibrator": None,
        }
        assert metrics["l2_ic"] == 0.132
        assert metrics["confidence_calibration"] == {
            "method": "isotonic",
            "ece_before": 0.087,
            "ece_after": 0.021,
            "n_samples": 1500,
        }

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "_load_gbm_meta")
    def test_meta_mode_keys_present_when_manifest_partial(self, mock_meta, mock_write, _v):
        # Older manifests written before isotonic_calibrator landed lack the
        # ECE fields. Substrate check asserts presence (not non-null), so the
        # keys must still appear with None values rather than be omitted.
        mock_meta.return_value = {
            "trained_date": "2026-04-15",
            "promoted": True,
            "meta_val_ic": 0.10,
            "momentum_test_ic": 0.05,
            "volatility_test_ic": 0.03,
        }
        wo.run(self._ctx())

        metrics = mock_write.call_args.args[3]
        assert "l1_ic" in metrics
        assert "l2_ic" in metrics
        assert "confidence_calibration" in metrics
        assert metrics["confidence_calibration"]["ece_before"] is None
        assert metrics["confidence_calibration"]["ece_after"] is None

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_non_meta_mode_does_not_emit_keys(self, _m, mock_write, _v):
        from inference.pipeline import PipelineContext
        ctx = PipelineContext(
            date_str="2026-05-09",
            bucket="bucket",
            dry_run=True,
            predictions=[],
            signals_data={},
            explicit_tickers=[],
            inference_mode="mse",
        )
        wo.run(ctx)
        metrics = mock_write.call_args.args[3]
        # Keys are scoped to meta mode — legacy modes don't have a Layer-2 IC
        # to report, so they should be absent rather than null.
        assert "l1_ic" not in metrics
        assert "l2_ic" not in metrics
        assert "confidence_calibration" not in metrics


class TestEmailGating:
    """Email is sent on exactly the terminal invocation — first run when
    no coverage-gap re-invoke is expected, or the supplemental re-invoke
    itself. Prevents duplicate morning briefings (two emails on 2026-04-24
    incident)."""

    def _ctx(self, *, explicit_tickers):
        from inference.pipeline import PipelineContext
        return PipelineContext(
            date_str="2026-04-24",
            bucket="bucket",
            dry_run=False,
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
                {"ticker": "B", "predicted_alpha": -0.01, "combined_rank": 2,
                 "predicted_direction": "DOWN", "prediction_confidence": 0.55},
            ],
            signals_data={},  # empty so the hard-fail coverage guard passes
            explicit_tickers=explicit_tickers,
        )

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "send_predictor_email", return_value=True)
    @patch("inference.coverage_check.compute_coverage_delta")
    def test_first_invocation_no_gap_sends(
        self, mock_delta, mock_send, _w, _m1, _m2,
    ):
        mock_delta.return_value = {
            "has_gap": False, "missing_count": 0, "missing_tickers": [],
        }
        wo.run(self._ctx(explicit_tickers=[]))
        assert mock_send.called

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "send_predictor_email", return_value=True)
    @patch("inference.coverage_check.compute_coverage_delta")
    def test_first_invocation_with_gap_defers(
        self, mock_delta, mock_send, _w, _m1, _m2,
    ):
        mock_delta.return_value = {
            "has_gap": True, "missing_count": 7,
            "missing_tickers": ["VLO", "LLY", "XEL", "BIIB", "ROST", "SNDK", "WDC"],
        }
        wo.run(self._ctx(explicit_tickers=[]))
        assert not mock_send.called

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "_read_existing_predictions", return_value=[])
    @patch.object(wo, "send_predictor_email", return_value=True)
    @patch("inference.coverage_check.compute_coverage_delta")
    def test_supplemental_invocation_always_sends(
        self, mock_delta, mock_send, _r, _w, _m1, _m2,
    ):
        # Even if the coverage-delta would report a gap, the supplemental
        # invocation is terminal — it must send the email regardless.
        mock_delta.return_value = {
            "has_gap": True, "missing_count": 1, "missing_tickers": ["ZZZ"],
        }
        wo.run(self._ctx(explicit_tickers=["VLO", "LLY"]))
        assert mock_send.called
        # And compute_coverage_delta should not be consulted on the
        # supplemental path (we already know we're terminal).
        assert not mock_delta.called

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "send_predictor_email", return_value=True)
    @patch("inference.coverage_check.compute_coverage_delta")
    def test_coverage_delta_failure_fails_open(
        self, mock_delta, mock_send, _w, _m1, _m2,
    ):
        mock_delta.side_effect = RuntimeError("S3 transient")
        wo.run(self._ctx(explicit_tickers=[]))
        # If we can't determine whether a re-invoke is coming, send the
        # email rather than risk silencing the morning briefing entirely.
        assert mock_send.called


# ── L234: effective-optimizer-params loader (email integration removed) ──
#
# config#856 slimmed the predictor email to a summary + console deep-link;
# the "Effective Optimizer Params" block (ROADMAP L234) no longer renders
# inline in the email — it's console-page territory now (see
# _build_predictor_email's docstring). The S3 loader itself is still a
# working best-effort utility (kept for a future console surfacing), so it
# is still covered directly here.


class TestLoadExecutorParamsForEmailHelper:
    def test_load_helper_swallows_s3_errors(self):
        """`_load_executor_params_for_email` must not raise — secondary
        observability path; primary briefing path must survive S3
        outage.
        """
        with patch("boto3.client", side_effect=RuntimeError("S3 down")):
            result = wo._load_executor_params_for_email("b")
        assert result is None

    def test_bucket_param_no_longer_surfaces_block_in_email(self):
        """Regression: the email must NOT inline the Effective Optimizer
        Params block even when a bucket + fake artifact are supplied — that
        content moved to the console page (config#856)."""
        predictions = [
            {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
             "predicted_direction": "UP", "prediction_confidence": 0.7,
             "p_up": 0.65, "p_down": 0.35},
        ]
        metrics = {"model_version": "v1", "ic_30d": 0.10, "inference_mode": "meta"}
        fake_params = {"min_score": 65.0, "max_position_pct": 0.10}
        with patch.object(wo, "_load_executor_params_for_email", return_value=fake_params):
            _subject, html, plain = wo._build_predictor_email(
                predictions, metrics, "2026-05-22",
                signals_data=None, veto_threshold=0.6, bucket="b",
            )
        assert "Effective Optimizer Params" not in html
        assert "EFFECTIVE OPTIMIZER PARAMS" not in plain


# ── config#2333: S3 write fail-loud on the primary dated artifact ────────
#
# The dated predictions/{date}.json key is the canonical prediction set the
# step function's CheckPredictorCoverage state and the executor both key
# off of. A silent S3 put failure there used to leave stale data with no
# signal to the pipeline. This write is now fail-loud (raises
# PipelineHardFail so the SF Catch fires); latest.json + metrics/latest.json
# remain best-effort but alert on failure instead of log-only.


class TestS3WriteFailLoud:
    """Test fail-loud on S3 dated-key failures, best-effort on secondary writes."""

    def setup_method(self):
        """Reset cache state before each test."""
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    @patch("inference.stages.write_output._s3_put_json")
    def test_dated_key_failure_raises_hardfall(self, mock_put):
        """Dated predictions write failure must raise PipelineHardFail."""
        from inference.pipeline import PipelineHardFail
        # First call (dated) fails, later calls not reached
        mock_put.side_effect = Exception("S3 connection error")
        predictions = [{"ticker": "AAPL", "prediction_confidence": 0.75}]
        with pytest.raises(PipelineHardFail):
            wo.write_predictions(predictions, "2026-04-08", "bucket", {"model_version": "v1"})

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    @patch("inference.stages.write_output._s3_put_json")
    @patch("ops_alerts.publish_ops_alert")
    def test_secondary_writes_failure_alerts_not_raises(self, mock_alert, mock_put):
        """Latest.json / metrics failures must alert but not raise."""
        predictions = [{"ticker": "AAPL", "prediction_confidence": 0.75}]
        # Dated succeeds, latest/metrics fail
        mock_put.side_effect = [
            None,  # dated succeeds
            Exception("IAM denied"),  # latest fails
            Exception("S3 throttle"),  # metrics fails
        ]
        # Should not raise
        wo.write_predictions(predictions, "2026-04-08", "bucket", {"model_version": "v1"})
        # Verify ops alerts were published for the two secondary failures
        assert mock_alert.call_count >= 2
        for _, kwargs in mock_alert.call_args_list:
            assert kwargs["severity"] == "warning"

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    @patch("inference.stages.write_output._s3_put_json")
    def test_all_writes_succeed_no_alerts(self, mock_put):
        """When all three writes succeed, no alerts."""
        mock_put.side_effect = [None, None, None]  # all succeed
        predictions = [{"ticker": "AAPL"}]
        with patch("ops_alerts.publish_ops_alert") as alert_spy:
            wo.write_predictions(predictions, "2026-04-08", "bucket", {})
            # No alert calls expected on full success
            alert_spy.assert_not_called()


# ── config#856: slim predictor email + console deep-link ────────────────


class TestPredictorReportUrl:
    def test_default_base(self):
        url = wo.predictor_report_url("2026-07-03")
        assert url.endswith(f"/{wo.PREDICTOR_SLUG}?date=2026-07-03")

    def test_custom_base_override(self):
        url = wo.predictor_report_url("2026-07-03", "https://stage.example.com/")
        assert url == f"https://stage.example.com/{wo.PREDICTOR_SLUG}?date=2026-07-03"


class TestSlimPredictorEmail:
    """The slim email is a summary + console deep-link (config#856) — the
    full per-ticker prediction table, research brief, and effective
    optimizer params live on the console Predictor page instead."""

    def _preds(self):
        return [
            {"ticker": "AAPL", "predicted_alpha": 0.02, "combined_rank": 1,
             "predicted_direction": "UP", "prediction_confidence": 0.72,
             "p_up": 0.7, "p_down": 0.3},
            {"ticker": "XOM", "predicted_alpha": -0.015, "combined_rank": 2,
             "predicted_direction": "DOWN", "prediction_confidence": 0.68,
             "p_up": 0.3, "p_down": 0.7},
        ]

    def test_subject_unchanged_shape(self):
        metrics = {"model_version": "v1", "ic_30d": 0.10, "inference_mode": "meta"}
        subject, _html, _plain = wo._build_predictor_email(
            self._preds(), metrics, "2026-07-03", veto_threshold=0.6,
        )
        assert "2026-07-03" in subject
        assert "1 UP / 1 DOWN" in subject

    def test_body_links_to_console_report(self):
        metrics = {"model_version": "v1", "ic_30d": 0.10, "inference_mode": "meta"}
        _subject, html, plain = wo._build_predictor_email(
            self._preds(), metrics, "2026-07-03", veto_threshold=0.6,
        )
        expected = wo.predictor_report_url("2026-07-03")
        assert expected in html
        assert expected in plain

    def test_summary_numbers_present(self):
        metrics = {"model_version": "meta-v3.2", "ic_30d": 0.1234, "inference_mode": "meta"}
        _subject, html, _plain = wo._build_predictor_email(
            self._preds(), metrics, "2026-07-03", veto_threshold=0.6,
        )
        assert "meta-v3.2" in html
        assert "0.1234" in html
        assert "UP / DOWN" in html

    def test_no_inline_prediction_table(self):
        """The slim email must NOT inline the per-ticker prediction table —
        that's console-page territory now."""
        metrics = {"model_version": "v1", "ic_30d": 0.10, "inference_mode": "meta"}
        _subject, html, _plain = wo._build_predictor_email(
            self._preds(), metrics, "2026-07-03", veto_threshold=0.6,
        )
        assert "AAPL" not in html
        assert "XOM" not in html
        assert "Res.Cal" not in html  # meta-model column header, no longer rendered

    def test_no_research_brief_inline(self):
        metrics = {"model_version": "v1", "ic_30d": 0.10, "inference_mode": "meta"}
        signals_data = {
            "market_regime": "bullish",
            "universe": [{"ticker": "AAPL", "score": 80}],
            "buy_candidates": [{"ticker": "AAPL", "score": 80, "signal": "ENTER"}],
        }
        _subject, html, _plain = wo._build_predictor_email(
            self._preds(), metrics, "2026-07-03",
            signals_data=signals_data, veto_threshold=0.6,
        )
        assert "Buy Candidates" not in html
        assert "Research Brief" not in html
        # Market regime is still surfaced in the summary (decision-relevant).
        assert "BULLISH" in html

    def test_unscored_buy_candidate_warning_still_pushes(self):
        """A red flag (actionable ticker with no GBM score) must still reach
        the inbox, not hide behind the console click."""
        metrics = {"model_version": "v1", "ic_30d": 0.10, "inference_mode": "meta"}
        signals_data = {
            "market_regime": "neutral",
            "universe": [],
            "buy_candidates": [{"ticker": "ZZZZ", "score": 90, "signal": "ENTER"}],
        }
        _subject, html, plain = wo._build_predictor_email(
            self._preds(), metrics, "2026-07-03",
            signals_data=signals_data, veto_threshold=0.6,
        )
        assert "DATA WARNING" in html
        assert "ZZZZ" in html
        assert "DATA WARNING" in plain
        assert "ZZZZ" in plain

    def test_veto_count_in_subject_and_summary(self):
        metrics = {"model_version": "v1", "ic_30d": 0.10, "inference_mode": "meta"}
        # Veto count is single-sourced from the authoritative gbm_veto boolean
        # (config#1815) — the only veto the executor acts on — not a display
        # heuristic. Ticker A is the vetoed name.
        preds = [
            {"ticker": "A", "predicted_alpha": -0.01, "combined_rank": 2,
             "predicted_direction": "DOWN", "prediction_confidence": 0.8,
             "gbm_veto": True},
            {"ticker": "B", "predicted_alpha": 0.01, "combined_rank": 1,
             "predicted_direction": "UP", "prediction_confidence": 0.6,
             "gbm_veto": False},
        ]
        subject, html, _plain = wo._build_predictor_email(
            preds, metrics, "2026-07-03", veto_threshold=0.6,
        )
        assert "1 veto" in subject
        assert "Vetoes" in html


class TestSendPredictorEmailConsoleBaseUrl:
    def test_console_base_url_passed_through_to_builder(self):
        captured = {}

        def _fake_build(*args, **kwargs):
            captured.update(kwargs)
            return "subj", "<html></html>", "plain"

        with patch.object(wo, "_build_predictor_email", side_effect=_fake_build), \
                patch("krepis.email_sender.send_email", return_value=True), \
                patch.object(wo.cfg, "EMAIL_SENDER", "s@x"), \
                patch.object(wo.cfg, "EMAIL_RECIPIENTS", ["o@x"]):
            wo.send_predictor_email(
                [], {"model_version": "v1"}, "2026-07-03",
                console_base_url="https://stage.example.com",
            )
        assert captured.get("console_base_url") == "https://stage.example.com"


# ── alpha-engine-config-I9019: relative dispersion gate + n_high_confidence ────
# Fixtures below are the MEASURED numbers from the issue's table
# (s3://alpha-engine-research/predictor/predictions/{date}.json ->
# output_distribution_gate.metrics, plus n_high_confidence), 2026-08-19..08-28.

class TestRelativeDispersionCheck:
    """_relative_dispersion_check — pure function, no S3."""

    PRE_PROMOTION_ALPHA_STDEV = [0.028204, 0.040751, 0.043427]  # 08-19, 08-20, 08-21

    def test_guard_red_without_the_fix(self):
        """Champion-challenger policy 7.4: prove the PRE-FIX gate (the absolute-
        floor-only `validate_live_batch_invariant_health`, with no relative check)
        reports passed=True on the exact 2026-08-24 collapse this PR closes. This
        is the gap alpha-engine-config-I9019 measured live: 5 consecutive
        passed=True days while alpha_stdev sat ~6x below its pre-promotion level.
        """
        # Alternating +/- 0.010437 over n=30 rows: population stdev (ddof=0)
        # is exactly 0.010437 — the measured 2026-08-24 alpha_stdev.
        predicted_alphas = [0.010437 if i % 2 == 0 else -0.010437 for i in range(30)]
        preds = [{"predicted_alpha": a} for a in predicted_alphas]
        old_result = validate_live_batch_invariant_health(preds)
        assert old_result.passed is True  # RED — the absolute floors alone miss this
        assert old_result.metrics["alpha_stdev"] == pytest.approx(0.010437, abs=1e-6)

    def test_2026_08_21_batch_passes_relative_check(self):
        # 08-21's alpha_stdev (0.043427) against the two trading days before it
        # (0.028204, 0.040751): median 0.034478, ratio ~1.26x — comfortably
        # above the 0.5x compression floor.
        result = _relative_dispersion_check(
            [0.028204, 0.040751], 0.043427, statistic="alpha_stdev", min_history_days=2,
        )
        assert result["applicable"] is True
        assert result["passed"] is True
        assert result["ratio"] > 1.0

    def test_2026_08_24_batch_refused_by_relative_check(self):
        # 08-24's alpha_stdev (0.010437) against the pre-promotion trailing
        # history (0.028204, 0.040751, 0.043427): median 0.040751,
        # ratio ~0.256x — the measured ~6x compression from
        # alpha-engine-config-I9019, well below the 0.5x floor -> FAILS.
        result = _relative_dispersion_check(
            self.PRE_PROMOTION_ALPHA_STDEV, 0.010437,
            statistic="alpha_stdev", min_history_days=3,
        )
        assert result["applicable"] is True
        assert result["passed"] is False
        assert result["ratio"] == pytest.approx(0.010437 / 0.040751, rel=1e-3)
        assert "compression floor" in result["reason"]

    def test_stdev_p_up_observe_only_check_also_refuses_08_24(self):
        # Same shape, on stdev_p_up (0.191162, 0.216746, 0.152755 -> 0.060170):
        # this is the OBSERVE-ONLY sibling check — callers decide not to gate on
        # it (recalibration-invariance, config#1373); the function itself has no
        # opinion about blocking.
        result = _relative_dispersion_check(
            [0.191162, 0.216746, 0.152755], 0.060170,
            statistic="stdev_p_up", min_history_days=3,
        )
        assert result["applicable"] is True
        assert result["passed"] is False

    def test_insufficient_history_does_not_fire(self):
        result = _relative_dispersion_check([0.04], 0.01, statistic="alpha_stdev")
        assert result["applicable"] is False
        assert result["passed"] is True

    def test_missing_today_value_does_not_fire(self):
        result = _relative_dispersion_check(
            [0.04, 0.03, 0.05, 0.04, 0.04], None, statistic="alpha_stdev",
        )
        assert result["applicable"] is False
        assert result["passed"] is True

    def test_zero_median_does_not_fire(self):
        result = _relative_dispersion_check(
            [0.0, 0.0, 0.0, 0.0, 0.0], 0.0, statistic="alpha_stdev",
        )
        assert result["applicable"] is False
        assert result["passed"] is True

    def test_exactly_at_floor_passes(self):
        result = _relative_dispersion_check(
            [0.10, 0.10, 0.10, 0.10, 0.10], 0.05, statistic="alpha_stdev",
        )
        assert result["applicable"] is True
        assert result["passed"] is True
        assert result["ratio"] == pytest.approx(0.5)

    def test_just_below_floor_fails(self):
        result = _relative_dispersion_check(
            [0.10, 0.10, 0.10, 0.10, 0.10], 0.0499, statistic="alpha_stdev",
        )
        assert result["applicable"] is True
        assert result["passed"] is False

    def test_non_numeric_history_entries_are_dropped(self):
        result = _relative_dispersion_check(
            [0.04, None, "bad", 0.05, 0.04, 0.04], 0.02, statistic="alpha_stdev",
        )
        assert result["history_n"] == 4


class TestNHighConfidenceZeroStreak:
    """_n_high_confidence_zero_streak — pure function, no S3."""

    def test_measured_five_day_streak(self):
        # today=2026-08-28 (0); history most-recent-first: 08-27..08-24 (all 0),
        # then 08-21 (5) breaks the streak — the measured 5-day zero-streak.
        streak = _n_high_confidence_zero_streak(0, [0, 0, 0, 0, 5, 5, 2])
        assert streak == 5

    def test_no_streak_when_today_nonzero(self):
        assert _n_high_confidence_zero_streak(2, [0, 0, 0]) == 0

    def test_streak_stops_at_missing_day(self):
        # today=0 (streak=1), yesterday=0 (streak=2), day before is missing -> stop.
        assert _n_high_confidence_zero_streak(0, [0, None, 0]) == 2

    def test_streak_stops_at_nonzero_day(self):
        assert _n_high_confidence_zero_streak(0, [0, 5, 0]) == 2

    def test_streak_includes_today_alone_when_no_history(self):
        assert _n_high_confidence_zero_streak(0, []) == 1


class TestLoadTrailingBatchHistory:
    """_load_trailing_batch_history — best-effort S3 read + fast circuit breaker."""

    def test_extracts_metrics_from_prior_batches(self):
        def make_response(alpha_stdev, stdev_p_up, n_high_confidence):
            payload = json.dumps({
                "output_distribution_gate": {
                    "metrics": {"alpha_stdev": alpha_stdev, "stdev_p_up": stdev_p_up},
                },
                "n_high_confidence": n_high_confidence,
            }).encode()
            body = MagicMock()
            body.read.return_value = payload
            return {"Body": body}

        from botocore.exceptions import ClientError

        def get_object(Bucket, Key):
            if Key == "predictor/predictions/2026-08-21.json":
                return make_response(0.043427, 0.191162, 5)
            if Key == "predictor/predictions/2026-08-20.json":
                return make_response(0.040751, 0.216746, 5)
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "x"}}, "GetObject")

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = get_object

        history = _load_trailing_batch_history(
            mock_s3, "bucket", "2026-08-24", lookback_calendar_days=10, max_days=10,
        )
        by_date = {h["date"]: h for h in history}
        assert by_date["2026-08-21"]["alpha_stdev"] == pytest.approx(0.043427)
        assert by_date["2026-08-21"]["stdev_p_up"] == pytest.approx(0.191162)
        assert by_date["2026-08-20"]["n_high_confidence"] == 5

    def test_nosuchkey_does_not_trip_the_circuit_breaker(self):
        from botocore.exceptions import ClientError
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "x"}}, "GetObject",
        )
        history = _load_trailing_batch_history(
            mock_s3, "bucket", "2026-08-24", lookback_calendar_days=10, max_days=10,
        )
        assert history == []
        assert mock_s3.get_object.call_count == 10  # tried every day, never broke early

    def test_circuit_breaker_aborts_fast_on_real_errors(self):
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = RuntimeError("connection refused")
        history = _load_trailing_batch_history(
            mock_s3, "bucket", "2026-08-24", lookback_calendar_days=20, max_days=10,
        )
        assert history == []
        assert mock_s3.get_object.call_count == wo._MAX_CONSECUTIVE_S3_FAILURES

    def test_a_success_resets_the_failure_counter(self):
        from botocore.exceptions import ClientError

        def make_response():
            payload = json.dumps({
                "output_distribution_gate": {"metrics": {"alpha_stdev": 0.04, "stdev_p_up": 0.2}},
                "n_high_confidence": 3,
            }).encode()
            body = MagicMock()
            body.read.return_value = payload
            return {"Body": body}

        calls = {"n": 0}

        def get_object(Bucket, Key):
            calls["n"] += 1
            # alternating fail/succeed: never 2 consecutive failures, so a
            # 2-failure circuit breaker must never trip.
            if calls["n"] % 2 == 0:
                return make_response()
            raise RuntimeError("transient")

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = get_object
        history = _load_trailing_batch_history(
            mock_s3, "bucket", "2026-08-24", lookback_calendar_days=8, max_days=10,
        )
        assert len(history) == 4  # calls 2, 4, 6, 8 all succeed; breaker never trips


class TestWritePredictionsRelativeDispersionGate:
    """Integration: write_predictions refuses the write when the relative-
    dispersion check fires (alpha-engine-config-I9019). Champion-challenger
    policy 7.4's guard-red-before-fix proof is
    TestRelativeDispersionCheck.test_guard_red_without_the_fix above.
    """

    @staticmethod
    def _history_response(alpha_stdev, stdev_p_up, n_high_confidence=5):
        payload = json.dumps({
            "output_distribution_gate": {
                "metrics": {"alpha_stdev": alpha_stdev, "stdev_p_up": stdev_p_up},
            },
            "n_high_confidence": n_high_confidence,
        }).encode()
        body = MagicMock()
        body.read.return_value = payload
        return {"Body": body}

    def _mock_s3_with_pre_promotion_history(self):
        from botocore.exceptions import ClientError
        history = {
            "2026-08-21": (0.043427, 0.191162),
            "2026-08-20": (0.040751, 0.216746),
            "2026-08-19": (0.028204, 0.152755),
        }

        def get_object(Bucket, Key):
            for d, (a, p) in history.items():
                if Key == f"predictor/predictions/{d}.json":
                    return self._history_response(a, p)
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "x"}}, "GetObject")

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = get_object
        return mock_s3

    @staticmethod
    def _predictions_with_alpha_stdev(target_stdev, n=30):
        # Alternating +/- target_stdev over n rows: population stdev (ddof=0,
        # what np.std / the gate use) is exactly target_stdev.
        return [
            {"ticker": f"T{i}", "predicted_alpha": target_stdev if i % 2 == 0 else -target_stdev}
            for i in range(n)
        ]

    def test_2026_08_24_collapse_is_refused(self, monkeypatch):
        monkeypatch.setattr(wo, "DISPERSION_HISTORY_MIN_DAYS", 3)
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = self._mock_s3_with_pre_promotion_history()
        predictions = self._predictions_with_alpha_stdev(0.010437)  # measured 08-24 alpha_stdev
        with patch.dict("sys.modules", {"boto3": mock_boto3}), \
                patch.object(wo.cfg, "OUTPUT_DISTRIBUTION_GATE_INFERENCE_BLOCKING", True, create=True):
            with pytest.raises(RuntimeError, match="alpha-engine-config-I9019"):
                write_predictions(predictions, "2026-08-24", "bucket", {})

    def test_2026_08_21_batch_is_not_refused(self, monkeypatch):
        monkeypatch.setattr(wo, "DISPERSION_HISTORY_MIN_DAYS", 3)
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = self._mock_s3_with_pre_promotion_history()
        predictions = self._predictions_with_alpha_stdev(0.043427)  # measured 08-21 alpha_stdev
        with patch.dict("sys.modules", {"boto3": mock_boto3}), \
                patch.object(wo.cfg, "OUTPUT_DISTRIBUTION_GATE_INFERENCE_BLOCKING", True, create=True), \
                patch("inference.stages.write_output._s3_put_json"):
            # date after the mocked history so 08-21/20/19 are all strictly trailing.
            write_predictions(predictions, "2026-08-22", "bucket", {"model_version": "v1"})

    def test_relative_dispersion_metrics_recorded_even_when_gate_not_blocking(self, monkeypatch, capsys):
        # blocking=False (the observed default) must still SEE the relative
        # check in the forensic trail — measurability (principle 7) doesn't
        # depend on the enforcement flag.
        monkeypatch.setattr(wo, "DISPERSION_HISTORY_MIN_DAYS", 3)
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = self._mock_s3_with_pre_promotion_history()
        predictions = self._predictions_with_alpha_stdev(0.010437)
        with patch.dict("sys.modules", {"boto3": mock_boto3}), \
                patch.object(wo.cfg, "OUTPUT_DISTRIBUTION_GATE_INFERENCE_BLOCKING", False, create=True):
            write_predictions(predictions, "2026-08-24", "bucket", {}, dry_run=True)
        out = capsys.readouterr().out
        predictions_json = json.loads(
            out.split("=== PREDICTIONS (dry-run) ===\n")[1].split("\n=== METRICS")[0]
        )
        gate = predictions_json["output_distribution_gate"]
        assert gate["passed"] is False
        assert gate["failed_check"] == "alpha_stdev_relative_compression"
        assert gate["metrics"]["relative_dispersion"]["alpha_stdev"]["passed"] is False
