"""config#1815 — α-derived binary direction, veto single-sourcing, IC header,
and rolling-metrics carry-forward.

Incident (2026-07-06): the morning brief read "2 UP / 26 DOWN | 14 vetoes"
while the model's predicted alphas split 12 positive / 16 negative and the
executor vetoed nothing. Three defects, each pinned here:

1. ``predicted_direction`` argmaxed the isotonic p_up, whose 0.5-crossing
   sits away from α=0 under the asymmetric 21d market-relative base rate
   (~60% of names underperform SPY) — positive-α names rendered DOWN and a
   ~3pt uniform macro shift flipped the whole headline (24:2 → 2:26).
   Fix: direction = sign(predicted alpha) via the single
   :func:`model.calibrator.derive_direction` helper at every derivation site.
2. THREE divergent veto definitions (subject count: α+rank; row tag: α+rank;
   badge: DOWN+confidence) vs the authoritative ``gbm_veto`` boolean
   (α+rank+confidence floor) the executor actually acts on. The email claimed
   14 executor overrides that never happened. Fix: every email surface reads
   ``gbm_veto``.
3. The daily inference metrics rebuild hardcoded ``hit_rate_30d_rolling=None``
   (+ meta-mode ic_30d/ic_ir_30d None), clobbering the backtester's weekly
   rolling patch — the IC header rendered "—" for months. Fix: carry-forward.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import inference.stages.write_output as wo
from inference.stages.write_output import _build_predictor_email, _load_rolling_metrics
from model.calibrator import PlattCalibrator, derive_direction


# ── 1. derive_direction: the single source of truth ─────────────────────────

class TestDeriveDirection:
    @pytest.mark.parametrize("alpha,expected", [
        (0.0113, "UP"),
        (1e-9, "UP"),
        (-0.0004, "DOWN"),
        (-0.0128, "DOWN"),
        (0.0, "DOWN"),     # no opinion is not a buy thesis
        (None, "DOWN"),
    ])
    def test_sign_semantics(self, alpha, expected):
        assert derive_direction(alpha) == expected

    def test_binary_only_no_flat(self):
        for a in np.linspace(-0.15, 0.15, 61):
            assert derive_direction(float(a)) in ("UP", "DOWN")


class TestCalibratorDirectionDecoupledFromBaseRate:
    """The incident shape: an isotonic calibrator that honestly maps small
    POSITIVE alphas below p_up=0.5 (bearish base rate) must no longer drag
    predicted_direction to DOWN."""

    def _bearish_calibrator(self) -> PlattCalibrator:
        cal = PlattCalibrator(method="isotonic")
        rng = np.random.default_rng(7)
        alphas = rng.uniform(-0.10, 0.10, 400)
        # Realized up-rate ~35% even for mildly positive alphas — mirrors the
        # 21d market-relative frame where most names underperform SPY.
        p_true = np.clip(0.35 + 1.5 * alphas, 0.02, 0.98)
        outcomes = (rng.uniform(size=400) < p_true).astype(int)
        cal.fit(alphas, outcomes)
        return cal

    def test_positive_alpha_is_up_even_when_p_up_below_half(self):
        cal = self._bearish_calibrator()
        out = cal.calibrate_prediction(0.0071, label_clip=0.15)  # GE 2026-07-06
        assert out["p_up"] < 0.5, "fixture must reproduce the base-rate shift"
        assert out["predicted_direction"] == "UP"

    def test_negative_alpha_is_down(self):
        cal = self._bearish_calibrator()
        out = cal.calibrate_prediction(-0.0128, label_clip=0.15)
        assert out["predicted_direction"] == "DOWN"

    def test_p_up_keeps_honest_base_rate_semantics(self):
        """p_up itself must NOT be redefined — only direction decouples."""
        cal = self._bearish_calibrator()
        out = cal.calibrate_prediction(0.0071, label_clip=0.15)
        assert 0.0 < out["p_up"] < 0.5
        assert out["p_flat"] == 0.0

    def test_unfitted_linear_fallback_uses_alpha_sign(self):
        cal = PlattCalibrator(method="isotonic")
        assert cal.calibrate_prediction(0.001)["predicted_direction"] == "UP"
        assert cal.calibrate_prediction(-0.001)["predicted_direction"] == "DOWN"


# ── 2. Email: every veto surface reads the authoritative gbm_veto ───────────

def _pred(ticker, alpha, rank, conf, gbm_veto, direction=None):
    return {
        "ticker": ticker,
        "predicted_alpha": alpha,
        "combined_rank": rank,
        "prediction_confidence": conf,
        "predicted_direction": direction or derive_direction(alpha),
        "gbm_veto": gbm_veto,
    }


class TestEmailVetoSingleSourcing:
    def _fixture(self):
        """4 names where every LEGACY display rule diverges from gbm_veto:
        - LOWCNF: negative α + bottom-half rank, but low confidence →
          legacy subject/row rules counted it; gbm_veto=False (the 2026-07-06
          all-28 shape).
        - BLOCKD: gbm_veto=True.
        - HIGHCONF-DOWN: DOWN + confidence ≥ threshold but positive rank-half →
          legacy badge rule showed VETO; gbm_veto=False.
        """
        return [
            _pred("WINNER", 0.011, 1, 0.11, False),
            _pred("HCONF", -0.001, 2, 0.90, False),   # legacy badge-rule veto
            _pred("LOWCNF", -0.0004, 3, 0.04, False),  # legacy subject-rule veto
            _pred("BLOCKD", -0.012, 4, 0.80, True),
        ]

    def _email(self, preds, metrics=None):
        with patch.object(wo, "_load_executor_params_for_email", return_value=None):
            return _build_predictor_email(
                preds, metrics or {"model_version": "meta-v3.0", "inference_mode": "meta"},
                "2026-07-06", signals_data=None, veto_threshold=0.65,
            )

    def test_subject_counts_gbm_veto_only(self):
        subject, _, _ = self._email(self._fixture())
        assert "1 veto" in subject and "vetoes" not in subject

    def test_subject_direction_counts_follow_alpha_sign(self):
        subject, _, _ = self._email(self._fixture())
        assert "1 UP / 3 DOWN" in subject

    def test_slim_email_has_no_per_ticker_veto_display(self):
        # config#856 slimmed this email: the per-ticker prediction table (and
        # its per-row gbm_veto badge) moved to the console Predictor page. The
        # authoritative per-ticker veto-single-sourcing invariant from
        # config#1815 is now enforced ON THE CONSOLE PAGE (crucible-dashboard
        # views/7_Predictor.py renders a `Veto` column single-sourced from the
        # same gbm_veto boolean, guarded by that repo's test). The email keeps
        # only the aggregate veto count, whose single-sourcing is guarded by
        # `test_subject_counts_gbm_veto_only` above — so no divergent display
        # rule can reappear in the email itself.
        _, html, _ = self._email(self._fixture())
        for t in ("BLOCKD", "HCONF", "LOWCNF", "WINNER"):
            assert t not in html
        assert "⚠ VETO" not in html  # no per-row veto markup inlined anymore

    def test_zero_vetoes_when_gbm_veto_all_false(self):
        preds = [p | {"gbm_veto": False} for p in self._fixture()]
        subject, html, _ = self._email(preds)
        assert "veto" not in subject.lower()
        for row in html.split("<tr>")[1:]:
            cell = row.split("</tr>")[0]
            assert "⚠ VETO" not in cell


class TestEmailIcHeader:
    def _email(self, metrics):
        with patch.object(wo, "_load_executor_params_for_email", return_value=None):
            return _build_predictor_email(
                [_pred("A", 0.01, 1, 0.1, False)], metrics, "2026-07-06",
            )

    def test_realized_ic_preferred(self):
        _, html, plain = self._email(
            {"model_version": "m", "inference_mode": "meta",
             "ic_30d": 0.1234, "meta_val_ic": 0.346}
        )
        assert "IC (30d)" in html and "0.1234" in html
        assert "IC (30d)" in plain

    def test_meta_val_ic_fallback(self):
        _, html, _ = self._email(
            {"model_version": "m", "inference_mode": "meta",
             "ic_30d": None, "meta_val_ic": 0.3461}
        )
        assert "IC (val)" in html and "0.3461" in html

    def test_dash_when_neither(self):
        _, html, _ = self._email({"model_version": "m", "inference_mode": "meta"})
        assert "—" in html


# ── 3. Rolling-metrics carry-forward ─────────────────────────────────────────

class TestRollingMetricsCarryForward:
    def test_carries_backtester_fields(self):
        existing = {
            "hit_rate_30d_rolling": 0.57, "ic_30d": 0.12, "ic_ir_30d": 0.8,
            "rolling_metrics_updated_at": "2026-07-05T12:00:00Z", "rolling_n": 42,
            "model_version": "stale-should-not-carry",
        }
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(existing).encode())
        }
        with patch("boto3.client", return_value=mock_s3):
            out = _load_rolling_metrics("bucket")
        assert out == {
            "hit_rate_30d_rolling": 0.57, "ic_30d": 0.12, "ic_ir_30d": 0.8,
            "rolling_metrics_updated_at": "2026-07-05T12:00:00Z", "rolling_n": 42,
        }
        assert "model_version" not in out

    def test_null_fields_not_carried(self):
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(
                {"hit_rate_30d_rolling": None, "ic_30d": 0.05}).encode())
        }
        with patch("boto3.client", return_value=mock_s3):
            out = _load_rolling_metrics("bucket")
        assert out == {"ic_30d": 0.05}

    def test_fetch_failure_warns_and_returns_empty(self, caplog):
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = RuntimeError("s3 down")
        with patch("boto3.client", return_value=mock_s3):
            with caplog.at_level("WARNING"):
                out = _load_rolling_metrics("bucket")
        assert out == {}
        assert any("carry-forward" in r.message for r in caplog.records)


# ── 4. L1 dead-component alert (training chokepoint) ─────────────────────────

class TestL1DeadComponentAlert:
    """The 2026-07-03 shape: momentum test IC 0.002 → wait, that is ABOVE the
    0.0 floor — the floor catches at/below zero (the OOS −0.011 shape). The
    alert fires on ≤ floor; the report card's 0.03-target RED remains the
    surface for weak-but-positive components. Pins: (a) a ≤-floor component
    lands in the manifest dict + fires log.error; (b) healthy components do
    not; (c) the alert never blocks promotion (alert-only by design)."""

    def _compute(self, mom_ic, vol_ic, floor=0.0):
        import config as cfg_mod
        return {
            name: round(ic, 6)
            for name, ic in (("momentum", mom_ic), ("volatility", vol_ic))
            if ic is not None and ic <= floor
        }

    def test_negative_ic_component_flagged(self):
        out = self._compute(-0.01092, 0.334)
        assert out == {"momentum": -0.01092}

    def test_zero_ic_component_flagged(self):
        assert "momentum" in self._compute(0.0, 0.334)

    def test_healthy_components_not_flagged(self):
        assert self._compute(0.002, 0.334) == {}

    def test_manifest_field_and_log_error_wired_in_trainer_source(self):
        """Text-invariant pin (mirrors the repo's wiring-test convention):
        the alert block must exist in meta_trainer between the completion
        log and the result dict, write the manifest field, use log.error
        (flow-doctor escalation surface), and reference the config floor."""
        from pathlib import Path
        src = (Path(__file__).parent.parent / "training" / "meta_trainer.py").read_text()
        assert "l1_components_below_ic_floor" in src
        assert "cfg.L1_COMPONENT_IC_ALERT_FLOOR" in src
        alert_pos = src.index("L1 component(s) at/below the IC alert floor")
        # log.error, not log.warning — must reach the flow-doctor dispatch tier
        block = src[alert_pos - 400: alert_pos + 100]
        assert "log.error" in block
        # Alert-only: the block must NOT touch the promotion decision
        ret_pos = src.index('"l1_components_below_ic_floor": l1_components_below_ic_floor')
        gate_region = src[alert_pos:ret_pos]
        assert "promoted =" not in gate_region, "alert must not mutate promotion"

    def test_config_floor_default_zero(self):
        import config as cfg_mod
        assert cfg_mod.L1_COMPONENT_IC_ALERT_FLOOR == 0.0
