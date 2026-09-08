"""The shadow artifact must record the calibration verdict for its own arm.

alpha-engine-config: on 2026-09-08 the variance fallback fired on three shadow
challengers and rewrote every served p_up from the linear heuristic. Verified
against the live artifacts: `predictions_shadow/spec-sota-combine-2026-07-24-
8578f8ae/2026-09-08.json` carries p_up values 0.4723 .. 0.5040, which is exactly
`0.5 + alpha/(2*0.05)` over that arm's alpha range — the heuristic output, not
the calibrator's. Yet `calibration_degradation` is absent from every shadow
payload, because `_write_shadow` copies only `predictions`.

So the fallback fired, the served numbers changed basis, and no durable artifact
recorded it: a detector that ran and told nobody. Any realized-edge or
calibration scoring over `predictions_shadow/` silently mixes the two bases.
"""

from __future__ import annotations

import json

import pytest

from inference.stages import shadow_versions as sv


class _Ctx:
    def __init__(self):
        self.date_str = "2026-09-08"
        self.bucket = "alpha-engine-research"
        self.predictions = [{"ticker": "AAA", "p_up": 0.4723, "predicted_alpha": -0.002775}]
        self.model_version = "meta-v3.0-8models"
        self.calibration_degradation = {
            "degraded": True, "basis": "linear_heuristic_fallback",
            "reason": "calibrator_collapse", "arm": "shadow",
            "shadow_version_id": "spec-sota-combine-2026-07-24-8578f8ae",
            "n_distinct_predicted_alpha": 6,
        }
        self.level_neutralization = {"enabled": True, "applied": True, "n_predictions": 29}


@pytest.fixture
def captured(monkeypatch):
    box = {}

    class _S3:
        def put_object(self, **kw):
            box["key"] = kw["Key"]
            box["payload"] = json.loads(kw["Body"].decode())

    class _Boto:
        @staticmethod
        def client(_name):
            return _S3()

    monkeypatch.setitem(__import__("sys").modules, "boto3", _Boto)
    return box


def test_shadow_payload_carries_the_calibration_verdict(captured):
    ctx = _Ctx()
    sv._write_shadow(ctx, "spec-sota-combine-2026-07-24-8578f8ae")
    deg = captured["payload"]["calibration_degradation"]
    assert deg["degraded"] is True
    assert deg["basis"] == "linear_heuristic_fallback"
    assert deg["shadow_version_id"] == "spec-sota-combine-2026-07-24-8578f8ae"
    assert deg["n_distinct_predicted_alpha"] == 6


def test_shadow_payload_carries_level_neutralization(captured):
    ctx = _Ctx()
    sv._write_shadow(ctx, "v")
    assert captured["payload"]["level_neutralization"]["applied"] is True


def test_the_field_is_present_even_when_nothing_degraded(captured):
    """An absent key is indistinguishable from a healthy one. It must be
    explicit, or a consumer reading `predictions_shadow/` cannot tell a
    calibrator-based p_up from a heuristic one."""
    ctx = _Ctx()
    ctx.calibration_degradation = None
    ctx.level_neutralization = None
    sv._write_shadow(ctx, "v")
    assert "calibration_degradation" in captured["payload"]
    assert captured["payload"]["calibration_degradation"] is None
    assert "level_neutralization" in captured["payload"]


def test_clone_stamps_the_version_id_on_the_shadow_context():
    """Without this the collapse record cannot name the arm it fired on."""
    from inference.pipeline import PipelineContext

    live = PipelineContext()
    live.date_str = "2026-09-08"
    shadow = sv._clone_for_shadow(
        live, weights_prefix="predictor/registry/vid/", version_id="vid",
    )
    assert shadow.shadow_version_id == "vid"
    assert live.shadow_version_id is None, "the live context must never be stamped"
