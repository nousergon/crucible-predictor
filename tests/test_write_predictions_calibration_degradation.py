"""The artifact must say when the batch was not served by the calibrator.

alpha-engine-config-I9086. Before this, a variance-fallback firing was an ERROR
line in a Lambda log and nothing else. `predictions/{date}.json` carried no
field distinguishing a batch whose p_up/direction/confidence came from the
linear heuristic from one the calibrator produced, so the executor, the
portfolio optimizer, the dashboard and EOD all consumed a degraded batch as a
healthy one. A silent-degradation path that leaves no mark on its own output is
unobserved, not healthy (principle 7).
"""
from __future__ import annotations

import json
import os
import statistics
import sys

from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.stages.write_output import DISPERSION_ABSOLUTE_FLOOR_ALPHA_STDEV

# alpha-engine-config-I9267: derive the spread from the dispersion gate's own
# absolute-floor constant — see the identical comment in
# test_write_predictions_inference_gate.py, the same fixture shape rotted
# there too.
_ALPHA_SPREAD_TARGET = DISPERSION_ABSOLUTE_FLOOR_ALPHA_STDEV * 1.5


def _predictions(n: int = 27):
    p_ups = [0.10 + (i / (n - 1)) * 0.75 for i in range(n)]
    raw_stdev = statistics.pstdev(p - 0.5 for p in p_ups)
    scale = _ALPHA_SPREAD_TARGET / raw_stdev
    preds = []
    for i, p_up in enumerate(p_ups):
        preds.append({
            "ticker": f"T{i:02d}",
            "p_up": round(p_up, 4),
            "p_down": round(1 - p_up, 4),
            "p_flat": 0.0,
            "predicted_alpha": (p_up - 0.5) * scale,
            "predicted_direction": "UP" if p_up >= 0.5 else "DOWN",
            "prediction_confidence": round(abs(p_up - 0.5) * 2.0, 4),
        })
    return preds


_DEGRADED = {
    "degraded": True,
    "basis": "linear_heuristic_fallback",
    "reason": "calibrator_collapse",
    "n_unique_p_up_pre_fallback": 2,
    "threshold": 3,
    "batch_size": 30,
    "calibrator_method": "isotonic",
}


def _capture(**kwargs):
    from inference.stages.write_output import write_predictions
    bodies = {}

    def _fake_put(s3, bucket, key, body, **_):
        bodies[key] = body

    with patch("inference.stages.write_output._s3_put_json", side_effect=_fake_put):
        write_predictions(
            _predictions(), "2026-08-28", "test-bucket",
            {"model_version": "test"}, **kwargs,
        )
    return bodies


def _find(bodies, needle):
    for key, body in bodies.items():
        if needle in key:
            return json.loads(body) if isinstance(body, str) else body
    raise AssertionError(f"no artifact key containing {needle!r} in {list(bodies)}")


def test_degradation_block_reaches_the_predictions_artifact():
    bodies = _capture(calibration_degradation=_DEGRADED)
    art = _find(bodies, "2026-08-28")
    assert art["calibration_degradation"] == _DEGRADED


def test_degradation_block_reaches_the_metrics_artifact():
    """A monitor must be able to read the degradation without pulling the
    whole prediction batch."""
    bodies = _capture(calibration_degradation=_DEGRADED)
    metrics = _find(bodies, "metrics")
    assert metrics["calibration_degradation"] == _DEGRADED


def test_field_is_present_and_null_when_not_supplied():
    """Additive per the S3 schema contract: the key always exists, so a
    consumer can distinguish 'not degraded' from 'producer too old to say'."""
    bodies = _capture()
    art = _find(bodies, "2026-08-28")
    assert "calibration_degradation" in art
    assert art["calibration_degradation"] is None
