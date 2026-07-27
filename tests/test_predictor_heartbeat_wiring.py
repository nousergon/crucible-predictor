"""Wiring test: inference.handler emits a flow-doctor heartbeat at end-of-run.

config#646 (Option A: dedicated emit_heartbeat() call site). The predictor's
successful default `predict` path must call
``get_flow_doctor().emit_heartbeat(bucket=...)`` with the research bucket so the
dashboard System Health consumer can read
``s3://alpha-engine-research/_flow_doctor/heartbeat/predictor/{date}.json``.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def stubbed_preflight(monkeypatch):
    """Replace PredictorPreflight with a no-op so tests don't hit S3."""
    pf = MagicMock()
    pf.return_value.run = MagicMock()
    fake_module = MagicMock(PredictorPreflight=pf)
    monkeypatch.setitem(sys.modules, "inference.preflight", fake_module)
    return pf


def _fake_context(function_name="local", arn=""):
    ctx = MagicMock()
    ctx.function_name = function_name
    ctx.invoked_function_arn = arn
    return ctx


def test_predict_success_emits_heartbeat_to_research_bucket(
    stubbed_preflight, monkeypatch,
):
    """On a successful predict run, the handler must obtain the flow-doctor
    singleton via get_flow_doctor() and call emit_heartbeat with the
    alpha-engine-research bucket (config#646)."""
    fake_daily = MagicMock()
    fake_daily.main = MagicMock()
    monkeypatch.setitem(sys.modules, "inference.daily_predict", fake_daily)

    import inference.handler as h

    fake_fd = MagicMock()
    # The handler does a function-local `from krepis.logging import
    # get_flow_doctor`, so patch at the source module (the name is re-resolved
    # from krepis.logging at call time, not off the handler module).
    with patch("krepis.logging.get_flow_doctor", return_value=fake_fd):
        result = h.handler({"date": "2026-04-15"}, _fake_context())

    assert result["statusCode"] == 200
    # Heartbeat written exactly once, to the research bucket.
    fake_fd.emit_heartbeat.assert_called_once()
    call = fake_fd.emit_heartbeat.call_args
    assert call.kwargs.get("bucket") == "alpha-engine-research"


def test_heartbeat_skipped_when_flow_doctor_unavailable(
    stubbed_preflight, monkeypatch,
):
    """get_flow_doctor() returning None must not blow up the run — the call is
    guarded by `if fd:` (get_flow_doctor returns None when flow-doctor is off)."""
    fake_daily = MagicMock()
    fake_daily.main = MagicMock()
    monkeypatch.setitem(sys.modules, "inference.daily_predict", fake_daily)

    import inference.handler as h

    with patch("krepis.logging.get_flow_doctor", return_value=None):
        result = h.handler({"date": "2026-04-15"}, _fake_context())

    assert result["statusCode"] == 200
