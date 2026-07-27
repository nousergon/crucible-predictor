"""Tests for predictor ops_alerts flow-doctor routing (config#1749)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ops_alerts import publish_ops_alert


@patch("ops_alerts.get_flow_doctor", return_value=None)
def test_publish_ops_alert_sns_only_when_flow_doctor_inactive(_mock_fd):
    with patch("krepis.alerts.publish") as publish_mock:
        publish_ops_alert("promoted", severity="warning", source="test")
    publish_mock.assert_called_once()
    assert publish_mock.call_args.kwargs["telegram"] is False


@patch("ops_alerts.get_flow_doctor")
def test_publish_ops_alert_routes_telegram_via_flow_doctor(mock_get_fd):
    mock_fd = MagicMock()
    mock_fd.notify_event.return_value = "rid-1"
    mock_get_fd.return_value = mock_fd
    with patch("krepis.alerts.publish") as publish_mock:
        publish_ops_alert(
            "inert rotation",
            severity="warning",
            source="predictor:model_zoo",
            dedup_key="k1",
        )
    assert publish_mock.call_args.kwargs["telegram"] is False
    mock_fd.notify_event.assert_called_once()
    assert mock_fd.notify_event.call_args.kwargs["severity"] == "warning"
