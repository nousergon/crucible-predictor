"""Dual-channel ops alerts for predictor (SNS + flow-doctor forum topics).

Migration arc: config#1740 T3 / config#1749 — retire raw ``telegram=True``
fan-out to Telegram General on model-zoo rotation alerts.
"""

from __future__ import annotations

import logging

from krepis.logging import get_flow_doctor

logger = logging.getLogger(__name__)


def _normalize_flow_doctor_severity(severity: str) -> str:
    normalized = severity.lower()
    if normalized == "warn":
        return "warning"
    return normalized


def publish_ops_alert(
    message: str,
    *,
    severity: str,
    source: str,
    dedup_key: str | None = None,
) -> None:
    """SNS via ``krepis.alerts.publish(telegram=False)`` + Telegram via flow-doctor."""
    from krepis.alerts import publish as alerts_publish

    alerts_publish(
        message=message,
        severity=severity,
        source=source,
        sns=True,
        telegram=False,
        dedup_key=dedup_key,
    )
    fd = get_flow_doctor()
    if fd is None:
        return
    try:
        subject = message.split("\n", 1)[0].replace("*", "").strip()
        if not subject:
            subject = f"Predictor alert [{severity.upper()}]"
        fd.notify_event(
            subject,
            body=message,
            severity=_normalize_flow_doctor_severity(severity),
            dedup_key=dedup_key or subject,
            context={"source": source},
        )
    except Exception as exc:
        logger.warning(
            "flow-doctor notify_event failed for ops alert (%s): %s — SNS already sent",
            source,
            exc,
        )
