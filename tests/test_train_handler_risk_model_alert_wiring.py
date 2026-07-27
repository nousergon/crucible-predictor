"""Source-level regression tests for config#2443: risk_model_persist failure
must be alertable, not silently swallowed.

Background: ``risk_model_persist`` (Step 2d2) is intentionally non-blocking —
an exception there must never abort the ~90-minute training run. Before this
fix, the ONLY trace of a failure was a ``log.warning(..., exc_info=True)``
line: the training summary gets no ``risk_model`` key, the S3 artifact
doesn't land, and the ARTIFACT_REGISTRY freshness monitor for these two
artifacts is a ``severity: warning`` row on a 10-calendar-day recency window
(config#1297) — it does not page on a single missed ``saturday_sf`` cycle. So
an entire week could pass with zero human-visible signal (config#2443's
2026-07-11 miss).

The fix routes the exception through ``ops_alerts.publish_ops_alert`` (the
same best-effort SNS+flow-doctor channel ``training/model_zoo.py``'s
rotation alerts already use), while keeping the block fully non-blocking:
alert-publish failure is itself caught and only WARN-logged.

Mirrors the source-level regression style of
``test_train_handler_triple_barrier_gate_wiring.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_REPO = Path(__file__).parent.parent


def _src(rel_path: str) -> str:
    return (_REPO / rel_path).read_text()


class TestTrainHandlerRiskModelAlertWiring:

    def test_failure_path_publishes_ops_alert(self):
        src = _src("training/train_handler.py")
        block_idx = src.find('"risk_model_persist failed (non-blocking): %s"')
        assert block_idx != -1
        from_ops_alerts_idx = src.find("from ops_alerts import publish_ops_alert", block_idx)
        publish_idx = src.find("publish_ops_alert(", block_idx)
        assert from_ops_alerts_idx != -1 and publish_idx != -1
        assert block_idx < from_ops_alerts_idx < publish_idx

    def test_alert_carries_a_per_date_dedup_key(self):
        src = _src("training/train_handler.py")
        assert 'dedup_key=f"risk_model_persist_failed_{date_str}"' in src

    def test_alert_severity_matches_registry_severity(self):
        # ARTIFACT_REGISTRY.yaml pins both risk_model artifacts at
        # severity: warning — keep the code-level alert in lockstep so a
        # future registry bump to `critical` is the obvious place to also
        # bump this literal (see alpha-engine-config#2443 disposition notes).
        src = _src("training/train_handler.py")
        block_idx = src.find("publish_ops_alert(")
        assert block_idx != -1
        window = src[block_idx:block_idx + 400]
        assert 'severity="warning"' in window

    def test_alert_publish_failure_does_not_escape(self):
        # The publish_ops_alert call must itself be wrapped so a flow-doctor/
        # SNS outage can never take down the (already non-blocking) training
        # run — mirrors every other publish_ops_alert call site in this repo
        # (training/model_zoo.py's four alert sites all use this posture).
        src = _src("training/train_handler.py")
        publish_idx = src.find("publish_ops_alert(")
        assert publish_idx != -1
        except_idx = src.find("except Exception:  # noqa: BLE001", publish_idx)
        assert except_idx != -1
        alert_fail_log_idx = src.find(
            "risk_model_persist: failure alert itself failed", except_idx,
        )
        assert alert_fail_log_idx != -1

    def test_alert_wiring_stays_inside_the_outer_non_blocking_except(self):
        # The new alert call must live INSIDE the existing
        # `except Exception as _rm_err:` handler for the risk_model_persist
        # block (not a sibling try at module scope) so it only ever fires
        # on an actual risk_model_persist failure, and its own failure can
        # never propagate out of that handler and abort training.
        src = _src("training/train_handler.py")
        outer_except_idx = src.find("except Exception as _rm_err:")
        assert outer_except_idx != -1
        publish_idx = src.find("publish_ops_alert(", outer_except_idx)
        gate_idx = src.find("Step 2e: Triple-barrier cutover gate", outer_except_idx)
        assert outer_except_idx != -1 and publish_idx != -1 and gate_idx != -1
        assert outer_except_idx < publish_idx < gate_idx

    def test_message_reports_date_and_non_blocking_posture(self):
        src = _src("training/train_handler.py")
        block_idx = src.find("publish_ops_alert(")
        window = src[block_idx:block_idx + 400]
        assert "date={date_str}" in window
        assert "non-blocking" in window
