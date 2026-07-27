"""Regression: observe-only calibration failures reach an ALARMED surface.

config#1684 (ARCHITECTURE.md §61 experiment/observe fail-hard doctrine).

``_run_meta_inference`` emits five OBSERVE-ONLY parallel-calibration fields
that each ride as null on a per-ticker predict/calibrate exception:
``expected_move_macro_aug`` (Stage 1c), ``expected_move_risk_aug`` (Stage 2b),
``meta_alpha_tb`` (Stage 3), ``barrier_win_prob`` (Task B), and ``p_up_platt``
(Platt shadow, config#1176). Before this change every one silently DEBUG-logged
and moved on — so weeks of variant_cutover_gate / calibration-soak evidence
could quietly develop holes with nothing a human ever sees. That is precisely
the WARN/DEBUG-swallow posture §61 overturns.

These producers run BEFORE the canonical predictions persist, so §61 says they
must NOT hard-raise (an observe failure cannot abort the safety-critical
prediction run) — instead they take the §61 carve-out (b): route the failure to
a surface with a consumer (SNS + flow-doctor via ``ops_alerts``), deduped so a
recurring gap doesn't storm but a newly-failing field still pages.

Source-level assertions (mirrors the sibling observe-field test files, which
avoid standing up arcticdb + fitted models).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO = Path(__file__).parent.parent


def _src(rel_path: str) -> str:
    return (_REPO / rel_path).read_text()


class TestFailureTrackerRecordsEveryObserveSwallow:
    """Each of the five observe-only swallows must append to the shared
    per-run failure tracker — not just log-and-forget."""

    def test_tracker_initialized_before_loop(self):
        src = _src("inference/stages/run_inference.py")
        assert "_observe_calib_failures: dict[str, list[str]] = {}" in src

    def test_all_five_observe_fields_recorded(self):
        src = _src("inference/stages/run_inference.py")
        for field in (
            "expected_move_macro_aug",
            "expected_move_risk_aug",
            "meta_alpha_tb",
            "barrier_win_prob",
            "p_up_platt",
        ):
            assert (
                f'_observe_calib_failures.setdefault(\n'
                f'                    "{field}", []).append(ticker)' in src
                or f'_observe_calib_failures.setdefault(\n'
                   f'                        "{field}", []).append(ticker)' in src
            ), f"{field} observe swallow does not record into the alarm tracker"


class TestObserveGapsReachAlarmedSurface:

    def test_publishes_ops_alert_after_loop(self):
        src = _src("inference/stages/run_inference.py")
        # Routed through the dual-channel (SNS + flow-doctor) ops surface,
        # NOT a bare log — this is the §61 carve-out (b).
        assert "from ops_alerts import publish_ops_alert" in src
        assert "publish_ops_alert(" in src
        assert 'source="predictor.inference.observe_calibration"' in src

    def test_alert_is_deduped_by_failing_field_set(self):
        # A recurring gap on the same field(s) dedups within the cooldown
        # window; a NEW failing field changes the key and pages afresh.
        src = _src("inference/stages/run_inference.py")
        assert 'dedup_key="predictor-observe-calib-gaps-" + "-".join(_fields)' in src

    def test_alert_only_fires_when_failures_exist(self):
        src = _src("inference/stages/run_inference.py")
        assert "if _observe_calib_failures:" in src

    def test_alert_references_the_doctrine(self):
        src = _src("inference/stages/run_inference.py")
        assert "§61" in src and "config#1684" in src


class TestAlertSendFailureDoesNotAbortInference:
    """§61 sets the priority: the canonical predictions are already built by
    the time we alert. A failure to *send* the observe-evidence alert must
    degrade to a WARN, never propagate and abort the safety-critical run."""

    def test_publish_wrapped_and_downgrades_to_warning(self):
        src = _src("inference/stages/run_inference.py")
        alert_idx = src.find('from ops_alerts import publish_ops_alert')
        assert alert_idx != -1
        tail = src[alert_idx:alert_idx + 1400]
        assert "except Exception as _alert_exc:" in tail
        assert "log.warning(" in tail

    def test_alert_block_runs_after_predictions_built(self):
        # The alert must sit after the per-ticker loop appends predictions,
        # so it reports the full run's gaps and cannot short-circuit the
        # canonical prediction build.
        src = _src("inference/stages/run_inference.py")
        append_idx = src.find("ctx.predictions.append(result)")
        alert_idx = src.find("from ops_alerts import publish_ops_alert")
        assert append_idx != -1 and alert_idx != -1
        assert alert_idx > append_idx, (
            "observe-gap alert must fire AFTER the prediction loop, not inside it"
        )
