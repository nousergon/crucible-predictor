"""Pin the #1854 retirement of the #902 bundled-DriftDetection block in
spot_train.sh.

Origin: nousergon/alpha-engine-config#902 bundled DriftDetection onto the
training spot to avoid a disproportionate standalone ~7-min-bootstrap spot
for a ~5-min workload. That bundle's own guard made it permanently dead once
``MODEL_ZOO_AUTO_PROMOTE_WINNER`` went live (2026-06-13): the guard always
reads AUTOPROMOTE=1 and skips, so the block ran nothing but its own
SKIPPING-warning branch. Placement B (bundling onto post-promotion
ModelZooSelect) was never built. Daily prediction-health drift now runs via
a dedicated weekday producer (config#1853; re-homed off the SF onto its own
EventBridge trigger by alpha-engine-config-I2722) instead — the weekly
bundled fallback this file used to pin is retired dead code
(config#1854).

These tests pin the ABSENCE of the retired block so a future edit cannot
silently reintroduce it.
"""
from __future__ import annotations

from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "infrastructure" / "spot_train.sh"


def _text() -> str:
    assert _SCRIPT.is_file(), f"spot_train.sh missing at {_SCRIPT}"
    return _SCRIPT.read_text()


def test_script_exists():
    assert _SCRIPT.is_file()


def test_bundled_drift_step_absent():
    t = _text()
    assert 'run_ssm "drift"' not in t
    assert "monitoring.drift_detector --alert" not in t


def test_drift_guard_files_absent():
    """The guard's /tmp staging files (drift-guard.py, spot-drift.sh) must
    not be re-introduced — do NOT re-add the standalone spot_drift_detection.sh
    launcher either (removed intentionally by config#902; see config#1854)."""
    t = _text()
    assert "/tmp/drift-guard.py" not in t
    assert "/tmp/spot-drift.sh" not in t
    assert "spot_drift_detection.sh" not in t


def test_drift_bundle_skip_metrics_absent():
    """The always-skip CloudWatch markers must not be emitted — the guard
    that wrote them (and the block it guarded) is gone, not just dormant."""
    t = _text()
    assert "DriftBundleSkippedAutoPromote" not in t
    assert "DriftBundleSkippedConfigError" not in t


def test_drift_detection_heartbeat_absent():
    """The bundled-spot drift-detection liveness heartbeat is retired along
    with the block it monitored; the daily weekday producer (config#1853)
    is responsible for its own liveness signal if one is ever needed
    (explicitly out of scope for config#1854 — see its P3 note)."""
    t = _text()
    assert "Process=drift-detection" not in t


def test_training_heartbeat_still_present():
    """The retirement must not collateral-damage the unrelated
    predictor-training completion heartbeat right before the removed block."""
    t = _text()
    assert 'Process=predictor-training' in t
    assert "Training complete." in t
