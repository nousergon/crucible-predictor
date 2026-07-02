"""Pin the #902 bundled-DriftDetection adoption in spot_train.sh.

Origin: nousergon/alpha-engine-config#902 — after the 2026-04-16 spot
migration, DriftDetection launched its OWN ~7-min-bootstrap spot
(nousergon-data/infrastructure/spot_drift_detection.sh) for a ~5-min
workload — disproportionate cost. It reads the champion baseline that
PredictorTraining produces, so it is bundled onto the SAME training spot:
the full-training path runs ``monitoring.drift_detector`` after training
succeeds, then the spot terminates. The standalone SF DriftDetection state
is collapsed in the sibling nousergon-data PR (Part of #902).

These tests pin the load-bearing semantics so a future refactor cannot
silently break them:

  * BLOCKING training / NON-BLOCKING drift: the bundled drift step lives
    AFTER the full-training run_ssm (so ``set -euo pipefail`` halts the
    script on a training failure BEFORE drift is reachable — training
    failure still halts), and its non-zero exit is SWALLOWED at BOTH the
    spot layer (inner ``exit 0``) and the dispatcher layer (``|| echo …``)
    so a drift ALERT (drift_detector ``sys.exit(1)``) never fails the run.
  * THE GUARD (root-cause, not a bandaid): the bundled drift audits the
    BASE champion (training_feature_stats.json). That equals the FINAL
    champion ONLY while MODEL_ZOO_AUTO_PROMOTE_WINNER is False. The step
    reads that flag via config.py and SKIPS + warns (never silently audits
    the wrong baseline) when auto-promote is ON — the operator is told to
    migrate to placement B (bundle onto ModelZooSelect, post-promotion).
  * bash 3.2 safety: the guard's paren/apostrophe-bearing config read is
    isolated in its own /tmp/drift-guard.py heredoc; the bundled bash body
    stays free of command substitution.
"""
from __future__ import annotations

import re
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "infrastructure" / "spot_train.sh"


def _text() -> str:
    assert _SCRIPT.is_file(), f"spot_train.sh missing at {_SCRIPT}"
    return _SCRIPT.read_text()


def test_script_exists():
    assert _SCRIPT.is_file()


def test_bundled_drift_step_present():
    t = _text()
    assert 'run_ssm "drift"' in t
    assert "monitoring.drift_detector --alert" in t


def test_drift_runs_after_full_training_runssm():
    """Ordering: the bundled drift step must come AFTER the full-training
    run_ssm — so `set -euo pipefail` halts the whole script on a training
    failure BEFORE drift is ever reached (training failure still HALTS)."""
    t = _text()
    train_idx = t.index('run_ssm "full-training"')
    drift_idx = t.index('run_ssm "drift"')
    assert train_idx < drift_idx, "drift must be bundled AFTER full-training"


def test_drift_is_non_blocking_at_both_layers():
    """Drift ALERT (drift_detector sys.exit(1)) must never fail the run:
    swallowed at the spot layer (inner exit 0) AND the dispatcher layer."""
    t = _text()
    # Dispatcher layer: the drift run_ssm swallows non-zero.
    assert re.search(
        r'run_ssm "drift".*?\|\| echo "WARNING: bundled drift step returned non-zero',
        t,
        re.S,
    ), "dispatcher-layer drift swallow (|| echo …) missing"
    # Spot layer: the inner script sets +e and always exits 0 after the detector.
    inner = t[t.index("<<'INNER'") : t.index("\nINNER\n")]
    assert "set +e" in inner
    assert inner.rstrip().endswith("exit 0"), "inner drift script must end with exit 0"


def test_guard_reads_auto_promote_flag_and_skips_when_on():
    """THE GUARD: read MODEL_ZOO_AUTO_PROMOTE_WINNER via config.py; when ON,
    SKIP the bundled run (never silently audit the wrong baseline) and point
    the operator at placement B."""
    t = _text()
    assert "config.MODEL_ZOO_AUTO_PROMOTE_WINNER" in t
    assert 'AUTOPROMOTE' in t
    # ON → skip + loud migrate warning.
    assert 'if [ "$AUTOPROMOTE" = "1" ]; then' in t
    assert "SKIPPING bundled drift" in t
    assert "placement B" in t and "ModelZooSelect" in t
    # config-read failure → skip fail-safe (never audit an unknown baseline).
    assert 'if [ "$AUTOPROMOTE" = "err" ]; then' in t


def test_guard_config_read_is_isolated_for_bash32_safety():
    """The guard's paren/apostrophe-bearing config read lives in its OWN
    /tmp/drift-guard.py heredoc; the bundled bash body uses `read` from a
    verdict file rather than $(...) so the enclosing $(cat <<'DRIFT' …)
    command substitution stays bash-3.2 paren-safe."""
    t = _text()
    assert "cat > /tmp/drift-guard.py <<'PYEOF'" in t
    # Extract the outer DRIFT heredoc body and assert it carries no bash-level
    # command substitution (the config read was moved into the nested python).
    body = t[t.index("<<'DRIFT'") : t.index("\nDRIFT\n")]
    # Strip the nested python heredoc (parens there are Python, and safe).
    body_wo_py = re.sub(r"<<'PYEOF'.*?\nPYEOF\n", "", body, flags=re.S)
    assert "$(" not in body_wo_py, "bash-level $(...) in DRIFT body breaks bash 3.2"


def test_drift_heartbeat_emitted():
    """The bundle preserves the drift-detection liveness heartbeat so the
    standalone launcher's Process=drift-detection signal is unbroken."""
    t = _text()
    assert 'Process=drift-detection' in t
