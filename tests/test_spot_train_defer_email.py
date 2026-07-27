"""Pins spot_train.sh to propagate PREDICTOR_DEFER_TRAINING_EMAIL and RUN_TOKEN
into the --full-only full-training workload env.

Brian's spec: the Saturday SF exports PREDICTOR_DEFER_TRAINING_EMAIL before
invoking spot_train.sh --full-only so the base champion-arch retrain defers its
per-run training email to the consolidated model-zoo digest. krepis 0.18.8+
requires --correlation-id (or $RUN_TOKEN) per fleet §116 rule 6 — the dashboard
box sets RUN_TOKEN via systemd, and spot_train.sh forwards it to every spot-side
heredoc. The full-training heredoc is single-quoted (cannot interpolate), so the
script computes export lines in the dispatcher and prepends them to the heredoc
body. This test catches a future edit that drops either propagation.
"""
from __future__ import annotations

from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "infrastructure" / "spot_train.sh"


def test_script_exists():
    assert _SCRIPT.is_file()


def test_computes_defer_export_from_dispatcher_env():
    text = _SCRIPT.read_text()
    # The dispatcher reads the var into a prependable export line.
    assert "PREDICTOR_DEFER_TRAINING_EMAIL" in text
    assert 'DEFER_EMAIL_EXPORT="export PREDICTOR_DEFER_TRAINING_EMAIL=' in text


def test_full_training_runssm_prepends_defer_export():
    text = _SCRIPT.read_text()
    # The full-training run_ssm body is prefixed with ${RUN_TOKEN_EXPORT}${DEFER_EMAIL_EXPORT}.
    assert 'run_ssm "full-training" "${RUN_TOKEN_EXPORT}${DEFER_EMAIL_EXPORT}${SHADOW_EXPORT}$(cat <<\'TRAIN\'' in text


def test_computes_run_token_export_from_dispatcher_env():
    text = _SCRIPT.read_text()
    # The dispatcher reads RUN_TOKEN into a prependable export line.
    assert "RUN_TOKEN" in text
    assert 'RUN_TOKEN_EXPORT="export RUN_TOKEN=' in text


def test_defer_export_empty_when_unset_keeps_bare_run_equivalent():
    # The else branch sets DEFER_EMAIL_EXPORT="" so a bare --full-only run (no
    # SF env) is byte-equivalent to before — no stray export injected.
    text = _SCRIPT.read_text()
    assert 'DEFER_EMAIL_EXPORT=""' in text
