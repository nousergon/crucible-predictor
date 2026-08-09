"""Structural + live-executed guards for the per-stage spot scripts split off
`spot_train.sh` (alpha-engine-config-I4497 / sf-pipeline-policy.md §2.1).

`spot_train.sh` stays the untouched rollback path; `_spot_common.sh` +
`spot_predictor_training.sh` / `spot_train_spec_dispatch.sh` /
`spot_model_zoo_select.sh` are the additive per-SF-state replacements
(crucible-predictor#436), one script per SF state (PredictorTraining /
TrainSpecDispatch / ModelZooSelect).

Regression class this file exists to catch: crucible-predictor#436 shipped
`spot_train_spec_dispatch.sh` and `spot_model_zoo_select.sh` treating an
unrecognized flag as a silent no-op WARNING rather than a hard failure —
which meant `--preflight-only` (appended by every nousergon-data SF command
via `$.preflight_args` on the Friday shell-run dry path, weekly-sf-policy.md
§7.1) was silently swallowed and the scripts fell through into REAL
training / REAL model-zoo selection+promotion instead of the read-only
dry probe. Fixed by (a) lifting the preflight-only probe into
`_spot_common.sh::maybe_run_preflight_only_and_exit()` — one place instead
of three — and (b) making every per-stage script's flag parser fail loud
(`exit 2`) on an unrecognized flag instead of warning-and-continuing.

Live-executed assertions here run a real `bash` subprocess but only through
code paths that exit BEFORE any spot_launch/AWS call (unknown-flag parse
failure, missing-required-arg failure) — verified safe: no AWS credentials
or network access required.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"
_COMMON = _INFRA / "_spot_common.sh"
_STAGE_SCRIPTS = [
    _INFRA / "spot_predictor_training.sh",
    _INFRA / "spot_train_spec_dispatch.sh",
    _INFRA / "spot_model_zoo_select.sh",
]
_MONOLITH = _INFRA / "spot_train.sh"


def _text(path: Path) -> str:
    return path.read_text()


# ── Existence + monolith untouched ───────────────────────────────────────────


def test_all_scripts_exist():
    assert _COMMON.is_file()
    assert _MONOLITH.is_file(), "monolith must stay in place as the rollback path"
    for script in _STAGE_SCRIPTS:
        assert script.is_file(), f"missing {script.name}"


@pytest.mark.parametrize("script", _STAGE_SCRIPTS)
def test_stage_scripts_are_executable(script):
    assert script.stat().st_mode & 0o111, f"{script.name} must be executable (chmod +x)"


# ── Sourcing + set -euo pipefail ─────────────────────────────────────────────


@pytest.mark.parametrize("script", [_COMMON, *_STAGE_SCRIPTS])
def test_set_euo_pipefail(script):
    text = _text(script)
    assert "set -euo pipefail" in text.splitlines()[:40], (
        f"{script.name} must declare 'set -euo pipefail' near the top"
    )


@pytest.mark.parametrize("script", _STAGE_SCRIPTS)
def test_stage_script_sources_common(script):
    text = _text(script)
    assert 'source "$SCRIPT_DIR/_spot_common.sh"' in text, (
        f"{script.name} must source _spot_common.sh rather than duplicating "
        f"spot-launch/SSM/cleanup infrastructure"
    )


def test_spot_common_refuses_no_direct_purpose_marker():
    # _spot_common.sh has no __main__-style guard historically; document the
    # intended usage instead of asserting a specific refusal mechanism this
    # repo doesn't implement (avoids inventing a contract the file doesn't
    # carry). See _spot_common.sh's own header comment for the source-only
    # convention this test would otherwise duplicate.
    text = _text(_COMMON)
    assert "Source this file from per-stage scripts" in text


# ── RUN_TOKEN forwarding — ONE place (crucible-predictor#428 incident) ──────


def test_run_token_export_computed_exactly_once_in_common():
    text = _text(_COMMON)
    assert text.count("_RUN_TOKEN_EXPORT=") == 2, (
        "RUN_TOKEN forwarding must be computed in exactly one place "
        "(the if/else pair in _spot_common.sh) — crucible-predictor#428 had "
        "to patch 5 separate heredocs for this; the whole point of the split "
        "is that it only needs patching once"
    )


@pytest.mark.parametrize("script", _STAGE_SCRIPTS)
def test_stage_scripts_do_not_recompute_run_token(script):
    text = _text(script)
    assert "_RUN_TOKEN_EXPORT=" not in text, (
        f"{script.name} must reuse _spot_common.sh's _RUN_TOKEN_EXPORT, not "
        f"recompute it locally — that would recreate the #428 five-heredoc "
        f"patch problem this split exists to remove"
    )
    if "_RUN_TOKEN_EXPORT" in text:
        assert "${_RUN_TOKEN_EXPORT}" in text, (
            f"{script.name} references _RUN_TOKEN_EXPORT but never forwards "
            f"it into a run_ssm heredoc prefix"
        )


# ── Preflight-only: the regression this file exists to catch ────────────────


def test_common_defines_shared_preflight_only_function():
    text = _text(_COMMON)
    assert "maybe_run_preflight_only_and_exit()" in text, (
        "the Friday shell_run dry-path probe must be a single shared "
        "function in _spot_common.sh, not copy-pasted per stage script"
    )
    assert 'PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-}"' in text, (
        "PREFLIGHT_ONLY must be unconditionally initialized (empty=disabled) "
        "at the top of _spot_common.sh, per the unconditional-init convention"
    )


def test_preflight_only_function_is_noop_when_unset():
    text = _text(_COMMON)
    start = text.index("maybe_run_preflight_only_and_exit() {")
    end = text.index("\n}", start)
    body = text[start:end]
    assert 'if [ -z "$PREFLIGHT_ONLY" ]; then' in body
    assert "return 0" in body, (
        "must be a true no-op (return 0, not exit) when PREFLIGHT_ONLY is "
        "unset, so a real run stays byte-behavior-identical"
    )


def test_preflight_only_probe_never_calls_training_or_promotion():
    text = _text(_COMMON)
    start = text.index("maybe_run_preflight_only_and_exit() {")
    end = text.index("\n}", start)
    body = text[start:end]
    code_lines = [
        ln
        for ln in body.splitlines()
        if not ln.lstrip().startswith(("print(", "log.info(", "#"))
    ]
    code = "\n".join(code_lines)
    forbidden = [
        "run_meta_training(",
        "train_main(",
        "from training.train_handler import main",
        "train_one_spec(",
        "run_rotation_and_select(",
        "run_select_only(",
        "put_object",
        "upload_file",
    ]
    for token in forbidden:
        assert token not in code, (
            f"preflight-only probe must NOT reference {token!r} — it would "
            f"break the no-train/no-promote/no-write invariant"
        )
    assert "TrainingPreflight" in body
    assert "list_symbols()" in body


@pytest.mark.parametrize("script", _STAGE_SCRIPTS)
def test_stage_script_supports_preflight_only_flag(script):
    """Every stage script must recognize --preflight-only and set
    PREFLIGHT_ONLY=1 — this is the exact defect crucible-predictor#436
    shipped for spot_train_spec_dispatch.sh / spot_model_zoo_select.sh: the
    flag was silently ignored as an 'unknown flag', so the Friday shell-run
    dry path (nousergon-data step_function.json's $.preflight_args, appended
    to every one of the three predictor SF states' commands) fell through
    into real training / real selection+promotion."""
    text = _text(script)
    assert "--preflight-only" in text, (
        f"{script.name} does not recognize --preflight-only — the Friday "
        f"shell-run dry path would silently run for real"
    )
    assert "PREFLIGHT_ONLY=1" in text, (
        f"{script.name} recognizes --preflight-only but never sets "
        f"PREFLIGHT_ONLY=1 — maybe_run_preflight_only_and_exit would no-op"
    )


@pytest.mark.parametrize("script", _STAGE_SCRIPTS)
def test_stage_script_calls_shared_preflight_check_after_deps_before_work(script):
    text = _text(script)
    assert "maybe_run_preflight_only_and_exit" in text, (
        f"{script.name} must call the shared preflight-only gate"
    )
    i_deps = text.index("install_deps")
    i_gate = text.index("maybe_run_preflight_only_and_exit\n")
    assert i_deps < i_gate, (
        f"{script.name}: maybe_run_preflight_only_and_exit must run AFTER "
        f"install_deps (same bootstrap path the real run uses) and BEFORE "
        f"any stage-specific work"
    )


# ── Fail loud on unrecognized flags (the incident's root cause) ─────────────


@pytest.mark.parametrize("script", _STAGE_SCRIPTS)
def test_no_silent_unknown_flag_swallow(script):
    text = _text(script)
    assert "ignoring unknown flag" not in text, (
        f"{script.name} must not silently ignore unrecognized flags — this "
        f"is exactly how --preflight-only went unhandled in crucible-"
        f"predictor#436. Unknown flags must hard-fail."
    )


@pytest.mark.parametrize("script", _STAGE_SCRIPTS)
def test_unknown_flag_live_hard_fails_before_any_aws_call(script):
    """Live-executed: real bash subprocess, zero AWS/network calls (the
    unknown-flag branch in every stage script's parse loop exits before
    spot_launch is ever reached)."""
    proc = subprocess.run(
        ["bash", str(script), "--this-flag-does-not-exist"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 2, (
        f"{script.name} --this-flag-does-not-exist: expected exit 2, got "
        f"{proc.returncode}. stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "unknown flag" in proc.stderr.lower()


def test_spec_dispatch_requires_spec_id_live():
    proc = subprocess.run(
        ["bash", str(_INFRA / "spot_train_spec_dispatch.sh")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 2
    assert "--spec-id is required" in proc.stderr


def test_model_zoo_select_requires_a_mode_live():
    proc = subprocess.run(
        ["bash", str(_INFRA / "spot_model_zoo_select.sh")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 2
    assert "--weekly or --select-only" in proc.stderr


# ── No stage-selection multiplexing flag survives on the new scripts ────────


@pytest.mark.parametrize(
    "script",
    [_INFRA / "spot_train_spec_dispatch.sh", _INFRA / "spot_model_zoo_select.sh"],
)
def test_no_mode_or_skip_stages_flag(script):
    """sf-pipeline-policy.md §2.1: no launcher may accept a flag whose
    purpose is selecting which SF stage to run. --weekly/--select-only on
    spot_model_zoo_select.sh select a SUB-MODE of the ModelZooSelect-family
    scripts (not a different SF state — ModelZooWeekly is not one of the
    three states this issue's table names), so it is exempted from this
    specific check; the structural-selection axis this test polices is
    SKIP_STAGES / ONLY_PHASES / --mode= style multiplexing."""
    text = _text(script)
    for forbidden in ("SKIP_STAGES", "ONLY_PHASES", "--skip-stages", "--mode="):
        assert forbidden not in text, f"{script.name} must not contain {forbidden!r}"
