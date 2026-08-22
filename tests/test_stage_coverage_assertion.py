"""Per-stage output assertion coverage (config-I7214, sf-pipeline-policy.md
§2.1 / §2.3a).

Brian ruled the end-of-run `StageCoverageAssert` SF state NON-SOTA: the
assertion belongs in each stage's OWN launcher script / Lambda handler,
asserting its declared output immediately before it exits. This module
enforces that every SF-wired launcher script and SF-wired Lambda gate action
owned by THIS repo actually carries the call, that it is wired to the
correct SF stage name, and that observe mode cannot be silently promoted.

Stage-name derivation (verified against the live definition 2026-08-13,
`nousergon-data/infrastructure/step_function.json`):

- ``PredictorTraining``  -> `spot_predictor_training.sh` (default full-only
  mode; the ONLY mode the SF invokes).
- ``TrainSpecDispatch``  -> `spot_train_spec_dispatch.sh`. NOT
  "ResolveZooSpecs" — that SF state is a separate inline
  `python -m training.model_zoo list-rotation-specs` SSM command with no
  launcher script of its own, so it carries no assertion from this repo.
  `TrainSpecDispatch` is the Task inside the `ModelZooTrainMap` Map state
  that actually launches this script per spec id.
- ``ModelZooSelect``     -> `spot_model_zoo_select.sh --select-only`. The
  script's `--weekly` mode is NOT reachable from the live SF (no
  "ModelZooWeekly" state exists there), so it is deliberately excluded
  below — asserting an undeclared registry stage would be a
  guaranteed-false MISSING, not a real signal.
- ``WeeklyRunDayGate`` / ``LibPinDriftCheck`` / ``PipelineContractCheck``
  -> the three gate actions of `alpha-engine-predictor-inference`
  (`inference/handler.py`). Each positively declares no durable output
  (`output: none` in ARTIFACT_REGISTRY.yaml`'s `pipeline_stages:` section,
  `alpha-engine-config-PR7229`) — the lib returns `COVERED_NO_OUTPUT`.
- ``RegimeSubstrate`` -> `regime/handler.py` (`alpha-engine-predictor-regime-substrate`).
- ``RegimeRetrospectiveEval`` -> `regime/retrospective_eval_handler.py`
  (`alpha-engine-predictor-regime-retrospective-eval`).

This is the enumeration side of the totality test below: derived from the
live SF definition where the state is reachable in this repo's checkout,
hardcoded only where deriving it live is impractical for a unit test (no
network / no `nousergon-data` checkout guaranteed in CI) — the docstring
above is the paper trail for each hardcoded entry, per
`bugclass_a_test_that_enumerates_what_exists_is_blind`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_INFRA = _ROOT / "infrastructure"

# ── Launcher scripts: stage name -> (script, must appear on this many lines) ──

_LAUNCHER_STAGE_CALLS = {
    "spot_predictor_training.sh": "PredictorTraining",
    "spot_train_spec_dispatch.sh": "TrainSpecDispatch",
}

# spot_model_zoo_select.sh carries the call once, gated to --select-only.
_MODEL_ZOO_SELECT_SCRIPT = "spot_model_zoo_select.sh"
_MODEL_ZOO_SELECT_STAGE = "ModelZooSelect"

_ASSERT_CALL_RE = re.compile(
    r'"\$LIB_PYTHON"\s+-m\s+krepis\.stage_coverage\s+assert\s+'
    r'--stage\s+"(?P<stage_var>\$_COVERAGE_STAGE)"\s+'
    r'--window-start\s+"\$_STAGE_WINDOW_START"'
)

#: alpha-engine-config-I8155: every shell invocation must ALSO pass
#: `--run-date "${EXECUTION_RUN_DATE:-}"` — the SF execution's own
#: run_date, unset-safe under `set -u` (`_spot_common.sh` sets
#: `set -euo pipefail`) so an unexported EXECUTION_RUN_DATE reaches the
#: CLI empty (loud, via the existing observe-mode guard) rather than
#: crashing the launcher on an unbound-variable error.
_RUN_DATE_FLAG_RE = re.compile(
    r'--run-date\s+"\$\{EXECUTION_RUN_DATE:-\}"'
)

_BARE_OR_TRUE_RE = re.compile(
    r'krepis\.stage_coverage\s+assert[^\n]*\|\|\s*true\b'
)


@pytest.mark.parametrize("script_name,stage", sorted(_LAUNCHER_STAGE_CALLS.items()))
def test_launcher_calls_stage_coverage_with_correct_stage(script_name: str, stage: str) -> None:
    """Each single-stage launcher carries the assertion, bound to its stage."""
    source = (_INFRA / script_name).read_text()
    assert _ASSERT_CALL_RE.search(source), (
        f"{script_name}: no `krepis.stage_coverage assert --stage "
        f'"$_COVERAGE_STAGE" --window-start "$_STAGE_WINDOW_START"` call found'
    )
    # _COVERAGE_STAGE must resolve to the correct stage name — assert the
    # literal assignment exists (unconditional single-mode script).
    assert re.search(
        rf'_COVERAGE_STAGE="{re.escape(stage)}"', source
    ), f"{script_name}: _COVERAGE_STAGE is not assigned {stage!r}"


def test_model_zoo_select_only_carries_the_assertion_gated_to_its_mode() -> None:
    """spot_model_zoo_select.sh: --select-only asserts ModelZooSelect; --weekly does not."""
    source = (_INFRA / _MODEL_ZOO_SELECT_SCRIPT).read_text()
    assert _ASSERT_CALL_RE.search(source), (
        f"{_MODEL_ZOO_SELECT_SCRIPT}: no stage-coverage assertion call found"
    )
    assert re.search(
        rf'_COVERAGE_STAGE="{re.escape(_MODEL_ZOO_SELECT_STAGE)}"', source
    ), f"{_MODEL_ZOO_SELECT_SCRIPT}: --select-only branch does not set _COVERAGE_STAGE={_MODEL_ZOO_SELECT_STAGE!r}"

    # The assertion call and the --select-only assignment must both fall
    # strictly after the --weekly branch's `exit 0`, i.e. inside the
    # `if [ "$MODE" = "select-only" ]` block, not the weekly one.
    weekly_block_end = source.index('echo "==> Model-zoo rotation complete')
    select_only_call = _ASSERT_CALL_RE.search(source)
    assert select_only_call.start() > weekly_block_end, (
        f"{_MODEL_ZOO_SELECT_SCRIPT}: the stage-coverage call appears before "
        "the --weekly block ends — it may have leaked into the wrong mode"
    )

    # --weekly explicitly does NOT get a stage assignment (no live SF state).
    weekly_block = source[: source.index("# ── Select-only")]
    assert '_COVERAGE_STAGE="ModelZooWeekly"' not in weekly_block, (
        "the --weekly branch must not assert a stage name absent from the "
        "live SF definition — see the module docstring"
    )


@pytest.mark.parametrize(
    "script_name",
    sorted(set(_LAUNCHER_STAGE_CALLS) | {_MODEL_ZOO_SELECT_SCRIPT}),
)
def test_launcher_does_not_swallow_the_assertion_with_bare_or_true(script_name: str) -> None:
    """The failure-visibility contract: `|| true` would make an absent module
    indistinguishable from a covered stage — the exact silence this
    mechanism exists to remove. Must use the loud `|| echo ... >&2` form.
    """
    source = (_INFRA / script_name).read_text()
    assert not _BARE_OR_TRUE_RE.search(source), (
        f"{script_name}: stage-coverage assertion is swallowed with `|| true` "
        "— use `|| echo \"WARNING: ...\" >&2` instead (config-I7214)"
    )
    # And the loud form is actually present alongside every assert call.
    for m in re.finditer(r'krepis\.stage_coverage\s+assert[^\n]*', source):
        line = m.group(0)
        assert "|| echo" in line and ">&2" in line, (
            f"{script_name}: an assertion call does not end in the required "
            f"loud `|| echo ... >&2` fallback: {line!r}"
        )


def _all_stage_coverage_assert_lines() -> list[tuple[Path, str]]:
    """`(path, line)` for every `krepis.stage_coverage assert` invocation
    across ALL of `infrastructure/*.sh` — not a hardcoded three-file list,
    so a fourth launcher gaining the call later is covered automatically
    (alpha-engine-config-I8155).
    """
    lines: list[tuple[Path, str]] = []
    for path in sorted(_INFRA.glob("*.sh")):
        source = path.read_text()
        for m in re.finditer(r'krepis\.stage_coverage\s+assert[^\n]*', source):
            lines.append((path, m.group(0)))
    return lines


def test_scan_found_stage_coverage_invocations() -> None:
    """Guard-the-guard: if `_all_stage_coverage_assert_lines` ever finds
    nothing, every test below it would vacuously pass — assert the scan
    actually found the three known invocations before trusting it.
    """
    found = _all_stage_coverage_assert_lines()
    assert len(found) >= 3, (
        f"expected at least 3 krepis.stage_coverage assert invocations "
        f"under infrastructure/*.sh, found {len(found)}: {found!r}"
    )


def test_every_shell_stage_coverage_invocation_passes_run_date() -> None:
    """alpha-engine-config-I8155: every shell invocation of
    `krepis.stage_coverage assert` must pass
    `--run-date "${EXECUTION_RUN_DATE:-}"` — the SF execution's own
    run_date, never defaulted from `$RUN_DATE` or `date` (a silent
    fall-back is the defect this fix removes).
    """
    for path, line in _all_stage_coverage_assert_lines():
        assert _RUN_DATE_FLAG_RE.search(line), (
            f"{path.name}: stage-coverage assertion does not pass "
            f'--run-date "${{EXECUTION_RUN_DATE:-}}": {line!r}'
        )


def test_no_shell_stage_coverage_invocation_passes_bare_run_date() -> None:
    """`$RUN_DATE` (unqualified) is an unreliable carrier fleet-wide — it is
    reassigned to the trading day elsewhere (`crucible-backtester`'s
    `_spot_common.sh`). No invocation may reference it, with or without the
    `EXECUTION_` prefix collapsed by a typo.
    """
    for path, line in _all_stage_coverage_assert_lines():
        assert "$RUN_DATE" not in line or "$EXECUTION_RUN_DATE" in line, (
            f"{path.name}: stage-coverage assertion references a bare "
            f"$RUN_DATE rather than $EXECUTION_RUN_DATE: {line!r}"
        )
        # Stronger check: the ONLY *_RUN_DATE reference on the line must be
        # the EXECUTION_-prefixed one — a bare "$RUN_DATE}" (e.g. from a
        # typo'd "${RUN_DATE:-}") would slip past the substring check above.
        assert not re.search(r'(?<!EXECUTION_)\$\{?RUN_DATE\b', line), (
            f"{path.name}: stage-coverage assertion references bare "
            f"$RUN_DATE instead of $EXECUTION_RUN_DATE: {line!r}"
        )


def test_stage_window_start_declared_once_in_common() -> None:
    """`_STAGE_WINDOW_START` is declared exactly once, in `_spot_common.sh`,
    and every launcher inherits it rather than re-deriving its own clock —
    otherwise two stages racing in the same run could disagree on "now".
    """
    common_source = (_INFRA / "_spot_common.sh").read_text()
    assert re.search(
        r'_STAGE_WINDOW_START="\$\{_STAGE_WINDOW_START:-\$\(date -u \+%Y-%m-%dT%H:%M:%SZ\)\}"',
        common_source,
    ), "_spot_common.sh does not declare _STAGE_WINDOW_START"

    for script_name in sorted(set(_LAUNCHER_STAGE_CALLS) | {_MODEL_ZOO_SELECT_SCRIPT}):
        source = (_INFRA / script_name).read_text()
        assert '_STAGE_WINDOW_START="${_STAGE_WINDOW_START' not in source, (
            f"{script_name}: redeclares _STAGE_WINDOW_START instead of "
            "inheriting it from _spot_common.sh"
        )


# ── Totality: every SF-wired launcher / Lambda action in this repo asserts ────

#: The complete set of stage names this repo's launchers/handlers must
#: assert, one entry per SF-wired call site. `ResolveZooSpecs` and the
#: `--weekly` mode of spot_model_zoo_select.sh are deliberately excluded —
#: see the module docstring.
_ALL_WIRED_STAGES = {
    "PredictorTraining",
    "TrainSpecDispatch",
    "ModelZooSelect",
    "WeeklyRunDayGate",
    "LibPinDriftCheck",
    "PipelineContractCheck",
    "RegimeSubstrate",
    "RegimeRetrospectiveEval",
}


def _all_stage_coverage_call_sites() -> set[str]:
    found: set[str] = set()
    for path in (_INFRA).glob("spot_*.sh"):
        source = path.read_text()
        for m in re.finditer(r'_COVERAGE_STAGE="([^"$]+)"', source):
            found.add(m.group(1))
    for path in [
        _ROOT / "inference" / "handler.py",
        _ROOT / "regime" / "handler.py",
        _ROOT / "regime" / "retrospective_eval_handler.py",
    ]:
        source = path.read_text()
        for m in re.finditer(r'assert_stage_coverage\(\s*\n?\s*"([^"]+)"', source):
            found.add(m.group(1))
    return found


def test_every_declared_wired_stage_has_a_call_site() -> None:
    """Totality: nothing in `_ALL_WIRED_STAGES` is silently unasserted.

    Derived from the live SF definition at authoring time (2026-08-13,
    `nousergon-data/infrastructure/step_function.json`) — see the module
    docstring for how each entry was verified. A future SF rename must
    update `_ALL_WIRED_STAGES` deliberately; this test cannot derive it
    live because CI does not guarantee a `nousergon-data` checkout, so it
    is the enumeration itself, not a proxy for one — an omission here is a
    real gap, not a false negative to explain away.
    """
    call_sites = _all_stage_coverage_call_sites()
    missing = _ALL_WIRED_STAGES - call_sites
    assert not missing, f"stages declared wired but with no call site: {sorted(missing)}"


def test_no_call_site_asserts_a_stage_outside_the_declared_set() -> None:
    """The inverse direction: a stray/renamed stage string is caught too."""
    call_sites = _all_stage_coverage_call_sites()
    extra = call_sites - _ALL_WIRED_STAGES
    assert not extra, (
        f"call site(s) assert stage(s) not in _ALL_WIRED_STAGES: {sorted(extra)} "
        "— update the enumeration (and its docstring justification) if this is intentional"
    )


# ── Promotion safety: nothing shipped may set --enforce ───────────────────────

_ENFORCE_RE = re.compile(r"--enforce\b|STAGE_COVERAGE_ENFORCE")


def test_no_call_site_sets_enforce_mode() -> None:
    """Promotion out of observe mode is a deliberate, reviewed diff — never a
    side effect of this PR. `--enforce` / `STAGE_COVERAGE_ENFORCE=1` must not
    appear anywhere this PR's call sites live.
    """
    checked = list(_INFRA.glob("spot_*.sh")) + [
        _INFRA / "_spot_common.sh",
        _ROOT / "inference" / "handler.py",
        _ROOT / "regime" / "handler.py",
        _ROOT / "regime" / "retrospective_eval_handler.py",
    ]
    for path in checked:
        source = path.read_text()
        assert not _ENFORCE_RE.search(source), (
            f"{path.name}: sets enforce mode — promotion must be its own "
            "reviewed diff, not shipped inside the observe-mode rollout "
            "(config-I7214)"
        )
