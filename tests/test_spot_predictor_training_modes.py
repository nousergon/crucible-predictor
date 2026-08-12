"""``spot_predictor_training.sh``: what the Saturday SF actually runs.

Two invariants, one file, because they are the same defect seen twice — a
per-stage script that took over from the monolith without taking over the
monolith's guards.

1. THE SMOKE STEP IS NOT ON THE CRITICAL PATH OF THE WEEKLY SF.

   The monolith ``spot_train.sh`` skipped the ``dry_run=True`` smoke step
   whenever ``MODE`` was ``full-only``::

       if [ "$MODE" != "full-only" ] && [ "$MODE" != "model-zoo-weekly" ] && ...

   and the weekly SF invokes the champion retrain in exactly that mode. When
   config#4442/#4497 split the monolith into per-stage scripts, ``full-only``
   became this script's DEFAULT and the guard did not come with it, so the
   split script ran smoke UNCONDITIONALLY. Every Saturday run therefore
   executed a full ``dry_run=True`` training pass and then a full
   ``dry_run=False`` one: double the spot wall time, double the ArcticDB read
   cost, and a code path that had been UNREACHABLE from the SF promoted to a
   hard prerequisite for reaching the real training. The 2026-08-11 run died
   in that dry run, having never attempted the work the stage exists to do.

2. THE INNER SSM CEILING IS STRICTLY LESS THAN THE OUTER STAGE BUDGET.

   ``run_ssm "full-training" ... "$MAX_RUNTIME_SECONDS"`` was 5400s — the same
   number as the SF stage's ``executionTimeout``, which must ALSO cover spot
   launch, the instance-running wait, the SSM-agent wait, config staging,
   bootstrap and pip install. An inner ceiling equal to the outer can never
   bind, so it can never name itself: the run dies at the outer boundary with
   no message attributing the timeout to training and no partial artifact. The
   whole value of a per-step ceiling is that the step that ran out of time is
   the step that says so.

   The assertion here is the RELATIONSHIP, not the literals — the outer budget
   is fed in through ``SF_EXECUTION_TIMEOUT`` and the inner ceiling must come
   out strictly below it, by a reserve, for whatever outer budget is supplied.

METHOD — the real script text, fake collaborators.

The script is copied into a temp directory next to a FAKE ``_spot_common.sh``
that defines every helper as a recorder. ``SCRIPT_DIR`` resolves to that temp
directory, so the script under test sources the fake and runs to completion
with no AWS involved. The text executed is the text in the repository, so
deleting the guard fails this test rather than merely changing a comment.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "infrastructure"
    / "spot_predictor_training.sh"
)

#: Every helper `spot_predictor_training.sh` expects `_spot_common.sh` to
#: provide, replaced by a recorder. `run_ssm` appends "<description> <timeout>"
#: to $_RECORD so the test can assert on the dispatch sequence.
_FAKE_COMMON = r"""
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-test-bucket}"
ALPHA_ENGINE_EXPERIMENT_ID="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"
BRANCH="${BRANCH:-main}"
INSTANCE_TYPES="${INSTANCE_TYPES:-r5.large}"
INSTANCE_TYPE=""
AMI_ID="ami-test"
SUBNETS="${SUBNETS:-subnet-test}"
LIB_PYTHON="${LIB_PYTHON:-/bin/true}"
MAX_SPOT_ATTEMPTS="${MAX_SPOT_ATTEMPTS:-2}"
SPOT_ATTEMPT="${SPOT_ATTEMPT:-1}"
SF_EXECUTION_TIMEOUT="${SF_EXECUTION_TIMEOUT:-}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-}"
_RUN_TOKEN_EXPORT=""
_INSTANCE_ID="i-0000000000test0000"

# The line under scrutiny in every "did the split drop a default?" question:
# the shared file still assigns this, which is why the script under test must
# capture any operator override BEFORE sourcing.
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-5400}"

_heartbeat_stop() { :; }
_heartbeat_start() { :; }
emit_heartbeat() { :; }
print_banner() { echo "== $1"; }
check_config_exists() { :; }
spot_launch() { :; }
cleanup() { :; }
wait_ssm_agent() { :; }
stage_config() { :; }
bootstrap_spot() { :; }
install_deps() { :; }
maybe_run_preflight_only_and_exit() {
  if [ -n "${PREFLIGHT_ONLY:-}" ]; then
    printf 'preflight\t0\n' >> "$_RECORD"
    exit 0
  fi
}
run_ssm() {
  printf '%s\t%s\n' "$1" "${3:-3600}" >> "$_RECORD"
}
"""


def _run(tmp_path: Path, args: list[str], env_extra: dict[str, str] | None = None):
    """Run the real script against the fake common file; return recorded steps.

    Returns ``(returncode, [(description, timeout_seconds), ...], stdout)``.
    """
    stage = tmp_path / "infrastructure"
    stage.mkdir(parents=True, exist_ok=True)
    (stage.parent / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(_SCRIPT, stage / _SCRIPT.name)
    (stage / "_spot_common.sh").write_text(_FAKE_COMMON)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    aws = bin_dir / "aws"
    aws.write_text("#!/usr/bin/env bash\nexit 0\n")
    aws.chmod(0o755)

    record = tmp_path / "record.tsv"
    record.write_text("")

    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(tmp_path),
        "_RECORD": str(record),
    }
    env.update(env_extra or {})

    proc = subprocess.run(
        ["bash", str(stage / _SCRIPT.name), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    steps = [
        (line.split("\t")[0], int(line.split("\t")[1]))
        for line in record.read_text().splitlines()
        if line.strip()
    ]
    return proc.returncode, steps, proc.stdout + proc.stderr


@pytest.fixture(autouse=True)
def _requires_bash():
    if shutil.which("bash") is None:  # pragma: no cover - bash is a hard dep
        pytest.skip("bash unavailable")


def test_the_sf_default_mode_does_not_run_the_dry_run_smoke(tmp_path: Path) -> None:
    """A bare invocation — what the weekly SF issues — runs full training ONLY.

    This is the assertion that fails against the unconditional form: before the
    guard was restored, the recorded steps were ``["smoke", "full-training"]``.
    """
    rc, steps, out = _run(tmp_path, [])
    assert rc == 0, out
    descriptions = [d for d, _ in steps]
    assert "smoke" not in descriptions, (
        "the dry_run=True smoke step ran on the SF's default (full-only) path — "
        f"recorded steps were {descriptions}. That doubles the spot wall time "
        "and the ArcticDB read cost, and puts the dry-run path on the critical "
        "path of the Saturday pipeline."
    )
    assert descriptions == ["full-training"], descriptions


def test_smoke_only_runs_the_smoke_and_stops(tmp_path: Path) -> None:
    """The standalone dry check is unaffected by the guard."""
    rc, steps, out = _run(tmp_path, ["--smoke-only"])
    assert rc == 0, out
    assert [d for d, _ in steps] == ["smoke"], steps


def test_smoke_first_keeps_the_monoliths_smoke_then_full_sequence(
    tmp_path: Path,
) -> None:
    """Nothing was lost in restoring the guard.

    The monolith's bare invocation ran smoke THEN full (its ``both`` mode).
    That capability survives as an explicit opt-in, so a human wanting the old
    belt-and-braces sequence still has it — it is simply no longer what the
    unattended weekly pipeline does by default.
    """
    rc, steps, out = _run(tmp_path, ["--smoke-first"])
    assert rc == 0, out
    assert [d for d, _ in steps] == ["smoke", "full-training"], steps


def test_preflight_only_still_exits_before_any_workload(tmp_path: Path) -> None:
    """config#4497's invariant: the Friday dry path spends no training compute."""
    rc, steps, out = _run(tmp_path, ["--preflight-only"])
    assert rc == 0, out
    assert [d for d, _ in steps] == ["preflight"], steps


@pytest.mark.parametrize("outer", [5400, 7200, 3600])
def test_inner_training_ceiling_is_strictly_below_the_outer_stage_budget(
    tmp_path: Path, outer: int
) -> None:
    """The ordering, asserted as a relationship rather than against literals.

    Parametrised over several outer budgets precisely so that pinning the old
    ``5400`` literal in either place cannot satisfy it: the inner ceiling has
    to TRACK the outer one, which is only true if it is derived from it.
    """
    rc, steps, out = _run(
        tmp_path, [], env_extra={"SF_EXECUTION_TIMEOUT": str(outer)}
    )
    assert rc == 0, out
    inner = dict(steps)["full-training"]
    assert inner < outer, (
        f"inner full-training ceiling {inner}s is not strictly below the outer "
        f"stage budget {outer}s. An inner ceiling that cannot bind can never "
        "name itself: the run dies at the outer boundary with no diagnostic."
    )
    # The margin must be a real reserve, not a token second — the outer budget
    # also has to cover launch, bootstrap and pip install before the workload
    # starts. 10 minutes is the floor; the script's declared reserve is larger.
    assert outer - inner >= 600, (
        f"reserve of {outer - inner}s is too small to cover spot launch "
        "(~7 min measured) plus bootstrap and pip install."
    )


def test_default_outer_budget_mirrors_the_step_function(tmp_path: Path) -> None:
    """The fallback literal is the SF's, and is documented as a mirror.

    When ``SF_EXECUTION_TIMEOUT`` is not exported the script falls back to a
    mirrored copy of the SF stage's ``executionTimeout``. That mirror is the
    weak point of the arrangement, so it is pinned here together with the
    pointer to its source of truth — if the SF budget moves and this does not,
    this test is where the drift is named.

    Source of truth: nousergon-data ``infrastructure/step_function.json``,
    state ``ResearchPredictorParallel/PredictorTraining``,
    ``Parameters.Parameters.executionTimeout``.
    """
    text = _SCRIPT.read_text()
    match = re.search(
        r'SF_STAGE_EXECUTION_TIMEOUT_SECONDS="\$\{SF_EXECUTION_TIMEOUT:-(\d+)\}"',
        text,
    )
    assert match, "the mirrored outer-budget default is not declared as expected"
    mirrored = int(match.group(1))
    assert mirrored == 5400, (
        "the mirrored SF stage executionTimeout changed. Confirm it against "
        "nousergon-data infrastructure/step_function.json before updating this."
    )
    rc, steps, out = _run(tmp_path, [])
    assert rc == 0, out
    assert dict(steps)["full-training"] < mirrored, json.dumps(steps)


def test_an_explicit_operator_override_still_wins(tmp_path: Path) -> None:
    """``MAX_RUNTIME_SECONDS=<n>`` from the environment is honoured.

    Non-obvious because ``_spot_common.sh`` assigns its own default to the same
    name: a post-source ``${MAX_RUNTIME_SECONDS:-...}`` in this script would be
    a no-op against an already-non-empty parameter, making an operator override
    indistinguishable from the shared default. The script captures the
    environment value BEFORE sourcing, which is what this pins.
    """
    rc, steps, out = _run(tmp_path, [], env_extra={"MAX_RUNTIME_SECONDS": "999"})
    assert rc == 0, out
    assert dict(steps)["full-training"] == 999, steps
