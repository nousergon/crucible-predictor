"""The EXIT trap must reach ``terminate-instances`` on a NON-reclaim failure.

The failure this pins, ``ne-weekly-freshness-pipeline`` execution
``watch-rerun-2026-08-10-9``, 2026-08-11:

``cleanup()`` asks ``krepis.ec2_spot relaunch-decision`` whether the spot was
reclaimed. Originally that CLI answered by EXIT CODE — ``0`` = relaunch,
``NO_RELAUNCH_EXIT_CODE`` (75) = hold. Hold was the designed answer for every
failure that is *not* an AWS reclaim, which is nearly all of them.

The launchers ran under ``set -e``, and the call was written::

    _decide_out="$(... relaunch-decision ... )"
    _decide_rc=$?

An assignment whose value comes from a command substitution is a simple command
whose exit status IS the substitution's, so on the ordinary hold answer errexit
fired and destroyed the shell *inside the EXIT trap*. ``_decide_rc=$?`` was
unreachable; so were ``terminate-instances`` and the staging cleanup below it.
And because ``set -e`` does not re-enter a trap it is already running, the abort
emitted nothing at all — the log simply stops.

Measured consequences on that run: cleanup's stdout ends after its own blank
line, ``==> Terminating spot instance ...`` never appears, CloudTrail records no
``TerminateInstances`` call for ``i-092854cbb6e62b753``, the r5.large ran until
its own systemd watchdog shut it down ~90 minutes later, the S3 staging prefix
survived, and the launcher exited 75 instead of the workload's real status.

This is the sibling of ``test_spot_heartbeat_not_leaked_on_failure.py``: that
one pins *why the SSM command stayed open*, this one pins *why the instance was
never terminated*. Both are on the same silent path and neither implies the
other — stopping the heartbeat lets the process exit promptly, but it still
exits before terminating anything unless this guard holds.

alpha-engine-config-I7009: ``relaunch-decision`` grew ``--json`` in
krepis-PR133 (released 0.51.0). With ``--json`` the verdict is a JSON field on
stdout and the CLI exits 0 whenever it reached ANY decision (hold included).
A non-zero exit with ``--json`` now means only "the CLI could not answer" (bad
input, AWS error) — NOT a verdict, and is itself treated as hold. These tests
pin BOTH outcomes: a JSON hold verdict on rc=0, and a CLI failure on non-zero
rc — both must still reach ``terminate-instances``.

The tests run the REAL ``cleanup`` out of the real script with ``aws`` and the
lib CLI stubbed, so they fail if the guard is reverted in either launcher.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"

#: (script, instance-id variable) for every launcher carrying a `cleanup` that
#: consults `relaunch-decision`. `_spot_common.sh` is sourced by the three
#: per-stage scripts; `spot_train.sh` is the monolith with its own copy.
_LAUNCHERS = (
    (_INFRA / "_spot_common.sh", "_INSTANCE_ID", "_S3_STAGING"),
    (_INFRA / "spot_train.sh", "INSTANCE_ID", "S3_STAGING"),
)


def _function_text(source: str, name: str) -> str | None:
    """Return a shell function's full text, brace-matched, or None if absent.

    Functions are lifted rather than the whole file sourced: `spot_train.sh` is
    a monolith whose top level launches a spot, and `_spot_common.sh` is a
    library. Lifting keeps both cases identical AND keeps the test honest —
    the text executed is the text in the repository.
    """
    marker = "\n%s() {" % name
    if marker not in source:
        return None
    start = source.index(marker) + 1
    depth = 0
    for idx in range(start, len(source)):
        if source[idx] == "{":
            depth += 1
        elif source[idx] == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise AssertionError(f"unbalanced braces in {name}()")


def _write_stub(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


@pytest.fixture()
def stub_bin(tmp_path: Path) -> Path:
    """A PATH directory where `aws` succeeds silently and does nothing."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _write_stub(stub_dir / "aws", "#!/usr/bin/env bash\nexit 0\n")
    return stub_dir


@pytest.fixture()
def hold_json_python(tmp_path: Path) -> Path:
    """A stand-in for ``$LIB_PYTHON`` that answers 'hold' via ``--json``.

    Prints a JSON verdict line with ``"relaunch": false`` and exits 0 — the
    shape the CLI produces post krepis-PR133 for a non-reclaim failure. Also
    doubles as the ``$LIB_PYTHON -c '...'`` JSON-field reader the migrated
    call sites pipe the verdict through.
    """
    stub = tmp_path / "hold-json-python"
    _write_stub(
        stub,
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-c" ]; then\n'
        "  exec python3 \"$@\"\n"
        "fi\n"
        'printf \'{"relaunch": false, "reason": "other", "classification": '
        '"other", "attempt": 1}\\n\'\n'
        "exit 0\n",
    )
    return stub


@pytest.fixture()
def cli_failure_python(tmp_path: Path) -> Path:
    """A stand-in for ``$LIB_PYTHON`` that fails to answer at all.

    Exits non-zero with no JSON on stdout — the post-I7009 contract for "the
    CLI could not answer" (bad input / AWS error), which must be treated as
    hold, exactly like a JSON hold verdict.
    """
    stub = tmp_path / "cli-failure-python"
    _write_stub(
        stub,
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-c" ]; then\n'
        "  exec python3 \"$@\"\n"
        "fi\n"
        "exit 3\n",
    )
    return stub


def _run_cleanup(
    script: Path,
    instance_var: str,
    staging_var: str,
    tmp_path: Path,
    stub_bin: Path,
    lib_python: Path,
) -> subprocess.CompletedProcess[str]:
    """Source the launcher for its function definitions only, install the
    real `cleanup` as the EXIT trap, then fail the script the way a failed
    SSM step does."""
    source = script.read_text()
    cleanup_text = _function_text(source, "cleanup")
    assert cleanup_text, f"{script.name}: no cleanup() to test"
    heartbeat_stop_text = _function_text(source, "_heartbeat_stop") or (
        "_heartbeat_stop() { :; }"
    )
    # alpha-engine-config-I7442: `cleanup()` in BOTH launchers now calls
    # `spot_common_teardown_staging`, defined once in `_spot_common.sh`.
    # `spot_train.sh` relies on the sourced copy rather than defining its
    # own, so it must be lifted here too or the harness's `cleanup` trap
    # hits "command not found" under `set -euo pipefail` — exactly the
    # silent-abort-inside-the-EXIT-trap class this file's own docstring
    # pins for `relaunch-decision`.
    common_source = (_INFRA / "_spot_common.sh").read_text()
    teardown_text = _function_text(common_source, "spot_common_teardown_staging")
    assert teardown_text, "_spot_common.sh: no spot_common_teardown_staging() to lift"

    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        # The condition under test: both launchers set exactly this.
        "set -euo pipefail\n"
        "_HEARTBEAT_PID=\"\"\n"
        f"{heartbeat_stop_text}\n"
        f"{teardown_text}\n"
        f"{cleanup_text}\n"
        "S3_BUCKET=test-bucket\n"
        "AWS_REGION=us-east-1\n"
        "SF_EXECUTION_TIMEOUT=\"\"\n"
        # State `cleanup` reads. Populated as `spot_launch` would.
        f"{instance_var}=i-0000000000test0000\n"
        f"{staging_var}=s3://test-bucket/tmp/spot_train/test\n"
        "_SPOT_NAME=test-stage\n"
        "_SSM_SLUG=spot-test\n"
        "_PROCESS_NAME=test-process\n"
        "MAX_RUNTIME_SECONDS=5400\n"
        "SPOT_ATTEMPT=1\n"
        "MAX_SPOT_ATTEMPTS=2\n"
        f"LIB_PYTHON={lib_python}\n"
        "_ORIG_ARGS=()\n"
        'trap \'cleanup "$_SSM_SLUG"\' EXIT\n'
        # The workload failed. This is what a failed `run_ssm` does.
        "exit 1\n"
    )
    harness.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{stub_bin}:{env['PATH']}"

    return subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


@pytest.mark.parametrize(
    ("script", "instance_var", "staging_var"),
    _LAUNCHERS,
    ids=[p.name for p, _, _ in _LAUNCHERS],
)
def test_cleanup_terminates_the_instance_on_json_hold_verdict(
    script: Path,
    instance_var: str,
    staging_var: str,
    tmp_path: Path,
    stub_bin: Path,
    hold_json_python: Path,
) -> None:
    """A JSON hold verdict (rc=0, relaunch:false) must not abort the EXIT trap
    and must reach terminate-instances, not relaunch."""
    if shutil.which("bash") is None:  # pragma: no cover - bash is a hard dep
        pytest.skip("bash unavailable")

    proc = _run_cleanup(
        script, instance_var, staging_var, tmp_path, stub_bin, hold_json_python
    )
    combined = proc.stdout + proc.stderr

    assert "==> Terminating spot instance" in combined, (
        f"{script.name}: cleanup did not reach terminate-instances on a JSON "
        f"hold verdict.\n--- output ---\n{combined}"
    )
    assert "Instance terminated" in combined, (
        f"{script.name}: cleanup did not finish teardown.\n"
        f"--- output ---\n{combined}"
    )
    # The workload's real status survives the trap.
    assert proc.returncode == 1, (
        f"{script.name}: expected the workload's exit status 1, got "
        f"{proc.returncode}."
    )
    # No relaunch — this was a hold verdict, not a reclaim.
    assert "Spot RECLAIMED" not in combined, (
        f"{script.name}: a hold verdict must not trigger a relaunch.\n"
        f"--- output ---\n{combined}"
    )


@pytest.mark.parametrize(
    ("script", "instance_var", "staging_var"),
    _LAUNCHERS,
    ids=[p.name for p, _, _ in _LAUNCHERS],
)
def test_cleanup_terminates_the_instance_on_cli_failure(
    script: Path,
    instance_var: str,
    staging_var: str,
    tmp_path: Path,
    stub_bin: Path,
    cli_failure_python: Path,
) -> None:
    """A CLI failure (non-zero rc, no JSON — "could not answer") must also be
    treated as hold: no abort of the EXIT trap, terminate-instances reached,
    no relaunch."""
    if shutil.which("bash") is None:  # pragma: no cover - bash is a hard dep
        pytest.skip("bash unavailable")

    proc = _run_cleanup(
        script, instance_var, staging_var, tmp_path, stub_bin, cli_failure_python
    )
    combined = proc.stdout + proc.stderr

    assert "==> Terminating spot instance" in combined, (
        f"{script.name}: cleanup aborted before terminate-instances on a CLI "
        f"failure — the spot instance leaks.\n--- output ---\n{combined}"
    )
    assert "Instance terminated" in combined, (
        f"{script.name}: cleanup did not finish teardown.\n"
        f"--- output ---\n{combined}"
    )
    assert "CLI failed to answer" in combined, (
        f"{script.name}: a CLI failure must be logged explicitly as "
        f"'could not answer', not silently swallowed.\n--- output ---\n{combined}"
    )
    assert proc.returncode == 1, (
        f"{script.name}: expected the workload's exit status 1, got "
        f"{proc.returncode}."
    )
    assert "Spot RECLAIMED" not in combined, (
        f"{script.name}: a CLI failure must not trigger a relaunch.\n"
        f"--- output ---\n{combined}"
    )


def test_relaunch_decision_call_uses_json_and_is_errexit_guarded() -> None:
    """Static backstop: the --json migration + errexit guard must be on the
    call, in every launcher.

    The behavioural tests above can only fail once the defect is reintroduced
    in a way that still reaches `cleanup`. This pins the shape directly, so a
    future refactor that reverts to the bare exit-code contract is caught even
    if the harness stops exercising that branch.
    """
    for script, _instance_var, _staging_var in _LAUNCHERS:
        source = script.read_text()
        assert "relaunch-decision" in source, f"{script.name}: call vanished"
        assert "--json" in source, (
            f"{script.name}: relaunch-decision call is missing --json "
            f"(alpha-engine-config-I7009) — the verdict is read from a JSON "
            f"field, not the exit code."
        )
        assert '2>/dev/null)" || _decide_rc=$?' in source, (
            f"{script.name}: the relaunch-decision command substitution is not "
            f"errexit-guarded. A CLI failure aborts the EXIT trap and the spot "
            f"instance is never terminated."
        )
        assert "\n    _decide_rc=$?\n" not in source, (
            f"{script.name}: `_decide_rc=$?` on its own line is unreachable "
            f"under `set -e` — the guard must be on the assignment itself."
        )
