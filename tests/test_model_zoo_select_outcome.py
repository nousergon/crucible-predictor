"""MODEL_ZOO_SELECT_OUTCOME v1 — a refusal is not a failure.

alpha-engine-config-I11106. The 2026-09-19 weekly SF terminated DEGRADED
because ``ModelZooSelect`` raised ``ArenaSlotUnservable``. The stage had done
exactly its job: it evaluated three M arms, applied the §5.3 serving
preconditions, and correctly concluded that nothing may be promoted. Before
this fix the code raised, the spot workload exited 1, and the SF could not
tell *the slot correctly declined to promote* from *the stage broke*.

The contract these tests pin, end to end:

* unservable slot — the arena cycle artifact is written, the workload prints
  exactly one ``MODEL_ZOO_SELECT_OUTCOME: unservable`` line, and the launcher
  exits 0.
* servable slot — exactly one ``MODEL_ZOO_SELECT_OUTCOME: decided`` line,
  exit 0.
* any genuine error — a workload exception, a resource kill, a timeout, or a
  run that reports no outcome at all — exits non-zero, unchanged from before,
  and prints NO outcome line.

The durable source of truth is ``arena/model/{date}.json::decision.status``;
the stdout marker is a cheap transport for the SF's ``Choice`` only (moving
that Choice onto a structured read is alpha-engine-config-I11101 deliverable
4). The Choice matches the string literally, so the line is anchored and
these tests pin its exact spelling.

**Nothing here weakens the refusal.** ``promotion_behavioral_veto`` refuses
exactly what it refused before, no arm becomes promotable, and the pointer
does not move — see
``tests/test_arena_model_slot.py::TestAnUnservableCycleIsAVerdict``.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("nousergon_lib.arena")

import config as cfg  # noqa: E402
from training import model_zoo as mz  # noqa: E402

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"
_LAUNCHER = _INFRA / "spot_model_zoo_select.sh"

#: The exact line the weekly SF's Choice matches on StandardOutputContent.
_MARKER = re.compile(r"^MODEL_ZOO_SELECT_OUTCOME: (decided|unservable)$", re.M)


# ── the python side: the verdict reaches the entrypoint's return ─────────────


class _FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = {}

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 signature
        import io
        import json as _json

        if Key not in self.objects:
            raise KeyError(Key)
        body = self.objects[Key]
        if not isinstance(body, (bytes, bytearray)):
            body = _json.dumps(body).encode()
        return {"Body": io.BytesIO(body)}

    def put_object(self, Bucket, Key, Body, **kwargs):  # noqa: N803
        self.puts[Key] = Body
        return {}

    def list_objects_v2(self, **kwargs):
        return {"Contents": []}


def _select(monkeypatch, *, status, pointer_version_id):
    """Run ``run_select_only`` with the arena decision forced to ``status``."""
    from tests.arena_test_helpers import (
        mock_arena_run_slot,
        mock_model_zoo_post_select_reads,
    )

    import model.registry as reg

    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(reg, "list_versions", lambda s3c, b, stage=None: [])
    monkeypatch.setattr(mz, "send_zoo_digest_email", lambda *a, **k: True)
    monkeypatch.setattr(mz, "_alert_inert_rotation", lambda *a, **k: None)
    monkeypatch.setattr(mz, "_emit_challengers_trained_metric", lambda n: None)
    mock_arena_run_slot(
        monkeypatch, pointer_version_id=pointer_version_id, moved=False, status=status,
    )
    mock_model_zoo_post_select_reads(monkeypatch)
    return mz.run_select_only(
        "bkt", date_str="2026-09-18", s3=_FakeS3(), specs=[],
        auto_promote_winner=False,
    )


def test_an_unservable_slot_returns_a_verdict_rather_than_raising():
    """RED before the fix: ``ArenaSlotUnservable`` propagated out of
    ``run_select_only``, aborting the rest of the selection tail as well."""
    with pytest.MonkeyPatch.context() as mp:
        board = _select(mp, status="unservable", pointer_version_id=None)
    assert board["arena_outcome"] == "unservable"
    assert board["promoted"] in (None, False)


@pytest.mark.parametrize("status", ["decided", "held", "bootstrap", "unmeasurable"])
def test_a_servable_slot_reports_decided(status):
    """Every status that leaves an arm serving reports ``decided``.

    ``unmeasurable`` included: the cycle measured nothing, which alarms in the
    log and in ``decision.status``, but the incumbent keeps serving — so it is
    neither a refusal nor a broken stage, and the two-valued v1 marker cannot
    carry the difference (alpha-engine-config-I11101 deliverable 4 closes it by
    moving the SF Choice onto a structured read of the artifact).
    """
    with pytest.MonkeyPatch.context() as mp:
        board = _select(mp, status=status, pointer_version_id="v-a")
    assert board["arena_outcome"] == "decided"


def test_an_outcomeless_select_is_refused_rather_than_reported_as_success(monkeypatch):
    """A select that reaches the end with no outcome is NOT exit 0.

    `no data` is never rendered as green (``principles.md`` §2.7): if the
    arena decision did not reach this entrypoint, the stage has reported
    nothing and must say so loudly rather than let the SF read a clean
    success.
    """
    monkeypatch.setattr(
        mz, "select_and_finalize",
        lambda *a, **k: {"date": "2026-09-18", "candidates": [], "promoted": None},
    )
    with pytest.raises(RuntimeError, match="arena_outcome"):
        mz.run_select_only("bkt", date_str="2026-09-18", s3=_FakeS3(), specs=[])


# ── the launcher side: the marker on stdout, and the exit code ───────────────


def _select_only_block() -> str:
    """The REAL ``--select-only`` branch, lifted from the launcher on disk.

    Lifted rather than sourced because the script's top level launches a spot
    instance. The text executed is the text in the repository, so a revert of
    the guard fails these tests.
    """
    source = _LAUNCHER.read_text()
    marker = '# ── Select-only (after Map joins) ──'
    assert marker in source, "the --select-only branch header moved"
    return source[source.index(marker):]


def _run_launcher_block(
    tmp_path: Path, *, ssm_stdout: str, ssm_rc: int = 0, lib_python: str = "/usr/bin/true",
):
    """Execute the lifted branch with the spot/SSM machinery stubbed out."""
    harness = tmp_path / "harness.sh"
    out_file = tmp_path / "ssm-stdout.txt"
    out_file.write_text(ssm_stdout)
    harness.write_text(
        "\n".join(
            [
                "set -euo pipefail",
                'MODE="select-only"',
                '_RUN_TOKEN_EXPORT=""',
                "MAX_RUNTIME_SECONDS=1",
                f'LIB_PYTHON="{lib_python}"',
                '_COVERAGE_STAGE="ModelZooSelect"',
                '_STAGE_WINDOW_START="2026-09-19T00:00:00Z"',
                'EXECUTION_RUN_DATE="2026-09-19"',
                "print_banner() { :; }",
                "emit_heartbeat() { :; }",
                # Reproduces krepis.ssm_dispatcher: the remote workload's
                # stdout is streamed to the launcher's stdout, and the exit
                # status is the workload's.
                'run_ssm() { cat "$FAKE_SSM_STDOUT"; return "$FAKE_SSM_RC"; }',
                "",
                _select_only_block(),
            ]
        )
    )
    env = dict(os.environ)
    env["FAKE_SSM_STDOUT"] = str(out_file)
    env["FAKE_SSM_RC"] = str(ssm_rc)
    return subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, env=env, cwd=tmp_path,
    )


_WORKLOAD_TAIL = (
    "============================================================\n"
    "  MODEL-ZOO SELECT\n"
    "  Winner:         None\n"
    "============================================================\n"
)


def test_an_unservable_run_prints_the_marker_once_and_exits_zero(tmp_path):
    res = _run_launcher_block(
        tmp_path,
        ssm_stdout=_WORKLOAD_TAIL + "MODEL_ZOO_SELECT_OUTCOME_RAW=unservable\n",
    )
    assert res.returncode == 0, res.stderr
    assert _MARKER.findall(res.stdout) == ["unservable"]


def test_a_decided_run_prints_the_marker_once_and_exits_zero(tmp_path):
    res = _run_launcher_block(
        tmp_path,
        ssm_stdout=_WORKLOAD_TAIL + "MODEL_ZOO_SELECT_OUTCOME_RAW=decided\n",
    )
    assert res.returncode == 0, res.stderr
    assert _MARKER.findall(res.stdout) == ["decided"]


def test_a_genuine_workload_failure_still_exits_non_zero_with_no_marker(tmp_path):
    """A resource kill / exception / timeout is unchanged: non-zero, no marker.

    The stdout carries no outcome line because the workload died before
    printing one — which is exactly the state the SF must keep treating as a
    broken stage.
    """
    res = _run_launcher_block(
        tmp_path,
        ssm_stdout="Traceback (most recent call last):\nMemoryError\n",
        ssm_rc=1,
    )
    assert res.returncode != 0
    assert _MARKER.findall(res.stdout) == []


def test_a_run_that_reports_no_outcome_is_not_a_success(tmp_path):
    """Exit 0 from the workload with no outcome line is a FAILURE here.

    Otherwise a workload that silently stopped reporting would read as a
    clean select to every machine consumer.
    """
    res = _run_launcher_block(tmp_path, ssm_stdout=_WORKLOAD_TAIL, ssm_rc=0)
    assert res.returncode != 0
    assert _MARKER.findall(res.stdout) == []
    assert "MODEL_ZOO_SELECT_OUTCOME_RAW" in res.stderr


def test_the_marker_is_an_anchored_line_of_its_own(tmp_path):
    """The SF Choice matches ``*MODEL_ZOO_SELECT_OUTCOME: unservable*`` on
    StandardOutputContent, so the line carries nothing else."""
    res = _run_launcher_block(
        tmp_path,
        ssm_stdout=_WORKLOAD_TAIL + "MODEL_ZOO_SELECT_OUTCOME_RAW=unservable\n",
    )
    lines = [ln for ln in res.stdout.splitlines() if "MODEL_ZOO_SELECT_OUTCOME:" in ln]
    assert lines == ["MODEL_ZOO_SELECT_OUTCOME: unservable"]


def test_the_workload_refuses_to_report_an_unknown_outcome():
    """The embedded workload never defaults a missing outcome to success."""
    source = _LAUNCHER.read_text()
    assert "_outcome = board.get('arena_outcome')" in source
    assert "if _outcome not in ('decided', 'unservable'):" in source
    assert "raise SystemExit(" in source
    # and the outcome line is the LAST thing it prints, because the SSM body
    # hands the launcher only the tail of the output (next test).
    assert source.index("MODEL_ZOO_SELECT_OUTCOME_RAW=%s") > source.index(
        "print('  Promoted:       %s' % board.get('promoted'))"
    )


def test_the_select_body_hands_ssm_only_a_bounded_tail():
    """rehearsal-2026-09-25-1: the select log passed SSM's ~24,000-character
    inline cap, SSM kept the HEAD and dropped the trailing outcome line, and a
    correct ``unservable`` verdict failed the stage as outcome-less.

    The SSM body must send the workload's output to a file and print only a
    tail comfortably under that cap, while still exiting with the workload's
    own status so a genuine failure stays a failure.
    """
    block = _select_only_block()
    body = block[block.index("<<'ZOOSEL'"):block.index("\nZOOSEL\n")]
    run_line = next(
        ln for ln in body.splitlines()
        if "krepis.ssm_log_capture run --slug spot-model-zoo-select" in ln
    )
    assert ">/tmp/spot-model-zoo-select.stdout 2>&1 || _ZOO_RC=$?" in run_line
    tail = re.search(r"^tail -c (\d+) /tmp/spot-model-zoo-select\.stdout$", body, re.M)
    assert tail, "the select body no longer prints a bounded tail"
    assert int(tail.group(1)) <= 16000
    assert body.rstrip().splitlines()[-1] == 'exit "$_ZOO_RC"'


def test_the_select_body_keeps_the_outcome_and_the_exit_code(tmp_path):
    """Run the SSM body's output handling for real, with a stand-in workload
    that prints far past SSM's inline cap before its outcome line."""
    block = _select_only_block()
    body = block[block.index("<<'ZOOSEL'") + len("<<'ZOOSEL'"):block.index("\nZOOSEL\n")]
    lines = body.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("_ZOO_RC=0"))
    tail_part = "\n".join(lines[start:])
    for rc in (0, 3):
        fake = tmp_path / "fake_py.sh"
        fake.write_text(
            "#!/bin/bash\n"
            "for i in $(seq 1 2000); do echo \"log line $i padding padding padding\"; done\n"
            "echo MODEL_ZOO_SELECT_OUTCOME_RAW=unservable\n"
            f"exit {rc}\n"
        )
        fake.chmod(0o755)
        script = tail_part.replace(
            '$PY -m krepis.ssm_log_capture run --slug spot-model-zoo-select '
            '--log /var/log/spot-model-zoo-select.log --bucket "$S3_BUCKET" -- '
            '$PY /tmp/spot-model-zoo-select.py',
            str(fake),
        ).replace("/tmp/spot-model-zoo-select.stdout", str(tmp_path / "out.txt"))
        res = subprocess.run(
            ["bash", "-c", "set -eo pipefail\n" + script],
            capture_output=True, text=True,
        )
        assert res.returncode == rc
        assert len(res.stdout) < 24000
        assert res.stdout.splitlines()[-1] == "MODEL_ZOO_SELECT_OUTCOME_RAW=unservable"


# ── alpha-engine-config-I11611: an idempotent re-run is a declared skip ──────
#
# Weekly failure class K09. The exactly-once guard (config#2252) makes a
# re-run of an already-finalized trading day write NOTHING. The stage-coverage
# assertion grades against this run's window, so before this fix it read both
# declared artifacts (leaderboard latest, trial log) WHOLLY STALE: measured on
# the scheduled Saturday f283a70c (cycle 2026-09-25, after Friday's rehearsal
# finalized the date) and on watch-rerun-2026-10-02-3. The launcher now passes
# the workload's own no-op declaration as `--not-applicable-reason`, which
# krepis turns from STALE into COVERED_NO_OUTPUT. It never excuses MISSING.

_NOOP_MARKER_KEY = "predictor/model_zoo/promotions/2026-10-02.json"


def _recording_lib_python(tmp_path: Path) -> tuple[str, Path]:
    """A LIB_PYTHON stand-in that records each argv element on its own line."""
    log = tmp_path / "lib-python-argv.txt"
    fake = tmp_path / "fake_lib_python.sh"
    fake.write_text(f"""#!/bin/bash
printf '%s\\n' "$@" >> "{log}"
""")
    fake.chmod(0o755)
    return str(fake), log


def _not_applicable_reason(log: Path) -> str:
    argv = log.read_text().splitlines()
    assert "krepis.stage_coverage" in argv, argv
    assert "--not-applicable-reason" in argv, (
        "the ModelZooSelect assertion no longer passes --not-applicable-reason"
    )
    return argv[argv.index("--not-applicable-reason") + 1]


def test_an_idempotent_rerun_declares_its_skip_to_the_coverage_assertion(tmp_path):
    lib_python, log = _recording_lib_python(tmp_path)
    res = _run_launcher_block(
        tmp_path,
        ssm_stdout=(
            _WORKLOAD_TAIL
            + f"MODEL_ZOO_SELECT_IDEMPOTENT_NOOP={_NOOP_MARKER_KEY}\n"
            + "MODEL_ZOO_SELECT_OUTCOME_RAW=unservable\n"
        ),
        lib_python=lib_python,
    )
    assert res.returncode == 0, res.stderr
    # The SF Choice contract is untouched by the declaration.
    assert _MARKER.findall(res.stdout) == ["unservable"]
    reason = _not_applicable_reason(log)
    assert reason.startswith("idempotent no-op:")
    assert _NOOP_MARKER_KEY in reason
    assert "config#2252" in reason


def test_a_fresh_select_declares_no_skip(tmp_path):
    """A select that did its work must be graded on its writes, so the
    declaration is empty, which krepis treats as no declaration at all."""
    lib_python, log = _recording_lib_python(tmp_path)
    res = _run_launcher_block(
        tmp_path,
        ssm_stdout=_WORKLOAD_TAIL + "MODEL_ZOO_SELECT_OUTCOME_RAW=decided\n",
        lib_python=lib_python,
    )
    assert res.returncode == 0, res.stderr
    assert _not_applicable_reason(log) == ""


@pytest.mark.parametrize(
    "line",
    [
        "MODEL_ZOO_SELECT_IDEMPOTENT_NOOP=",
        'MODEL_ZOO_SELECT_IDEMPOTENT_NOOP=x" --enforce "y',
        "MODEL_ZOO_SELECT_IDEMPOTENT_NOOP=a key with spaces",
        "  MODEL_ZOO_SELECT_IDEMPOTENT_NOOP=" + _NOOP_MARKER_KEY,
    ],
)
def test_a_malformed_noop_line_is_not_a_declaration(tmp_path, line):
    """Only a whole, anchored line naming a plain S3 key counts. Anything else
    leaves the run graded on its writes, so a stray log line cannot excuse a
    stage that really did not refresh its outputs."""
    lib_python, log = _recording_lib_python(tmp_path)
    res = _run_launcher_block(
        tmp_path,
        ssm_stdout=_WORKLOAD_TAIL + line + "\nMODEL_ZOO_SELECT_OUTCOME_RAW=decided\n",
        lib_python=lib_python,
    )
    assert res.returncode == 0, res.stderr
    assert _not_applicable_reason(log) == ""


def test_the_workload_declares_the_noop_only_from_the_guard_and_before_the_outcome():
    source = _LAUNCHER.read_text()
    assert "if board.get('idempotent_noop'):" in source
    assert (
        "print('MODEL_ZOO_SELECT_IDEMPOTENT_NOOP=%s' % board.get('promotion_marker_key'))"
        in source
    )
    # The outcome line stays LAST (test_the_workload_refuses_to_report_an_unknown_outcome).
    assert source.index("MODEL_ZOO_SELECT_IDEMPOTENT_NOOP=%s") < source.index(
        "MODEL_ZOO_SELECT_OUTCOME_RAW=%s"
    )


def test_the_noop_board_names_the_marker_the_launcher_reports():
    """The workload prints ``promotion_marker_key`` from the no-op board, so
    the board must carry it, keyed by the same helper the guard reads."""
    import inspect

    src = inspect.getsource(mz._verify_promotion_marker_noop)
    assert '"idempotent_noop": True' in src
    assert '"promotion_marker_key": _promotion_marker_key(date_str)' in src
    assert mz._promotion_marker_key("2026-10-02") == _NOOP_MARKER_KEY
