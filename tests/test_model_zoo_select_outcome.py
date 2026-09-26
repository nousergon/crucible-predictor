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


def _run_launcher_block(tmp_path: Path, *, ssm_stdout: str, ssm_rc: int = 0):
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
                'LIB_PYTHON="/usr/bin/true"',
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
    # and the outcome line is the LAST thing it prints: SSM caps inline
    # StandardOutputContent at 24KB and rotates the buffer past it.
    assert source.index("MODEL_ZOO_SELECT_OUTCOME_RAW=%s") > source.index(
        "print('  Promoted:       %s' % board.get('promoted'))"
    )


# ── alpha-engine-config-I11605: the outcome line past the inline SSM cap ──────


def _run_launcher_block_with_s3_copy(tmp_path: Path, *, inline: str, s3_copy: str | None):
    """Like ``_run_launcher_block``, plus a stubbed ``aws`` that serves the
    uncapped S3 copy of the workload's stdout from a local directory."""
    s3_root = tmp_path / "s3-ssm-output"
    if s3_copy is not None:
        leaf = s3_root / "cmd-id" / "i-0abc" / "awsrunShellScript" / "0.awsrunShellScript"
        leaf.mkdir(parents=True)
        (leaf / "stdout").write_text(s3_copy)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_aws = bindir / "aws"
    # `aws s3 cp <src> <dest> --recursive ...` -> copy the fake prefix to dest.
    fake_aws.write_text(
        "#!/usr/bin/env bash\n"
        '[ "$1 $2" = "s3 cp" ] || exit 1\n'
        f'[ -d "{s3_root}" ] && cp -r "{s3_root}/." "$4"\n'
        "exit 0\n"
    )
    fake_aws.chmod(0o755)
    harness = tmp_path / "harness.sh"
    out_file = tmp_path / "ssm-stdout.txt"
    out_file.write_text(inline)
    harness.write_text(
        "\n".join(
            [
                "set -euo pipefail",
                'MODE="select-only"',
                '_RUN_TOKEN_EXPORT=""',
                "MAX_RUNTIME_SECONDS=1",
                'LIB_PYTHON="/usr/bin/true"',
                '_COVERAGE_STAGE="ModelZooSelect"',
                '_STAGE_WINDOW_START="2026-09-19T00:00:00Z"',
                'EXECUTION_RUN_DATE="2026-09-19"',
                '_S3_STAGING="s3://bkt/tmp/spot_train/run"',
                "_ZOO_S3_RETRY_SLEEP=0",
                "print_banner() { :; }",
                "emit_heartbeat() { :; }",
                'run_ssm() { cat "$FAKE_SSM_STDOUT"; return 0; }',
                "",
                _select_only_block(),
            ]
        )
    )
    env = dict(os.environ)
    env["FAKE_SSM_STDOUT"] = str(out_file)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    return subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, env=env, cwd=tmp_path,
    )


_TRUNCATED_INLINE = "x" * 200 + "\n--output truncated--\n"


def test_an_outcome_past_the_inline_cap_is_read_from_the_uncapped_s3_copy(tmp_path):
    """MEASURED on rehearsal-2026-09-25-1: the workload printed
    ``MODEL_ZOO_SELECT_OUTCOME_RAW=unservable`` last, the inline copy was
    truncated before it, and the stage failed as outcome-less."""
    res = _run_launcher_block_with_s3_copy(
        tmp_path,
        inline=_TRUNCATED_INLINE,
        s3_copy=_WORKLOAD_TAIL + "MODEL_ZOO_SELECT_OUTCOME_RAW=unservable\n",
    )
    assert res.returncode == 0, res.stderr
    assert _MARKER.findall(res.stdout) == ["unservable"]
    assert "uncapped S3 copy" in res.stderr


def test_an_uncapped_copy_without_an_outcome_still_fails(tmp_path):
    """The fallback reads the workload's own line; it never defaults one."""
    res = _run_launcher_block_with_s3_copy(
        tmp_path, inline=_TRUNCATED_INLINE, s3_copy=_WORKLOAD_TAIL,
    )
    assert res.returncode != 0
    assert _MARKER.findall(res.stdout) == []


def test_a_missing_s3_copy_still_fails(tmp_path):
    res = _run_launcher_block_with_s3_copy(tmp_path, inline=_TRUNCATED_INLINE, s3_copy=None)
    assert res.returncode != 0
    assert _MARKER.findall(res.stdout) == []
