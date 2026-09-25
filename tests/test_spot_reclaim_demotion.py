"""A relaunch after a spot reclaim must not walk back into the reclaimed pool,
the final attempt must go on-demand, and a launch-time refusal must reach the
EXIT handler (alpha-engine-config-I11573).

This is the predictor port of nousergon-data's
``tests/test_spot_reclaim_demotion.py`` (I11565, nousergon-data PR1944, and
I11574, nousergon-data PR1958).

THE INCIDENT (rehearsal-2026-09-24-1, DataPhase1). Four consecutive boxes, all
c6i.large, all ended ``Server.SpotInstanceTermination``: each relaunch walked
``INSTANCE_TYPES`` from its head. This repo's launchers walk their own
rotation (r5.large first) the same way, and nothing ever escalated to
on-demand. Separately, every per-stage script here armed its cleanup trap on
the line AFTER ``spot_launch``, so an ec2_spot capacity (64) or spot-quota
(65) refusal exited with no handler at all.

These tests drive the REAL launch path (``_spot_common.sh``'s
``spot_launch`` and the ``cleanup`` EXIT trap, including the
relaunch ``exec``) with stand-ins for ``aws`` and for ``$LIB_PYTHON`` (the
krepis CLIs). The stand-in ``krepis.ec2_spot launch`` records its argv and
"launches" the head of ``--types`` in the head of ``--subnets``, which is what
the real rotation does when capacity is available, so the recorded argv is
the ordering the real launcher would have bought.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"
_HELPER = _INFRA / "_spot_relaunch.sh"
_COMMON = _INFRA / "_spot_common.sh"

_TYPES = "r5.large,r5a.large,r6i.large,m5.large"
_SUBNETS = "subnet-a61ec0fb,subnet-1e58307a,subnet-789d3857,subnet-c670118d,subnet-7cff7c43,subnet-e07166ec"

#: Every per-stage script that launches through ``spot_launch``.
_STAGE_SCRIPTS = (
    "spot_model_zoo_select.sh",
    "spot_predictor_training.sh",
    "spot_train_spec_dispatch.sh",
)


@pytest.fixture(autouse=True)
def _requires_bash():
    if shutil.which("bash") is None:  # pragma: no cover - bash is a hard dep
        pytest.skip("bash unavailable")


# ── spot_demote_csv (pure) ───────────────────────────────────────────────────


def _demote(lst: str, demoted: str) -> str:
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail; source "{_HELPER}"; spot_demote_csv "$1" "$2"',
            "_",
            lst,
            demoted,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


@pytest.mark.parametrize(
    ("lst", "demoted", "expected"),
    [
        ("a,b,c", "", "a,b,c"),
        ("a,b,c", "a", "b,c,a"),
        # Reclaim order is kept; the most recently reclaimed pool goes LAST.
        ("a,b,c,d", "a,c", "b,d,a,c"),
        ("a,b,c,d", "a,c,a", "b,d,c,a"),
        # A demoted value the list does not contain is never ADDED.
        ("a,b,c", "z", "a,b,c"),
        # A single-type override keeps its only type (never dropped).
        ("c6i.large", "c6i.large", "c6i.large"),
        # Everything reclaimed: pure reorder, nothing lost.
        ("a,b", "b,a", "b,a"),
        # No substring confusion between dotted names.
        ("c5.large,c5a.large", "c5.large", "c5a.large,c5.large"),
    ],
)
def test_spot_demote_csv(lst: str, demoted: str, expected: str) -> None:
    assert _demote(lst, demoted) == expected


# ── stand-ins ────────────────────────────────────────────────────────────────

_FAKE_PYTHON = r"""#!/usr/bin/env bash
if [ "$1" = "-c" ]; then exec "$REAL_PYTHON" "$@"; fi
if [ "$1" = "-m" ] && [ "$2" = "krepis.ec2_spot" ] && [ "$3" = "launch" ]; then
  shift 3
  printf '%s\n' "$*" >> "$FAKE_DIR/launch.log"
  n=$(wc -l < "$FAKE_DIR/launch.log" | tr -d ' ')
  iid="i-fake$n"
  types="" subnets="" market="spot"
  while [ $# -gt 0 ]; do
    case "$1" in
      --types) types="$2"; shift 2 ;;
      --subnets) subnets="$2"; shift 2 ;;
      --no-spot) market="on-demand"; shift ;;
      *) shift ;;
    esac
  done
  printf '%s\t%s\n' "${types%%,*}" "${subnets%%,*}" > "$FAKE_DIR/pool-$iid"
  echo "$market" > "$FAKE_DIR/market-$iid"
  echo "$iid"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "krepis.ec2_spot" ] && [ "$3" = "relaunch-decision" ]; then
  iid="" attempt=1 max=2
  while [ $# -gt 0 ]; do
    case "$1" in
      --instance-id) iid="$2"; shift 2 ;;
      --attempt) attempt="$2"; shift 2 ;;
      --max-attempts) max="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  # An on-demand box has no spot request and is never classified a reclaim.
  cls="${FAKE_CLASSIFICATION:-reclaim}"
  [ "$(cat "$FAKE_DIR/market-$iid" 2>/dev/null)" = "on-demand" ] && cls="other"
  if [ "$cls" = "reclaim" ] && [ "$attempt" -lt "$max" ]; then
    echo '{"attempts_remaining": 1, "classification": "reclaim", "reason": "reclaim", "relaunch": true, "verdict": "relaunch"}'
  else
    echo "{\"attempts_remaining\": 0, \"classification\": \"$cls\", \"reason\": \"hold\", \"relaunch\": false, \"verdict\": \"hold\"}"
  fi
  exit 0
fi
# krepis.spot_evidence teardown, anything else: no-op.
exit 0
"""

_FAKE_AWS = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_DIR/aws.log"
if [ "$1" = "ec2" ] && [ "$2" = "describe-instances" ]; then
  iid=""
  while [ $# -gt 0 ]; do
    case "$1" in --instance-ids) iid="$2"; shift 2 ;; *) shift ;; esac
  done
  if [ -f "$FAKE_DIR/pool-$iid" ]; then cat "$FAKE_DIR/pool-$iid"; else echo "None"; fi
  exit 0
fi
exit 0
"""

# A per-stage launcher shaped exactly like spot_predictor_training.sh's
# control flow: source, declare identity, launch (spot_launch arms the EXIT
# trap itself, I11573), run a workload that dies.
_STAGE = """#!/usr/bin/env bash
set -euo pipefail
source "{infra}/_spot_common.sh"
_SPOT_NAME="${{_SPOT_NAME:-predictor-training}}"
_SSM_SLUG="${{_SSM_SLUG:-spot-full-training}}"
_PROCESS_NAME="${{_PROCESS_NAME:-predictor-training}}"
MAX_RUNTIME_SECONDS="${{MAX_RUNTIME_SECONDS:-5400}}"
_ORIG_ARGS=("$@")
spot_launch
echo "ATTEMPT $SPOT_ATTEMPT ran on $_INSTANCE_ID"
exit "${{FAKE_WORKLOAD_RC:-7}}"
"""


class _Rig:
    def __init__(self, tmp_path: Path) -> None:
        self.dir = tmp_path / "fake"
        self.dir.mkdir()
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.state = tmp_path / "state"
        (self.bin / "aws").write_text(_FAKE_AWS)
        (self.bin / "aws").chmod(0o755)
        self.python = tmp_path / "fake-python"
        self.python.write_text(_FAKE_PYTHON)
        self.python.chmod(0o755)
        self.stage = tmp_path / "stage.sh"
        self.stage.write_text(_STAGE.format(infra=_INFRA))
        self.stage.chmod(0o755)
        self.home = tmp_path / "home"
        self.home.mkdir()

    def env(self, **extra: str) -> dict[str, str]:
        env = {
            "PATH": f"{self.bin}:/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(self.home),
            "LIB_PYTHON": str(self.python),
            "REAL_PYTHON": sys.executable,
            "FAKE_DIR": str(self.dir),
            "SPOT_RECLAIM_STATE_DIR": str(self.state),
            "SPOT_DESCRIBE_RETRY_SECONDS": "0",
            "EXECUTION_RUN_DATE": "2026-09-24",
        }
        env.update(extra)
        return env

    def run(self, script: Path, *args: str, **extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(script), *args],
            capture_output=True,
            text=True,
            env=self.env(**extra),
            timeout=120,
        )

    def launches(self) -> list[dict[str, object]]:
        log = self.dir / "launch.log"
        out = []
        for line in log.read_text().splitlines() if log.exists() else []:
            argv = line.split()
            out.append(
                {
                    "types": argv[argv.index("--types") + 1].split(","),
                    "subnets": argv[argv.index("--subnets") + 1].split(","),
                    "on_demand": "--no-spot" in argv,
                    "tags": [argv[i + 1] for i, a in enumerate(argv) if a == "--extra-tag"],
                }
            )
        return out

    def terminated(self) -> list[str]:
        log = self.dir / "aws.log"
        lines = log.read_text().splitlines() if log.exists() else []
        return [ln for ln in lines if ln.startswith("ec2 terminate-instances")]


@pytest.fixture
def rig(tmp_path: Path) -> _Rig:
    return _Rig(tmp_path)


def _why(proc: subprocess.CompletedProcess) -> str:
    return f"\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


# ── reclaim demotion + on-demand escalation (I11565 port) ────────────────────


def test_first_attempt_is_unchanged_spot_in_declared_order(rig: _Rig) -> None:
    proc = rig.run(rig.stage, RUN_TOKEN="exec-1", FAKE_WORKLOAD_RC="0")
    assert proc.returncode == 0, _why(proc)
    [launch] = rig.launches()
    assert launch["types"] == _TYPES.split(",")
    assert launch["subnets"] == _SUBNETS.split(",")
    assert launch["on_demand"] is False
    # Provenance is recorded on EVERY launch, as launch_with_fallback does, so
    # a spot launch is countable as spot_ok rather than provenance_missing.
    assert launch["tags"] == ["LaunchMarket=spot", "LaunchReason=spot_ok"]


def test_rehearsal_shape_relaunch_demotes_the_reclaimed_pool_and_goes_on_demand(
    rig: _Rig,
) -> None:
    """Attempt 1 reclaimed -> attempt 2/2 is the final attempt: on-demand, with
    the reclaimed type and subnet at the back of the rotation."""
    proc = rig.run(rig.stage, RUN_TOKEN="rehearsal-2026-09-24-1", MAX_SPOT_ATTEMPTS="2")
    first, second = rig.launches()

    assert first["types"][0] == "r5.large" and first["on_demand"] is False
    assert second["on_demand"] is True, _why(proc)
    assert second["tags"] == ["LaunchMarket=on-demand", "LaunchReason=force_on_demand"]
    assert second["types"][0] != "r5.large"
    assert second["types"][-1] == "r5.large"
    assert sorted(second["types"]) == sorted(_TYPES.split(","))
    assert second["subnets"][-1] == "subnet-a61ec0fb"
    assert sorted(second["subnets"]) == sorted(_SUBNETS.split(","))
    # The on-demand attempt's own failure is not a reclaim: no third launch,
    # and the launcher's exit status is the workload's.
    assert len(rig.launches()) == 2
    assert proc.returncode == 7, _why(proc)
    assert len(rig.terminated()) == 2, _why(proc)


def test_default_budget_is_one_spot_then_on_demand(rig: _Rig) -> None:
    """This repo's default budget is MAX_SPOT_ATTEMPTS=2, so a reclaim on the
    first attempt goes straight to the on-demand rung."""
    proc = rig.run(rig.stage, RUN_TOKEN="exec-default")
    launches = rig.launches()
    assert [launch["on_demand"] for launch in launches] == [False, True], _why(proc)
    assert proc.returncode == 7, _why(proc)


def test_three_attempt_budget_goes_on_demand_only_on_the_last(rig: _Rig) -> None:
    proc = rig.run(rig.stage, RUN_TOKEN="exec-3", MAX_SPOT_ATTEMPTS="3")
    l1, l2, l3 = rig.launches()
    assert [launch["on_demand"] for launch in (l1, l2, l3)] == [False, False, True], _why(proc)
    assert (l1["types"][0], l2["types"][0]) == ("r5.large", "r5a.large")
    assert l3["types"][-2:] == ["r5.large", "r5a.large"]


def test_sf_reissue_inherits_the_demotion(rig: _Rig) -> None:
    """A re-run of the script from attempt 1 on the same launcher box, in the
    same execution, must start OFF the pools this execution saw reclaimed,
    including the reclaim whose verdict was hold (budget exhausted)."""
    # MAX_SPOT_ATTEMPTS=1: attempt 1 is reclaimed and the verdict is HOLD, so
    # only the record carries the lesson forward.
    first = rig.run(rig.stage, RUN_TOKEN="exec-reissue", MAX_SPOT_ATTEMPTS="1")
    assert first.returncode == 7, _why(first)
    assert "spot reclaim recorded: r5.large@subnet-a61ec0fb" in first.stderr, _why(first)

    reissue = rig.run(rig.stage, RUN_TOKEN="exec-reissue", MAX_SPOT_ATTEMPTS="1")
    launch_1, launch_2 = rig.launches()
    assert launch_1["types"][0] == "r5.large"
    assert launch_2["types"][0] == "r5a.large", _why(reissue)
    assert launch_2["types"][-1] == "r5.large"
    assert launch_2["subnets"][0] == "subnet-1e58307a"
    assert launch_2["subnets"][-1] == "subnet-a61ec0fb"
    # A reissue's first attempt is not a relaunch: still spot.
    assert launch_2["on_demand"] is False


def test_another_execution_does_not_inherit_the_demotion(rig: _Rig) -> None:
    rig.run(rig.stage, RUN_TOKEN="exec-A", MAX_SPOT_ATTEMPTS="1")
    rig.run(rig.stage, RUN_TOKEN="exec-B", MAX_SPOT_ATTEMPTS="1", FAKE_WORKLOAD_RC="0")
    _, other = rig.launches()
    assert other["types"] == _TYPES.split(",")


def test_adhoc_runs_without_run_token_do_not_share_demotions(rig: _Rig) -> None:
    rig.run(rig.stage, MAX_SPOT_ATTEMPTS="1")
    rig.run(rig.stage, MAX_SPOT_ATTEMPTS="1", FAKE_WORKLOAD_RC="0")
    _, second = rig.launches()
    assert second["types"] == _TYPES.split(",")


def test_adhoc_relaunch_chain_still_demotes_without_run_token(rig: _Rig) -> None:
    rig.run(rig.stage, MAX_SPOT_ATTEMPTS="2")
    _, second = rig.launches()
    assert second["types"][-1] == "r5.large"
    assert second["on_demand"] is True


def test_workload_failure_is_not_retried_and_records_nothing(rig: _Rig) -> None:
    proc = rig.run(rig.stage, RUN_TOKEN="exec-bug", FAKE_CLASSIFICATION="other")
    assert proc.returncode == 7, _why(proc)
    assert len(rig.launches()) == 1
    rig.run(rig.stage, RUN_TOKEN="exec-bug", FAKE_CLASSIFICATION="other")
    _, reissue = rig.launches()
    assert reissue["types"] == _TYPES.split(",")
    assert "spot reclaim recorded" not in proc.stderr


def test_final_attempt_on_demand_can_be_turned_off(rig: _Rig) -> None:
    rig.run(
        rig.stage,
        RUN_TOKEN="exec-nood",
        MAX_SPOT_ATTEMPTS="2",
        SPOT_FINAL_ATTEMPT_ON_DEMAND="0",
    )
    first, second = rig.launches()
    assert second["on_demand"] is False
    assert second["tags"] == ["LaunchMarket=spot", "LaunchReason=spot_ok"]
    # Demotion still applies.
    assert second["types"][-1] == "r5.large"
    # ...and the spot final attempt, reclaimed again, still gives up at MAX.
    assert len(rig.launches()) == 2


def test_single_type_override_still_launches_its_type(rig: _Rig) -> None:
    rig.run(rig.stage, RUN_TOKEN="exec-one", MAX_SPOT_ATTEMPTS="2", INSTANCE_TYPES="c6i.xlarge")
    first, second = rig.launches()
    assert first["types"] == second["types"] == ["c6i.xlarge"]
    assert second["on_demand"] is True


# ── launch-time refusals: exit 64 / 65 before the workload (I11574 port) ─────


def _refuse_first_launch(rig: _Rig, rc: int) -> None:
    """The FIRST ``krepis.ec2_spot launch`` of the run exits ``rc`` without an
    instance id (64 = every pool refused capacity, 65 = spot quota hit)."""
    fake = rig.python.read_text().replace(
        'if [ "$1" = "-m" ] && [ "$2" = "krepis.ec2_spot" ] && [ "$3" = "launch" ]; then\n  shift 3\n',
        'if [ "$1" = "-m" ] && [ "$2" = "krepis.ec2_spot" ] && [ "$3" = "launch" ]; then\n  shift 3\n'
        f'  if [ ! -s "$FAKE_DIR/launch.log" ]; then printf \'%s\\n\' "$*" >> "$FAKE_DIR/launch.log"; exit {rc}; fi\n',
    )
    rig.python.write_text(fake)


def test_stage_capacity_exhausted_at_launch_relaunches_on_demand(rig: _Rig) -> None:
    """ec2_spot exit 64 inside ``spot_launch`` used to exit the stage BEFORE
    its cleanup trap was installed: no classification, no relaunch, no
    on-demand rung. ``spot_launch`` now arms the trap itself."""
    _refuse_first_launch(rig, 64)
    proc = rig.run(
        rig.stage, RUN_TOKEN="exec-stage-cap", MAX_SPOT_ATTEMPTS="2", FAKE_WORKLOAD_RC="0"
    )
    launches = rig.launches()
    assert len(launches) == 2, _why(proc)
    first, second = launches
    assert first["on_demand"] is False
    assert second["on_demand"] is True, _why(proc)
    assert second["tags"] == [
        "LaunchMarket=on-demand",
        "LaunchReason=capacity_exhausted",
    ]
    # Nothing was reclaimed, so nothing is demoted.
    assert second["types"] == _TYPES.split(",")
    assert proc.returncode == 0, _why(proc)
    assert "ATTEMPT 2 ran on i-fake2" in proc.stdout, _why(proc)


def test_stage_spot_quota_at_launch_relaunches_on_demand(rig: _Rig) -> None:
    """ec2_spot exit 65 (account-wide spot quota) is a launch-time refusal too;
    on-demand capacity is a separate quota, so the relaunch goes on-demand
    tagged with lib spot_dispatch's ``quota_exceeded`` reason."""
    _refuse_first_launch(rig, 65)
    proc = rig.run(
        rig.stage, RUN_TOKEN="exec-stage-quota", MAX_SPOT_ATTEMPTS="2", FAKE_WORKLOAD_RC="0"
    )
    launches = rig.launches()
    assert len(launches) == 2, _why(proc)
    assert launches[1]["on_demand"] is True, _why(proc)
    assert launches[1]["tags"] == [
        "LaunchMarket=on-demand",
        "LaunchReason=quota_exceeded",
    ]
    assert proc.returncode == 0, _why(proc)


def test_stage_spot_quota_goes_on_demand_before_the_final_attempt(rig: _Rig) -> None:
    """A spot quota is account-wide: another spot attempt cannot succeed, so
    the very next attempt is on-demand even with budget left."""
    _refuse_first_launch(rig, 65)
    proc = rig.run(
        rig.stage, RUN_TOKEN="exec-quota-3", MAX_SPOT_ATTEMPTS="3", FAKE_WORKLOAD_RC="0"
    )
    launches = rig.launches()
    assert len(launches) == 2, _why(proc)
    assert launches[1]["on_demand"] is True, _why(proc)


def test_stage_capacity_refusal_with_budget_left_retries_spot(rig: _Rig) -> None:
    """Capacity exhaustion is not account-wide: with budget left the next
    attempt is spot again; only the final attempt escalates."""
    _refuse_first_launch(rig, 64)
    proc = rig.run(
        rig.stage, RUN_TOKEN="exec-cap-3", MAX_SPOT_ATTEMPTS="3", FAKE_WORKLOAD_RC="0"
    )
    launches = rig.launches()
    assert len(launches) == 2, _why(proc)
    assert launches[1]["on_demand"] is False, _why(proc)
    assert proc.returncode == 0, _why(proc)


def test_stage_launch_refusal_on_the_final_attempt_gives_up(rig: _Rig) -> None:
    _refuse_first_launch(rig, 64)
    proc = rig.run(rig.stage, RUN_TOKEN="exec-cap-1", MAX_SPOT_ATTEMPTS="1")
    assert len(rig.launches()) == 1
    assert proc.returncode == 64, _why(proc)
    assert "persisted across all 1 attempt(s)" in proc.stderr, _why(proc)
    # Nothing launched, so nothing is terminated.
    assert rig.terminated() == [], _why(proc)


def test_spot_launch_arms_the_exit_trap_before_launching() -> None:
    """Structural: the trap is armed INSIDE ``spot_launch``, ahead of the
    krepis launch call, so no stage script can place it too late, and no
    stage script re-arms it after the launch."""
    text = _COMMON.read_text()
    body = text[text.index("spot_launch() {") :]
    body = body[: body.index("\n# ── Staging teardown")]
    trap_line = "trap 'cleanup \"$_SSM_SLUG\"' EXIT"
    assert trap_line in body
    assert body.index(trap_line) < body.index("-m krepis.ec2_spot launch")
    for stage in _STAGE_SCRIPTS:
        stage_text = (_INFRA / stage).read_text()
        launch_at = stage_text.index("\nspot_launch ")
        assert "trap 'cleanup" not in stage_text[launch_at:], (
            f"{stage}: re-arming the trap after spot_launch is where I11574 hid"
        )


def test_the_common_file_sources_the_one_helper() -> None:
    text = _COMMON.read_text()
    assert '/_spot_relaunch.sh"' in text
    assert "spot_launch_plan" in text
    assert '"${_SPOT_PLAN_MARKET_ARGS[@]}"' in text
    assert "spot_record_reclaim" in text
    assert "export SPOT_RELAUNCH_CAUSE" in text
    assert os.access(_HELPER, os.R_OK)
