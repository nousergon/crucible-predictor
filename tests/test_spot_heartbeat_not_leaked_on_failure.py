"""A background heartbeat must not outlive the script that started it.

The failure this pins, `ne-weekly-freshness-pipeline` execution
`watch-rerun-2026-08-10-9`, 2026-08-11 (alpha-engine-config-I6948):

===========================================  ==========  ==================
Event                                        Time (PT)   Source
===========================================  ==========  ==================
``PredictorTraining`` scheduled, timeout 5400  15:22:14  SF history ev. 136
smoke step FAILED, ``No module named 'flow_doctor'``  15:24:36  spot diagnostics
SSM SIGKILL, ``ResponseCode 137``, ``ExecutionTimedOut``  16:52:14  poll
===========================================  ==========  ==================

The workload died **142 seconds** in. The remaining 87 minutes were spent
holding a pipe open.

``_heartbeat_start`` backgrounds ``krepis.heartbeat emit``, which inherits the
launcher's stdout. Under Step Functions that stdout is the pipe
``krepis.ssm_log_capture`` reads until EOF, and EOF requires *every* writer to
close. Each per-stage script calls ``_heartbeat_stop`` only on the path where
the phase succeeds, so a ``set -e`` abort in between reached the EXIT trap with
the heartbeat still running — and the trap did not stop it.

Two consequences, and the second is the expensive one:

1. the SSM command ran to its full budget and was killed at it, so the SF
   recorded a timeout for a stage that had already failed;
2. ``krepis.ssm_log_capture`` ships to S3 *after* its read loop, so a loop that
   never returns ships nothing. The run's own diagnostics died with it, which is
   why the timeout was diagnosed as "the workload is too slow" — the exact
   inverse of what happened.

krepis 0.50.0 breaks the hang at the chokepoint and kills the orphan group
(``krepis-PR132``). This file pins the launcher's own half: the heartbeat is
stopped on EVERY exit path, so the breaker never has to fire and a stopped
heartbeat keeps meaning "the phase ended".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"
_COMMON = _INFRA / "_spot_common.sh"

#: Every launcher that backgrounds a heartbeat. `_spot_common.sh` carries the
#: shared `cleanup` for the three per-stage scripts; `spot_train.sh` is the
#: monolith and defines its own.
_SCRIPTS_WITH_CLEANUP = (_COMMON, _INFRA / "spot_train.sh")


def _body_of(source: str, func: str) -> str:
    """Return the text of a shell function body, brace-matched."""
    start = source.index("\n%s() {" % func) + 1
    depth = 0
    for idx in range(start, len(source)):
        if source[idx] == "{":
            depth += 1
        elif source[idx] == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise AssertionError("unterminated function %s" % func)


@pytest.mark.parametrize("script", _SCRIPTS_WITH_CLEANUP, ids=lambda p: p.name)
class TestCleanupStopsTheHeartbeat:
    def test_the_exit_trap_stops_the_heartbeat(self, script: Path):
        body = _body_of(script.read_text(encoding="utf-8"), "cleanup")
        assert "_heartbeat_stop" in body, (
            "%s: cleanup() is the EXIT trap and every heartbeat stop outside it "
            "is on a success path — a failing stage leaks the heartbeat, which "
            "holds the SSM log pipe open to the full executionTimeout "
            "(config-I6948)" % script.name
        )

    def test_it_stops_before_anything_that_can_block(self, script: Path):
        # Ordering is the point. cleanup() then calls `aws ec2 terminate-
        # instances`, `aws s3 rm`, and (in _spot_common) a spot relaunch
        # decision — any of which can hang or exit non-zero. A stop placed
        # after them is not reached on the paths that need it.
        body = _body_of(script.read_text(encoding="utf-8"), "cleanup")
        stop_at = body.index("_heartbeat_stop")
        for later in ("terminate-instances", "aws s3 rm", "relaunch-decision"):
            if later in body:
                assert stop_at < body.index(later), (
                    "%s: _heartbeat_stop must precede %r in cleanup()"
                    % (script.name, later)
                )


class TestTheLeakPathThisFileExistsFor:
    """The conditions that made the leak reachable, so a regression is visible."""

    def test_the_stage_scripts_start_a_heartbeat_they_do_not_stop_on_failure(self):
        # Not a defect once cleanup() stops it — but if this stops being true,
        # the guard above is testing something that no longer exists, and a
        # test whose subject has vanished is a false green.
        stage = _INFRA / "spot_predictor_training.sh"
        source = stage.read_text(encoding="utf-8")
        assert "_heartbeat_start" in source
        # Every _heartbeat_stop in the per-stage script sits at top level, i.e.
        # only reached when the preceding run_ssm returned 0.
        assert source.count("_heartbeat_start") == source.count("_heartbeat_stop")

    def test_the_heartbeat_is_backgrounded_and_therefore_inherits_stdout(self):
        body = _body_of(_COMMON.read_text(encoding="utf-8"), "_heartbeat_start")
        assert re.search(r"krepis\.heartbeat emit .*&\s*$", body, re.M), (
            "the leak mechanism is the trailing `&`: a backgrounded child "
            "inherits the launcher's stdout, which under SF is the pipe "
            "krepis.ssm_log_capture reads until EOF"
        )
