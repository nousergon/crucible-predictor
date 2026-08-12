"""The deps step must not be the only thing that knows what pip did.

The failure this pins (2026-08-11, `ne-weekly-freshness-pipeline` execution
`watch-rerun-2026-08-10-9`)::

    RuntimeError: flow-doctor is not installed but a flow_doctor_yaml was
    provided. Install via nousergon-lib[flow_doctor] or add
    flow-doctor[diagnosis] to requirements: No module named 'flow_doctor'

`flow-doctor` is reachable from two lines of ``requirements.txt`` — the
``krepis[flow_doctor]`` pin and the ``nousergon-lib[...,flow-doctor,...]``
pin — and `krepis.logging.setup_logging` raises hard in a deployed runtime
when it is absent, by design: a silently-degraded error monitor is worse
than a stopped pipeline.

The install that was supposed to provide it exited **0**. Its entire
surviving output, because the step piped pip through ``tail -1``, was::

    WARNING: Running pip as the 'root' user can result in broken permissions

Nothing in that record distinguishes "resolved the extra" from "skipped it"
from "never saw it", and the import failure lands a minute later in a
different SSM step with no upstream to read. pip reports a dropped extra as
a WARNING on a *successful* exit, so the one line `tail -1` keeps is the one
line guaranteed not to carry it.

The fleet copy of this step is
``krepis.spot_bootstrap.render_install_deps``; the two are kept in step
until this repo consumes the rendered version (config-I6949).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_COMMON = Path(__file__).resolve().parent.parent / "infrastructure" / "_spot_common.sh"


@pytest.fixture(scope="module")
def install_deps_body() -> str:
    text = _COMMON.read_text()
    # Anchor on the definition, not the header comment that also names it.
    start = text.index("\ninstall_deps() {")
    end = text.index("\n}", start)
    return text[start:end]


def test_pip_output_is_not_discarded_by_tail(install_deps_body: str):
    assert "| tail -1" not in install_deps_body


def test_a_failed_install_dumps_the_captured_log(install_deps_body: str):
    # Preserving the log buys nothing if the failure path does not print it —
    # the SSM step output is the only surface anyone reads after the fact.
    assert 'tail -80 "$_pip_log" >&2' in install_deps_body
    assert "exit 1" in install_deps_body


def test_a_dropped_extra_is_surfaced_on_a_successful_exit(install_deps_body: str):
    # The exact shape of the 2026-08-11 failure: rc=0, extra missing.
    assert "does not provide the extra" in install_deps_body


def test_the_environment_is_checked_before_the_import_that_would_fail(
    install_deps_body: str,
):
    assert "pip check" in install_deps_body
