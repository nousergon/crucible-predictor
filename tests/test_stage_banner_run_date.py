"""Launcher banners label themselves with the cycle date, never the UTC day.

alpha-engine-config-I11475. The 2026-09-23 weekly rehearsal (run_date
2026-09-23) crossed 00:00 UTC, and the model-zoo-select banner read
``MODEL-ZOO SELECT-ONLY — 2026-09-24`` because every launcher printed
``$(date +%Y-%m-%d)``, the box's UTC calendar day. The same launcher's
stage-coverage assertion files its verdict under ``$EXECUTION_RUN_DATE``, so
one stage named two different days.

``_spot_common.sh::stage_run_date`` is the one definition, mirroring
nousergon-data ``infrastructure/_stage_window.sh::stage_run_date``:
``$EXECUTION_RUN_DATE`` first, then the America/New_York day, never UTC.
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"
_COMMON = _INFRA / "_spot_common.sh"

#: Every launcher that prints a dated banner. The SF runs the first two
#: (PredictorTraining, ModelZooSelect); spot_train.sh is the manual launcher.
_BANNER_LAUNCHERS = (
    "spot_predictor_training.sh",
    "spot_model_zoo_select.sh",
    "spot_train.sh",
)


def _stage_run_date_fn() -> str:
    """Lift ``stage_run_date`` out of ``_spot_common.sh``.

    The function is lifted rather than the whole file sourced: sourcing it runs
    the file's global defaults under ``set -euo pipefail``, which is not what
    this test is about.
    """
    m = re.search(r"^stage_run_date\(\) \{\n.*?^\}\n", _COMMON.read_text(), re.S | re.M)
    assert m, "stage_run_date() not found in _spot_common.sh"
    return m.group(0)


def _run(env_extra: dict[str, str], *, drop: tuple[str, ...] = ()) -> str:
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(env_extra)
    proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell input
        ["bash", "-c", _stage_run_date_fn() + "stage_run_date"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return proc.stdout


def test_stage_run_date_is_the_sf_run_date_when_exported():
    assert _run({"EXECUTION_RUN_DATE": "2026-09-23"}) == "2026-09-23"


@pytest.mark.parametrize("unset", [True, False])
def test_stage_run_date_falls_back_to_the_exchange_day_not_utc(unset):
    drop = ("EXECUTION_RUN_DATE",) if unset else ()
    extra = {} if unset else {"EXECUTION_RUN_DATE": ""}
    before = datetime.now(ZoneInfo("America/New_York")).date()
    got = _run(extra, drop=drop)
    after = datetime.now(ZoneInfo("America/New_York")).date()
    assert got in {before.isoformat(), after.isoformat()}


def test_fallback_uses_new_york_time_not_the_box_clock():
    """Between 00:00 UTC and midnight ET the UTC day is already tomorrow on
    the exchange's calendar. Run the helper under a shell TZ that is 18 hours
    ahead of New York (a different calendar day for most of every day); the
    result must still be New York's day, not the shell clock's."""
    ny_before = datetime.now(ZoneInfo("America/New_York")).date()
    got = _run({"TZ": "Pacific/Kiritimati"}, drop=("EXECUTION_RUN_DATE",))
    ny_after = datetime.now(ZoneInfo("America/New_York")).date()
    assert got in {ny_before.isoformat(), ny_after.isoformat()}


@pytest.mark.parametrize("launcher", _BANNER_LAUNCHERS)
def test_launcher_banner_uses_the_cycle_date(launcher):
    text = (_INFRA / launcher).read_text()
    banner_lines = [
        line
        for line in text.splitlines()
        if line.lstrip().startswith("echo") and " — $(" in line
    ]
    assert banner_lines, f"{launcher}: no dated banner line found"
    for line in banner_lines:
        assert "$(stage_run_date)" in line, line
        assert "$(date" not in line, line
