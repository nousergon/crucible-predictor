"""A systemd unit whose ExecStart never returns must not be Type=oneshot.

The failure this pins (2026-08-11, ``ne-weekly-freshness-pipeline`` execution
``watch-rerun-2026-08-10-5``): ``infrastructure/_spot_common.sh``'s bootstrap
wrote an ``ec2-spot-watchdog.service`` declaring::

    Type=oneshot
    ExecStart=/usr/local/bin/ec2-spot-watchdog.sh   # while true; do ... done
    RemainAfterExit=yes

``systemctl start`` on a ``Type=oneshot`` unit BLOCKS until ExecStart exits,
and for oneshot units ``TimeoutStartSec`` defaults to infinity — so the
bootstrap hung until SSM gave up: ``status=TimedOut``, stderr ending on the
``systemctl enable`` symlink line with nothing after it. PredictorTraining
failed, took Branch B of the weekly pipeline down, and the spot instance was
paid for a boot that never ran a workload.

``nousergon-data`` carries a twin of this bootstrap helper, hit the identical
defect on 2026-08-10, and fixed it in nousergon-data#1294 with this test as
the guard. The defect stayed live here for another day because the fix was
applied to the instance rather than the class — this file is the mirror
(AGENTS.md: when a SOTA pattern exists in another alpha-engine repo, mirror
it rather than inventing a parallel shortcut).

Unit files here are written INSIDE bash heredocs, so no systemd linter sees
them. This test parses them out of the shell scripts and checks the one
property whose violation hangs a pipeline: an endless ExecStart under a
service type that waits for it to finish.

Scope note: a genuinely one-shot job (a timer-driven backup, a drift check)
is correctly ``Type=oneshot`` and is not flagged — the test keys on the
ExecStart script's own body containing an unbounded loop, not on the type.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INFRA = _REPO_ROOT / "infrastructure"

# A service type whose `systemctl start` waits for ExecStart to exit.
_BLOCKING_TYPES = {"oneshot"}
# Shapes that never return on their own.
_ENDLESS = (
    re.compile(r"while\s+true"),
    re.compile(r"while\s+:\s*;"),
    re.compile(r"for\s*\(\(\s*;\s*;\s*\)\)"),
)


def _shell_scripts() -> list[Path]:
    return sorted(p for p in _INFRA.rglob("*.sh") if p.is_file())


def _units_in(text: str) -> list[tuple[str, str]]:
    """Return (unit_body, whole_file_text) for each embedded unit file."""
    return [(m.group(0), text) for m in re.finditer(r"\[Unit\].*?\[Install\]", text, re.S)]


def _execstart_target(unit: str) -> str | None:
    m = re.search(r"^ExecStart=(\S+)", unit, re.M)
    return m.group(1) if m else None


def _service_type(unit: str) -> str:
    m = re.search(r"^Type=(\S+)", unit, re.M)
    return m.group(1) if m else "simple"


def _body_of(script_path: str, text: str) -> str:
    """The inline heredoc body written to *script_path* in the same file.

    Bootstrap scripts write their ExecStart target in the same heredoc that
    declares the unit, e.g. ``cat > /usr/local/bin/x.sh <<'WDSH' ... WDSH``.
    """
    name = re.escape(script_path)
    m = re.search(rf"cat\s*>\s*{name}\s*<<'?(\w+)'?\n(.*?)\n\1", text, re.S)
    return m.group(2) if m else ""


@pytest.mark.parametrize("script", _shell_scripts(), ids=lambda p: p.name)
def test_endless_execstart_is_never_a_blocking_service_type(script: Path):
    text = script.read_text()
    for unit, whole in _units_in(text):
        target = _execstart_target(unit)
        if not target:
            continue
        body = _body_of(target, whole)
        if not body:
            continue  # ExecStart lives outside this file — nothing to assert
        if not any(rx.search(body) for rx in _ENDLESS):
            continue
        stype = _service_type(unit)
        assert stype not in _BLOCKING_TYPES, (
            f"{script.name}: {target} runs an unbounded loop but its unit "
            f"declares Type={stype}. `systemctl start` will block until "
            f"ExecStart exits — which it never does — and the caller dies "
            f"when its own timeout kills it (2026-08-11: the weekly SF's "
            f"PredictorTraining bootstrap, status=TimedOut). Use Type=simple."
        )


def test_the_spot_watchdog_unit_is_a_simple_service():
    """Anchored assertion on the specific unit that hung the pipeline.

    ``_spot_common.sh`` no longer carries this unit as literal text
    (alpha-engine-config-I4992/I6922 cutover: ``bootstrap_spot()`` now calls
    ``krepis.spot_bootstrap render``) — asserted against the RENDERED script
    instead, which is what actually reaches the spot box.
    """
    from tests._spot_bootstrap_render_support import rendered_script

    text = rendered_script()
    units = [u for u, _ in _units_in(text) if "ec2-spot-watchdog" in u or "EC2 Spot Watchdog" in u]
    assert units, "ec2-spot-watchdog unit not found in the rendered bootstrap"
    for unit in units:
        assert _service_type(unit) == "simple", unit
        assert "RemainAfterExit" not in unit, (
            "RemainAfterExit belongs to oneshot units; a supervised daemon "
            "uses Restart= instead"
        )


def test_the_watchdog_enable_call_is_time_bounded():
    """A future regression must fail loudly in seconds, not silently at the
    SSM budget. The 2026-08-11 hang produced NO stderr past the symlink line,
    which is why it read as a slow bootstrap rather than a blocked systemctl.

    Asserted against the RENDERED script — see the docstring on the test
    above for why ``_spot_common.sh``'s own text no longer carries this.
    """
    from tests._spot_bootstrap_render_support import rendered_script

    text = rendered_script()
    assert re.search(r"timeout\s+\d+\s+systemctl\s+enable\s+--now\s+ec2-spot-watchdog", text), (
        "the watchdog enable call must be wrapped in `timeout N systemctl "
        "enable --now ec2-spot-watchdog` with an explicit error message"
    )
