"""A spot bootstrap must INSTALL what it asserts is present.

The failure this pins (2026-08-11, `ne-weekly-freshness-pipeline` execution
`watch-rerun-2026-08-10-6`): the bootstrap in ``infrastructure/_spot_common.sh``
carried

    command -v python3.12 >/dev/null || { echo "ERROR: python3.12 not found"; exit 1; }

and nothing that installs it. The AL2023 spot AMI does not ship python3.12, so
PredictorTraining's bootstrap exited 1 the moment it reached that line —
"ERROR: python3.12 not found" — taking Branch B of the weekly pipeline down for
the second consecutive run.

It was latent behind a louder defect: until #461 the watchdog unit blocked the
bootstrap before execution ever got this far, so fixing the hang immediately
exposed this. A component that has never completed carries stacked defects,
each hidden behind the last.

``nousergon-data`` hit and fixed the identical defect in its twin of this file
(nousergon-data#1296); this test was the mirror.

An assertion that a tool exists is a PRECONDITION on the image. A bootstrap's
job is to establish preconditions, not to require them — so every `command -v`
guard must be preceded, in the script that actually runs, by something that
provides the binary.

## Subject moved: the rendered script, not this repo's source

``bootstrap_spot()`` no longer carries the `dnf install`/`command -v` text
inline (alpha-engine-config-I4992/I6922 cutover) — it constructs a
``krepis.spot_bootstrap render`` call. This module now derives the launcher's
actual arguments from the live source and asserts the invariant below against
the script krepis actually renders, which is what an SSM command on the spot
box really executes.
"""

from __future__ import annotations

import re

import pytest

from krepis.spot_bootstrap import PYTHON

from tests._spot_bootstrap_render_support import rendered_script

# How a bootstrap may legitimately provide a binary.
_PROVIDERS = ("dnf install", "yum install", "apt-get install", "curl -", "pip install")


@pytest.fixture(scope="module")
def rendered() -> str:
    return rendered_script()


def _fatal_command_v_guards(script: str) -> list[tuple[str, int]]:
    """(binary, offset) for each `command -v X` guard that exits non-zero."""
    out = []
    for m in re.finditer(r"command -v (\S+)[^\n]*\|\|[^\n]*exit 1", script):
        out.append((m.group(1), m.start()))
    return out


def test_every_asserted_binary_is_installed_first(rendered: str):
    guards = _fatal_command_v_guards(rendered)
    assert guards, (
        "no fatal `command -v` guard found in the rendered bootstrap — if it "
        "stopped asserting its interpreter, this test has lost its subject "
        "and must be updated deliberately, not deleted"
    )
    for binary, offset in guards:
        before = rendered[:offset]
        provided = any(
            p in line and binary.split(".")[0] in line
            for line in before.splitlines()
            for p in _PROVIDERS
        )
        assert provided, (
            f"the rendered bootstrap asserts `{binary}` is present but never "
            f"installs it — a bootstrap establishes preconditions, it does not "
            f"require them. This is the 2026-08-11 MorningEnrich failure "
            f"(`ERROR: python3.12 not found`), inherited by DataPhase1 and "
            f"RAGIngestion too."
        )


def test_bootstrap_installs_python312_explicitly(rendered: str):
    """Anchored: 3.12 specifically, since requirements.txt resolves against it
    and a silent fall back to the system python3 is its own drift class."""
    assert PYTHON == "python3.12", (
        f"krepis.spot_bootstrap.PYTHON is {PYTHON!r}, not python3.12 — this "
        "repo's requirements.txt is resolved against 3.12 specifically; a "
        "renderer bump to a different interpreter version needs a matching "
        "audit of every wheel in requirements.txt before it ships here"
    )
    assert re.search(rf"dnf install[^\n]*{re.escape(PYTHON)}\b", rendered), (
        f"the rendered bootstrap must install {PYTHON} explicitly"
    )


@pytest.mark.parametrize("tool", ["git", "gcc"])
def test_bootstrap_installs_the_tools_the_later_steps_use(rendered: str, tool: str):
    """`git clone` runs in this same script, and requirements.txt builds wheels
    from source — both were in the monolith's install line."""
    install_lines = [ln for ln in rendered.splitlines() if "dnf install" in ln]
    assert any(tool in ln for ln in install_lines), (
        f"{tool} is used by the bootstrap or the deps step but is not installed "
        "in the rendered script"
    )
