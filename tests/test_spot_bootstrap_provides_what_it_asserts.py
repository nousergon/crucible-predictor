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
(nousergon-data#1296); this test is the mirror.

An assertion that a tool exists is a PRECONDITION on the image. A bootstrap's
job is to establish preconditions, not to require them — so every `command -v`
guard here must be preceded, in the same script, by something that provides the
binary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _REPO_ROOT / "infrastructure" / "_spot_common.sh"

# How a bootstrap may legitimately provide a binary.
_PROVIDERS = ("dnf install", "yum install", "apt-get install", "curl -", "pip install")


def _bootstrap_block() -> str:
    text = _COMMON.read_text()
    m = re.search(r"bootstrap_spot\(\)\s*\{(.*?)\n\}", text, re.S)
    assert m, "bootstrap_spot() not found in _spot_common.sh"
    return m.group(1)


def _fatal_command_v_guards(block: str) -> list[tuple[str, int]]:
    """(binary, offset) for each `command -v X` guard that exits non-zero."""
    out = []
    for m in re.finditer(r"command -v (\S+)[^\n]*\|\|[^\n]*exit 1", block):
        out.append((m.group(1), m.start()))
    return out


def test_every_asserted_binary_is_installed_first():
    block = _bootstrap_block()
    guards = _fatal_command_v_guards(block)
    assert guards, (
        "no fatal `command -v` guard found — if the bootstrap stopped asserting "
        "its interpreter, this test has lost its subject and must be updated "
        "deliberately, not deleted"
    )
    for binary, offset in guards:
        before = block[:offset]
        provided = any(
            p in line and binary.split(".")[0] in line
            for line in before.splitlines()
            for p in _PROVIDERS
        )
        assert provided, (
            f"_spot_common.sh bootstrap asserts `{binary}` is present but never "
            f"installs it — a bootstrap establishes preconditions, it does not "
            f"require them. This is the 2026-08-11 MorningEnrich failure "
            f"(`ERROR: python3.12 not found`), inherited by DataPhase1 and "
            f"RAGIngestion too."
        )


def test_bootstrap_installs_python312_explicitly():
    """Anchored: 3.12 specifically, since requirements.txt resolves against it
    and a silent fall back to the system python3 is its own drift class."""
    block = _bootstrap_block()
    assert re.search(r"dnf install[^\n]*python3\.12", block), (
        "the bootstrap must install python3.12 explicitly"
    )


@pytest.mark.parametrize("tool", ["git", "gcc"])
def test_bootstrap_installs_the_tools_the_later_steps_use(tool: str):
    """`git clone` runs in this same block, and requirements.txt builds wheels
    from source — both were in the monolith's install line."""
    block = _bootstrap_block()
    install_lines = [ln for ln in block.splitlines() if "dnf install" in ln]
    assert any(tool in ln for ln in install_lines), (
        f"{tool} is used by the bootstrap or the deps step but is not installed"
    )
