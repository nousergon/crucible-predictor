"""The spot_train.sh launcher's ``-m krepis.{ec2_spot,ssm_dispatcher}``
CLI callsites must EXECUTE, not merely import (config#1649).

Bug class this guards (config#1646, 2026-07-03): nousergon-lib 0.81.0
turned ``nousergon_lib.ec2_spot`` / ``nousergon_lib.ssm_dispatcher`` /
``nousergon_lib.ssm_log_capture`` into guard-less re-export shims
(``import krepis.<mod> as _mod; sys.modules[__name__] = _mod`` with no
``if __name__ == "__main__":`` block). Under ``python -m`` (runpy) such a
shim imports, rebinds, falls off the end — **exit 0, nothing runs, no
output**. The 2026-05-27 wrapper-layer fix (nousergon-data#602) moved the
SF's ``ssm_log_capture`` caller to the canonical ``krepis`` path; this
launcher's ``ec2_spot``/``ssm_dispatcher`` callsites were the LAUNCHER-layer
instance of the exact same class and are migrated here (config#1649).

String-pinning wiring tests (``test_spot_train_ssm_lib_chokepoint.py``,
``test_spot_train_reclaim_relaunch.py``) prove the launcher's source text
NAMES the krepis module — they provably cannot prove the module is
*executable* under ``python -m`` (importable-but-not-executable is exactly
how the 2026-07-03 incident hid). This file mirrors
``nousergon-data/tests/test_ssm_log_capture_wrapper_executes.py``: it
extracts the exact module paths ``infrastructure/spot_train.sh`` invokes
via ``-m <module>.{ec2_spot,ssm_dispatcher}`` and proves each one's CLI
actually parses argv and dispatches — entirely offline (no AWS calls): a
guard-less shim prints NOTHING and exits 0 for ``--help`` or a bare
invocation; a real, executing CLI prints usage/errors and exits non-zero
where argparse requires it. That divergence is the structural guard.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "infrastructure" / "spot_train.sh"

# `python -m <module>.{ec2_spot,ssm_dispatcher}` as invoked by the launcher
# (LIB_PYTHON is a runtime path, not part of the module name).
_MODULE_RE = re.compile(r"-m\s+([\w.]+\.(?:ec2_spot|ssm_dispatcher))\b")


def _invoked_modules() -> set[str]:
    """Every ec2_spot / ssm_dispatcher module path the launcher ACTUALLY
    invokes on a non-comment line (comment-only mentions — e.g. historical
    prose citing the old `-m nousergon_lib.ec2_spot` shim no-op, config#1646
    — must not be mistaken for a real callsite)."""
    if not _SCRIPT.exists():
        return set()
    modules: set[str] = set()
    for raw in _SCRIPT.read_text().splitlines():
        if raw.strip().startswith("#"):
            continue
        modules.update(_MODULE_RE.findall(raw))
    return modules


def test_spot_train_script_references_at_least_one_krepis_module():
    """Sanity: extraction must find both chokepoints — an empty/partial
    set would make the executability tests below vacuously pass (the
    silent-no-op of tests)."""
    modules = _invoked_modules()
    assert modules, (
        "No `-m <module>.{ec2_spot,ssm_dispatcher}` invocations found in "
        "spot_train.sh — either the launcher no longer uses the lib CLI "
        "(update this test's extraction) or the regex went stale."
    )
    assert "krepis.ec2_spot" in modules, (
        "spot_train.sh no longer invokes `-m krepis.ec2_spot` — the "
        "config#1649 migration off the guard-less nousergon_lib re-export "
        "shim appears to have regressed."
    )
    assert "krepis.ssm_dispatcher" in modules, (
        "spot_train.sh no longer invokes `-m krepis.ssm_dispatcher` — the "
        "config#1649 migration off the guard-less nousergon_lib re-export "
        "shim appears to have regressed."
    )


def test_spot_train_does_not_invoke_nousergon_lib_cli():
    """Regression guard: the launcher must not fall back to `-m
    nousergon_lib.{ec2_spot,ssm_dispatcher}`. On lib >=0.81.0 those paths
    are guard-less re-export shims — silent exit-0 no-ops under `python
    -m` (the exact 2026-07-03 failure mode, one layer up)."""
    text = _SCRIPT.read_text()
    offenders = [
        (n, line)
        for n, line in enumerate(text.splitlines(), start=1)
        if not line.strip().startswith("#")
        and re.search(r"-m\s+nousergon_lib\.(?:ec2_spot|ssm_dispatcher)\b", line)
    ]
    assert not offenders, (
        f"Found {len(offenders)} non-comment `-m nousergon_lib."
        f"{{ec2_spot,ssm_dispatcher}}` invocation(s) in spot_train.sh:\n"
        + "\n".join(f"  line {n}: {line.strip()}" for n, line in offenders)
        + "\n\nRoute through the krepis CLI directly (config#1649) — the "
        "nousergon_lib re-export shim is guard-less on lib >=0.81.0."
    )


@pytest.mark.parametrize("module", sorted(_invoked_modules()))
def test_krepis_module_help_executes(module):
    """`python -m <module> --help` must print real usage text and exit 0.

    A guard-less re-export shim (`sys.modules[__name__] = _mod`, no
    `__main__` guard) never reaches argparse under runpy: it exits 0 with
    EMPTY stdout regardless of the argv given, including `--help`. A real
    CLI's `--help` prints a non-empty `usage:` block. This assertion is
    the offline-safe analog of the sentinel-execution proof in
    nousergon-data's `test_ssm_log_capture_wrapper_executes.py` — no AWS
    credentials or network access required.
    """
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"`python -m {module} --help` exited {proc.returncode} "
        f"(stderr tail={proc.stderr[-300:]!r})"
    )
    assert "usage:" in proc.stdout.lower(), (
        f"`python -m {module} --help` produced no usage text "
        f"(stdout={proc.stdout!r}). A guard-less re-export shim (the "
        "config#1646 bug class) imports, rebinds sys.modules, and falls "
        "off the end WITHOUT ever reaching argparse — silent exit 0 with "
        "empty stdout for any argv, including --help."
    )


@pytest.mark.parametrize("module", sorted(_invoked_modules()))
def test_krepis_module_bare_invocation_fails_loud(module):
    """`python -m <module>` with NO subcommand must exit non-zero with a
    stderr error naming the missing required subcommand.

    Both `krepis.ec2_spot` and `krepis.ssm_dispatcher` declare their
    subparsers with `required=True`; a bare invocation hits that argparse
    guard and exits 2. A guard-less shim exits 0 with empty stderr for
    ANY argv — this is the same silent-no-op signature `--help` catches,
    proved from the opposite (missing-required-arg) direction.
    """
    proc = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0, (
        f"`python -m {module}` (no subcommand) exited 0 — a real CLI with "
        "a required subparser must fail loud on a missing subcommand. "
        "Exit 0 here is the guard-less re-export shim's silent no-op "
        "signature (config#1646 bug class)."
    )
    assert proc.stderr.strip(), (
        f"`python -m {module}` (no subcommand) exited "
        f"{proc.returncode} but printed nothing to stderr — expected an "
        "argparse 'the following arguments are required' error. Empty "
        "output on a non-zero exit is still consistent with a shim that "
        "crashed for an unrelated reason; this asserts the CLI actually "
        "ran its argument parser."
    )
