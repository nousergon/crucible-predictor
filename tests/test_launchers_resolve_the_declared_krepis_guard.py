"""Every spot launcher in this repo resolves through the ops-owned krepis guard.

**The defect** (`alpha-engine-config-I6931`, second half tracked as
`alpha-engine-config-I7343`). Until 2026-08-14 every spot launcher in the fleet
defaulted its interpreter to a co-tenant's checkout::

    LIB_PYTHON="${LIB_PYTHON:-/home/ec2-user/alpha-engine-dashboard/.venv/bin/python}"

— NINE such lines across FIVE repos. So the version of ``krepis.ec2_spot``,
``krepis.ssm_dispatcher`` and ``krepis.spot_bootstrap`` that launches every spot
workload of all three Step Functions pipelines was governed by
``crucible-dashboard/requirements.txt``, which no merge in this repo can see, and
an absent or too-old venv produced a ``ModuleNotFoundError`` at launch rather
than a statement of what was wrong.

**The fix** is one ops-owned wrapper, ``/opt/nousergon/bin/lib-python``
(`nous-ergon-ops-PR676`), which execs the box's DECLARED krepis venv and aborts
with ``EX_CONFIG`` (78) naming the version it found when that venv is absent or
below the launcher floor. It never falls back. Each launcher's diff is the one
line naming it — writing the guard per launcher would be nine copies of one
contract across five repos, which is the `alpha-engine-config-I6922` defect one
layer down.

**What this test holds.** That the launchers in THIS repo keep resolving through
the guard, and that no launcher re-acquires a private fallback. Every assertion
is derived from the scripts on disk — no line numbers, because these files move.
"""

from __future__ import annotations

import re
from pathlib import Path

_INFRA = Path(__file__).resolve().parents[1] / "infrastructure"

#: The one path every launcher repo's LIB_PYTHON default must name. Declared and
#: installed by nous-ergon-ops (``bin/lib-python`` +
#: ``bin/install-box-config.sh``); asserted identically in all five repos.
GUARD = "/opt/nousergon/bin/lib-python"

#: The pre-I7343 default. Its reappearance anywhere executable is the regression.
CO_TENANT = "/home/ec2-user/alpha-engine-dashboard/.venv/bin/python"

#: Launcher scripts this repo is known to own. A rename must update this list;
#: a DELETION must not silently drop coverage, which is what the membership
#: assertion below catches. Filenames only — never line numbers.
KNOWN_LAUNCHERS = {'_spot_common.sh', 'spot_train.sh'}

_ASSIGN = re.compile(r'^([ \t]*)LIB_PYTHON=(.*)$', re.M)
_DEFAULTED = re.compile(r'^[ \t]*LIB_PYTHON="\$\{LIB_PYTHON:-([^}]*)\}"[ \t]*$')


def _shell_scripts() -> list[Path]:
    """The SPOT-LAUNCHER surface, derived from the naming convention rather than
    listed: ``spot_*.sh`` plus the ``_spot_common.sh`` they source.

    Scoped this way on purpose. Other scripts under ``infrastructure/`` — the
    dashboard box's own health, alert and deploy units — legitimately name that
    box's venv for their own service; they are not launching a spot and I7343
    does not touch them. A filename denylist would need editing every time a box
    service is added; this derivation does not.
    """
    return sorted(
        p
        for p in _INFRA.rglob("*.sh")
        if p.is_file() and (p.name.startswith("spot_") or p.name.startswith("_spot"))
    )


def _assignment_sites() -> dict[str, list[str]]:
    """{script name: [each raw LIB_PYTHON assignment line]} — derived, not listed."""
    out: dict[str, list[str]] = {}
    for path in _shell_scripts():
        hits = [m.group(0) for m in _ASSIGN.finditer(path.read_text())]
        if hits:
            out[path.name] = hits
    return out


def _executable_lines(path: Path) -> list[str]:
    """Non-comment, non-blank lines. Comments legitimately quote the old default
    (the whole rationale is about it), so the sweeps run over code only."""
    return [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_every_known_launcher_still_declares_its_interpreter():
    """A launcher that stops assigning LIB_PYTHON has not become safe — it has
    become one that inherits whatever the caller's environment happens to hold,
    which is the pre-guard behaviour with no line to grep for."""
    found = _assignment_sites()
    missing = KNOWN_LAUNCHERS - set(found)
    assert not missing, (
        f"launchers with no LIB_PYTHON assignment: {sorted(missing)}. If one was "
        "renamed, update KNOWN_LAUNCHERS; if one was deleted, this test's coverage "
        "shrank and that must be a deliberate, reviewed diff."
    )


def test_every_launcher_defaults_to_the_ops_owned_guard():
    """The load-bearing assertion: the default names the guard, in every script
    that assigns it — including any launcher added after this test was written."""
    for name, lines in sorted(_assignment_sites().items()):
        for line in lines:
            match = _DEFAULTED.match(line)
            assert match, (
                f"{name}: LIB_PYTHON assignment is not the "
                'LIB_PYTHON="${LIB_PYTHON:-<path>}" form: {line.strip()!r}. The '
                "override idiom is what lets a rehearsal point at another "
                "interpreter without editing the script."
            )
            assert match.group(1) == GUARD, (
                f"{name}: LIB_PYTHON defaults to {match.group(1)!r}, not the "
                f"ops-owned guard {GUARD!r}. Pointing a launcher at a repo-local "
                "or co-tenant venv restores the alpha-engine-config-I6931 defect: "
                "the krepis version that launches every spot stage becomes "
                "whatever that checkout happens to hold, with no declared floor."
            )


def test_the_env_var_override_is_preserved():
    """``LIB_PYTHON=... script.sh`` must still win. The guard is the DEFAULT, not
    a hardcode — a rehearsal or a second box needs the override."""
    for name, lines in sorted(_assignment_sites().items()):
        for line in lines:
            assert "${LIB_PYTHON:-" in line, (
                f"{name}: LIB_PYTHON is hardcoded, losing the env override: "
                f"{line.strip()!r}"
            )


def test_no_launcher_falls_back_to_a_co_tenant_checkout():
    """The whole defect was a silent fallback to whichever checkout was newest.

    A launcher that keeps a second candidate path — as a fallback branch, a
    ``||``, or a bare reference — recreates it invisibly, and does so while a
    declaration exists to point at, which is worse than the original.
    """
    for path in _shell_scripts():
        offenders = [line for line in _executable_lines(path) if CO_TENANT in line]
        assert not offenders, (
            f"{path.name}: executable line(s) still name the co-tenant venv "
            f"{CO_TENANT!r}: {offenders}. The launcher must resolve through "
            f"{GUARD!r} and nothing else — the guard aborts rather than "
            "degrading, and a fallback here silently removes that."
        )


def test_no_launcher_reimplements_the_guard_locally():
    """Nine copies of one fail-loud contract across five repos is exactly the
    `alpha-engine-config-I6922` defect. The version check lives in
    ``bin/lib-python``, in the repo that owns the box's provisioning — a
    launcher-side ``krepis.__version__`` comparison is that defect returning."""
    for path in _shell_scripts():
        text = path.read_text()
        for line in _executable_lines(path):
            assert "krepis.__version__" not in line, (
                f"{path.name}: a launcher-local krepis version check "
                f"({line.strip()!r}) duplicates the contract "
                f"{GUARD!r} already enforces for all five repos."
            )
        del text
