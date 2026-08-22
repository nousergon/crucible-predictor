"""The spot bootstrap in this repo must not drift from the fleet's canonical one.

## Why this test exists

``infrastructure/_spot_common.sh`` exists twice in the fleet — here and in
``nousergon-data`` — as two independently-written implementations of the same
spot bootstrap. Neither was forked from the other; both were built by
applying the same written standard (ARCHITECTURE.md §111), eight days apart,
and the standard named a *shape* and no implementation. They then diverged.

What that cost, measured on the weekly Step Function over 2026-08-10/11
(alpha-engine-config-I6922):

| Defect | Fixed in `nousergon-data` | Reached here |
|---|---|---|
| watchdog unit ``Type=oneshot`` blocks its own bootstrap | `#1294` | `#461`, 16h later — killed `watch-rerun-2026-08-10-5` |
| bootstrap ASSERTS ``python3.12`` instead of installing it | `#1296` | `#462`, 23h later — killed `-6`, four hours after `-5` |

Both defects live in the two blocks this test guards. Each port cost a failed
weekly run to discover, because the second copy's defect stays invisible until
the first copy's is fixed and the pipeline gets far enough to reach it.

## What it asserts, and why it is not a second hardcoded copy

The invariants are **derived from** ``krepis.spot_bootstrap``, the fleet's
canonical renderer (shipped 0.46.0, alpha-engine-config-I6922 phase 1) — not
restated here. ``krepis`` is already a pinned dependency of this repo, so the
check is hermetic and needs no network and no sibling checkout.

Deriving rather than restating is the whole point: a third repo, or a change to
the canonical renderer, moves this test's expectations automatically. A test
that hardcoded ``Type=simple`` would be a *third* copy of the invariant and
would drift exactly like the first two.

## Why the Bash copy no longer exists

``krepis.spot_bootstrap`` could not replace this heredoc while the launcher
resolved its interpreter as::

    LIB_PYTHON="${LIB_PYTHON:-/home/ec2-user/alpha-engine-dashboard/.venv/bin/python}"

— the *shared application host's* venv, whose krepis version was governed by
``crucible-dashboard/requirements.txt``, a file no merge in this repo can see.
A cutover therefore carried a runtime dependency this repo could not pin, and
when that venv was behind, the bootstrap did not degrade — it died with
``ModuleNotFoundError`` on every spot stage of every pipeline.

That blocker cleared (`alpha-engine-config-I7343`, 2026-08-14). The launcher
now resolves through ``/opt/nousergon/bin/lib-python``, the ops-owned guard
over the box's DECLARED krepis venv, which aborts with ``EX_CONFIG`` (78)
naming the version it found when that venv is below the launcher floor — a
declared, enforced, fail-loud floor, which is what I6931 owed. Every launcher
interpreter that can run this repo at all now satisfies krepis's own pin
(``requirements.txt`` / ``requirements-lambda.txt``: ``krepis[flow-doctor]
==0.59.0``).

The cutover this docstring called for has landed: ``bootstrap_spot()`` now
constructs a ``krepis.spot_bootstrap render`` CLI call instead of an inline
heredoc, and this module proves the deletion changed nothing — it derives the
launcher's actual arguments from the live source, renders them through the
real ``krepis.spot_bootstrap.render_bootstrap`` (the same function the
launcher invokes at runtime), and asserts the invariants below against that
RENDERED output, never a restated copy of it.
"""

from __future__ import annotations

import re

import pytest

from krepis.spot_bootstrap import (
    PYTHON,
    SYSTEMCTL_ENABLE_TIMEOUT_SEC,
    SpotBootstrapSpec,
    render_bootstrap,
)

from tests._spot_bootstrap_render_support import bootstrap_spot_block, rendered_script

# A spec's values do not matter here: this test compares the parts of the
# rendered script that are the SAME for every workload — the watchdog unit and
# the interpreter install. The repo-specific tail (clone target, config
# destinations) is asserted by the per-repo tests beside this one.
_CANONICAL = render_bootstrap(
    SpotBootstrapSpec(repo_url="https://example.invalid/x.git", checkout="/tmp/x")
)


@pytest.fixture(scope="module")
def bootstrap_block() -> str:
    """The RENDERED script ``bootstrap_spot()`` sends to ``run_ssm`` today.

    Not the raw shell source: the invariants below (watchdog unit shape,
    interpreter install ordering) are about what actually reaches the spot
    instance, and that is now krepis's output, not this repo's text. Now that
    both sides of the class-level watchdog/interpreter tests below are
    produced by the same ``render_bootstrap`` call, their protection has
    moved: they still fail if a future krepis change regresses the unit or
    the interpreter block, and ``test_bootstrap_spot_carries_no_inline_heredoc``
    catches this repo constructing its own copy again instead of calling
    krepis.
    """
    return rendered_script()


def test_bootstrap_spot_carries_no_inline_heredoc():
    """The heredoc cannot creep back in.

    Structural guard for the cutover (alpha-engine-config-I4992/I6922):
    ``bootstrap_spot()`` must construct a renderer call, never restate the
    watchdog unit, the ``dnf install``, or the ``git clone`` inline.
    """
    block = bootstrap_spot_block()
    for forbidden in (
        "<<'BOOTSTRAP'",
        "dnf install",
        "git clone",
        "systemctl is-enabled ec2-spot-watchdog",
    ):
        assert forbidden not in block, (
            f"bootstrap_spot() contains {forbidden!r} — the inline heredoc has "
            "crept back in. This function must only construct a "
            "`krepis.spot_bootstrap render` call and hand the result to "
            "run_ssm; the shared, non-repo-specific bootstrap logic belongs "
            "in krepis, never restated here."
        )
    assert "krepis.spot_bootstrap" in block, (
        "bootstrap_spot() no longer invokes krepis.spot_bootstrap — the "
        "cutover has been reverted without restoring this test's premise"
    )


def _service_directives(script: str) -> set[str]:
    """``KEY=VALUE`` lines in the systemd unit's ``[Service]`` section."""
    m = re.search(r"\[Service\]\n(.*?)(?=\n\[|\nUNIT\b)", script, re.S)
    assert m, "no [Service] section found in the systemd unit"
    return {
        line.strip()
        for line in m.group(1).splitlines()
        if "=" in line and not line.strip().startswith("#")
    }


# ── The watchdog unit ────────────────────────────────────────────────────────


def test_watchdog_unit_carries_every_directive_the_canonical_unit_declares(
    bootstrap_block: str,
):
    """Derived, not restated — krepis owns what the unit must say.

    The 2026-08-10 hang was a missing/incorrect ``Type``. Asserting the whole
    ``[Service]`` section rather than that one field means the next divergence
    in this unit — whatever field it lands on — also fails here.
    """
    canonical = _service_directives(_CANONICAL)
    local = _service_directives(bootstrap_block)
    missing = canonical - local
    assert not missing, (
        f"{_SPOT_COMMON.name}'s ec2-spot-watchdog unit is missing "
        f"{sorted(missing)}, which krepis.spot_bootstrap declares. This is the "
        "alpha-engine-config-I6922 drift class: the two fleet copies of this "
        "unit diverged and a weekly run paid for it. Reconcile against "
        "krepis.spot_bootstrap._watchdog_block()."
    )


def test_watchdog_unit_is_never_oneshot(bootstrap_block: str):
    """The anchored instance, kept explicit because it cost two weekly runs.

    ``systemctl start`` on a ``Type=oneshot`` unit BLOCKS until ``ExecStart``
    exits, and ``TimeoutStartSec`` defaults to infinity for oneshot. This
    unit's ``ExecStart`` is an endless supervision loop, so the unit declared
    ``Type=oneshot`` + ``RemainAfterExit=yes`` hung every bootstrap until SSM
    SIGKILLed it at its budget — rc=137 at exactly the timeout, with stderr
    ending on the ``systemctl enable`` symlink line and nothing after it.

    Scoped to the PARSED ``[Service]`` directives, never a substring search of
    the script: both copies explain this defect in a comment that contains the
    literal ``Type=oneshot``, and a naive ``not in`` reports the explanation as
    the defect. A checker that cannot tell a comment from a directive fails in
    the alarming direction on the very file that documents the fix.
    """
    assert "Type=oneshot" not in _service_directives(bootstrap_block), (
        "the ec2-spot-watchdog unit declares Type=oneshot; its ExecStart never "
        "returns, so `systemctl enable --now` blocks forever and the whole "
        "bootstrap dies under SIGKILL with no output (nousergon-data#1294, "
        "crucible-predictor#461)"
    )


def test_enabling_the_watchdog_is_bounded_by_the_canonical_timeout(
    bootstrap_block: str,
):
    """A misdeclared unit must fail in seconds WITH a message.

    The bound is what turns "bootstrap is slow" into "systemctl is blocked
    forever". Its value comes from krepis so the two copies cannot pick
    different budgets.
    """
    assert re.search(
        rf"timeout\s+{SYSTEMCTL_ENABLE_TIMEOUT_SEC}\s+systemctl\s+enable\s+--now\s+ec2-spot-watchdog",
        bootstrap_block,
    ), (
        "`systemctl enable --now ec2-spot-watchdog` must be wrapped in "
        f"`timeout {SYSTEMCTL_ENABLE_TIMEOUT_SEC}` "
        "(krepis.spot_bootstrap.SYSTEMCTL_ENABLE_TIMEOUT_SEC), so a future "
        "unit-shape regression fails fast and loud instead of consuming the "
        "whole SSM budget in silence"
    )


def test_a_failed_enable_explains_itself_and_aborts(bootstrap_block: str):
    """Silence is what made the 2026-08-10 hang unreadable for a day."""
    m = re.search(
        r"timeout\s+\d+\s+systemctl\s+enable\s+--now\s+ec2-spot-watchdog\s*\|\|\s*\{(.*?)\}",
        bootstrap_block,
        re.S,
    )
    assert m, "the guarded enable must have a `|| { ... }` failure branch"
    branch = m.group(1)
    assert "exit 1" in branch, "a watchdog that cannot be enabled must abort the bootstrap"
    assert ">&2" in branch, "the failure must reach stderr, which is what SSM captures"


# ── The interpreter ──────────────────────────────────────────────────────────


def test_the_interpreter_is_installed_not_merely_asserted(bootstrap_block: str):
    """A bootstrap establishes preconditions; it does not require them.

    The per-stage split replaced the monolith's ``dnf install`` with a bare
    ``command -v`` assertion, encoding an AMI contract nothing provides. The
    AL2023 spot AMI does not ship python3.12.
    """
    assert re.search(rf"dnf install[^\n]*{re.escape(PYTHON)}\b", bootstrap_block), (
        f"the bootstrap must install {PYTHON} explicitly — the AL2023 spot AMI "
        f"does not ship it (nousergon-data#1296, crucible-predictor#462)"
    )


def test_the_interpreter_check_is_a_postcondition_not_a_precondition(
    bootstrap_block: str,
):
    """Ordering is the actual invariant — both copies had the check, one had it alone."""
    install = re.search(rf"dnf install[^\n]*{re.escape(PYTHON)}\b", bootstrap_block)
    check = re.search(rf"command -v {re.escape(PYTHON)}\b", bootstrap_block)
    assert install and check, "expected both a dnf install and a command -v guard"
    assert install.start() < check.start(), (
        f"`command -v {PYTHON}` appears BEFORE the dnf install that provides it. "
        "That is a precondition on an image we do not build — which is exactly "
        "the defect that killed MorningEnrich and PredictorTraining on "
        "consecutive weekly runs."
    )


def test_a_missing_interpreter_after_install_is_fatal(bootstrap_block: str):
    """No silent fallback to the AMI's system python3.

    requirements.txt is resolved against 3.12; a fallback resolves different
    wheels and fails later, elsewhere, with the install log gone.
    """
    m = re.search(
        rf"command -v {re.escape(PYTHON)} >/dev/null \|\| \{{(.*?)\}}", bootstrap_block
    )
    assert m, f"the post-install `command -v {PYTHON}` guard is missing its abort branch"
    assert "exit 1" in m.group(1)
