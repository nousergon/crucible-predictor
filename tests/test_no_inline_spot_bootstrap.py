"""No spot launcher in this repo may bootstrap a spot instance inline.

## The class this guards (alpha-engine-config-I7372, G16 spot-bootstrap cutover)

``infrastructure/spot_train.sh`` was the last file in this repo carrying an
inline bootstrap heredoc — the watchdog-unit + interpreter-install +
``git clone`` block ``infrastructure/_spot_common.sh``'s ``bootstrap_spot()``
already collapsed onto ``krepis.spot_bootstrap`` (alpha-engine-config-
I4992/I6922). ``spot_train.sh`` now sources ``_spot_common.sh`` and calls
that same function rather than restating it a second time (see the
``source`` line in ``spot_train.sh`` for why it does not adopt
``_spot_common.sh``'s helpers wholesale).

Every per-step heredoc's ``command -v python3.12 >/dev/null && PY=python3.12
|| PY=python3`` silent interpreter fallback (crucible-predictor#462 class) is
also gone: a fallback resolves ``requirements.txt`` against a different
interpreter than it was pinned for, and does so silently, minutes later, in
another process.

## Derived, not enumerated

The primary check below calls ``krepis.spot_bootstrap.scan_for_inline_
bootstraps`` — the fleet's canonical classifier, driven by *behaviour*
(watchdog + interpreter-install + checkout + dispatch signatures), not a
filename list. A copy under a new name, in a new directory, with a new
function prefix, still matches, because what it *does* is unchanged.

## A known limitation of the primary check, and the two checks that cover it

``scan_for_inline_bootstraps`` clears an entire file the moment it finds a
``-m krepis.spot_bootstrap`` reference anywhere in that file's text (its
``_DELEGATES`` short-circuit) — filed as ``alpha-engine-config-I7378``
(regional rather than file-level clearing; a krepis fix, not something this
repo may make, per the G16 brief's "do not edit krepis"). A file that both
delegates AND still carries a restated heredoc elsewhere in the same file
would read as clean to the primary check. The two derived checks below do
not depend on that delegate short-circuit and catch what it would miss:

1. ``test_no_silent_interpreter_fallback_anywhere`` — a plain grep-shaped
   assertion, unaffected by whether the file also delegates.
2. ``test_no_heredoc_reintroduces_a_home_ec2_user_git_clone`` — looks inside
   every heredoc body for a ``git clone`` into ``/home/ec2-user/``, which is
   exactly the shape a delegate+heredoc fork would carry.

## A second, narrower gap: three LIVE launchers, out of this PR's scope

Running check 1 tree-wide (before the carve-out below existed) also found
the identical silent-fallback class in three files this PR does NOT touch —
``spot_predictor_training.sh``, ``spot_model_zoo_select.sh``,
``spot_train_spec_dispatch.sh`` — the LIVE per-stage launchers the weekly SF
actually invokes (unlike ``spot_train.sh``, retained only as that split's
rollback path). Higher blast radius than this PR's assigned file, so it is
tracked as its own follow-up (``alpha-engine-config-I7380``) rather than
bundled here. ``_KNOWN_LIVE_LAUNCHER_GAP`` below is the carve-out; it must
shrink to empty, never grow, and this test fails outright (no matching
carve-out) the moment the fallback appears anywhere else.

## Proof this test fires (see the PR body for the actual before/after run)

A heredoc-shaped file was temporarily reintroduced under
``infrastructure/`` carrying the watchdog/interpreter/clone/dispatch
signatures, all three tests below turned red, the file was removed, and all
three turned green again.
"""

from __future__ import annotations

import re
from pathlib import Path

from krepis.spot_bootstrap import scan_for_inline_bootstraps

REPO_ROOT = Path(__file__).resolve().parents[1]

# Silent interpreter selection: `command -v python3.12 ... || VAR=python3`
# (or `PIP="python3 -m pip"`, etc.) — any shape that falls back to a bare
# `python3`/`pip` invocation instead of asserting python3.12 or failing loud.
_SILENT_FALLBACK = re.compile(
    r"command\s+-v\s+python3\.12[^\n]*\|\|\s*\w+=(?:\"?python3\b|\"?python3\s+-m\s+pip)"
)

# A `git clone` naming a shell variable for the URL/branch, inside a heredoc
# body targeting the spot's home directory — the crucible-predictor#463 shape.
_HEREDOC_CLONE = re.compile(
    r"<<-?\s*'?(\w+)'?\n(.*?)\n\1\b", re.S
)
_CLONE_HOME_EC2_USER = re.compile(r"git\s+clone\b[^\n]*/home/ec2-user/")


#: alpha-engine-config-I7380: the silent-fallback class survives here, in the
#: LIVE per-stage launchers, tracked as its own follow-up rather than bundled
#: into this PR (which is scoped to spot_train.sh + the _spot_common.sh
#: reference helper it reuses). Every path here must be one this test found
#: BEFORE the carve-out was added, verified against a filed issue — never
#: added speculatively.
_KNOWN_LIVE_LAUNCHER_GAP = frozenset(
    {
        "infrastructure/spot_predictor_training.sh",
        "infrastructure/spot_model_zoo_select.sh",
        "infrastructure/spot_train_spec_dispatch.sh",
    }
)


def _shell_files() -> list[Path]:
    out: list[Path] = []
    for subdir in ("infrastructure", "scripts", "bin"):
        base = REPO_ROOT / subdir
        if not base.is_dir():
            continue
        out.extend(sorted(p for p in base.rglob("*") if p.is_file() and p.suffix in (".sh", ".bash")))
    return out


def test_scan_for_inline_bootstraps_reports_nothing():
    """The primary, behavioural check: derived from the tree, not enumerated."""
    findings = scan_for_inline_bootstraps(REPO_ROOT)
    assert not findings, (
        "inline spot bootstrap(s) found outside krepis.spot_bootstrap: "
        + "; ".join(str(f) for f in findings)
        + " — collapse onto infrastructure/_spot_common.sh's bootstrap_spot() "
        "(or krepis.spot_bootstrap.render_bootstrap directly) instead of "
        "restating the watchdog/interpreter-install/clone shape."
    )


def test_no_silent_interpreter_fallback_anywhere():
    """Regional check the I7378 delegate short-circuit cannot blind.

    Every SSM step this repo dispatches must assert python3.12 (installed by
    bootstrap_spot()) or fail loud — never silently resolve requirements.txt
    wheels against whatever `python3` happens to mean on the AMI.
    """
    hits: list[str] = []
    for path in _shell_files():
        rel = str(path.relative_to(REPO_ROOT))
        text = path.read_text(encoding="utf-8")
        for m in _SILENT_FALLBACK.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append(f"{rel}:{line_no}")
    unexpected = [h for h in hits if not any(h.startswith(k + ":") for k in _KNOWN_LIVE_LAUNCHER_GAP)]
    assert not unexpected, (
        f"silent python3.12->python3 interpreter fallback found at {unexpected} — "
        "replace with a strict `command -v python3.12 || exit 1` guard "
        "(crucible-predictor#462 class, alpha-engine-config-I7372). If this is "
        "one of the alpha-engine-config-I7380 files, it belongs in "
        "_KNOWN_LIVE_LAUNCHER_GAP with a comment, not silently ignored."
    )
    # The carve-out must track exactly what alpha-engine-config-I7380 covers —
    # never wider than what is actually still broken, so a real fix there
    # shrinks this set instead of leaving a stale exemption.
    still_gapped = {k for k in _KNOWN_LIVE_LAUNCHER_GAP if any(h.startswith(k + ":") for h in hits)}
    stale = _KNOWN_LIVE_LAUNCHER_GAP - still_gapped
    assert not stale, (
        f"{sorted(stale)} no longer trip the silent-fallback pattern — "
        "alpha-engine-config-I7380 appears fixed; shrink "
        "_KNOWN_LIVE_LAUNCHER_GAP (and close/verify the issue) instead of "
        "leaving a stale carve-out"
    )


def test_no_heredoc_reintroduces_a_home_ec2_user_git_clone():
    """Regional check for the delegate+heredoc fork the I7378 gap would miss.

    A file that references `-m krepis.spot_bootstrap` anywhere clears the
    primary scan wholesale (I7378); this looks inside every heredoc body for
    the specific defect shape (crucible-predictor#463: a git-clone line
    naming a shell variable, targeting the spot's home directory) regardless
    of whether the file also delegates elsewhere.
    """
    hits: list[str] = []
    for path in _shell_files():
        text = path.read_text(encoding="utf-8")
        for heredoc in _HEREDOC_CLONE.finditer(text):
            body = heredoc.group(2)
            if _CLONE_HOME_EC2_USER.search(body):
                line_no = text.count("\n", 0, heredoc.start()) + 1
                hits.append(f"{path.relative_to(REPO_ROOT)}:{line_no}")
    assert not hits, (
        f"a heredoc reintroducing a /home/ec2-user/ git-clone was found at "
        f"{hits} — the shared, non-repo-specific bootstrap logic belongs in "
        "krepis.spot_bootstrap, never restated inline (crucible-predictor#463)"
    )


def test_spot_train_delegates_to_shared_bootstrap_spot():
    """Anchored on the specific file this arc's brief scoped to.

    ``spot_train.sh`` must source ``_spot_common.sh`` and call its
    ``bootstrap_spot()`` rather than defining a second render call or a
    restated heredoc of its own.
    """
    text = (REPO_ROOT / "infrastructure" / "spot_train.sh").read_text(encoding="utf-8")
    assert re.search(r'source\s+"\$SCRIPT_DIR/_spot_common\.sh"', text), (
        "spot_train.sh no longer sources _spot_common.sh — the bootstrap_spot() "
        "reuse this cutover relies on has been reverted"
    )
    assert re.search(r"^\s*bootstrap_spot\s*$", text, re.M), (
        "spot_train.sh no longer calls bootstrap_spot() — has the shared-"
        "helper reuse been replaced with a restated heredoc or a second "
        "krepis.spot_bootstrap render call?"
    )
    assert "<<BOOTSTRAP" not in text and "<<'BOOTSTRAP'" not in text, (
        "spot_train.sh's own BOOTSTRAP heredoc has reappeared"
    )
    # The heredoc this replaced armed a spot-side hard-timeout timer inline
    # (`systemd-run --on-active=${MAX_RUNTIME_SECONDS} ... shutdown -h now`).
    # bootstrap_spot() now carries that guarantee via --max-runtime-seconds
    # (see test_max_runtime_seconds_is_passed_and_resolves in
    # test_spot_bootstrap_env_closure.py for the shared-helper side); this
    # asserts spot_train.sh's own half — that MAX_RUNTIME_SECONDS is set to a
    # real, non-empty default BEFORE bootstrap_spot() is ever called, so the
    # cap the deleted heredoc used to arm is not silently un-shipped here.
    assert re.search(r'^MAX_RUNTIME_SECONDS="\$\{MAX_RUNTIME_SECONDS:-\d+\}"', text, re.M), (
        "spot_train.sh no longer declares a non-empty default for "
        "MAX_RUNTIME_SECONDS before bootstrap_spot() is called — the spot-"
        "side hard-timeout timer the deleted inline heredoc used to arm via "
        "`systemd-run --on-active=...` would silently go unarmed"
    )
    bootstrap_call_idx = text.index("\nbootstrap_spot\n")
    max_runtime_idx = text.index('MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-')
    assert max_runtime_idx < bootstrap_call_idx, (
        "MAX_RUNTIME_SECONDS is declared AFTER bootstrap_spot() is called — "
        "the hard-timeout cap would be unset at render time"
    )
