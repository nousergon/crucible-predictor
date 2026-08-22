"""Pin ``requirements.txt`` and ``requirements-lambda.txt`` to the same
alpha-engine-lib version.

The Lambda image is built from ``requirements-lambda.txt`` (per the
Dockerfile); local dev + the training spot use ``requirements.txt``.
Drift between them is the antipattern that:

  - 2026-05-07 shipped lib v0.2.4 to prod after the project pin had
    already moved to v0.5.5 (see comment block at the top of
    ``requirements-lambda.txt``).
  - 2026-05-12 broke the predictor canary on PR #147 — the project
    pin moved to v0.12.0 (which ships ``alpha_engine_lib.secrets``)
    but the Lambda pin stayed at v0.9.1, so the Lambda image had no
    ``secrets`` module and crashed with ``ModuleNotFoundError``.

This test re-greps both files on every CI run so a future commit that
bumps one without the other fails here, not in a canary.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Distribution name renamed alpha-engine-lib -> nousergon-lib at lib 0.60.0
# (config#1245 / #1172). Accept either spelling across the crossing; the
# version-equality assertion below is the load-bearing part.
# alpha-engine-config-I7301: TAGS ONLY. This pattern used to accept
# `|(?:[0-9a-f]{40})` as well, and that alternative is why a commit-SHA pin
# shipped in crucible-predictor#422 and stayed green here for 13 days.
#
# A SHA pin passes every check this file makes — both files carried the same
# SHA, so the lockstep equality held — while breaking a check ONE REPO OVER
# that nothing in this repo runs: the weekly SF's `check_lib_pin_drift` gate
# asserts `backtester pin == predictor pin`, and it cannot compare a SHA to
# `crucible-backtester`'s `vX.Y.Z`. From 2026-07-31 the gate reported
# `parity_ok: null` on every invocation and fell through to its fail-open,
# hiding a real parity break (backtester v0.124.5 against this repo's SHA at
# ~v0.124.16).
#
# So the tag form is not a style preference — it is the wire format of a
# cross-repo invariant, and this guard is the only place a SHA can be caught
# before it reaches production.
_LIB_PIN_RE = re.compile(
    r"(?:alpha-engine-lib|nousergon-lib)\[[^\]]*\]\s*@\s*git\+"
    r"https://github\.com/nousergon/nousergon-lib@"
    r"(v[0-9]+\.[0-9]+\.[0-9]+)"
)

# The same line ending in a raw commit SHA. Matched only so this guard can
# fail with the reason instead of an unhelpful "could not find a pin".
# Mirrors inference/lib_pin_drift.py::_LIB_SHA_PIN_RE.
_LIB_SHA_PIN_RE = re.compile(
    r"(?:alpha-engine-lib|nousergon-lib)\[[^\]]*\]\s*@\s*git\+"
    r"https://github\.com/nousergon/nousergon-lib@([0-9a-f]{7,40})\b"
)

_PIN_FILES = ("requirements.txt", "requirements-lambda.txt")


def _read_lib_pin(filename: str) -> str:
    text = (_REPO_ROOT / filename).read_text()
    match = _LIB_PIN_RE.search(text)
    if match is None:
        sha = _LIB_SHA_PIN_RE.search(text)
        assert sha is None, (
            f"{filename} pins nousergon-lib by commit SHA "
            f"({sha.group(1)[:12]}...), not a vX.Y.Z tag.\n\n"
            f"Pin by tag. A SHA is invisible to the weekly Step Function's "
            f"cross-repo co-install parity gate (check_lib_pin_drift), which "
            f"compares this repo's pin to crucible-backtester's and cannot "
            f"resolve a commit to a version. A SHA pin there does not fail "
            f"loudly — it degrades the gate to fail-open on every run, which "
            f"is how a real parity break ran unnoticed from 2026-07-31 to "
            f"2026-08-13 (alpha-engine-config-I7301)."
        )
        raise AssertionError(
            f"could not find a nousergon-lib pin in {filename} — the regex "
            f"expects ``nousergon-lib[extras] @ "
            f"git+https://github.com/nousergon/nousergon-lib@vX.Y.Z``"
        )
    return match.group(1)


def test_requirements_and_lambda_pins_match():
    """Both files must pin nousergon-lib to the same vX.Y.Z tag."""
    root_pin = _read_lib_pin("requirements.txt")
    lambda_pin = _read_lib_pin("requirements-lambda.txt")
    assert root_pin == lambda_pin, (
        f"alpha-engine-lib pin drift: requirements.txt={root_pin!r} but "
        f"requirements-lambda.txt={lambda_pin!r}. Both files must pin to "
        f"the same tag — they're two views of the same dependency "
        f"graph. Drift broke the predictor canary on 2026-05-12 (PR #147 / "
        f"hotfix fix/lambda-lib-pin-v0.12.0)."
    )


# ── I7171 -> I7301: the acknowledged-SHA exception is retired ───────────────
#
# crucible-predictor#493 held the line at ONE acknowledged SHA
# (c907a044...) while the re-pin was outstanding, on the reading that a tag
# re-pin "would silently swap in ~100 unrelated commits' worth of library
# code". That was the right call to make with the pin still in place, and it
# is superseded here because the re-pin landed and the swap does not go that
# direction:
#
#   git log --oneline v0.124.57..c907a04
#     -> the three commits are the feat/signals-fallback-chain branch, and
#        that feature LANDED on main (squash-merged, which is why the tip is
#        unreachable from any tag while signals.py is byte-identical in both)
#   git diff --numstat v0.124.57 c907a04 | awk '$1>0 && $2==0'   -> empty
#   git diff --stat    v0.124.57 c907a04 | tail -1
#     -> 71 files changed, 333 insertions(+), 9132 deletions(-)
#
# No file had content in the SHA that v0.124.57 lacks; the SHA was strictly
# behind. Re-pinning restored code rather than swapping any away.
#
# The exception is removed rather than left inert because its rationale
# block asserted the opposite, and a stale comment on a passing test is how
# the next reader re-derives a wrong conclusion.
# `test_no_deploy_artifact_pins_the_lib_by_commit_sha` below is strictly
# stronger: it rejects EVERY SHA, including the formerly-acknowledged one.


def _parse_pip_pins(filename: str) -> dict[str, str]:
    """Parse a requirements file into {package_name: specifier}."""
    pins = {}
    text = (_REPO_ROOT / filename).read_text()
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "@" in line:
            # git+https format (e.g., nousergon-lib) — skip (handled separately)
            continue
        # PEP 508 format: package_name[extras]specifier (e.g., pandas>=2.0.0,<3)
        match = re.match(r"^([a-zA-Z0-9_-]+)", line)
        if match:
            pkg = match.group(1)
            specifier = line[len(pkg):].lstrip("[]").strip()
            if specifier and not specifier.startswith("#"):
                pins[pkg] = specifier
    return pins


def test_shared_package_specifiers_match():
    """Packages in both requirements files must have identical specifiers."""
    root_pins = _parse_pip_pins("requirements.txt")
    lambda_pins = _parse_pip_pins("requirements-lambda.txt")

    shared_packages = set(root_pins.keys()) & set(lambda_pins.keys())
    mismatches = {
        pkg: (root_pins[pkg], lambda_pins[pkg])
        for pkg in shared_packages
        if root_pins[pkg] != lambda_pins[pkg]
    }

    assert not mismatches, (
        f"Package specifier drift between requirements.txt and "
        f"requirements-lambda.txt: {dict(mismatches)}. Both files must pin "
        f"shared packages to identical specifiers — drift allows the Lambda "
        f"image to resolve a different dependency set than local dev "
        f"(config#2346)."
    )


# ── alpha-engine-config-I7301: the tag form is a cross-repo contract ─────────

def test_no_deploy_artifact_pins_the_lib_by_commit_sha():
    """A SHA pin is rejected here, where it is cheap to see.

    The 13-day window this closes: `crucible-predictor#422` (2026-07-31)
    replaced the tag with `c907a044bb1553815225327bc56644050543b6f2`. Every
    guard in this repo passed — the two files agreed, so lockstep held — and
    the damage landed in a different repo's Step Function, where the co-install
    parity gate could not compare a SHA to `crucible-backtester`'s `v0.124.5`
    and fell through to fail-open on every weekly run.

    Worth stating plainly, because it is what made the SHA survive review: the
    commit was not even on `nousergon-lib` main. `git merge-base --is-ancestor`
    fails against `origin/main`; it is a PR-branch commit reachable only
    through `refs/pull/*`. Production installed a tree that never landed.
    """
    for filename in _PIN_FILES:
        text = (_REPO_ROOT / filename).read_text()
        sha = _LIB_SHA_PIN_RE.search(text)
        assert sha is None, (
            f"{filename} pins nousergon-lib by commit SHA "
            f"{sha.group(1)[:12]}... — pin by vX.Y.Z tag instead "
            f"(alpha-engine-config-I7301)."
        )


def test_the_pin_regex_does_not_accept_a_sha():
    """Pin the guard itself.

    The hole was not a missing test, it was a permissive regex: `_LIB_PIN_RE`
    carried a `|[0-9a-f]{40}` alternative, so the SHA pin matched and every
    assertion downstream compared SHA-to-SHA and passed. Removing the
    alternative is the fix; this asserts it stays removed.
    """
    sha_line = (
        "nousergon-lib[arcticdb,flow-doctor] @ "
        "git+https://github.com/nousergon/nousergon-lib"
        "@c907a044bb1553815225327bc56644050543b6f2"
    )
    tag_line = (
        "nousergon-lib[arcticdb,flow-doctor] @ "
        "git+https://github.com/nousergon/nousergon-lib@v0.124.57"
    )
    assert _LIB_PIN_RE.search(sha_line) is None
    assert _LIB_SHA_PIN_RE.search(sha_line) is not None
    assert _LIB_PIN_RE.search(tag_line).group(1) == "v0.124.57"


def test_a_sha_pinned_file_fails_with_the_reason_not_a_parse_miss(tmp_path, monkeypatch):
    """`_read_lib_pin` must name the SHA, not report a missing pin.

    A "could not find a nousergon-lib pin" message on a file that visibly
    contains one sends the reader after the regex instead of the pin — the
    same could-not-measure-reported-as-something-else shape this whole issue
    is about.
    """
    (tmp_path / "requirements.txt").write_text(
        "numpy>=2.0\n"
        "nousergon-lib[arcticdb] @ git+https://github.com/nousergon/"
        "nousergon-lib@c907a044bb1553815225327bc56644050543b6f2\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "_REPO_ROOT", tmp_path)
    with pytest.raises(AssertionError, match="pins nousergon-lib by commit SHA"):
        _read_lib_pin("requirements.txt")
