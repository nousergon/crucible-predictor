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
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Distribution name renamed alpha-engine-lib -> nousergon-lib at lib 0.60.0
# (config#1245 / #1172). Accept either spelling across the crossing; the
# version-equality assertion below is the load-bearing part.
_LIB_PIN_RE = re.compile(
    r"(?:alpha-engine-lib|nousergon-lib)\[[^\]]*\]\s*@\s*git\+"
    r"https://github\.com/nousergon/nousergon-lib@"
    r"((?:v[0-9]+\.[0-9]+\.[0-9]+)|(?:[0-9a-f]{40}))"
)


def _read_lib_pin(filename: str) -> str:
    text = (_REPO_ROOT / filename).read_text()
    match = _LIB_PIN_RE.search(text)
    assert match is not None, (
        f"could not find alpha-engine-lib pin in {filename} — the regex "
        f"expects ``alpha-engine-lib[extras] @ git+https://.../alpha-engine-lib@vX.Y.Z`` "
        f"or ``...@<commit-hash>``"
    )
    return match.group(1)


def test_requirements_and_lambda_pins_match():
    """Both files must pin alpha-engine-lib to the same tag or commit."""
    root_pin = _read_lib_pin("requirements.txt")
    lambda_pin = _read_lib_pin("requirements-lambda.txt")
    assert root_pin == lambda_pin, (
        f"alpha-engine-lib pin drift: requirements.txt={root_pin!r} but "
        f"requirements-lambda.txt={lambda_pin!r}. Both files must pin to "
        f"the same tag or commit — they're two views of the same dependency "
        f"graph. Drift broke the predictor canary on 2026-05-12 (PR #147 / "
        f"hotfix fix/lambda-lib-pin-v0.12.0)."
    )


# ── I7171: a SHA pin must be an explicit, tracked exception ────────────────
#
# _LIB_PIN_RE above accepts EITHER a tag or a raw commit SHA — necessary
# because crucible-predictor has pinned nousergon-lib by SHA since #422. But
# an UNACKNOWLEDGED SHA pin defeats inference/lib_pin_drift.py's Saturday-SF
# floor check (alpha-engine-config-I7171): a commit SHA cannot be compared
# to MIN_LIB_VERSION, so that gate silently degrades to "unmeasured" for as
# long as any SHA pin exists (lib_pin_drift.SHA_PINNED).
#
# The correct fix is re-pinning to a released tag + rejecting SHA pins
# outright. That was NOT done here: verified 2026-08-13 —
# `git -C nousergon-lib tag --contains c907a044bb1553815225327bc56644050543b6f2`
# returns nothing, and the commit sits on a diverged line (103 commits
# behind v0.124.57, 3 commits v0.124.57 does not have) — so a tag re-pin
# right now would silently swap in ~100 unrelated commits' worth of library
# code, which is a real dependency bump belonging in its own reviewed,
# tested PR, not a gate-reason fix.
#
# Until that re-pin lands, this test holds the line at the ONE acknowledged
# SHA: any OTHER SHA pin (a silent swap, or reintroduction after the tag
# re-pin) fails here instead of silently reopening the unmeasured gate.
_ACKNOWLEDGED_SHA_PIN = "c907a044bb1553815225327bc56644050543b6f2"


def test_sha_pin_is_the_one_acknowledged_exception_not_a_silent_new_one():
    """A SHA pin is allowed ONLY if it is the acknowledged one.

    Tracked by alpha-engine-config-I7171 — re-pin to a vX.Y.Z tag once
    nousergon-lib ships a release reachable from this commit (or once the
    fix this commit carries is superseded on a tagged line), then delete
    this exception and _LIB_SHA_PIN_RE's callers can go back to rejecting
    SHA pins unconditionally.
    """
    for filename in ("requirements.txt", "requirements-lambda.txt"):
        pin = _read_lib_pin(filename)
        if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", pin):
            continue  # tag pin — always fine, comparable to the floor
        assert pin == _ACKNOWLEDGED_SHA_PIN, (
            f"{filename} pins nousergon-lib by an UNACKNOWLEDGED commit SHA "
            f"({pin!r}). A SHA pin defeats inference/lib_pin_drift.py's "
            f"cross-repo floor check (alpha-engine-config-I7171) — either "
            f"re-pin to a vX.Y.Z tag, or update _ACKNOWLEDGED_SHA_PIN in "
            f"this file (and its docstring) with the rationale for the "
            f"new SHA."
        )


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
