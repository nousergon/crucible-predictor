"""Diff the *resolved* versions of packages shared between requirements.txt
and requirements-lambda.txt, installed into two separate clean venvs.

``tests/test_lib_pin_lockstep.py`` already catches drift in the *specifier
text* between the two files. That's necessary but not sufficient: pip's
resolver considers the whole file together, so two files with identical
specifier text for a shared package can still resolve to different
transitive versions once the surrounding package set differs (as it does
here — requirements-lambda.txt intentionally omits training-only deps).
This script closes that gap by comparing what each file ACTUALLY resolves
to, not just what it asks for (config#3025 dim6).

Usage: check_requirements_resolved_lockstep.py <root-freeze.txt> <lambda-freeze.txt>
Exits 1 and prints every mismatched shared package on divergence.
"""

from __future__ import annotations

import sys


def _parse_freeze(path: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "@" in line:
                # VCS installs (e.g. nousergon-lib @ git+https://...) are
                # covered by test_lib_pin_lockstep.py's tag-equality check.
                continue
            if "==" not in line:
                continue
            pkg, _, version = line.partition("==")
            pins[pkg.lower()] = version
    return pins


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <root-freeze.txt> <lambda-freeze.txt>", file=sys.stderr)
        return 2

    root_pins = _parse_freeze(sys.argv[1])
    lambda_pins = _parse_freeze(sys.argv[2])

    shared = sorted(set(root_pins) & set(lambda_pins))
    mismatches = {
        pkg: (root_pins[pkg], lambda_pins[pkg])
        for pkg in shared
        if root_pins[pkg] != lambda_pins[pkg]
    }

    if mismatches:
        print(
            "Resolved-version drift between requirements.txt and "
            "requirements-lambda.txt for shared packages (identical "
            "specifier text can still resolve differently once the "
            "surrounding package set diverges):",
            file=sys.stderr,
        )
        for pkg, (root_v, lambda_v) in mismatches.items():
            print(f"  {pkg}: requirements.txt={root_v} requirements-lambda.txt={lambda_v}", file=sys.stderr)
        return 1

    print(f"OK: {len(shared)} shared packages resolve to identical versions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
