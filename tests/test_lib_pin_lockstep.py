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
    r"(?:alpha-engine-lib|nousergon-lib)\[[^\]]*\]\s*@\s*git\+https://github\.com/nousergon/nousergon-lib@(v[0-9]+\.[0-9]+\.[0-9]+)"
)


def _read_lib_pin(filename: str) -> str:
    text = (_REPO_ROOT / filename).read_text()
    match = _LIB_PIN_RE.search(text)
    assert match is not None, (
        f"could not find alpha-engine-lib pin in {filename} — the regex "
        f"expects ``alpha-engine-lib[extras] @ git+https://.../alpha-engine-lib@vX.Y.Z``"
    )
    return match.group(1)


def test_requirements_and_lambda_pins_match():
    """Both files must pin alpha-engine-lib to the same tag."""
    root_pin = _read_lib_pin("requirements.txt")
    lambda_pin = _read_lib_pin("requirements-lambda.txt")
    assert root_pin == lambda_pin, (
        f"alpha-engine-lib pin drift: requirements.txt={root_pin!r} but "
        f"requirements-lambda.txt={lambda_pin!r}. Both files must pin to "
        f"the same tag — they're two views of the same dependency graph. "
        f"Drift broke the predictor canary on 2026-05-12 (PR #147 / hotfix "
        f"fix/lambda-lib-pin-v0.12.0)."
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
