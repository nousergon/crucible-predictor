"""Regression: no module in this repo reads a secret via ``os.environ.get``.

After the 2026-05-12 ``.env`` → SSM migration (PR 4 of the arc), every
secret-bearing call site routes through ``krepis.secrets.get_secret()``.
This test re-greps the codebase on every CI run so a future commit can't
silently re-introduce an ``os.environ.get("POLYGON_API_KEY")`` style read.

Non-secret env vars (``LANGCHAIN_PROJECT``, ``EMAIL_SENDER``,
``PREDICTOR_PARAMS_CACHE``, etc.) are allowed for now — they migrate to
alpha-engine-config YAML in PR 8 of the arc.

The legacy ``ssm_secrets.py`` bulk-load shim was retired in PR 9 of the
.env→SSM arc (config#890); the allowlist below is empty and the invariant
applies to every module. Re-add a filename here only if a new deliberate
shim is ever introduced.

alpha-engine-config-I7925: this repo-tree scan is blind to a first-party
*dependency* reading a pinned secret via ``os.environ.get`` from
``site-packages`` — exactly how ``nousergon_lib.preflight`` reading
``GITHUB_TOKEN`` bypassed this very invariant and halted preopen trading
(alpha-engine-config-I7924). ``test_no_secret_environ_reads_in_installed_dependencies``
below closes that gap using the scanner shared via
``nousergon_lib.testing.secret_scan`` (lifted there rather than copied a
third time — this repo, nousergon-data, and nousergon-lib's own suite all
call it; see nousergon-lib#345).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nousergon_lib.testing.secret_scan import scan_installed_packages

_REPO_ROOT = Path(__file__).resolve().parent.parent

# First-party packages this repo installs whose installed tree must also be
# clean of a pinned-secret os.environ.get read — the repo-tree scan below
# cannot see inside site-packages.
_DEPENDENCY_PACKAGES = ("nousergon_lib", "krepis")

_PINNED_SECRETS = frozenset(
    [
        "ANTHROPIC_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_API_KEY",
        "VOYAGE_API_KEY",
        "POLYGON_API_KEY",
        "FMP_API_KEY",
        "FINNHUB_API_KEY",
        "FRED_API_KEY",
        "GMAIL_APP_PASSWORD",
        "GITHUB_TOKEN",
        "RAG_DATABASE_URL",
        "EDGAR_IDENTITY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "EMAIL_SENDER",
        "EMAIL_RECIPIENTS",
    ]
)

_ALLOWED_FILES: frozenset[str] = frozenset()

_ENV_READ_RE = re.compile(
    r'os\.(?:environ\.get|getenv)\(\s*["\']([A-Z_][A-Z0-9_]*)["\']'
)


def _iter_python_files():
    for path in _REPO_ROOT.rglob("*.py"):
        parts = set(path.parts)
        if parts & {".venv", "build", "tests", "node_modules", "package"}:
            continue
        if path.name in _ALLOWED_FILES:
            continue
        yield path


def test_no_secret_environ_reads():
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_python_files():
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _ENV_READ_RE.finditer(line):
                name = match.group(1)
                if name in _PINNED_SECRETS:
                    violations.append((path.relative_to(_REPO_ROOT), lineno, name))
    assert not violations, (
        "Found os.environ.get reads of pinned secrets — use "
        "`from krepis.secrets import get_secret` instead:\n"
        + "\n".join(f"  {p}:{ln}  {name}" for p, ln, name in violations)
    )


def test_no_secret_environ_reads_in_installed_dependencies():
    """The repo-tree scan above cannot see an installed dependency's source.

    ``nousergon_lib.preflight._github_auth_headers()`` reading
    ``GITHUB_TOKEN`` via a literal ``os.environ.get`` from site-packages is
    exactly the surface that let an expired credential halt preopen trading
    (alpha-engine-config-I7924) while this repo's own scan reported clean.
    """
    violations, missing = scan_installed_packages(_DEPENDENCY_PACKAGES, _PINNED_SECRETS)
    if violations:
        raise AssertionError(
            "Found os.environ.get reads of pinned secrets inside an "
            "INSTALLED first-party dependency — this repo's own tree scan "
            "cannot see this surface:\n"
            + "\n".join(f"  {v}" for v in violations)
        )
    if missing:
        pytest.skip(
            "first-party package(s) not importable in this environment — "
            f"the invariant is unverified against them this run: {', '.join(missing)}"
        )
