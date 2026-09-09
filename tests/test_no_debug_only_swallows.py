"""Class guard: no `except Exception` handler in this repo's core source may
swallow the failure into `logger.debug(...)` or a bare `pass` (alpha-engine-
config-I10031, class present fleet-wide per alpha-engine-config-I10226).

Thin call-site over the lifted detector, `nousergon_lib.testing.
debug_swallow_guard` (`nousergon-lib-PR398`) — the AST walk, the allowlist
diff, and the self-containment check live there now, not here. See that
module's docstring for the exact class this catches (a bare `except
Exception:`/`except Exception as e:` whose ENTIRE body is a single
`logger.debug(...)` call or `pass`) and `crucible-executor`'s original
`tests/test_no_debug_only_swallows.py` for the pattern this mirrors.

This repo has no single top-level package (unlike `executor/` in
crucible-executor) — core source is scanned as one directory per top-level
module plus the repo root's own `*.py` files, merged into one dict. `tests/`,
`scripts/` (ops tooling) and `infrastructure/` (deploy/IaC) are out of scope,
matching crucible-executor's convention of scanning only the production
package(s), not tooling around them.
"""

from __future__ import annotations

from pathlib import Path

from nousergon_lib.testing.debug_swallow_guard import (
    check_against_allowlist,
    check_allowlist_entries_self_contained,
    find_debug_only_swallows,
    load_allowlist,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ALLOWLIST_PATH = _REPO_ROOT / ".debug-swallow-allowlist.yaml"

# One entry per top-level source directory this repo's production code lives
# in, plus the repo root itself (config.py, data_manifest.py, ops_alerts.py,
# polygon_client.py, retry.py, stage_coverage_safety.py). `inference/stages/`
# is scanned separately since the detector is non-recursive.
_SOURCE_DIRS = [
    _REPO_ROOT,
    _REPO_ROOT / "analysis",
    _REPO_ROOT / "data",
    _REPO_ROOT / "inference",
    _REPO_ROOT / "inference" / "stages",
    _REPO_ROOT / "labeling",
    _REPO_ROOT / "model",
    _REPO_ROOT / "monitoring",
    _REPO_ROOT / "regime",
    _REPO_ROOT / "risk_model",
    _REPO_ROOT / "store",
    _REPO_ROOT / "training",
]


def _all_swallow_sites() -> dict[str, set[int]]:
    merged: dict[str, set[int]] = {}
    for d in _SOURCE_DIRS:
        for path, lines in find_debug_only_swallows(d, repo_root=_REPO_ROOT).items():
            if lines:
                merged.setdefault(path, set()).update(lines)
    return merged


def _load_allowlist() -> list[dict]:
    return load_allowlist(_ALLOWLIST_PATH)


def test_no_new_debug_only_swallows_outside_allowlist():
    """Every debug-only-or-pass `except Exception` swallow in this repo's
    core source is either fixed (raised, or recorded at WARNING/ERROR+) or
    has a non-expired, matching entry in `.debug-swallow-allowlist.yaml`."""
    live_sites = _all_swallow_sites()
    allowlist = _load_allowlist()
    failures = check_against_allowlist(live_sites, allowlist)
    assert not failures, "\n".join(failures)


def test_allowlist_entries_are_self_contained():
    """Every entry names a reason, an expiry, and a tracking issue — a
    swallow with no named recording surface is not a swallow, it is a
    deletion (alpha-engine-config-I10031 deliverable 2)."""
    failures = check_allowlist_entries_self_contained(_load_allowlist())
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
