"""Every top-level first-party package `inference/handler.py` imports at
runtime must have a matching ``COPY <pkg>/ <pkg>/`` line in the Dockerfile.

Root-caused by config#1282 (PR #305): ``check_drift`` added
``from monitoring.drift_detector import check_drift`` but the Dockerfile
was never updated to copy ``monitoring/`` into the Lambda image. The module
existed in the repo and passed CI (which imports it directly off disk) but
was absent from the deployed container — every ``action=check_drift``
invocation 500'd with ``ModuleNotFoundError`` in prod, undetected because
deploy.sh's canary only exercises ``dry_run=true``, never ``check_drift``.

``tests/test_regime_dockerfile_wiring.py`` already pins this contract for
the ``regime/`` module specifically. This test generalizes it: it derives
the required package set directly from handler.py's imports, so a future
handler action importing a NEW top-level package fails CI immediately
instead of shipping a silent 500 to prod.

**Demoted to a fast pre-check (config#2334).** This regex only derives
DIRECT ``from X import ...`` lines in ``inference/handler.py`` — it is
blind to ``import X`` style and to TRANSITIVE imports (a package
imported by a module ``handler.py`` imports, rather than by
``handler.py`` itself). config#2334 found the exact bug class one hop
deeper this way (``labeling/``/``risk_model/``, not currently reachable
— see ``tests/test_dockerfile_import_closure.py``). Two stronger layers
now sit above this one:

  1. ``tests/test_dockerfile_import_closure.py`` — AST-based, walks the
     FULL first-party import graph (module scope + deferred) reachable
     from all three Lambda entrypoints, not just handler.py's direct
     imports.
  2. ``.github/workflows/ci.yml``'s ``docker-import-closure`` job —
     builds the real image and imports the real entrypoint modules
     inside the built container, the ground-truth runtime check.

This test is kept (not deleted) because it's near-instant and gives a
precise, minimal repro of the specific historical incident even when
the broader checks above are being modified.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HANDLER = REPO_ROOT / "inference" / "handler.py"

# First-party top-level packages already known to ship in the image via
# their own directory-level COPY (or, for `inference`, are the handler's
# own package and always present). Anything imported from a base name NOT
# in this set but corresponding to a real repo-root directory must have a
# matching COPY line, or this test's own coverage would blind-spot it too.
_FIRST_PARTY_DIRS = {
    p.name for p in REPO_ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()
}

_IMPORT_RE = re.compile(r"^\s*from (\w+)(?:\.\w+)* import ", re.MULTILINE)


def _first_party_packages_imported_by_handler() -> set[str]:
    text = HANDLER.read_text()
    bases = {m.group(1) for m in _IMPORT_RE.finditer(text)}
    return bases & _FIRST_PARTY_DIRS


def test_dockerfile_copies_every_package_handler_imports() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    imported = _first_party_packages_imported_by_handler()
    assert imported, (
        "Expected inference/handler.py to import at least one first-party "
        "package (e.g. inference, monitoring) — regex or package detection "
        "may have broken."
    )
    missing = {
        pkg for pkg in imported
        if f"COPY {pkg}/ {pkg}/" not in dockerfile
    }
    assert not missing, (
        f"Dockerfile is missing COPY line(s) for package(s) {sorted(missing)}, "
        f"which inference/handler.py imports at runtime. Without "
        f"'COPY <pkg>/ <pkg>/', the Lambda image ships without the module "
        f"and the handler 500s with ModuleNotFoundError in prod (config#1282 "
        f"class of bug) — deploy.sh's canary won't catch it since it only "
        f"exercises dry_run=true, not every handler action."
    )
