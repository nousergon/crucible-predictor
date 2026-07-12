"""Full-closure static pin: every first-party import reachable from a
Lambda-deployed entrypoint — module-scope OR deferred/function-local —
must have a matching Dockerfile COPY.

Background
----------
``tests/test_dockerfile_handler_import_completeness.py`` (PR352,
config#1282) derives the required COPY set from a scoped regex over
``inference/handler.py``'s own DIRECT ``from X import ...`` lines. That
regex has two blind spots:

  1. ``import X`` style (no ``from``) is invisible to it.
  2. TRANSITIVE imports are invisible — a package imported by a module
     that ``inference/handler.py`` imports (rather than by
     ``handler.py`` itself) is not derived at all.

config#2334 found the exact bug class one hop deeper via #2 during a
grep audit: ``labeling/`` and ``risk_model/`` are first-party packages
with no Dockerfile COPY, and they ARE imported by modules that live
inside COPY'd directories (``data/label_generator.py`` imports
``labeling.triple_barrier``; ``training/risk_model_persist.py`` imports
``risk_model``). Verified NOT a live gap (see below) — but the next
handler action that reaches either would replay the PR352 prod-500
exactly.

Deferred imports ARE walked (unlike the nousergon-data reference)
--------------------------------------------------------------------
nousergon-data's ``test_dockerfile_copies_match_deployed_imports.py``
(the strongest in-fleet static reference for this pattern) deliberately
walks MODULE-SCOPE imports only (``tree.body``), treating deferred
imports as out of scope. That convention does not fit here: the
original PR352 bug this whole test file lineage exists to prevent
(``monitoring.drift_detector`` inside ``check_drift``) is ITSELF a
deferred, function-local import — every first-party import in
``inference/handler.py`` except the module-top ``krepis.logging`` one
is deferred by design (per-action dispatch keeps cold-start fast; see
handler.py's action dispatch block). A module-scope-only walk would
have missed the very bug that motivated this test suite. So this test
walks the FULL AST (``ast.walk``, not just ``tree.body``) — every
``import X`` / ``from X import ...`` anywhere in a reachable file's
tree, regardless of nesting.

Submodule-precise resolution
-----------------------------
A naive "package X is imported somewhere, so every file under X/ is
reachable" over-approximation would wrongly mark e.g.
``data/label_generator.py`` reachable just because *some* other file
under ``data/`` is imported by the entrypoints — even though nothing
actually imports ``label_generator`` specifically. This walker
resolves each dotted import (``from data.label_generator import ...``)
to the SPECIFIC file it names, and only that file is queued for
further traversal — so the closure stays precise.

Why this test does NOT add COPY lines for labeling/ or risk_model/
--------------------------------------------------------------------
Both import sites are inside ``training/meta_trainer.py`` and
``training/train_handler.py``, and ``training/`` itself — while
physically ``COPY``'d into the image — is never imported (module-scope
or deferred) by any of the three Lambda CMDs this image serves
(``inference.handler.handler``, ``regime.handler.lambda_handler``,
``regime.retrospective_eval_handler.lambda_handler``; see
infrastructure/deploy.sh). Training runs on EC2 spot
(infrastructure/spot_train.sh, a separate git checkout, NOT the Docker
image) — so ``training/`` shipping in the Lambda image is inert cargo,
not a runtime dependency of anything invoked there. Confirmed via
repo-wide grep: no non-test, non-``training/`` file imports
``data.label_generator``, ``labeling``, or ``risk_model`` anywhere.

This test pins that closure so the fact stays true: it walks the REAL
full import graph (module-scope + deferred) from each Lambda
entrypoint and fails loudly the moment a Lambda-reachable module gains
ANY import of a first-party package/module that isn't itself COPY'd.
If ``labeling``/``risk_model`` (or any other package) ever become
reachable — e.g. a future handler action calls into ``training/`` —
this test fails and says exactly which COPY line to add.

The robust runtime backstop is the built-image import-walk CI job
(``.github/workflows/ci.yml``'s ``docker-import-closure`` job), which
actually imports the real entrypoint modules inside the built
container — this static test is the fast pre-check that runs on every
PR without a Docker build; the CI job is the ground truth.

Pattern source
--------------
Ports nousergon-data's ``test_dockerfile_copies_match_deployed_imports.py``
AST-based full-closure walk to crucible-predictor, widened from
module-scope-only to full-AST since this repo's handler imports are
deferred by convention (see above), and made submodule-precise so a
COPY'd package with one unreachable submodule doesn't false-negative.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"

# The three Lambda CMDs this image serves (infrastructure/deploy.sh /
# setup-regime-*.sh). The full import graph reachable from these files
# is the real "deployed closure."
_ENTRYPOINTS = (
    "inference/handler.py",
    "regime/handler.py",
    "regime/retrospective_eval_handler.py",
)


def _dockerfile_copied_dirs() -> set[str]:
    """Top-level directory names COPY'd by the Dockerfile, e.g.
    ``COPY monitoring/ monitoring/`` -> ``{"monitoring"}``."""
    out: set[str] = set()
    for line in DOCKERFILE.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "COPY" and parts[1].endswith("/"):
            out.add(parts[1].rstrip("/"))
    return out


def _dockerfile_copied_single_files() -> set[str]:
    """Single-file root modules COPY'd by the Dockerfile, e.g.
    ``COPY retry.py .`` -> ``{"retry"}``."""
    out: set[str] = set()
    for line in DOCKERFILE.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "COPY" and parts[1].endswith(".py"):
            out.add(Path(parts[1]).stem)
    return out


def _local_packages() -> set[str]:
    """Repo-root directories that are first-party Python packages."""
    return {
        p.name for p in REPO_ROOT.iterdir()
        if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith(".")
    }


def _local_single_file_modules() -> set[str]:
    """Repo-root single-file first-party modules (no package dir)."""
    return {p.stem for p in REPO_ROOT.glob("*.py")}


def _dotted_import_targets(py_file: Path) -> set[str]:
    """Every dotted import target ANYWHERE in the file — module scope
    AND deferred (function-local / inside try/if blocks). For
    ``import a.b.c`` yields ``"a.b.c"``; for ``from a.b import c, d``
    yields ``"a.b"``, ``"a.b.c"``, ``"a.b.d"`` (the last two cover the
    ``from pkg import submodule`` idiom, e.g.
    ``from data import label_generator``, where ``c``/``d`` may
    themselves be submodules rather than symbols)."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.add(node.module)
                for alias in node.names:
                    out.add(f"{node.module}.{alias.name}")
    return out


def _resolve_first_party(dotted: str, local_pkgs: set[str], local_single_files: set[str]) -> Path | None:
    """Resolve a dotted import path to the most specific existing
    first-party .py file (submodule file, or package ``__init__.py``)
    under the repo root. Returns None if the root isn't first-party, or
    no filesystem entity matches any prefix of the dotted path."""
    parts = dotted.split(".")
    root = parts[0]
    if root not in local_pkgs and root not in local_single_files:
        return None
    # Try the most specific prefix first: a.b.c as a file a/b/c.py,
    # then a/b.py, then a.py (single-file root module), then fall back
    # to the package __init__.py for whichever prefix is a real dir.
    for end in range(len(parts), 0, -1):
        as_file = REPO_ROOT.joinpath(*parts[:end - 1], f"{parts[end - 1]}.py")
        if as_file.is_file():
            return as_file
        as_pkg_init = REPO_ROOT.joinpath(*parts[:end], "__init__.py")
        if as_pkg_init.is_file():
            return as_pkg_init
    return None


def _top_level_root(dotted: str) -> str:
    return dotted.split(".")[0]


def _reachable_closure() -> dict[Path, str]:
    """BFS from the three Lambda entrypoints over the FULL import graph
    (module-scope + deferred), resolving each dotted import to the
    SPECIFIC first-party file it names. Returns a mapping of every
    reachable file to the top-level package/module root name that
    would need a Dockerfile COPY line to satisfy it (the file's own
    root, for reporting purposes).
    """
    local_pkgs = _local_packages()
    local_single_files = _local_single_file_modules()

    seen: dict[Path, str] = {}
    queue: list[tuple[Path, str]] = [(REPO_ROOT / e, _top_level_root(e.replace("/", "."))) for e in _ENTRYPOINTS]

    while queue:
        py, root = queue.pop()
        if py in seen or not py.exists():
            continue
        seen[py] = root
        for dotted in _dotted_import_targets(py):
            target = _resolve_first_party(dotted, local_pkgs, local_single_files)
            if target is None or target in seen:
                continue
            queue.append((target, _top_level_root(dotted)))

    return seen


def _missing_copies(reachable: dict[Path, str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Given the reachable closure, find every (root, importer) pair
    where root is a first-party package/module NOT covered by a
    Dockerfile COPY line. Splits into directory-style vs single-file
    misses for reporting.
    """
    copied_dirs = _dockerfile_copied_dirs()
    copied_single_files = _dockerfile_copied_single_files()
    local_pkgs = _local_packages()
    local_single_files = _local_single_file_modules()

    missing_dirs: dict[str, list[str]] = {}
    missing_files: dict[str, list[str]] = {}
    for py, _self_root in reachable.items():
        rel = str(py.relative_to(REPO_ROOT))
        for dotted in _dotted_import_targets(py):
            root = _top_level_root(dotted)
            if root in local_pkgs and root not in copied_dirs:
                missing_dirs.setdefault(root, []).append(rel)
            elif root in local_single_files and root not in copied_single_files:
                missing_files.setdefault(root, []).append(rel)
    return missing_dirs, missing_files


def test_entrypoints_exist() -> None:
    """Sanity check the entrypoint list itself hasn't silently gone
    stale (e.g. a handler renamed/removed) — if this trips, the whole
    closure walk below is vacuous."""
    missing = [e for e in _ENTRYPOINTS if not (REPO_ROOT / e).exists()]
    assert not missing, f"Lambda entrypoint file(s) no longer exist: {missing}"


def test_dockerfile_copies_every_import_in_deployed_closure() -> None:
    """Walk the real full import graph (module scope + deferred) from
    every Lambda entrypoint and assert every first-party package/module
    referenced, anywhere in that closure, has a matching Dockerfile
    COPY line.

    This is the transitive generalization of
    ``test_dockerfile_handler_import_completeness.py``: it does not
    stop at handler.py's own direct imports, it does not stop at
    module-scope imports, and it covers all three Lambda entrypoints
    (inference + both regime Lambdas), not just the inference one.
    """
    reachable = _reachable_closure()
    assert reachable, (
        "Closure walk found zero reachable files from the Lambda "
        "entrypoints — BFS or Dockerfile COPY parsing is broken."
    )

    missing_dirs, missing_files = _missing_copies(reachable)

    assert not missing_dirs and not missing_files, (
        "The following first-party imports are reachable (module scope "
        "or deferred) from a Lambda entrypoint (inference.handler / "
        "regime.handler / regime.retrospective_eval_handler) but have "
        "no matching Dockerfile COPY line — the deployed image will "
        "ModuleNotFoundError at import time (config#1282 / PR352 bug "
        "class):\n"
        + "\n".join(
            f"  - COPY {pkg}/ {pkg}/   (imported by: {', '.join(sorted(set(files)))})"
            for pkg, files in sorted(missing_dirs.items())
        )
        + ("\n" if missing_dirs and missing_files else "")
        + "\n".join(
            f"  - COPY {mod}.py .   (imported by: {', '.join(sorted(set(files)))})"
            for mod, files in sorted(missing_files.items())
        )
    )


def test_labeling_and_risk_model_are_not_in_the_deployed_closure() -> None:
    """Pins the config#2334 finding: ``labeling/`` and ``risk_model/``
    are first-party packages WITHOUT a Dockerfile COPY, and that is
    currently correct because neither is reachable from any Lambda
    entrypoint — both are only reached via imports inside ``training/``
    (meta_trainer.py, risk_model_persist.py), and ``training/`` is
    never imported by an entrypoint (training runs on EC2 spot, not
    this image).

    If this test starts failing, it means ``labeling`` or ``risk_model``
    HAS become reachable from a Lambda entrypoint (e.g. a new handler
    action imports something under training/) — at that point add the
    matching ``COPY labeling/ labeling/`` / ``COPY risk_model/
    risk_model/`` line(s) to the Dockerfile rather than adjusting this
    test.
    """
    reachable_files = set(_reachable_closure())
    reachable_names = {str(p.relative_to(REPO_ROOT)) for p in reachable_files}
    labeling_files = {
        str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "labeling").rglob("*.py")
    }
    risk_model_files = {
        str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "risk_model").rglob("*.py")
    }

    assert not (reachable_names & labeling_files), (
        f"labeling/ has become reachable from a Lambda entrypoint: "
        f"{sorted(reachable_names & labeling_files)}. Add `COPY "
        "labeling/ labeling/` to the Dockerfile — do not weaken this "
        "test."
    )
    assert not (reachable_names & risk_model_files), (
        f"risk_model/ has become reachable from a Lambda entrypoint: "
        f"{sorted(reachable_names & risk_model_files)}. Add `COPY "
        "risk_model/ risk_model/` to the Dockerfile — do not weaken "
        "this test."
    )


def test_training_package_is_not_reachable_from_any_lambda_entrypoint() -> None:
    """Documents + pins WHY labeling/risk_model are exempt: nothing
    under ``training/`` is reachable from any Lambda entrypoint at all
    (module scope or deferred). If this test fails, ``training/``'s
    exemption above is no longer valid and the whole closure — including
    ``labeling``/``risk_model`` — needs its Dockerfile COPY story
    re-examined, not just this test's assumption.
    """
    reachable_files = set(_reachable_closure())
    reachable_names = {str(p.relative_to(REPO_ROOT)) for p in reachable_files}
    training_files = {
        str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "training").rglob("*.py")
    }
    hit = reachable_names & training_files
    assert not hit, (
        f"training/ has become reachable from a Lambda entrypoint: "
        f"{sorted(hit)}. This invalidates the assumption behind "
        "test_labeling_and_risk_model_are_not_in_the_deployed_closure "
        "— training/'s own imports (labeling, risk_model, analysis) "
        "may now need Dockerfile COPY lines too. Investigate before "
        "adjusting any test."
    )
