#!/usr/bin/env python3
"""Fail on any dependency extra written with an underscore.

WHY THIS EXISTS
---------------
PEP 685 normalisation of extras landed in pip 23.3. Before that, pip compared
the *requested* extra string against the distribution's ``Provides-Extra``
metadata without normalising ``_`` to ``-``. setuptools has always normalised
on the writing side, so a project declaring::

    [project.optional-dependencies]
    flow_doctor = ["flow-doctor[diagnosis,s3]>=0.8.1,<0.10"]

publishes ``Provides-Extra: flow-doctor``. A consumer asking for
``krepis[flow_doctor]`` therefore matches nothing on old pip — and pip reports
that as a **WARNING on a successful exit**, so the install is silently short a
package and the failure surfaces much later as an ImportError, in a different
process, on a different machine.

Measured 2026-08-12 against ``krepis[flow_doctor]==0.38.0`` (dry-run resolve):

    pip 23.2.1  ->  18 packages, "WARNING: krepis 0.38.0 does not provide the
                    extra 'flow_doctor'", exit 0.  flow-doctor ABSENT.
    pip 23.3.2  ->  30 packages, flow-doctor present.
    pip 24.0    ->  30 packages, flow-doctor present.
    pip 25.0    ->  30 packages, flow-doctor present.

Amazon Linux 2023's ``python3.12-pip`` is **23.2.1**, so every spot bootstrap in
this fleet installs on the broken side of that boundary. That is how a
five-times-repaired weekly pipeline arrived at a sixth cause (config#6963): the
training smoke died on ``ModuleNotFoundError: No module named 'flow_doctor'``
raised out of ``krepis.logging.setup_logging``, from an environment pip had
reported as installed successfully.

The hyphenated form resolves correctly on EVERY pip version. It is the correct
spelling unconditionally, not a workaround for old pip — which is why this
check is a hard failure rather than a warning, and why it is not conditioned on
the local pip version. CI runs on a modern pip and would never reproduce the
production failure; the point of a static check is that it does not have to.

Scope: requirement *requesters* only. A ``[project.optional-dependencies]``
table KEY may legitimately be written either way (setuptools normalises it on
publish), so declaration tables are not flagged.

THE DENOMINATOR IS THE WHOLE TREE
---------------------------------
The first revision of this file globbed ``requirements*.txt`` at the repo ROOT
and never looked at workflow YAML. Both were blind spots with a live defect in
them on the day the guard shipped:

* ``.github/workflows/deploy.yml`` ran ``pip install "krepis[flow_doctor]>=0.7.0"``
  — surviving PR #465, which fixed every requester this checker could see.
* ``nousergon-data/infrastructure/lambdas/scheduled-groom-dispatcher/requirements.txt``
  requested ``nousergon-lib[flow-doctor,github_app]`` — a *nested* requirements
  file, and a second underscored extra (``github_app``) nobody had looked for
  because the sweep was written around the one symbol that had already failed.

A checker reporting "clean" for files it never opened is worse than no checker:
it converts an unexamined tree into a green check. So the requirements scan is
recursive, workflow YAML is scanned as an inline surface, and ``rc=2`` on an
empty denominator (already present) is the backstop for the case where the walk
finds nothing at all.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# A requirement's extras group: NAME[extra1,extra2] — capture the bracket body.
_EXTRAS_RE = re.compile(r"^\s*(?P<name>[A-Za-z0-9._-]+)\s*\[(?P<extras>[^\]]*)\]")

# Files whose lines are requirement REQUESTERS. RECURSIVE: a nested
# `infrastructure/lambdas/<fn>/requirements.txt` installs on exactly the same
# resolver as a root one, and is where the surviving instance of this defect was
# found (see "THE DENOMINATOR IS THE WHOLE TREE").
_REQUIREMENTS_GLOBS = ("**/requirements*.txt",)

# Directories never walked: vendored/installed trees carry third-party
# requirements files this repo does not author and cannot fix, and a concurrent
# agent worktree is a different checkout's problem. Kept as an explicit,
# reviewable list rather than a heuristic — anything not named here IS scanned,
# so the default direction is "look at it".
_PRUNED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        "site-packages",
        "build",
        "dist",
        ".worktrees",
    }
)

# Install commands embedded in CI workflow YAML are requesters like any other,
# and they resolve on the runner's pip. Matching is restricted to lines that
# actually invoke an installer so an unrelated YAML value containing brackets
# cannot be mistaken for a requirement.
_WORKFLOW_GLOBS = (".github/workflows/*.yml", ".github/workflows/*.yaml")
_INSTALL_CMD_RE = re.compile(r"\b(?:pip3?|uv pip|python3?\s+-m\s+pip)\s+install\b")

# In pyproject/Dockerfiles the requester appears inside a string or a pip
# command rather than at line start, so match the bracket form anywhere.
_INLINE_RE = re.compile(r"(?P<name>[A-Za-z0-9._-]+)\[(?P<extras>[^\]]*)\]")
# Dockerfiles carry `RUN pip install "pkg[extra] @ git+https://..."` — the same
# requester shape inside a shell string. They matter MORE than requirements
# files here, not less: a Docker base image pins its own pip, so a Dockerfile
# that resolves correctly today breaks silently the day that pin moves back.
_INLINE_GLOBS = ("**/pyproject.toml", "**/Dockerfile", "**/Dockerfile.*")


def _offending_extras(extras_body: str) -> list[str]:
    return [e.strip() for e in extras_body.split(",") if "_" in e.strip()]


def is_pruned(path: Path, root: Path) -> bool:
    """True when any path component between ``root`` and ``path`` is pruned."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return any(part in _PRUNED_DIRS for part in rel.parts)


def _walk(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Every file under ``root`` matching ``patterns``, pruned and deduped."""
    found: dict[Path, None] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file() or is_pruned(path, root):
                continue
            found[path] = None
    return sorted(found)


def _scan_workflow(path: Path) -> list[str]:
    """Scan CI workflow YAML for requesters inside installer invocations.

    Only lines that actually run an installer are considered: a workflow is
    full of bracketed values (matrix entries, expressions) that are not
    requirements, and a checker that flags those gets switched off.
    """
    problems: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0]
        if not _INSTALL_CMD_RE.search(line):
            continue
        for m in _INLINE_RE.finditer(line):
            bad = _offending_extras(m.group("extras"))
            if bad:
                problems.append(
                    f"{path}:{lineno}: {m.group('name')} requests extra(s) "
                    f"{bad} with an underscore — write "
                    f"{[b.replace('_', '-') for b in bad]}"
                )
    return problems


def _scan_requirements(path: Path) -> list[str]:
    problems: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = _EXTRAS_RE.match(line)
        if not m:
            continue
        bad = _offending_extras(m.group("extras"))
        if bad:
            problems.append(
                f"{path}:{lineno}: {m.group('name')} requests extra(s) "
                f"{bad} with an underscore — write {[b.replace('_', '-') for b in bad]}"
            )
    return problems


def _scan_inline(path: Path) -> list[str]:
    """Scan pyproject-style files, where requesters appear inside strings.

    A ``[project.optional-dependencies]`` table KEY needs no special-casing:
    ``_INLINE_RE`` requires the ``[`` to follow the name with no separator, and
    a TOML key is always followed by ``=`` before its list. So ``flow_doctor =
    ["krepis[flow-doctor]"]`` yields exactly one match — ``krepis[flow-doctor]``,
    the requester — and the key is invisible to the regex by construction.

    An earlier revision of this function carried an explicit skip for keys and
    it suppressed the requester on the same line instead, so a genuinely broken
    ``flow_doctor = ["krepis[flow_doctor]"]`` passed. Covered by
    ``test_declaration_table_key_is_not_a_requester``, both directions.
    """
    problems: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0]
        for m in _INLINE_RE.finditer(line):
            bad = _offending_extras(m.group("extras"))
            if bad:
                problems.append(
                    f"{path}:{lineno}: {m.group('name')} requests extra(s) "
                    f"{bad} with an underscore — write "
                    f"{[b.replace('_', '-') for b in bad]}"
                )
    return problems


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent

    problems: list[str] = []
    scanned = 0
    for path in _walk(root, _REQUIREMENTS_GLOBS):
        scanned += 1
        problems.extend(_scan_requirements(path))
    for path in _walk(root, _INLINE_GLOBS):
        scanned += 1
        problems.extend(_scan_inline(path))
    for path in _walk(root, _WORKFLOW_GLOBS):
        scanned += 1
        problems.extend(_scan_workflow(path))

    if scanned == 0:
        # A checker that silently scans nothing reports "clean" for a repo it
        # never opened. That is the failure shape this whole class is about.
        print(f"lint_extras: ERROR: no dependency files found under {root}", file=sys.stderr)
        return 2

    if problems:
        print(
            "lint_extras: underscored extras found — pip <23.3 drops these "
            "SILENTLY on a successful exit (see this file's docstring):",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"lint_extras: OK — {scanned} dependency file(s) scanned, no underscored extras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
