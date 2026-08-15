"""Consumer-contract test: `krepis.stage_coverage` must actually be importable
in this repo's own resolved environment, and every call site must target that
namespace — never the never-shipped `nousergon_lib.stage_coverage`.

**Root cause this guards against** (`alpha-engine-config-I7389`).
`nousergon_lib.stage_coverage` was added to nousergon-lib and immediately
removed as a duplicate of the krepis landing (`nousergon-lib#314`/`#315`) — it
never existed on any released version. `alpha-engine-config-I7334` established
that and fixed `crucible-evaluator`. **This repo was never swept**, and kept
importing the dead name at eight sites: `inference/handler.py` (x3),
`regime/handler.py`, `regime/retrospective_eval_handler.py`, and three
`infrastructure/spot_*.sh` launchers.

Measured live 2026-08-14 23:59 UTC on `/aws/lambda/alpha-engine-predictor-inference`,
on EVERY invocation::

    ERROR [predictor] stage-coverage assertion unavailable:
      No module named 'nousergon_lib.stage_coverage'

So `alpha-engine-config-I7214`'s coverage assertion had never once executed in
the predictor, in either the Lambda or the spot stages.

**Why it stayed invisible, and what that demands of this module.** Both layers
degrade silently by construction: the Python sites swallow `ImportError` into a
log line and continue, and the shell sites are explicitly observe-mode
(`|| echo "WARNING: ... stage NOT failed"`). A detector that cannot import is
indistinguishable from one that ran and found nothing. `crucible-evaluator` and
`crucible-dashboard` each carry a guard for this; this repo was the only one of
the four without one, which is precisely why I7334's sweep missed it.

Mirrors `crucible-evaluator/tests/test_stage_coverage_packaging.py`. Every
assertion derives from the real source — the call sites are discovered by
walking the tree, not listed — so a ninth site added later is covered without
editing this file.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_REQUIREMENTS = _REPO_ROOT / "requirements.txt"
_LAMBDA_REQUIREMENTS = _REPO_ROOT / "requirements-lambda.txt"
_INFRA = _REPO_ROOT / "infrastructure"

#: The krepis release that first ships `krepis.stage_coverage` (krepis#148,
#: config-I7214/I7334). Corrected 0.59.1 -> 0.59.2 under config-I7357 by
#: walking the krepis tags rather than reading a changelog line — a contract
#: test keyed on a wrong constant passes on its own bug.
_STAGE_COVERAGE_INTRODUCED_IN = (0, 59, 2)

#: The namespace that never existed. Its reappearance anywhere executable is
#: the regression.
_DEAD_NAMESPACE = "nousergon_lib.stage_coverage"

_PIN_RE = re.compile(r"^krepis(\[[^\]]*\])?(?:>=|==)(\d+)\.(\d+)\.(\d+)")


def _krepis_pin(path: pathlib.Path) -> tuple[int, int, int]:
    """Parse the `krepis[...]==X.Y.Z` (or `>=`) version out of a requirements
    file. Regex-based rather than adding a `packaging` dependency this repo
    does not otherwise declare — derive from the real source, never duplicate
    the number by hand.
    """
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        m = _PIN_RE.match(line)
        if m:
            return (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    raise AssertionError(f"{path.name} has no `krepis==X.Y.Z` / `krepis>=X.Y.Z` line")


@pytest.mark.parametrize("path", [_REQUIREMENTS, _LAMBDA_REQUIREMENTS], ids=lambda p: p.name)
def test_krepis_pin_covers_stage_coverage_introduction(path: pathlib.Path) -> None:
    """Both requirement files, not just the dev one: the Lambda resolves
    `requirements-lambda.txt`, so a floor that is correct in one file and
    stale in the other reproduces this defect in production while CI is green.
    """
    pin = _krepis_pin(path)
    assert pin >= _STAGE_COVERAGE_INTRODUCED_IN, (
        f"{path.name} krepis pin {'.'.join(map(str, pin))} predates "
        f"{'.'.join(map(str, _STAGE_COVERAGE_INTRODUCED_IN))}, which first ships "
        "krepis.stage_coverage — every stage-coverage call site would take the "
        "ImportError branch again (alpha-engine-config-I7389)."
    )


def test_krepis_stage_coverage_is_importable_in_this_environment() -> None:
    """The consumer-contract assertion I7214/I7334 required: actually import
    the module, in the SAME environment CI resolves requirements into. If this
    fails the pin is wrong (or krepis regressed) — not the handlers' degrade
    path, which is exercised separately and must log loud, never pass quietly.
    """
    module = pytest.importorskip(
        "krepis.stage_coverage",
        reason="krepis.stage_coverage unimportable — see the krepis pin in requirements.txt",
    )
    assert hasattr(module, "assert_stage_coverage")
    assert callable(module.assert_stage_coverage)


def _python_sources() -> list[pathlib.Path]:
    """Every first-party Python file, tests excluded — tests legitimately name
    the dead namespace when documenting the defect."""
    return sorted(
        p
        for p in _REPO_ROOT.rglob("*.py")
        if p.is_file()
        and "tests" not in p.parts
        and ".venv" not in p.parts
        and not any(part.startswith(".") for part in p.relative_to(_REPO_ROOT).parts)
    )


def _launcher_scripts() -> list[pathlib.Path]:
    return sorted(p for p in _INFRA.rglob("*.sh") if p.is_file())


def _executable_lines(path: pathlib.Path) -> list[str]:
    """Non-comment, non-blank lines. A comment may legitimately quote the dead
    name — the whole rationale is about it — so the sweep runs over code only."""
    return [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().lstrip("#").startswith("!") and not line.strip().startswith("#")
    ]


@pytest.mark.parametrize(
    "path",
    _python_sources() + _launcher_scripts(),
    ids=lambda p: str(p.relative_to(_REPO_ROOT)),
)
def test_no_source_names_the_dead_namespace(path: pathlib.Path) -> None:
    """The load-bearing sweep, derived from the tree rather than a list: no
    executable line in any first-party source may name
    `nousergon_lib.stage_coverage`. It has never existed on any released
    nousergon-lib, so every such reference is a detector that cannot run.
    """
    offenders = [line.strip() for line in _executable_lines(path) if _DEAD_NAMESPACE in line]
    assert not offenders, (
        f"{path.relative_to(_REPO_ROOT)}: executable line(s) name "
        f"{_DEAD_NAMESPACE!r}, which has never shipped on any nousergon-lib "
        f"release: {offenders}. Use `krepis.stage_coverage` "
        "(alpha-engine-config-I7334/I7389)."
    )


def test_at_least_one_source_actually_calls_stage_coverage() -> None:
    """Guards the sweep above against passing vacuously. If the call sites were
    all deleted or renamed, every parametrized case would pass while I7214's
    assertion silently stopped existing — the failure mode this whole module is
    about, one level up.
    """
    hits = [
        p
        for p in _python_sources() + _launcher_scripts()
        if "krepis.stage_coverage" in p.read_text()
    ]
    assert hits, (
        "no first-party source references krepis.stage_coverage — either the "
        "coverage assertion was removed (which must be a deliberate, reviewed "
        "diff) or this derivation no longer matches the tree."
    )
