"""The underscored-extras guard, and proof it fails when the fix is reverted.

`overseer-policy.md` #13: a guard is not a guard until it has been observed
failing. The reverted-tree cases below are that observation — they reconstruct
the exact spelling that took the weekly pipeline down (alpha-engine-config-I6963)
and assert the checker rejects it.

Root cause being guarded: pip <23.3 does not normalise `_` to `-` in a
*requested* extra, while setuptools always normalises the *declared* one. So
`krepis[flow_doctor]` matches nothing against `Provides-Extra: flow-doctor`, and
pip reports that as a WARNING on a SUCCESSFUL exit. Measured 2026-08-12:
pip 23.2.1 resolves 18 packages with flow-doctor absent; 23.3.2 / 24.0 / 25.0
resolve 30 with it present. Amazon Linux 2023 ships pip 23.2.1.

Most cases call ``main()`` in-process so the checker's own lines are measured by
coverage; one case shells out, because the CLI exit-code contract is what CI
actually depends on and an in-process call would not exercise it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LINTER = REPO_ROOT / "scripts" / "lint_extras.py"

# Some repos run this suite INSIDE the built Lambda image, where the Dockerfile
# copies the application package and not `scripts/`. Importing the linter at
# module scope there raises FileNotFoundError during COLLECTION, which fails the
# whole suite rather than one test — measured on crucible-evaluator's
# `docker-image-tests` job, /var/task/scripts/lint_extras.py absent.
#
# Skipping is safe here and only here: this file tests a REPO-HYGIENE checker,
# and the packaged image is not the repo. The check itself is not weakened —
# `lint-extras` runs it against the real tree in its own CI job, so a genuinely
# broken linter still fails the build. The skip is loud (a stated reason) and
# scoped to the one condition where the file cannot exist by design; it never
# fires in a checkout, where `test_this_repo_is_clean` would catch a deletion.
if not LINTER.exists():  # pragma: no cover - image-context guard
    pytest.skip(
        f"{LINTER} absent — running inside a packaged image, not a checkout. "
        "The linter is exercised against the real tree by the `lint-extras` CI job.",
        allow_module_level=True,
    )

_spec = importlib.util.spec_from_file_location("lint_extras", LINTER)
assert _spec and _spec.loader
lint_extras = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint_extras)


def _rc(target: Path) -> int:
    """Run the checker in-process against `target`."""
    return lint_extras.main(["lint_extras.py", str(target)])


def _write(tmp_path: Path, name: str, body: str) -> None:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_this_repo_is_clean():
    """The real tree passes — the fix is in place and stays in place."""
    assert _rc(REPO_ROOT) == 0


def test_cli_contract_holds_out_of_process():
    """CI invokes this as a subprocess and gates on its exit code."""
    result = subprocess.run(
        [sys.executable, str(LINTER), str(REPO_ROOT)], capture_output=True, text=True
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_rejects_the_reverted_spelling(tmp_path, capsys):
    """Revert the fix and the guard must fail. This is the #13 observation."""
    _write(tmp_path, "requirements.txt", "krepis[flow_doctor]==0.38.0\n")
    assert _rc(tmp_path) == 1, "the guard PASSED on the spelling that caused I6963"
    err = capsys.readouterr().err
    assert "flow_doctor" in err
    assert "flow-doctor" in err, "the message must name the correct spelling"


def test_rejects_an_underscore_among_several_extras(tmp_path):
    """The live defect had the underscore mid-list, not alone in the bracket."""
    _write(
        tmp_path,
        "requirements.txt",
        "nousergon-lib[arcticdb,flow_doctor,contracts] @ git+https://example/x@v1\n",
    )
    assert _rc(tmp_path) == 1


def test_accepts_the_hyphenated_form(tmp_path):
    """The correct spelling on every pip version must not be flagged."""
    _write(
        tmp_path,
        "requirements.txt",
        "krepis[flow-doctor]==0.38.0\n"
        "nousergon-lib[arcticdb,flow-doctor,quant-xs,contracts] @ git+https://example/x@v1\n",
    )
    assert _rc(tmp_path) == 0


def test_ignores_commented_out_requirements(tmp_path):
    """A commented example must not fail the build."""
    _write(
        tmp_path,
        "requirements.txt",
        "# historical: krepis[flow_doctor] used to be written this way\n"
        "krepis[flow-doctor]==0.38.0\n",
    )
    assert _rc(tmp_path) == 0


def test_declaration_table_key_is_not_a_requester(tmp_path):
    """`flow_doctor = [...]` in optional-dependencies is a KEY, not a requester.

    setuptools normalises it on publish, so the key is legitimate either way —
    but the requester inside the VALUE is still a defect and must be caught.
    An earlier revision skipped the whole line and missed it.
    """
    _write(tmp_path, "requirements.txt", "krepis[flow-doctor]==0.38.0\n")
    _write(
        tmp_path,
        "pyproject.toml",
        '[project.optional-dependencies]\nflow_doctor = ["krepis[flow-doctor]"]\n',
    )
    assert _rc(tmp_path) == 0, "a declaration KEY must not be flagged"

    _write(
        tmp_path,
        "pyproject.toml",
        '[project.optional-dependencies]\nflow_doctor = ["krepis[flow_doctor]"]\n',
    )
    assert _rc(tmp_path) == 1, "the requester inside the VALUE must be flagged"


def test_scans_dockerfiles_including_nested(tmp_path):
    """Dockerfiles are requesters too, and they matter more, not less.

    A Docker base image pins its own pip, so a Dockerfile resolving correctly
    today breaks silently the day that pin moves back. The live defects in
    crucible-research / crucible-backtester / nousergon-data are all in
    Dockerfiles, several nested one directory down.
    """
    _write(tmp_path, "requirements.txt", "krepis[flow-doctor]==0.38.0\n")
    _write(
        tmp_path,
        "Dockerfile",
        'RUN pip install "nousergon-lib[arcticdb,flow_doctor,rag] @ git+https://e/x@v1"\n',
    )
    assert _rc(tmp_path) == 1, "a top-level Dockerfile requester must be caught"

    (tmp_path / "Dockerfile").unlink()
    _write(
        tmp_path,
        "lambda_health/Dockerfile",
        'RUN pip install "nousergon-lib[flow_doctor] @ git+https://e/x@v1"\n',
    )
    assert _rc(tmp_path) == 1, "a nested Dockerfile requester must be caught"


def test_dockerfile_variants_are_scanned(tmp_path):
    """`Dockerfile.alerts` is a real filename in crucible-research."""
    _write(tmp_path, "requirements.txt", "krepis[flow-doctor]==0.38.0\n")
    _write(
        tmp_path,
        "Dockerfile.alerts",
        'RUN pip install "nousergon-lib[flow_doctor] @ git+https://e/x@v1"\n',
    )
    assert _rc(tmp_path) == 1


def test_scanning_nothing_is_an_error_not_a_pass(tmp_path):
    """A checker that opened no files must not report clean.

    This fleet's failure registry is full of checks that reported success over
    an empty denominator; the guard must not join them.
    """
    assert _rc(tmp_path) == 2, "a check with nothing to check is not a passing check"


# ── The denominator (alpha-engine-config-I6963, second pass) ─────────────────
#
# The first revision globbed `requirements*.txt` at the repo ROOT and never
# opened workflow YAML. Both blind spots held a live defect on the day the guard
# shipped:
#
#   .github/workflows/deploy.yml      `pip install "krepis[flow_doctor]>=0.7.0"`
#   nousergon-data .../scheduled-groom-dispatcher/requirements.txt
#                                     `nousergon-lib[flow-doctor,github_app]`
#
# The checker reported "OK — 4 dependency file(s) scanned" over this repo while
# the first of those sat in file five. Detection blindness outranks the defects
# it hides (engagement-protocol-policy.md §5), so these pin the WALK, not the
# spelling.


def test_nested_requirements_files_are_scanned(tmp_path):
    """A lambda's own requirements.txt installs on the same resolver.

    The surviving instance of I6963 was in
    `infrastructure/lambdas/scheduled-groom-dispatcher/requirements.txt` — and
    it was a SECOND underscored extra (`github_app`), invisible to a sweep
    written around the one symbol that had already failed.
    """
    _write(tmp_path, "requirements.txt", "krepis[flow-doctor]==0.38.0\n")
    _write(
        tmp_path,
        "infrastructure/lambdas/scheduled-groom-dispatcher/requirements.txt",
        "nousergon-lib[flow-doctor,github_app] @ git+https://e/x@v1\n",
    )
    assert _rc(tmp_path) == 1, "a nested requirements file must be scanned"


def test_workflow_install_steps_are_scanned(tmp_path):
    """CI installs resolve on the runner's pip and are requesters like any other."""
    _write(tmp_path, "requirements.txt", "krepis[flow-doctor]==0.38.0\n")
    _write(
        tmp_path,
        ".github/workflows/deploy.yml",
        "jobs:\n  deploy:\n    steps:\n"
        '      - run: pip install "krepis[flow_doctor]>=0.7.0"\n',
    )
    assert _rc(tmp_path) == 1, "a workflow install step must be scanned"


def test_workflow_lines_that_are_not_installs_are_left_alone(tmp_path):
    """A checker that flags matrix entries and expressions gets switched off."""
    _write(tmp_path, "requirements.txt", "krepis[flow-doctor]==0.38.0\n")
    _write(
        tmp_path,
        ".github/workflows/ci.yml",
        "on:\n  push:\n"
        "jobs:\n  test:\n"
        "    steps:\n"
        "      - run: echo ${{ matrix.py_ver }}\n"
        "      - run: pytest tests/test_a[case_one]\n"
        '      - run: pip install "krepis[flow-doctor]>=0.7.0"\n',
    )
    assert _rc(tmp_path) == 0


def test_vendored_trees_are_not_scanned(tmp_path):
    """Third-party requirements this repo cannot fix must not fail its CI.

    The prune list is the only thing standing between a recursive walk and a
    permanently red check on a developer machine with a .venv in the tree.
    """
    _write(tmp_path, "requirements.txt", "krepis[flow-doctor]==0.38.0\n")
    _write(
        tmp_path,
        ".venv/lib/python3.12/site-packages/somepkg/requirements.txt",
        "somepkg[bad_extra]==1.0\n",
    )
    _write(tmp_path, "node_modules/x/requirements.txt", "y[bad_extra]==1.0\n")
    assert _rc(tmp_path) == 0


def test_the_walk_is_wider_than_the_repo_root(tmp_path):
    """Counts the denominator directly, so a silent narrowing is visible."""
    _write(tmp_path, "requirements.txt", "krepis[flow-doctor]==0.38.0\n")
    _write(tmp_path, "infrastructure/lambdas/a/requirements.txt", "requests==2.0\n")
    _write(tmp_path, "sub/pyproject.toml", "[project]\nname='x'\n")
    _write(tmp_path, ".github/workflows/ci.yml", "jobs: {}\n")
    files = set()
    for globs in (
        lint_extras._REQUIREMENTS_GLOBS,
        lint_extras._INLINE_GLOBS,
        lint_extras._WORKFLOW_GLOBS,
    ):
        files.update(lint_extras._walk(tmp_path, globs))
    assert len(files) == 4, sorted(str(f) for f in files)
