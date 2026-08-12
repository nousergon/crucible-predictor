"""The underscored-extras guard, and proof it fails when the fix is reverted.

`overseer-policy.md` #13: a guard is not a guard until it has been observed
failing. The reverted-tree case below is that observation — it reconstructs the
exact spelling that took the weekly pipeline down (config#6963) and asserts the
checker rejects it.

Root cause being guarded: pip <23.3 does not normalise `_` to `-` in a
*requested* extra, while setuptools always normalises the *declared* one. So
`krepis[flow_doctor]` matches nothing against `Provides-Extra: flow-doctor`, and
pip reports that as a WARNING on a SUCCESSFUL exit. Measured 2026-08-12:
pip 23.2.1 resolves 18 packages with flow-doctor absent; 23.3.2 / 24.0 / 25.0
resolve 30 with it present. Amazon Linux 2023 ships pip 23.2.1.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LINTER = REPO_ROOT / "scripts" / "lint_extras.py"


def _run(target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LINTER), str(target)],
        capture_output=True,
        text=True,
    )


def test_this_repo_is_clean():
    """The real tree passes — the fix is in place and stays in place."""
    result = _run(REPO_ROOT)
    assert result.returncode == 0, (
        f"lint_extras rejects this repo's own dependency files:\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_rejects_the_reverted_spelling(tmp_path):
    """Revert the fix and the guard must fail. This is the #13 observation."""
    (tmp_path / "requirements.txt").write_text(
        "krepis[flow_doctor]==0.38.0\n", encoding="utf-8"
    )
    result = _run(tmp_path)
    assert result.returncode == 1, (
        "the guard PASSED on the exact spelling that caused config#6963 — "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "flow_doctor" in result.stderr
    assert "flow-doctor" in result.stderr, "the message must name the correct spelling"


def test_rejects_an_underscore_among_several_extras(tmp_path):
    """The live defect had the underscore mid-list, not alone in the bracket."""
    (tmp_path / "requirements.txt").write_text(
        "nousergon-lib[arcticdb,flow_doctor,contracts] @ git+https://example/x@v1\n",
        encoding="utf-8",
    )
    assert _run(tmp_path).returncode == 1


def test_accepts_the_hyphenated_form(tmp_path):
    """The correct spelling on every pip version must not be flagged."""
    (tmp_path / "requirements.txt").write_text(
        "krepis[flow-doctor]==0.38.0\n"
        "nousergon-lib[arcticdb,flow-doctor,quant-xs,contracts] @ git+https://example/x@v1\n",
        encoding="utf-8",
    )
    assert _run(tmp_path).returncode == 0


def test_ignores_commented_out_requirements(tmp_path):
    """A commented example must not fail the build."""
    (tmp_path / "requirements.txt").write_text(
        "# historical note: krepis[flow_doctor] used to be written this way\n"
        "krepis[flow-doctor]==0.38.0\n",
        encoding="utf-8",
    )
    assert _run(tmp_path).returncode == 0


def test_declaration_table_key_is_not_a_requester(tmp_path):
    """`flow_doctor = [...]` in optional-dependencies is a KEY.

    setuptools normalises it on publish, so it is legitimate either way. Only
    the requester inside the value is a defect — asserted here to be caught.
    """
    (tmp_path / "requirements.txt").write_text("krepis[flow-doctor]==0.38.0\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project.optional-dependencies]\n"
        'flow_doctor = ["krepis[flow-doctor]"]\n',
        encoding="utf-8",
    )
    assert _run(tmp_path).returncode == 0, "a declaration KEY must not be flagged"

    (tmp_path / "pyproject.toml").write_text(
        "[project.optional-dependencies]\n"
        'flow_doctor = ["krepis[flow_doctor]"]\n',
        encoding="utf-8",
    )
    assert _run(tmp_path).returncode == 1, "the requester inside the VALUE must be flagged"


def test_scanning_nothing_is_an_error_not_a_pass(tmp_path):
    """A checker that opened no files must not report clean.

    This fleet's failure registry is full of checks that reported success over
    an empty denominator; the guard must not join them.
    """
    result = _run(tmp_path)
    assert result.returncode == 2, (
        f"empty tree reported rc={result.returncode}, expected 2 — "
        "a check with nothing to check is not a passing check"
    )


def test_scans_dockerfiles_including_nested(tmp_path):
    """Dockerfiles are requesters too, and they matter more, not less.

    A Docker base image pins its own pip, so a Dockerfile resolving correctly
    today breaks silently the day that pin moves back. The live defects in
    crucible-research / crucible-backtester / nousergon-data are all in
    Dockerfiles, several of them nested one directory down (lambda_health/,
    lambda_concordance/), so the nested case is asserted explicitly.
    """
    (tmp_path / "requirements.txt").write_text("krepis[flow-doctor]==0.38.0\n", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text(
        'RUN pip install --no-cache-dir "nousergon-lib[arcticdb,flow_doctor,rag] '
        '@ git+https://example/x@v1"\n',
        encoding="utf-8",
    )
    assert _run(tmp_path).returncode == 1, "a top-level Dockerfile requester must be caught"

    (tmp_path / "Dockerfile").unlink()
    nested = tmp_path / "lambda_health"
    nested.mkdir()
    (nested / "Dockerfile").write_text(
        'RUN pip install "nousergon-lib[flow_doctor] @ git+https://example/x@v1"\n',
        encoding="utf-8",
    )
    assert _run(tmp_path).returncode == 1, "a nested Dockerfile requester must be caught"


def test_dockerfile_variants_are_scanned(tmp_path):
    """`Dockerfile.alerts` is a real filename in crucible-research."""
    (tmp_path / "requirements.txt").write_text("krepis[flow-doctor]==0.38.0\n", encoding="utf-8")
    (tmp_path / "Dockerfile.alerts").write_text(
        'RUN pip install "nousergon-lib[flow_doctor] @ git+https://example/x@v1"\n',
        encoding="utf-8",
    )
    assert _run(tmp_path).returncode == 1
