"""Every variable the bootstrap reads must be one the launcher supplies.

The failure this pins (2026-08-11, `ne-weekly-freshness-pipeline` execution
`watch-rerun-2026-08-10-7`)::

    Using: Python 3.12.12
    fatal: repository '' does not exist

``bootstrap_spot()`` sends a **single-quoted** heredoc, so its body is literal
on the spot instance — nothing in it is expanded launcher-side. Variables
therefore reach the instance only through the ``_spot_env_export`` prefix. The
body read ``${REPO_URL}``; the prefix exported ``S3_STAGING``, ``BRANCH`` and
``ALPHA_ENGINE_EXPERIMENT_ID`` but not ``REPO_URL``, so the clone ran against
the empty string.

The comment directly above the function claimed otherwise —

    the launcher-side export prefix sets S3_STAGING/BRANCH/REPO_URL

— which is exactly why nobody caught it by reading: the documentation and the
code disagreed, and the code won.

This was the THIRD defect in this bootstrap in one day, each revealed by fixing
the one ahead of it (#461 watchdog hang, #462 missing python3.12, this). A step
that has never run to completion hides its next failure behind its current one,
so the sequence is expected rather than surprising — and the useful response is
a check on the whole class rather than one more literal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_COMMON = Path(__file__).resolve().parents[1] / "infrastructure" / "_spot_common.sh"
_TEXT = _COMMON.read_text(encoding="utf-8")

# Set by the SSM/Lambda runtime or by the heredoc's own `export` line, so the
# launcher does not have to supply them.
_AMBIENT = {
    "HOME", "PATH", "PWD", "USER", "SHELL", "TMPDIR",
    "AWS_REGION", "AWS_DEFAULT_REGION", "XDG_CACHE_HOME",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_EXECUTION_ENV",
}


def _bootstrap_parts() -> tuple[str, str]:
    """(export_prefix, heredoc_body) for bootstrap_spot()."""
    fn = re.search(r"bootstrap_spot\(\)\s*\{(.*?)\n\}", _TEXT, re.S)
    assert fn, "bootstrap_spot() not found in _spot_common.sh"
    block = fn.group(1)

    prefix = re.search(r'_spot_env_export="export ([^"]*)"', block)
    assert prefix, "the launcher-side export prefix is gone — every variable the heredoc reads is now unset"

    body = re.search(r"<<'BOOTSTRAP'\n(.*?)\nBOOTSTRAP", block, re.S)
    assert body, "the BOOTSTRAP heredoc is gone or is no longer single-quoted"
    return prefix.group(1), body.group(1)


def _exported_names(prefix: str) -> set[str]:
    return set(re.findall(r"(\w+)=", prefix))


def _read_names(body: str) -> set[str]:
    """Variables the body READS without a `${X:-default}` fallback.

    A defaulted read is safe by construction — the point of this check is the
    bare `${X}` that silently becomes the empty string.
    """
    names = set()
    for match in re.finditer(r"\$\{(\w+)([^}]*)\}", body):
        name, tail = match.group(1), match.group(2)
        if tail.startswith(":-") or tail.startswith("-"):
            continue  # has a default
        names.add(name)
    for match in re.finditer(r"\$(\w+)\b", body):
        names.add(match.group(1))
    return names


def _assigned_in_body(body: str) -> set[str]:
    return set(re.findall(r"^\s*(?:export\s+)?(\w+)=", body, re.M)) | set(
        re.findall(r"export ([\w= ]+)", body)
    )


def test_the_heredoc_is_single_quoted():
    """The premise of this whole check: nothing expands launcher-side."""
    _bootstrap_parts()  # asserts the <<'BOOTSTRAP' form


@pytest.mark.parametrize("name", sorted(_read_names(_bootstrap_parts()[1])))
def test_every_variable_the_bootstrap_reads_is_supplied(name):
    prefix, body = _bootstrap_parts()
    if name in _AMBIENT:
        return

    exported = _exported_names(prefix)
    assigned = set()
    for line in body.splitlines():
        assigned |= set(re.findall(r"^\s*(?:export\s+)?(\w+)=", line))
        assigned |= set(re.findall(r"export\s+((?:\w+=\S*\s*)+)", line) and
                        re.findall(r"(\w+)=", line) or [])

    assert name in exported or name in assigned, (
        f"the bootstrap heredoc reads ${{{name}}} but the launcher never exports "
        f"it and the body never assigns it. The heredoc is single-quoted, so it "
        f"resolves to the EMPTY STRING on the instance — that is how the clone "
        f"ran as `git clone ... ''` and died with "
        f"`fatal: repository '' does not exist` (2026-08-11). Add {name} to "
        f"_spot_env_export, or give the read a ${{{name}:-default}}."
    )


def test_repo_url_specifically_is_exported():
    """Anchored assertion on the variable that broke."""
    prefix, _ = _bootstrap_parts()
    assert "REPO_URL=" in prefix, (
        "REPO_URL is not in the launcher-side export prefix; the clone will run "
        "against the empty string"
    )


def test_the_note_above_the_function_matches_what_is_exported():
    """The stale comment is what made this survive review.

    It listed REPO_URL as exported while the code did not export it. A comment
    that names the mechanism must track it.
    """
    note = re.search(r"export prefix sets ([^\n]*)", _TEXT)
    if not note:
        pytest.skip("the explanatory NOTE was removed; nothing to keep honest")
    prefix, _ = _bootstrap_parts()
    exported = _exported_names(prefix)
    for claimed in re.findall(r"[A-Z][A-Z0-9_]+", note.group(1)):
        assert claimed in exported, (
            f"the NOTE above bootstrap_spot() claims {claimed} is exported, but "
            f"the prefix exports {sorted(exported)}. The code won that "
            f"disagreement once already (2026-08-11); fix whichever is wrong."
        )
