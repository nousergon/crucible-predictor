"""The IAM drift check may not run on the ``pull_request`` path.

Brian ruling 2026-08-10: a pull request whose only failing check is a drift
check must read GREEN.

A drift check compares codified source against LIVE AWS, so its answer is a
property of the account at the moment it runs, never of the branch under
review. ``--pr-diff-aware`` already excused half of that — drift on roles the
PR itself changes, which is red by construction because the codified JSON
deliberately differs from live until ``apply.sh`` runs post-merge
(alpha-engine-config#3492). What it left is drift on roles the PR does NOT
touch: real, but caused by someone else at some other time and unclearable by
this author. Reddening a PR for it is the gate-only-the-blocked-actor-can-clear
trap, and it teaches the reader to merge past a red drift check — which is how
a real finding goes unread (nous-ergon-ops-I563, ``alarm-red-by-construction``).

Nothing is lost: the comparison runs on ``push: [main]``, on the daily 09:35
sweep, and on ``workflow_dispatch``, in full hard-fail mode.

Deliberately parsed as text rather than with PyYAML: this repo's
``requirements.txt`` does not carry pyyaml, and a guard that silently skips
when a dependency is absent is not a guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
IAM_DRIFT = WORKFLOWS / "iam-drift-check.yml"

# A workflow naming any of these reads live AWS and compares it against
# codified source.
LIVE_DRIFT_INVOCATIONS = ("check-drift.py", "check-definition-drift.py")


def _trigger_block(text: str) -> str:
    """The `on:` mapping, as raw text, up to the next top-level key."""
    m = re.search(r"^on:\n((?:[ \t]+.*\n|\n)*)", text, re.MULTILINE)
    assert m, "no `on:` block found"
    return m.group(1)


def test_iam_drift_check_has_no_pull_request_trigger():
    text = IAM_DRIFT.read_text(encoding="utf-8")
    triggers = _trigger_block(text)
    assert not re.search(r"^\s{2}pull_request:", triggers, re.MULTILINE), (
        "iam-drift-check.yml grades live AWS against codified source. On a PR "
        "it can only report drift the author did not cause and cannot clear "
        "(Brian ruling 2026-08-10). Keep it on push:[main] + schedule."
    )
    # Removing it from PRs is only correct because it still runs elsewhere.
    assert re.search(r"^\s{2}push:", triggers, re.MULTILINE), "no push trigger"
    assert re.search(r"^\s{4}branches: \[main\]", triggers, re.MULTILINE)
    assert re.search(r"^\s{2}schedule:", triggers, re.MULTILINE), "no daily sweep"
    assert "check-drift.py" in text, "the checker invocation is gone entirely"


@pytest.mark.parametrize(
    "path", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name
)
def test_no_workflow_runs_a_live_drift_check_on_prs(path: Path):
    """The class guard — a future workflow cannot reintroduce this quietly."""
    text = path.read_text(encoding="utf-8")
    if not re.search(r"^\s{2}pull_request:", _trigger_block(text), re.MULTILINE):
        return
    hit = next((s for s in LIVE_DRIFT_INVOCATIONS if s in text), None)
    if hit is None:
        return
    # A step explicitly excused from the PR path is the same fix applied per
    # step instead of per workflow.
    if re.search(r"github\.event_name\s*!=\s*'pull_request'", text):
        return
    pytest.fail(
        f"{path.name} runs {hit} on the pull_request path. A live-AWS drift "
        "comparison grades the account, not the diff, and reddens PRs their "
        "authors cannot fix (Brian ruling 2026-08-10). Move it to push:[main] "
        "+ schedule, or guard the step with "
        "`if: github.event_name != 'pull_request'`."
    )
