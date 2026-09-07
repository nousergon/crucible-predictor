"""alert-message-lint's warn-only step must survive a CRASH, not just a finding.

alpha-engine-config-I10147. Measured 2026-09-07: metron run 33718023593
crashed this step with `ModuleNotFoundError: No module named 'yaml'` under
`set -euo pipefail`, and the crash failed the CI job -- indistinguishable from
an enforced finding, and it blocked seven metron dependency PRs for four days.
Separately, `alert_message_lint.py::main` (nousergon-data) returns 1 on its
own UNMEASURED paths (bad --repo-root, a GuardError while scanning) even
under --warn-only -- only the *findings* path honors the flag. Either way,
warn-only must mean warn-only: the step must capture the lint invocation's
exit code and turn ANY nonzero exit into a `::warning` annotation, then exit 0
unconditionally -- recording the crash, never silencing it.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "alert-class-pr-guard.yml"

MARKER = "- name: A PR-added alert message must be actionable (warn-only)"


def _lint_step_run_block() -> str:
    text = WORKFLOW.read_text()
    assert MARKER in text, f"{WORKFLOW} lost its alert-message-lint step"
    return text[text.index(MARKER):]


def test_the_workflow_still_carries_the_lint_step():
    assert WORKFLOW.is_file(), f"{WORKFLOW} missing"
    assert MARKER in WORKFLOW.read_text()


def test_a_nonzero_lint_exit_cannot_propagate_through_set_e():
    block = _lint_step_run_block()
    assert "set +e" in block, (
        f"{WORKFLOW.name}'s lint step invokes the script without disabling "
        "`set -e` first -- a crash or an internal UNMEASURED exit(1) fails "
        "the job, defeating warn-only (alpha-engine-config-I10147)"
    )
    assert "rc=$?" in block, (
        f"{WORKFLOW.name}'s lint step does not capture the invocation's exit "
        "code -- a nonzero exit cannot be reported without it"
    )
    assert block.rstrip().splitlines()[-1].strip() == "exit 0", (
        f"{WORKFLOW.name}'s lint step does not end with an unconditional "
        "`exit 0` -- warn-only must never fail the job"
    )


def test_a_crash_is_recorded_not_silenced():
    block = _lint_step_run_block()
    assert 'if [ "$rc" -ne 0 ]' in block, (
        f"{WORKFLOW.name}'s lint step does not branch on a nonzero exit code"
    )
    after = block.split('if [ "$rc" -ne 0 ]', 1)[1]
    assert "::warning" in after, (
        f"{WORKFLOW.name}'s lint step swallows a nonzero exit silently -- a "
        "crash must still emit a ::warning annotation naming it, per "
        "alpha-engine-config-I10147"
    )
