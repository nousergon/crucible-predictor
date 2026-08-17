"""A failure path must not delete the evidence its own message points at.

**The 2026-08-15 weekly-SF failure (alpha-engine-config-I7396 -> I7442).** The
`PredictorBacktest` stage died and printed:

    ERROR: SSM step 'predictor-backtest' terminal status=Failed ...
      -- full remote log: s3://alpha-engine-research/tmp/spot_predictor-backtest/
        20260815T123311Z-i-08a4371deec28ef07/ssm-output/

Four lines later, the same exit path printed *"Instance terminated; S3 staging
cleaned."* The prefix the error named was **empty** by the time anyone read it,
and so was its parent.

That copy is not redundant. SSM's ``GetCommandInvocation`` returns only the
**first** 24 KB of stdout, so on any long stage the tail -- which is where a
traceback lives -- exists nowhere else. A message pointing at evidence the same
exit path just removed is worse than no message.

**What changed at I7442 in this repo.** Both launchers' unguarded
``aws s3 rm "$_S3_STAGING"``/``aws s3 rm "$S3_STAGING"`` teardown lines are
gone. Teardown now runs through ``krepis.spot_evidence teardown`` via the
shared ``spot_common_teardown_staging()`` helper in ``_spot_common.sh``
(mirroring crucible-backtester's #675 fix, per policy-shared-code), which
copies the staging prefix to ``_spot_evidence/`` and deletes staging only if
that copy succeeded -- so the ordering is a property of that helper's call
graph rather than of a branch repeated at every call site, and it holds for
every launcher in this repo at once. ``spot_train.sh`` keeps its own
``cleanup()`` (it is the retained pre-split rollback path, config-I4442/I4497)
but bridges into the SAME helper rather than restating the retain-before-
delete logic inline.

These tests therefore pin the CLASS property: no launcher in this repo may
carry an unguarded staging delete, and the teardown must degrade to retention
rather than to deletion.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parent.parent / "infrastructure"
COMMON = INFRA / "_spot_common.sh"
SPOT_TRAIN = INFRA / "spot_train.sh"


@pytest.fixture(scope="module")
def body() -> str:
    assert COMMON.is_file(), f"{COMMON} missing"
    return COMMON.read_text(encoding="utf-8")


def _fn_body(source: str, name: str) -> str:
    """Return a shell function's text, brace-matched."""
    marker = f"\n{name}() {{"
    start = source.index(marker) + 1
    depth = 0
    for idx in range(start, len(source)):
        if source[idx] == "{":
            depth += 1
        elif source[idx] == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise AssertionError(f"unterminated function {name}")


class TestNoLauncherDeletesItsOwnStaging:
    """The class guard. Fixing one call site of a systemic defect is not a fix."""

    def test_no_shell_script_in_infrastructure_removes_S3_STAGING(self):
        offenders = []
        for path in sorted(INFRA.glob("*.sh")):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "aws s3 rm" in stripped and "S3_STAGING" in stripped:
                    offenders.append(f"{path.name}:{n}: {stripped}")
        assert not offenders, (
            "an unguarded staging delete is back -- this is the "
            "alpha-engine-config-I7442 defect, and it destroys the only "
            "un-truncated copy of a failure's output:\n  "
            + "\n  ".join(offenders)
        )

    def test_every_launcher_that_stages_also_tears_down_through_the_chokepoint(self):
        for path in sorted(INFRA.glob("spot_*.sh")):
            text = path.read_text(encoding="utf-8")
            if "S3_STAGING=" not in text:
                continue  # sources _spot_common.sh and uses _S3_STAGING, covered below
            assert "spot_common_teardown_staging" in text, (
                f"{path.name} provisions its own staging prefix but never "
                "hands it to the shared teardown helper"
            )


class TestTeardownGoesThroughTheChokepoint:
    def test_common_cleanup_calls_the_shared_teardown(self, body):
        cleanup_fn = _fn_body(body, "cleanup")
        assert 'spot_common_teardown_staging "$exit_code"' in cleanup_fn

    def test_spot_train_cleanup_bridges_into_the_same_helper(self):
        # spot_train.sh keeps its own cleanup() (the retained pre-split
        # rollback path) but must not reimplement the retain-before-delete
        # logic -- it bridges its un-prefixed S3_STAGING into the shared
        # helper defined once in _spot_common.sh.
        source = SPOT_TRAIN.read_text(encoding="utf-8")
        cleanup_fn = _fn_body(source, "cleanup")
        assert "spot_common_teardown_staging" in cleanup_fn
        assert '_S3_STAGING="$S3_STAGING"' in cleanup_fn, (
            "spot_train.sh's cleanup() must bridge its own S3_STAGING into "
            "the shared helper's _S3_STAGING, the same bridge pattern this "
            "file already uses at its bootstrap_spot() call site"
        )

    def test_the_teardown_helper_invokes_krepis(self, body):
        fn = _fn_body(body, "spot_common_teardown_staging")
        assert "krepis.spot_evidence teardown" in fn
        assert "--exit-code" in fn, (
            "the workload's exit status is what decides whether evidence is "
            "preserved; without it every run looks like a success"
        )
        assert "--staging" in fn and "--slug" in fn

    def test_an_unavailable_chokepoint_degrades_to_RETENTION_not_deletion(self, body):
        """The merge-order safety property, and the fail-safe direction.

        A box whose krepis pin predates `spot_evidence` must keep the evidence,
        never fall back to the delete this whole change exists to remove.
        """
        fn = _fn_body(body, "spot_common_teardown_staging")
        assert "RETAINED" in fn
        code = "\n".join(
            line for line in fn.splitlines() if not line.strip().startswith("#")
        )
        assert "aws s3 rm" not in code

    def test_the_teardown_never_aborts_the_exit_path(self, body):
        fn = _fn_body(body, "spot_common_teardown_staging")
        # Every path through the helper (no staging, success, and the
        # unavailable-chokepoint fallback) must `return 0` -- a janitor that
        # can change the trap's exit status masks the workload's own failure.
        assert fn.count("return 0") >= 3, (
            "spot_common_teardown_staging must return 0 on every path so it "
            f"cannot change cleanup()'s exit status; found body:\n{fn}"
        )


class TestTheResourceLimitFlagWaitsForThePinBump:
    """`--resource-limit` is deliberately ABSENT, and that is load-bearing.

    It is a NEW `krepis.ssm_dispatcher` flag (krepis-PR161). `$LIB_PYTHON` is
    the dispatch box's venv, pinned to a krepis release that predates it, so
    argparse would reject the unknown flag and EVERY SSM step would fail on
    merge — in exactly the window before the pin bump. `krepis.spot_evidence`
    degrades safely when absent (the teardown retains); this flag has no safe
    degradation at all.

    It lands with the pin bump, `alpha-engine-config-I7556`. Until then this
    test is what stops it being reintroduced by someone reading
    sf-pipeline-policy §3 obligation 3 and adding the obvious line.
    """

    def test_no_launcher_passes_the_flag_yet(self):
        offenders = []
        for path in sorted(INFRA.glob("*.sh")):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip().startswith("#"):
                    continue
                if "--resource-limit" in line:
                    offenders.append(f"{path.name}:{n}")
        assert not offenders, (
            "--resource-limit is passed to krepis.ssm_dispatcher before the "
            "dispatch box's krepis pin ships it (alpha-engine-config-I7556). "
            "An unknown argparse flag fails EVERY SSM step: "
            + ", ".join(offenders)
        )
