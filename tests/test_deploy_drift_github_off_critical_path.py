"""``alpha-engine-config-I7927``, completing what ``crucible-predictor#538``
started: **no field the preopen halting gate reads may require GitHub.**

WHY #538 WAS NOT THE WHOLE THING
--------------------------------
#538 moved ``sf_drift``'s expected definition to the deploy's own S3 record, so
that verdict no longer needs ``api.github.com``. ``cf_drift`` was left as it
was — a comparison of the CloudFormation stack's ``git-sha`` tag against
``main`` HEAD. And ``DeployDriftGate`` in ``nousergon-data
infrastructure/step_function_daily.json`` carries::

    Not IsPresent($.drift_result.Payload.cf_drift) -> HandleFailure

``cf_drift`` is omitted when ``upstream is None`` — the same 2026-08-21 shape
that cost a trading session (``I7924``). So after #538 a GitHub outage at
05:15 PT still halted the trading day, one Choice branch further down, however
cleanly ``sf_drift`` had been obtained. Both fields had to become answerable
in-region, or neither did.

``test_a_total_github_outage_halts_nothing`` is the assertion that says the
issue is actually closed, and it fails on ``origin/main`` as of #538.

WHERE THE IN-REGION EXPECTED SHA COMES FROM
-------------------------------------------
No new artifact and no new writer. ``deploy-infrastructure.sh`` stamps
``[git:<sha>] `` into the ``Comment`` of the very bytes it uploads, so the S3
object #538 already reads *says which deploy wrote it*. That same run tags the
CloudFormation stack with the same SHA. Reading the stamp back out is the whole
mechanism.

IT ALSO CLOSES DELIVERABLE 4
----------------------------
The S3 copy became load-bearing for a verdict that halts trading, and nothing
asserted it describes the deploy that is actually live — upload happens in the
deploy's step 2, ``update-state-machine`` in step 3, and
``check-definition-drift.py`` exists because a third party can write that key.
Both artifacts carry the SHA of the deploy that wrote them, so they can be asked
whether they agree. A disagreement is a *measured* incoherence, not an
unmeasurable state, so it degrades and pages rather than withdrawing the
verdict — withdrawing it would turn a rare inconsistency into a halted session,
which is the harm ``I7799`` exists to prevent.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import inference.deploy_drift as dd

_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
_OTHER = "b" * 40
_DEFINITION = {
    "Comment": "preopen",
    "StartAt": "DeployDriftCheck",
    "States": {"DeployDriftCheck": {"Type": "Succeed"}},
}


def _stamped(sha: str, doc: dict | None = None) -> dict:
    out = json.loads(json.dumps(doc if doc is not None else _DEFINITION))
    out["Comment"] = f"[git:{sha}] {out.get('Comment', '')}".rstrip()
    return out


def _run(*, live_sha=_SHA, s3_definition=..., stack_sha=_SHA, upstream=None,
         live_definition=...):
    """Drive ``check_deploy_drift`` with every I/O seam pinned.

    ``upstream=None`` is the default on purpose: these tests are about what the
    probe can still answer when GitHub answers nothing at all.
    """
    if s3_definition is ...:
        s3_definition = _stamped(_SHA)
    if live_definition is ...:
        live_definition = _stamped(_SHA)
    with patch.object(dd, "_read_sf_comment",
                      return_value=f"[git:{live_sha}] preopen" if live_sha else "no stamp"), \
         patch.object(dd, "_read_stack_tag", return_value=stack_sha), \
         patch.object(dd, "_fetch_origin_main_sha", return_value=upstream), \
         patch.object(dd, "_read_sf_definition", return_value=live_definition), \
         patch.object(dd, "_fetch_s3_definition", return_value=s3_definition), \
         patch.object(dd, "_fetch_repo_definition", return_value=None) as gh:
        result = dd.check_deploy_drift(region="us-east-1", account_id="711398986525")
    return result, gh


# ── The issue's actual close condition ───────────────────────────────────────

def test_a_total_github_outage_halts_nothing():
    """The 2026-08-21 input, one branch further down than #538 covered.

    ``DeployDriftGate`` halts on an absent ``sf_drift`` OR an absent
    ``cf_drift``. Both must be present, and both must be false, with GitHub
    answering nothing.
    """
    result, gh = _run(upstream=None)
    assert result["sf_drift"] is False
    assert result["cf_drift"] is False, (
        "cf_drift was omitted with GitHub down — DeployDriftGate's "
        "`Not IsPresent(cf_drift) -> HandleFailure` branch halts the trading "
        "day, so the dependency was never actually off the critical path"
    )
    assert result["has_drift"] is False
    assert result["sf_definition_source"] == "s3"
    assert result["cf_drift_source"] == "s3"
    assert result["upstream_sha"] is None
    gh.assert_not_called()


def test_cf_drift_is_measured_against_the_deploy_not_against_main():
    result, _ = _run(stack_sha=_OTHER, upstream=None)
    assert result["cf_drift"] is True
    assert result["cf_drift_source"] == "s3"
    assert result["cf_drift_reason"] == "sha_mismatch"


def test_s3_is_preferred_for_cf_drift_even_when_github_is_available():
    """If GitHub stayed on the halting path whenever it happened to be up, the
    dependency would still be there and would still fail on the one morning it
    mattered. Same argument #538 made for ``sf_drift``."""
    result, _ = _run(upstream="c" * 40, stack_sha=_SHA)
    assert result["cf_drift_source"] == "s3"
    assert result["cf_drift"] is False
    # main having moved is not lost — it is said by deploy_stamp_stale, which
    # degrades rather than halts.
    assert result["deploy_stamp_stale"] is True


def test_a_terminal_stack_still_fires_cf_drift():
    """The 2026-04-20 ROLLBACK_COMPLETE class is untouched by any of this."""
    sentinel = dd.StackStateError(reason="stack_in_terminal_state",
                                  detail="x is in ROLLBACK_COMPLETE",
                                  stack_status="ROLLBACK_COMPLETE")
    result, _ = _run(stack_sha=sentinel, upstream=None)
    assert result["cf_drift"] is True
    assert result["cf_drift_reason"] == "stack_in_terminal_state"
    assert result["cf_drift_source"] is None


# ── Fail-closed is not relaxed ───────────────────────────────────────────────

def test_no_expected_sha_from_either_source_still_omits_cf_drift():
    """S3 unreadable AND GitHub silent: nothing was measured, so the key is
    ABSENT and the SF's IsPresent guard halts — config-I7048, Brian's
    2026-08-09 ruling config#6615, unchanged."""
    result, _ = _run(s3_definition=None, upstream=None)
    assert "cf_drift" not in result
    assert "sf_drift" not in result
    assert result["cf_drift_reason"] == "fetch_failed"


def test_an_unstamped_s3_copy_does_not_grant_a_cf_verdict():
    """A legacy or hand-written S3 object with no ``[git:]`` Comment carries no
    expected SHA. That is not a pass — it falls back, and with GitHub silent it
    omits."""
    result, _ = _run(s3_definition={"Comment": "no stamp", "StartAt": "X",
                                    "States": {"X": {"Type": "Succeed"}}},
                     upstream=None)
    assert result["s3_deploy_sha"] is None
    assert "cf_drift" not in result


def test_a_legacy_untagged_stack_is_still_a_confirmed_non_drift():
    result, _ = _run(stack_sha=None, upstream=None)
    assert result["cf_drift"] is False
    assert result["cf_drift_reason"] == "no_git_sha_tag_legacy"
    assert result["cf_drift_source"] is None


def test_github_fallback_survives_when_s3_is_unreadable():
    result, _ = _run(s3_definition=None, upstream=_SHA, stack_sha=_SHA)
    assert result["cf_drift"] is False
    assert result["cf_drift_source"] == "github"


# ── Deliverable 4: the S3 expectation must describe the live deploy ──────────

def test_an_s3_copy_from_another_deploy_degrades_and_pages_without_halting():
    result, _ = _run(live_sha=_SHA, s3_definition=_stamped(_OTHER),
                     live_definition=_stamped(_SHA), upstream=None)
    assert result["deploy_stamp_stale"] is True
    assert result["deploy_stamp_stale_reason"] == "s3_copy_stamp_mismatch"
    # It DEGRADES. Withdrawing the verdict here would turn a rare inconsistency
    # into a halted trading session — the harm I7799 exists to prevent.
    assert "sf_drift" in result
    assert result["s3_deploy_sha"] == _OTHER


def test_matching_stamps_leave_the_stale_reason_alone():
    result, _ = _run(upstream=_SHA)
    assert result["deploy_stamp_stale"] is False
    assert result["deploy_stamp_stale_reason"] == "in_sync"


def test_the_incoherence_signal_never_changes_the_halting_verdict():
    """Whatever the stamps say, ``sf_drift`` is the body comparison and nothing
    else. Strictly additive — this cannot make the gate halt where it did not."""
    edited = _stamped(_SHA)
    edited["States"]["Injected"] = {"Type": "Succeed"}
    drifted, _ = _run(live_definition=edited, s3_definition=_stamped(_OTHER),
                      upstream=None)
    clean, _ = _run(s3_definition=_stamped(_OTHER), upstream=None)
    assert drifted["sf_drift"] is True
    assert clean["sf_drift"] is False


# ── The weekly pipeline is untouched ─────────────────────────────────────────

def test_the_weekly_pipeline_keeps_stamp_semantics():
    """No market-open deadline there — a lost weekly run costs a rerun, not a
    session — so neither the S3 definition read nor the in-region cf_drift
    reference applies."""
    with patch.object(dd, "_read_sf_comment", return_value=f"[git:{_SHA}] weekly"), \
         patch.object(dd, "_read_stack_tag", return_value=_SHA), \
         patch.object(dd, "_fetch_origin_main_sha", return_value=_OTHER), \
         patch.object(dd, "_fetch_s3_definition") as s3:
        result = dd.check_deploy_drift(
            region="us-east-1", account_id="711398986525",
            sf_name="ne-weekly-freshness-pipeline",
        )
    s3.assert_not_called()
    assert result["sf_drift"] is True
    assert result["cf_drift_source"] == "github"


@pytest.mark.parametrize("field", ["cf_drift_source", "s3_deploy_sha"])
def test_the_new_fields_are_always_present(field):
    """A field emitted only sometimes is a field nobody can key on. Both are
    diagnostic, so unlike the drift booleans they are present in every
    polarity — including on the stamp-only weekly path."""
    result, _ = _run(s3_definition=None, upstream=None)
    assert field in result
