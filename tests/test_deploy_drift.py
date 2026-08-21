"""Unit tests for inference.deploy_drift — SF + CF drift probe.

Exercises the pure-Python compare logic + the boto3-call surface with
stubs. The GitHub-fetch helper (``_fetch_origin_main_sha``) is owned
by alpha-engine-lib and tested there; this module re-imports it so
``patch.object(dd, "_fetch_origin_main_sha", ...)`` keeps mocking the
same symbol the production code calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import inference.deploy_drift as dd


# ── Pure helpers ─────────────────────────────────────────────────────────────

def test_extract_sf_sha_happy_path():
    assert dd._extract_sf_sha("[git:abc123def456] rest of comment") == "abc123def456"


def test_extract_sf_sha_short_sha_ok():
    # 7-char SHAs (git short hashes) are valid
    assert dd._extract_sf_sha("[git:abc1234] stuff") == "abc1234"


def test_extract_sf_sha_no_prefix_returns_none():
    assert dd._extract_sf_sha("No git prefix here") is None


def test_extract_sf_sha_empty_comment_returns_none():
    assert dd._extract_sf_sha("") is None


def test_extract_sf_sha_handles_whitespace():
    assert dd._extract_sf_sha("   [git:deadbeef12345] some comment") == "deadbeef12345"


def test_extract_sf_sha_rejects_nonhex():
    # 'g' is not hex — prevents accidental match on non-SHA strings
    assert dd._extract_sf_sha("[git:abcdefg] garbage") is None


# ── SHA-match logic ──────────────────────────────────────────────────────────

def test_shas_match_exact():
    assert dd._shas_match("abc123" * 6 + "abcd", "abc123" * 6 + "abcd") is True


def test_shas_match_short_prefix():
    # Short deployed SHA (e.g. 12 chars) matches if it's a prefix of upstream
    assert dd._shas_match("abc123def456", "abc123def456ffffffffffff12345678ffff9999") is True


def test_shas_match_mismatch():
    assert dd._shas_match("abc123def456", "deadbeef56781111111111111111111111111111") is False


def test_shas_match_none_deployed_passes():
    # Missing stamp → can't prove drift → don't block
    assert dd._shas_match(None, "deadbeef" * 5) is True


def test_shas_match_none_upstream_passes():
    assert dd._shas_match("abc123def456", None) is True


def test_shas_match_malformed_deployed_passes():
    # <7 char stamp is malformed, treat as missing
    assert dd._shas_match("abc", "deadbeef" * 5) is True


# ── config-I7799: the definition comparison is OFF unless a test asks ────────
#
# `check_deploy_drift` now reads the live definition and fetches the repo's for
# the weekday pipelines. Left unpatched these two would reach boto3 and
# raw.githubusercontent from a unit test, and the pre-I7799 tests below would
# pass only because both happened to fail. Default them to "could not compare"
# — the documented fallback-to-stamp path — so every test below states which
# world it is in.

@pytest.fixture(autouse=True)
def _no_definition_comparison_by_default():
    # I7927 adds a THIRD source — S3 — and it must be defaulted here too, or
    # every test that does not patch it reaches boto3 for real and passes only
    # because that happened to fail. That is precisely the failure mode this
    # fixture was written for.
    with patch.object(dd, "_read_sf_definition", return_value=None), \
         patch.object(dd, "_fetch_s3_definition", return_value=None), \
         patch.object(dd, "_fetch_repo_definition", return_value=None):
        yield


# ── check_deploy_drift composition ───────────────────────────────────────────

@patch.object(dd, "_read_sf_comment", return_value="[git:deadbeef12345] weekday pipeline")
@patch.object(dd, "_read_stack_tag", return_value="deadbeef12345abcdef0123456789012345abcdef")
@patch.object(dd, "_fetch_origin_main_sha", return_value="deadbeef12345abcdef0123456789012345abcdef")
def test_no_drift_when_everything_matches(mock_fetch, mock_tag, mock_comment):
    result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["has_drift"] is False
    assert result["sf_drift"] is False
    assert result["cf_drift"] is False
    assert result["sf_stamp_present"] is True
    assert result["stack_stamp_present"] is True


@patch.object(dd, "_read_sf_comment", return_value="[git:aaaa111aaaa1] stale")
@patch.object(dd, "_read_stack_tag", return_value="bbbb222bbbb2cccccccccccccccccccccccccccc")
@patch.object(dd, "_fetch_origin_main_sha", return_value="bbbb222bbbb2cccccccccccccccccccccccccccc")
def test_sf_drift_detected(mock_fetch, mock_tag, mock_comment):
    result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["sf_drift"] is True
    assert result["cf_drift"] is False
    assert result["has_drift"] is True


@patch.object(dd, "_read_sf_comment", return_value="[git:bbbb222bbbb2] ok")
@patch.object(dd, "_read_stack_tag", return_value="aaaa111aaaa1dddddddddddddddddddddddddddd")
@patch.object(dd, "_fetch_origin_main_sha", return_value="bbbb222bbbb2cccccccccccccccccccccccccccc")
def test_cf_drift_detected(mock_fetch, mock_tag, mock_comment):
    result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["sf_drift"] is False
    assert result["cf_drift"] is True
    assert result["has_drift"] is True


@patch.object(dd, "_read_sf_comment", return_value=None)
@patch.object(dd, "_read_stack_tag", return_value=None)
@patch.object(dd, "_fetch_origin_main_sha", return_value="a" * 40)
def test_missing_stamps_do_not_trigger_drift(mock_fetch, mock_tag, mock_comment):
    # First SF deploy before stamping shipped won't have a prefix; first CF
    # deploy without git-sha tag won't have the tag. Don't block these paths.
    # alpha-engine-config-I7048: a MISSING local stamp is a confirmed,
    # already-measured state (nothing to compare) — distinct from an
    # unmeasured upstream fetch failure — so sf_drift/cf_drift stay a
    # definite, present False here (upstream IS reachable in this fixture;
    # see test_github_outage_is_no_drift below for the genuinely
    # unmeasured — upstream unreachable — case).
    result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["has_drift"] is False
    assert result["sf_stamp_present"] is False
    assert result["stack_stamp_present"] is False
    assert result["sf_drift"] is False
    assert result["cf_drift"] is False
    assert result["sf_drift_reason"] == "no_git_sha_stamp_legacy"
    assert result["cf_drift_reason"] == "no_git_sha_tag_legacy"


@patch.object(dd, "_read_sf_comment", return_value="[git:abc1234] old")
@patch.object(dd, "_read_stack_tag", return_value="abc1234")
@patch.object(dd, "_fetch_origin_main_sha", return_value=None)
def test_github_outage_is_no_drift(mock_fetch, mock_tag, mock_comment):
    # Can't compare against upstream, but stamps DO exist → genuinely
    # UNMEASURED, not "confirmed no drift". alpha-engine-config-I7048: this
    # test previously pinned the bug it now guards against — sf_drift/
    # cf_drift must be OMITTED here (not a present False), so the SF's
    # IsPresent-guarded DeployDriftGate Choice (config#6615, fail CLOSED on
    # unknown) actually halts on a GitHub outage instead of silently
    # trading through an unverified deploy.
    result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["upstream_sha"] is None
    assert "sf_drift" not in result
    assert "cf_drift" not in result
    assert result["sf_drift_reason"] == "fetch_failed"
    assert result["cf_drift_reason"] == "fetch_failed"
    # has_drift is a best-effort summary field the SF Choice does NOT read
    # (it reads sf_drift/cf_drift directly) — it degrades to False when
    # both components are unmeasured, same non-gating status as before.
    assert result["has_drift"] is False


# ── AWS read surface ─────────────────────────────────────────────────────────

def test_read_sf_comment_parses_description():
    mock_sfn = MagicMock()
    mock_sfn.describe_state_machine.return_value = {
        "definition": json.dumps({"Comment": "[git:abc123] foo", "States": {}})
    }
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_sfn
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        comment = dd._read_sf_comment("arn:aws:states:us-east-1:1:stateMachine:x")
    assert comment == "[git:abc123] foo"


def test_read_sf_comment_returns_none_on_error():
    mock_sfn = MagicMock()
    mock_sfn.describe_state_machine.side_effect = Exception("boom")
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_sfn
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        comment = dd._read_sf_comment("arn:...")
    assert comment is None


def test_read_stack_tag_happy():
    mock_cfn = MagicMock()
    mock_cfn.describe_stacks.return_value = {
        "Stacks": [{
            "Tags": [
                {"Key": "git-sha", "Value": "deadbeef"},
                {"Key": "other", "Value": "thing"},
            ]
        }]
    }
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_cfn
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        sha = dd._read_stack_tag("alpha-engine-orchestration")
    assert sha == "deadbeef"


def test_read_stack_tag_returns_none_when_tag_absent():
    mock_cfn = MagicMock()
    mock_cfn.describe_stacks.return_value = {
        "Stacks": [{"Tags": [{"Key": "other", "Value": "x"}]}]
    }
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_cfn
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        sha = dd._read_stack_tag("stack")
    assert sha is None


# ── config-I7799: content comparison for the weekday pipelines ───────────────
#
# Brian ruling 2026-08-20, option (b). The 2026-08-20 preopen halted because
# three merges to nousergon-data had not deployed — none of which touched the
# preopen definition. The live orchestration was byte-identical to main's and
# correct; the session went unmanaged anyway.

_UPSTREAM = "bbbb222bbbb2cccccccccccccccccccccccccccc"
_STALE_STAMP = "[git:aaaa111aaaa1] stale stamp, unrelated merges"

_DEFINITION = {
    "Comment": "Alpha Engine preopen",
    "StartAt": "A",
    "States": {"A": {"Type": "Pass", "End": True}},
}


def _stamped(doc, sha):
    """What the deploy uploads: the same doc with a [git:<sha>] Comment."""
    out = json.loads(json.dumps(doc))
    out["Comment"] = f"[git:{sha}] {doc.get('Comment', '')}".rstrip()
    return out


def test_canonical_definition_ignores_the_deploy_stamp():
    assert (
        dd._canonical_definition(_stamped(_DEFINITION, "abc1234"))
        == dd._canonical_definition(_DEFINITION)
    )


def test_canonical_definition_ignores_key_order():
    reordered = {
        "States": {"A": {"End": True, "Type": "Pass"}},
        "StartAt": "A",
        "Comment": "Alpha Engine preopen",
    }
    assert dd._canonical_definition(reordered) == dd._canonical_definition(_DEFINITION)


def test_canonical_definition_drops_a_comment_that_was_only_a_stamp():
    """`Comment = f'[git:{sha}] {orig}'.rstrip()` on an empty orig leaves the
    stamp alone, which must normalise to the repo's absent Comment."""
    bare = {"StartAt": "A", "States": {"A": {"Type": "Pass", "End": True}}}
    assert (
        dd._canonical_definition({**bare, "Comment": "[git:abc1234]"})
        == dd._canonical_definition(bare)
    )


def test_canonical_definition_sees_a_real_difference():
    changed = json.loads(json.dumps(_DEFINITION))
    changed["States"]["A"]["Type"] = "Succeed"
    assert dd._canonical_definition(changed) != dd._canonical_definition(_DEFINITION)


@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=_UPSTREAM)
def test_stale_stamp_with_identical_definition_degrades_instead_of_halting(
    mock_fetch, mock_tag, mock_comment,
):
    """The 2026-08-20 case exactly: unrelated merges undeployed, preopen
    definition unchanged. Must NOT halt, must still SAY the stamp is stale."""
    with patch.object(dd, "_read_sf_definition",
                      return_value=_stamped(_DEFINITION, "aaaa111aaaa1")), \
         patch.object(dd, "_fetch_repo_definition", return_value=_DEFINITION):
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert result["sf_drift"] is False
    assert result["has_drift"] is False
    assert result["deploy_stamp_stale"] is True
    assert result["sf_definition_compared"] is True
    assert result["sf_drift_reason"] == "definition_identical"
    assert result["sf_definition_path"] == "infrastructure/step_function_daily.json"


@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=_UPSTREAM)
def test_a_real_definition_change_still_halts(mock_fetch, mock_tag, mock_comment):
    changed = json.loads(json.dumps(_DEFINITION))
    changed["States"]["A"] = {"Type": "Fail", "Error": "Nope"}
    with patch.object(dd, "_read_sf_definition",
                      return_value=_stamped(_DEFINITION, "aaaa111aaaa1")), \
         patch.object(dd, "_fetch_repo_definition", return_value=changed):
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert result["sf_drift"] is True
    assert result["has_drift"] is True
    assert result["sf_drift_reason"] == "definition_mismatch"


@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=_UPSTREAM)
def test_unreachable_repo_definition_falls_back_to_the_stamp_verdict(
    mock_fetch, mock_tag, mock_comment,
):
    """A missing comparison is never a pass (sf-pipeline-policy §2.3a rule 2).
    The fallback is the stamp, which halts strictly more often."""
    with patch.object(dd, "_read_sf_definition",
                      return_value=_stamped(_DEFINITION, "aaaa111aaaa1")), \
         patch.object(dd, "_fetch_repo_definition", return_value=None):
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert result["sf_drift"] is True
    assert result["sf_definition_compared"] is False
    assert result["sf_definition_reason"] == "repo_definition_unreachable"


@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=_UPSTREAM)
def test_unreadable_live_definition_falls_back_to_the_stamp_verdict(
    mock_fetch, mock_tag, mock_comment,
):
    with patch.object(dd, "_read_sf_definition", return_value=None), \
         patch.object(dd, "_fetch_repo_definition", return_value=_DEFINITION):
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert result["sf_drift"] is True
    assert result["sf_definition_compared"] is False
    assert result["sf_definition_reason"] == "live_definition_unreadable"


@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=_UPSTREAM)
def test_weekly_pipeline_keeps_stamp_semantics(mock_fetch, mock_tag, mock_comment):
    """No market-open deadline there — a lost weekly run costs a rerun, not a
    session, so the broader stamp signal stays the halt condition."""
    with patch.object(dd, "_read_sf_definition",
                      return_value=_stamped(_DEFINITION, "aaaa111aaaa1")), \
         patch.object(dd, "_fetch_repo_definition", return_value=_DEFINITION):
        result = dd.check_deploy_drift(
            region="us-east-1", account_id="123",
            sf_name="ne-weekly-freshness-pipeline",
        )

    assert result["sf_drift"] is True
    assert result["sf_definition_compared"] is False
    assert result["sf_definition_path"] is None
    assert result["sf_definition_reason"] == "stamp_only_pipeline"


@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=None)
def test_github_outage_still_omits_sf_drift(mock_fetch, mock_tag, mock_comment):
    """config-I7048's invariant survives: with a stamp present, upstream
    unfetchable AND no expected definition readable, there is nothing to
    compare, so the key is omitted and the SF's IsPresent guard fails closed.

    The REASON moved with I7927 — `live_definition_unreadable` rather than
    `upstream_sha_unavailable` — because the upstream SHA is no longer the
    first thing the comparison needs. The invariant under test is the omission,
    and it is unchanged.
    """
    with patch.object(dd, "_read_sf_definition", return_value=None):
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert "sf_drift" not in result
    assert result["sf_definition_compared"] is False
    assert result["sf_definition_source"] == "none"
    assert result["sf_definition_reason"] == "live_definition_unreadable"


@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=None)
def test_a_github_outage_no_longer_stops_the_halting_verdict(
    mock_fetch, mock_tag, mock_comment,
):
    """**The point of alpha-engine-config-I7927.**

    This is the exact 2026-08-21 input — stamp present, upstream unfetchable —
    and the probe now reaches a real `sf_drift` verdict anyway, from S3, with
    GitHub never consulted for it. On 2026-08-21 this input halted trading.
    """
    with patch.object(dd, "_read_sf_definition",
                      return_value=_stamped(_DEFINITION, "aaaa111aaaa1")), \
         patch.object(dd, "_fetch_s3_definition", return_value=_DEFINITION), \
         patch.object(dd, "_fetch_repo_definition") as gh:
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert result["sf_drift"] is False
    assert result["sf_definition_compared"] is True
    assert result["sf_definition_source"] == "s3"
    assert result["sf_definition_reason"] == "definition_identical"
    # GitHub was not asked for the expected definition at all.
    gh.assert_not_called()


# ── alpha-engine-config-I7924: the 2026-08-21 preopen halt ──────────────────
# An expired GITHUB_TOKEN made _fetch_origin_main_sha return None, so the probe
# omitted sf_drift as unmeasured and DeployDriftGate's fail-closed branch
# stopped the trading day 3.4 seconds in. nousergon-lib v0.124.79 makes the
# fetch survive the rejected credential; these tests pin what the probe must
# then SAY, so the surviving call does not become a silent dependence on the
# anonymous fallback.

@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
def test_deploy_stamp_stale_is_omitted_when_upstream_could_not_be_fetched(
    mock_tag, mock_comment,
):
    """The literal 2026-08-21 payload had `deploy_stamp_stale: false` beside
    `upstream_sha: null` — nothing compared, rendered as a verified clean.

    It must be ABSENT, exactly like sf_drift/cf_drift, with the reason code
    carrying why.
    """
    with patch.object(dd, "_fetch_origin_main_sha", return_value=None):
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert "deploy_stamp_stale" not in result
    assert result["deploy_stamp_stale_reason"] == "fetch_failed"
    assert "sf_drift" not in result
    assert "cf_drift" not in result
    assert result["upstream_sha"] is None


@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=_UPSTREAM)
def test_deploy_stamp_stale_reason_is_in_sync_when_measured_clean(
    mock_fetch, mock_tag, mock_comment,
):
    with patch.object(dd, "_read_sf_definition",
                      return_value=_stamped(_DEFINITION, "aaaa111aaaa1")), \
         patch.object(dd, "_fetch_repo_definition", return_value=_DEFINITION):
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert result["deploy_stamp_stale"] is True
    assert result["deploy_stamp_stale_reason"] == "sha_mismatch"


def test_a_rejected_github_credential_is_named_in_the_payload():
    """The token is still expired even though the probe now answers. If the
    payload did not say so, the fix would trade a loud halt for a silent
    dependence on the unauthenticated fallback."""
    def _fetch(repo, branch="main", *, stats=None, **_kw):
        if stats is not None:
            stats["github_credential_rejected"] = True
            stats["github_credential_status"] = 401
        return _UPSTREAM

    with patch.object(dd, "_read_sf_comment", return_value=f"[git:{_UPSTREAM}] in sync"), \
         patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM), \
         patch.object(dd, "_fetch_origin_main_sha", side_effect=_fetch), \
         patch.object(dd, "_read_sf_definition",
                      return_value=_stamped(_DEFINITION, _UPSTREAM[:12])), \
         patch.object(dd, "_fetch_repo_definition", return_value=_DEFINITION):
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert result["github_credential_rejected"] is True
    assert result["github_credential_status"] == 401
    # …and the run is otherwise clean: a bad token must not halt trading.
    assert result["sf_drift"] is False
    assert result["has_drift"] is False


@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=_UPSTREAM)
def test_a_healthy_credential_leaves_no_rejection_key(
    mock_fetch, mock_tag, mock_comment,
):
    """Absence must never be readable as 'checked and healthy' — the key is
    present ONLY on an actual rejection."""
    with patch.object(dd, "_read_sf_definition",
                      return_value=_stamped(_DEFINITION, "aaaa111aaaa1")), \
         patch.object(dd, "_fetch_repo_definition", return_value=_DEFINITION):
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert "github_credential_rejected" not in result
    assert "github_credential_status" not in result


# ── alpha-engine-config-I7927: S3 is the expected definition, GitHub is not ──
# The halting verdict must not depend on a live api.github.com call. The deploy
# is the sole writer (sf-pipeline-policy 2.4) and already uploads the exact
# bytes it feeds to update-state-machine, so it has already written down the
# answer this probe was re-deriving from the upstream the deploy was built from.

@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=_UPSTREAM)
def test_s3_is_preferred_even_when_github_is_perfectly_available(
    mock_fetch, mock_tag, mock_comment,
):
    """Not merely a fallback — the default. If GitHub stayed on the halting
    path whenever it happened to be up, the dependency would still be there and
    would still fail on the one morning it mattered."""
    with patch.object(dd, "_read_sf_definition",
                      return_value=_stamped(_DEFINITION, "aaaa111aaaa1")), \
         patch.object(dd, "_fetch_s3_definition", return_value=_DEFINITION), \
         patch.object(dd, "_fetch_repo_definition") as gh:
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert result["sf_definition_source"] == "s3"
    gh.assert_not_called()


@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=_UPSTREAM)
def test_a_real_definition_change_still_halts_when_read_from_s3(
    mock_fetch, mock_tag, mock_comment,
):
    """The guarantee the whole probe exists for, on the new source: a live
    definition that differs from what the deploy published still halts."""
    changed = json.loads(json.dumps(_DEFINITION))
    changed["States"]["A"] = {"Type": "Fail", "Error": "HandPatched"}
    with patch.object(dd, "_read_sf_definition",
                      return_value=_stamped(changed, "aaaa111aaaa1")), \
         patch.object(dd, "_fetch_s3_definition", return_value=_DEFINITION):
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert result["sf_drift"] is True
    assert result["has_drift"] is True
    assert result["sf_definition_source"] == "s3"
    assert result["sf_definition_reason"] == "definition_mismatch"


@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=_UPSTREAM)
def test_github_is_the_fallback_when_s3_is_unreadable(
    mock_fetch, mock_tag, mock_comment,
):
    """S3 down is not a reason to drop to the stamp while a better source is
    still reachable — the pre-I7927 path is strictly more information."""
    with patch.object(dd, "_read_sf_definition",
                      return_value=_stamped(_DEFINITION, "aaaa111aaaa1")), \
         patch.object(dd, "_fetch_s3_definition", return_value=None), \
         patch.object(dd, "_fetch_repo_definition", return_value=_DEFINITION):
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert result["sf_definition_compared"] is True
    assert result["sf_definition_source"] == "github"
    assert result["sf_drift"] is False


@patch.object(dd, "_read_sf_comment", return_value=_STALE_STAMP)
@patch.object(dd, "_read_stack_tag", return_value=_UPSTREAM)
@patch.object(dd, "_fetch_origin_main_sha", return_value=None)
def test_both_sources_gone_still_fails_closed(mock_fetch, mock_tag, mock_comment):
    """S3 unreadable AND no upstream SHA — strictly rarer than the 2026-08-21
    single-source failure, and still omits sf_drift rather than granting a
    pass (sf-pipeline-policy 2.3a rule 2)."""
    with patch.object(dd, "_read_sf_definition",
                      return_value=_stamped(_DEFINITION, "aaaa111aaaa1")), \
         patch.object(dd, "_fetch_s3_definition", return_value=None):
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")

    assert "sf_drift" not in result
    assert result["sf_definition_source"] == "none"
    assert result["sf_definition_reason"] == "expected_definition_unavailable"


def test_the_s3_bucket_is_not_configurable_by_ambient_environment(monkeypatch):
    """I7925's lesson applied one module over: a probe that can be pointed
    somewhere else by an ambient value is a probe whose answer is not about the
    thing you think it is."""
    monkeypatch.setenv("DEFINITION_BUCKET", "attacker-bucket")
    monkeypatch.setenv("S3_BUCKET", "attacker-bucket")
    assert dd._DEFINITION_BUCKET == "alpha-engine-research"


def test_the_s3_key_is_the_repo_path():
    """deploy-infrastructure.sh uploads to the same key as the repo path, so
    the two never need to be kept in sync separately."""
    assert (dd._SF_DEFINITION_PATHS["ne-preopen-trading-pipeline"]
            == "infrastructure/step_function_daily.json")
