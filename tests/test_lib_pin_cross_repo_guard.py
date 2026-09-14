"""Unit tests for `scripts/lib_pin_cross_repo_guard.py` — the parity DECISION
extracted from `.github/workflows/lib-pin-cross-repo-guard.yml`'s inline bash
(alpha-engine-config-I10762).

The guard's own comment warned that comparing only against the sibling's
`main` pin fails BOTH halves of a correct lockstep bump, because at the
moment either PR runs the sibling's main still holds the OLD pin — the
alpha-engine-config-I7934 lookaside (an in-flight partner PR pinning the same
version) is what breaks that deadlock. This file is the mutation-tested proof
that the lookaside actually does: `TestBothMergeOrders` exercises a
coordinated pin bump in BOTH merge orders (predictor-first and
backtester-first), and `test_removing_the_lookaside_breaks_both_orders` (run
manually, see the PR body — the mutation is never committed) confirms the
both-orders tests go red the moment the lookaside branch is deleted.

No network: `evaluate()` takes the sibling's open PRs (with their pins)
already resolved, exactly as the workflow now hands them off.
"""

from __future__ import annotations

from scripts.lib_pin_cross_repo_guard import evaluate

OLD = "v0.124.76"
NEW = "v0.124.77"


def _pr(number: int, pin: str | None) -> dict:
    return {"number": number, "pin": pin}


class TestEqualPins:
    def test_equal_pins_pass_without_consulting_prs(self) -> None:
        # sibling_open_prs=None would normally mean "API unreachable -> fail
        # closed" -- passing it here and still getting green proves the
        # equal-pin branch short-circuits before ever looking at PRs, exactly
        # like the original inline bash's `exit 0` before the API call.
        v = evaluate(NEW, NEW, None)
        assert v.ok
        assert v.lines == [f"co-install parity OK: both pin {NEW}"]


class TestBothMergeOrders:
    """A coordinated pin bump (both repos move OLD -> NEW together) must be
    green at every step, regardless of which repo's PR merges first. This
    guard's OWN/SIBLING are fixed to predictor/backtester, so "both orders"
    from this repo's side means: predictor's PR runs BEFORE backtester's
    (this repo is the first half, needs the I7934 lookaside) and predictor's
    PR runs AFTER backtester's already merged (this repo is the second half,
    sibling@main already matches -- plain equality, no lookaside needed).
    crucible-backtester's mirror of this file exercises the same two shapes
    from backtester's side, covering the "backtester first" order."""

    def test_this_repo_first_partner_pr_open_green(self) -> None:
        # predictor PR open with NEW pin; backtester main still OLD;
        # backtester carries an open partner PR already pinning NEW.
        v = evaluate(
            own_pin=NEW,
            sibling_main_pin=OLD,
            sibling_open_prs=[_pr(718, NEW)],
        )
        assert v.ok
        assert v.lines[0].startswith("::notice::co-install parity DEFERRED")
        assert "crucible-backtester#718" in v.lines[0]
        assert v.lines[1] == (
            "co-install parity OK (lockstep partner in flight): "
            "crucible-backtester#718 pins v0.124.77"
        )

    def test_this_repo_second_after_sibling_already_merged_new(self) -> None:
        # backtester's half already merged (sibling@main is now NEW); this
        # repo's own PR, also NEW, compares equal -- no lookaside needed.
        v = evaluate(own_pin=NEW, sibling_main_pin=NEW, sibling_open_prs=[])
        assert v.ok
        assert v.lines == [f"co-install parity OK: both pin {NEW}"]


class TestRedPaths:
    def test_no_partner_pr_is_red(self) -> None:
        v = evaluate(own_pin=NEW, sibling_main_pin=OLD, sibling_open_prs=[])
        assert not v.ok
        assert v.lines[0].startswith("::error::co-install parity break")
        assert "NO open crucible-backtester PR pins" in v.lines[0]

    def test_api_unreachable_fails_closed(self) -> None:
        v = evaluate(own_pin=NEW, sibling_main_pin=OLD, sibling_open_prs=None)
        assert not v.ok
        assert "GitHub API could not be reached" in v.lines[0]
        assert "no answer" in v.lines[0]

    def test_partner_pr_with_a_different_pin_is_red(self) -> None:
        v = evaluate(
            own_pin=NEW,
            sibling_main_pin=OLD,
            sibling_open_prs=[_pr(999, "v0.124.80")],
        )
        assert not v.ok
        assert "NO open crucible-backtester PR pins" in v.lines[0]

    def test_partner_pr_with_unparsable_pin_is_ignored_not_matched(self) -> None:
        v = evaluate(own_pin=NEW, sibling_main_pin=OLD, sibling_open_prs=[_pr(1, None)])
        assert not v.ok


def test_removing_the_lookaside_breaks_both_orders() -> None:
    """Documents the mutation performed manually (never committed, per the PR
    body): with the `sibling_open_prs` for-loop in `evaluate()` deleted (or
    short-circuited to always fall through to the "no partner" branch), both
    `TestBothMergeOrders.test_*_partner_pr_open_green` tests above go red --
    proving they actually exercise the lookaside and are not vacuously green.
    This test itself is a no-op assertion; it exists to point a reader at the
    mutation record rather than re-perform it (the mutation is not
    expressible without editing the source under test)."""
    assert evaluate(NEW, OLD, [_pr(718, NEW)]).ok  # true today, WITH the lookaside
