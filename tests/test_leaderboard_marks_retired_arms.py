"""A retired arm's historical shadow rows must not silently mix with a live
arm's on the realized-edge leaderboard (alpha-engine-config-I10180).

Retiring the five degenerate ``spec-sota-combine-2026-07-*`` arms stops NEW
shadow writes. Their existing ``predictions_shadow/{version_id}/*.json`` remain
— the bundles and their output are evidence and are not deleted — so the reader
has to say which rows belong to a retired arm, or a leaderboard reader compares
a retired constant-alpha arm's IC against a live one's and cannot tell.

And a retired arm must not raise the ``unmeasurable`` page: it stops writing
shadows BY DESIGN. Paging that would be a new false page created by the fix for
a false page.
"""

from __future__ import annotations

from analysis.observe_leaderboard import build_observe_leaderboard
from model.retired_arms import RETIRED_SHADOW_VERSIONS

RETIRED = "spec-sota-combine-2026-07-24-8578f8ae"
LIVE = "v3.0-meta-2026-09-04-cc3271ea"


def _pairs(vid, dates):
    return [
        {"date": d, "ticker": "AAA", "predicted_alpha": 0.01,
         "champion_version_id": vid}
        for d in dates
    ]


def _build(**kw):
    return build_observe_leaderboard(
        bucket="b", write_to_s3=False, write_latest=False,
        date_str="2026-09-08", prices_by_ticker={}, sector_map={},
        live_pairs=[], **kw,
    )


def _entry(payload, vid):
    return next(e for e in payload["entries"] if e["version_id"] == vid)


def test_a_retired_arms_row_is_marked_retired_with_its_reason():
    out = _build(
        shadow_pairs_by_version={RETIRED: _pairs(RETIRED, ["2026-09-05"]),
                                 LIVE: _pairs(LIVE, ["2026-09-05"])},
        registered_challenger_ids=[LIVE],
        shadow_dates_by_version={RETIRED: ["2026-09-05"], LIVE: ["2026-09-05"]},
    )
    e = _entry(out, RETIRED)
    assert e["retired"] is True
    assert e["retired_reason"] == RETIRED_SHADOW_VERSIONS[RETIRED]


def test_a_live_arms_row_says_retired_false_rather_than_omitting_the_field():
    """An absent field is indistinguishable from a healthy one — the whole
    reason this leaderboard emits explicit rows at all."""
    out = _build(
        shadow_pairs_by_version={LIVE: _pairs(LIVE, ["2026-09-05"])},
        registered_challenger_ids=[LIVE],
        shadow_dates_by_version={LIVE: ["2026-09-05"]},
    )
    assert _entry(out, LIVE)["retired"] is False
    assert _entry(out, LIVE)["retired_reason"] is None


def test_a_retired_arm_never_raises_the_unmeasurable_page():
    """It stops writing shadows by design. Paging that is a new false page."""
    out = _build(
        shadow_pairs_by_version={RETIRED: [], LIVE: _pairs(LIVE, ["2026-09-08"])},
        registered_challenger_ids=[LIVE, RETIRED],
        shadow_dates_by_version={RETIRED: [], LIVE: ["2026-09-08"]},
    )
    e = _entry(out, RETIRED)
    assert e["measurability"] == "retired"
    assert "retired" in (e["verdict_reason"] or "").lower()


def test_a_retired_arm_does_not_define_the_cohort_other_arms_are_judged_against():
    """The cohort is 'what a live arm should have written'. At the moment of
    retirement the five degenerate arms hold the NEWEST shadow-write dates —
    they wrote on 2026-09-08, the day they were retired — so leaving them in
    the cohort would mark every genuinely healthy live arm `unmeasurable` for
    the next three cycles, purely because it did not match a retired arm's
    residue."""
    out = _build(
        shadow_pairs_by_version={LIVE: _pairs(LIVE, ["2026-09-05"])},
        registered_challenger_ids=[LIVE, RETIRED],
        shadow_dates_by_version={
            RETIRED: ["2026-09-06", "2026-09-07", "2026-09-08"],
            LIVE: ["2026-09-05"],
        },
    )
    assert _entry(out, LIVE)["measurability"] == "measured"


def test_a_genuinely_starved_live_arm_still_pages():
    """The retirement carve-out must not swallow the signal it sits next to."""
    other = "v3.0-meta-2026-08-28-01cf7e1a"
    out = _build(
        shadow_pairs_by_version={LIVE: _pairs(LIVE, ["2026-09-08"]), other: []},
        registered_challenger_ids=[LIVE, other],
        shadow_dates_by_version={LIVE: ["2026-09-08"], other: []},
    )
    assert _entry(out, other)["measurability"] == "unmeasurable"
