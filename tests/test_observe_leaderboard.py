"""Tests for the realized-edge observe leaderboard (config #671/#702/L4539).

Pins:
  - realized rank-IC scoring of a shadow version vs champion over matured pairs;
  - since I9319/I9322 the leaderboard MEASURES only — no promotion-readiness
    verdict; the M-slot pointer is decided by ``nousergon_lib.arena``;
  - 'too young' / 'insufficient outcomes' are reported in ``verdict_reason``,
    not as gates;
  - the leaderboard NEVER promotes / allocates (it is measurement only — there
    is no registry write path in this module).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.observe_leaderboard as ol


def _prices(n_days=120):
    """A rising-AAA / flat-SPY fixture so realized 21d alpha is well-defined and
    positive, with enough forward room for the 21d window to close."""
    idx = pd.date_range("2026-03-01", periods=n_days, freq="B")
    return {
        "AAA": pd.Series(np.linspace(100, 160, n_days), index=idx),
        "BBB": pd.Series(np.linspace(100, 90, n_days), index=idx),
        "SPY": pd.Series(np.linspace(100, 110, n_days), index=idx),
    }


def _pred_rows(alpha_by_ticker, dates):
    """Build (date, ticker, predicted_alpha) rows for a set of dates/tickers."""
    rows = []
    for d in dates:
        for tkr, a in alpha_by_ticker.items():
            rows.append({"date": d, "ticker": tkr,
                         "predicted_alpha": a, "realized_alpha": None})
    return rows


def _weekly_dates(n_weeks, per_week=5):
    """ISO dates spanning ``n_weeks`` distinct weeks (Mon-Fri each), early enough
    that the 21d forward window closes within the fixture's price history."""
    start = pd.Timestamp("2026-03-02")  # a Monday
    out = []
    for w in range(n_weeks):
        for d in range(per_week):
            out.append((start + pd.Timedelta(weeks=w, days=d)).date().isoformat())
    return out


def _census(vid, dates):
    """Bypass the registered-challenger S3 census in unit tests."""
    return {
        "registered_challenger_ids": [vid],
        "shadow_dates_by_version": {vid: dates},
    }


def test_leaderboard_scores_realized_ic_and_beats_champion():
    dates = _weekly_dates(5)
    good = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)
    champ = _pred_rows({"AAA": 0.1, "BBB": 0.9}, dates)
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v-good": good},
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
        **_census("v-good", dates),
    )
    e = res["entries"][0]
    assert e["version_id"] == "v-good"
    assert e["realized_rank_ic"] is not None
    assert e["realized_rank_ic"] > 0
    assert res["champion"]["realized_rank_ic"] < 0
    assert e["beats_champion_realized"] is True
    assert e["n_weeks_coverage"] >= 4
    assert e["n_matured_outcomes"] >= ol.MIN_REALIZED_OUTCOMES
    assert "ready_for_full_promotion" not in e
    assert "ready_version_ids" not in res
    assert res["decision_authority"]["artifact"] == "arena/model/{date}.json"


def test_leaderboard_soak_too_young_is_a_point_estimate_only():
    """Two weeks still yields a per-version IC when pairs exist — but the
    leaderboard no longer gates promotion-readiness on soak length (I9319)."""
    dates = _weekly_dates(2)
    good = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)
    champ = _pred_rows({"AAA": 0.1, "BBB": 0.9}, dates)
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v-young": good},
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
        **_census("v-young", dates),
    )
    e = res["entries"][0]
    assert e["realized_rank_ic"] is not None
    assert e["n_weeks_coverage"] == 2
    assert "arena/model/{date}.json" in e["verdict_reason"]
    assert "ready_for_full_promotion" not in e


def test_leaderboard_loses_to_champion():
    dates = _weekly_dates(5)
    bad = _pred_rows({"AAA": 0.1, "BBB": 0.9}, dates)
    champ = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v-bad": bad},
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
        **_census("v-bad", dates),
    )
    e = res["entries"][0]
    assert e["beats_champion_realized"] is False
    assert "realized rank-IC" in e["verdict_reason"]


def test_leaderboard_unmatured_outcomes_reported():
    last = _prices(120)["AAA"].index[-1]
    near_end = [(last - pd.Timedelta(days=k)).date().isoformat() for k in range(5)]
    rows = _pred_rows({"AAA": 0.9, "BBB": 0.1}, near_end)
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v-unmatured": rows},
        live_pairs=rows, prices_by_ticker=_prices(), sector_map={},
        **_census("v-unmatured", near_end),
    )
    e = res["entries"][0]
    assert e["realized_rank_ic"] is None
    assert "unmeasurable" in e["verdict_reason"]


def test_leaderboard_is_measurement_only_no_registry_writes():
    dates = _weekly_dates(5)
    rows = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v": rows}, live_pairs=rows,
        prices_by_ticker=_prices(), sector_map={},
        **_census("v", dates),
    )
    assert "s3_key" not in res
    import inspect

    src = inspect.getsource(ol)
    assert "promote_to_champion" not in src
    assert "register_to_observe" not in src
    assert res["decision_authority"]["note"].startswith("This leaderboard MEASURES")


def test_leaderboard_writes_dated_and_latest():
    dates = _weekly_dates(5)
    rows = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)

    class _S3:
        def __init__(self):
            self.puts = []

        def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
            self.puts.append(Key)

    s3 = _S3()
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=True, write_latest=True, s3_client=s3,
        date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v": rows}, live_pairs=rows,
        prices_by_ticker=_prices(), sector_map={},
    )
    assert f"{ol.OUTPUT_PREFIX}/2026-06-13.json" in s3.puts
    assert f"{ol.OUTPUT_PREFIX}/latest.json" in s3.puts
    assert res["s3_key"] == f"{ol.OUTPUT_PREFIX}/2026-06-13.json"


# ── config#1052: champion realized-edge NOISE-CHASING monitor ─────────────────


def test_champion_monitor_healthy_when_realized_edge_positive():
    dates = _weekly_dates(5)
    champ = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)
    res = ol.build_champion_realized_monitor(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
    )
    assert res["kind"] == "champion_realized_monitor"
    assert res["champion"]["realized_rank_ic"] > 0
    assert res["champion"]["n_matured_outcomes"] >= ol.MIN_REALIZED_OUTCOMES
    assert res["chasing_noise"] is False
    assert "healthy" in res["verdict_reason"]


def test_champion_monitor_flags_noise_when_realized_ic_non_positive():
    dates = _weekly_dates(5)
    champ = _pred_rows({"AAA": 0.1, "BBB": 0.9}, dates)
    res = ol.build_champion_realized_monitor(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
    )
    assert res["champion"]["realized_rank_ic"] <= 0
    assert res["chasing_noise"] is True
    assert "NOISE WATCH" in res["verdict_reason"]


def test_champion_monitor_verdict_none_when_too_few_outcomes():
    last = _prices(120)["AAA"].index[-1]
    near_end = [(last - pd.Timedelta(days=k)).date().isoformat() for k in range(3)]
    champ = _pred_rows({"AAA": 0.9, "BBB": 0.1}, near_end)
    res = ol.build_champion_realized_monitor(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
    )
    assert res["chasing_noise"] is None
    assert "insufficient matured outcomes" in res["verdict_reason"]


def test_champion_monitor_is_measurement_only_no_actuator():
    import inspect

    src = inspect.getsource(ol.build_champion_realized_monitor)
    assert "promote_to_champion" not in src
    assert "register_to_observe" not in src


def test_champion_monitor_writes_dated_and_latest():
    dates = _weekly_dates(5)
    champ = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)

    class _S3:
        def __init__(self):
            self.puts = []

        def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
            self.puts.append(Key)

    s3 = _S3()
    res = ol.build_champion_realized_monitor(
        bucket="b", write_to_s3=True, write_latest=True, s3_client=s3,
        date_str="2026-06-13", horizon_days=21,
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
    )
    assert f"{ol.OUTPUT_PREFIX}/2026-06-13.json" in s3.puts
    assert f"{ol.OUTPUT_PREFIX}/latest.json" in s3.puts
    assert res["s3_key"] == f"{ol.OUTPUT_PREFIX}/2026-06-13.json"


# ── alpha-engine-config-I9336: measurability vs. merely-thin ───────────────

class TestMeasurabilityForShadowArm:
    """measurability_for_shadow_arm — pure function, no S3. Mirrors
    crucible-research/scoring/leaderboard_scoring.py::measurability_for
    (crucible-research-PR767)."""

    def test_empty_cohort_is_measured(self):
        assert ol.measurability_for_shadow_arm([], []) == (ol.MEASURABILITY_MEASURED, None)

    def test_zero_scored_with_cohort_is_unmeasurable(self):
        m, reason = ol.measurability_for_shadow_arm([], ["2026-08-21", "2026-08-28"])
        assert m == ol.MEASURABILITY_UNMEASURABLE
        assert "0 of 2" in reason

    def test_scored_within_recent_lag_is_measured(self):
        m, reason = ol.measurability_for_shadow_arm(
            ["2026-08-28"], ["2026-08-14", "2026-08-21", "2026-08-28"],
        )
        assert m == ol.MEASURABILITY_MEASURED
        assert reason is None

    def test_scored_outside_recent_lag_is_unmeasurable(self):
        # Measured 2026-08-29: sota-directional-combine's last shadow write
        # (2026-08-13) is well outside the recent cohort window.
        m, reason = ol.measurability_for_shadow_arm(
            ["2026-08-13"], ["2026-08-14", "2026-08-21", "2026-08-28"],
        )
        assert m == ol.MEASURABILITY_UNMEASURABLE
        assert "2026-08-13" in reason


def test_leaderboard_registered_challenger_with_no_shadow_ever_gets_explicit_unmeasurable_row():
    """alpha-engine-config-I9336 deliverable 2: a registered challenger that
    has NEVER written a shadow prediction must appear as an explicit
    UNMEASURABLE row, never be silently absent from the leaderboard."""
    dates = _weekly_dates(5)
    scored = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)
    champ = _pred_rows({"AAA": 0.1, "BBB": 0.9}, dates)
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v-scored": scored},
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
        registered_challenger_ids=["v-scored", "v-never-shadowed"],
        shadow_dates_by_version={"v-scored": dates, "v-never-shadowed": []},
    )
    by_vid = {e["version_id"]: e for e in res["entries"]}
    assert "v-never-shadowed" in by_vid
    e = by_vid["v-never-shadowed"]
    assert e["measurability"] == ol.MEASURABILITY_UNMEASURABLE
    assert "ready_for_full_promotion" not in e
    assert "UNMEASURABLE" in e["verdict_reason"]
    assert e["n_matured_outcomes"] == 0


def test_leaderboard_stopped_arm_becomes_unmeasurable_not_thin():
    """An arm with plenty of OLD scored dates but none in the recent cohort
    window is UNMEASURABLE — the exact I9307-class defect this mirrors (a
    starved arm rendering identically to one that merely has little)."""
    dates = _weekly_dates(5)
    scored = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)
    champ = _pred_rows({"AAA": 0.1, "BBB": 0.9}, dates)
    stale_dates = ["2026-01-05", "2026-01-06", "2026-01-07"]  # long before `dates`
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v-active": scored, "v-stopped": []},
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
        registered_challenger_ids=["v-active", "v-stopped"],
        shadow_dates_by_version={"v-active": dates, "v-stopped": stale_dates},
    )
    by_vid = {e["version_id"]: e for e in res["entries"]}
    assert by_vid["v-stopped"]["measurability"] == ol.MEASURABILITY_UNMEASURABLE
    assert by_vid["v-active"]["measurability"] == ol.MEASURABILITY_MEASURED


def test_leaderboard_alerts_once_per_build_naming_starved_arms(monkeypatch):
    calls = []
    monkeypatch.setattr("ops_alerts.publish_ops_alert", lambda **kw: calls.append(kw))
    dates = _weekly_dates(5)
    good = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)
    champ = _pred_rows({"AAA": 0.1, "BBB": 0.9}, dates)
    ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v-good": good},
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
        registered_challenger_ids=["v-good", "v-starved"],
        shadow_dates_by_version={"v-good": dates, "v-starved": []},
    )
    assert len(calls) == 1
    assert "v-starved" in calls[0]["message"]
    assert calls[0]["severity"] == "warning"


def test_leaderboard_no_alert_when_nothing_starved(monkeypatch):
    calls = []
    monkeypatch.setattr("ops_alerts.publish_ops_alert", lambda **kw: calls.append(kw))
    dates = _weekly_dates(5)
    good = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)
    champ = _pred_rows({"AAA": 0.1, "BBB": 0.9}, dates)
    ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v-good": good},
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
        registered_challenger_ids=["v-good"],
        shadow_dates_by_version={"v-good": dates},
    )
    assert calls == []


class TestListShadowDatesForVersion:
    def test_extracts_dates_from_listing(self):
        class _FakeS3:
            def get_paginator(self, name):
                return self

            def paginate(self, Bucket, Prefix):
                return [{"Contents": [
                    {"Key": f"{Prefix}2026-08-27.json"},
                    {"Key": f"{Prefix}2026-08-28.json"},
                ]}]

        dates = ol._list_shadow_dates_for_version("b", "v1", s3_client=_FakeS3())
        assert dates == ["2026-08-27", "2026-08-28"]

    def test_read_failure_returns_empty(self):
        class _BrokenS3:
            def get_paginator(self, name):
                raise RuntimeError("boom")

        assert ol._list_shadow_dates_for_version("b", "v1", s3_client=_BrokenS3()) == []


class TestListRegisteredChallengerIds:
    def test_reads_via_list_versions(self, monkeypatch):
        monkeypatch.setattr(
            "model.registry.list_versions",
            lambda s3, bucket, stage=None: [{"version_id": "v1"}, {"version_id": "v2"}],
        )
        ids = ol._list_registered_challenger_ids("b", s3_client=object())
        assert ids == ["v1", "v2"]

    def test_read_failure_returns_empty(self, monkeypatch):
        def _raise(*a, **k):
            raise RuntimeError("boom")
        monkeypatch.setattr("model.registry.list_versions", _raise)
        assert ol._list_registered_challenger_ids("b", s3_client=object()) == []
