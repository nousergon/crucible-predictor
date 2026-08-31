"""alpha-engine-config-I8219 — the champion's own realized record must be
attributable to the arm that produced it.

**The defect these pin.** ``analysis/observe_leaderboard._load_shadow_pairs``
carried ``champion_version_id`` onto every row; ``_load_live_pairs`` — the
loader every LIVE consumer reads — silently omitted it. Measured live
2026-08-31 on ``predictor/model_zoo/observe_leaderboard/latest.json``:
``serving_champion_attributed.attribution_status == "unstamped_predictions"``
and ``realized_rank_ic_by_version == [{"version_id": null, ...}]``, against
prediction artifacts that carry the stamp.

The same rows feed ``training/arena_model_slot.build_series``, which since
-I9319 decides the live champion pointer: a version no arm claims is dropped,
so the M slot ranked its arms on shadow output alone with the serving
champion's entire live record discarded, while emitting a well-formed
``arena_cycle``.

Each test below fails against the pre-fix code (policy §7.4).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.observe_leaderboard as ol  # noqa: E402


class _FakeS3:
    """Serves a fixed key->payload map; anything else raises like S3 does."""

    def __init__(self, objects: dict):
        self._objects = objects

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 signature
        if Key not in self._objects:
            raise KeyError(Key)
        body = json.dumps(self._objects[Key]).encode()

        class _B:
            @staticmethod
            def read():
                return body

        return {"Body": _B()}

    def get_paginator(self, _name):
        objects = self._objects

        class _P:
            @staticmethod
            def paginate(Bucket, Prefix):  # noqa: N803
                yield {"Contents": [{"Key": k} for k in objects if k.startswith(Prefix)]}

        return _P()


def _predictions(date, *, champion_version_id=None, tickers=("AAA", "BBB")):
    doc = {"date": date, "predictions": [
        {"ticker": t, "predicted_alpha": 0.01} for t in tickers
    ]}
    if champion_version_id is not None:
        doc["champion_version_id"] = champion_version_id
    return doc


def _marker(run_date, champion_after):
    return {"run_date": run_date, "champion_version_id_after": champion_after}


# ── the promotion-history resolver ──────────────────────────────────────────

_HISTORY = [
    {"run_date": "2026-08-14", "champion_version_id": "v3.0-meta-2026-08-14-119e069b"},
    {"run_date": "2026-08-21", "champion_version_id": "v3.0-meta-2026-08-21-7d3d1cce"},
    {"run_date": "2026-08-28", "champion_version_id": "v3.0-meta-2026-08-14-119e069b"},
]


def test_champion_serving_on_reproduces_the_live_stamps():
    """Validated against live artifacts 2026-08-31 — the rule is 'newest marker
    STRICTLY EARLIER than the prediction date', because the rotation writes its
    marker for the trading day whose preopen inference has already run."""
    assert ol.champion_serving_on("2026-08-26", _HISTORY) == "v3.0-meta-2026-08-21-7d3d1cce"
    assert ol.champion_serving_on("2026-08-31", _HISTORY) == "v3.0-meta-2026-08-14-119e069b"
    # The marker's own run_date is NOT covered by that marker.
    assert ol.champion_serving_on("2026-08-21", _HISTORY) == "v3.0-meta-2026-08-14-119e069b"
    # No marker precedes it -> honest absence, never a guess.
    assert ol.champion_serving_on("2026-07-01", _HISTORY) is None


def test_load_promotion_history_is_sorted_and_survives_a_bad_marker():
    s3 = _FakeS3({
        "predictor/model_zoo/promotions/2026-08-21.json": _marker("2026-08-21", "B"),
        "predictor/model_zoo/promotions/2026-08-14.json": _marker("2026-08-14", "A"),
        "predictor/model_zoo/promotions/2026-08-28.json": {"run_date": "2026-08-28"},
    })
    hist = ol.load_promotion_history("bkt", s3_client=s3)
    assert [r["run_date"] for r in hist] == ["2026-08-14", "2026-08-21"]


# ── the defect itself ───────────────────────────────────────────────────────

def test_live_pairs_carry_the_artifacts_own_stamp():
    """RED pre-fix: `_load_live_pairs` built rows without champion_version_id."""
    s3 = _FakeS3({
        "predictor/predictions/2026-08-26.json": _predictions(
            "2026-08-26", champion_version_id="v3.0-meta-2026-08-21-7d3d1cce"),
    })
    pairs = ol._load_live_pairs("bkt", 400, s3_client=s3, promotion_history=[])
    assert pairs, "fixture date must be inside the window"
    assert {p["champion_version_id"] for p in pairs} == {"v3.0-meta-2026-08-21-7d3d1cce"}
    assert {p["champion_version_id_source"] for p in pairs} == {ol.ATTR_STAMPED}


def test_pre_stamp_predictions_are_attributed_from_promotion_history():
    """Without this every live row before 2026-08-25 is unattributable, so the
    serving arm has no matured outcome of its own for ~30 days after the stamp
    shipped — -I8219's unmeasurability left in place while looking fixed."""
    s3 = _FakeS3({
        "predictor/predictions/2026-08-26.json": _predictions("2026-08-26"),
    })
    pairs = ol._load_live_pairs("bkt", 400, s3_client=s3, promotion_history=_HISTORY)
    assert {p["champion_version_id"] for p in pairs} == {"v3.0-meta-2026-08-21-7d3d1cce"}
    assert {p["champion_version_id_source"] for p in pairs} == {ol.ATTR_PROMOTION_HISTORY}


def test_the_stamp_wins_over_the_history_when_they_disagree():
    """The artifact's own record of itself beats a reconstruction of it."""
    s3 = _FakeS3({
        "predictor/predictions/2026-08-26.json": _predictions(
            "2026-08-26", champion_version_id="STAMPED-WINS"),
    })
    pairs = ol._load_live_pairs("bkt", 400, s3_client=s3, promotion_history=_HISTORY)
    assert {p["champion_version_id"] for p in pairs} == {"STAMPED-WINS"}
    assert {p["champion_version_id_source"] for p in pairs} == {ol.ATTR_STAMPED}


def test_a_date_no_marker_precedes_is_unattributed_not_guessed():
    s3 = _FakeS3({
        "predictor/predictions/2026-07-01.json": _predictions("2026-07-01"),
    })
    pairs = ol._load_live_pairs("bkt", 400, s3_client=s3, promotion_history=_HISTORY)
    assert {p["champion_version_id"] for p in pairs} == {None}
    assert {p["champion_version_id_source"] for p in pairs} == {ol.ATTR_UNATTRIBUTED}


# ── the class fix: one row constructor, so a third loader cannot drift ──────

def test_both_prediction_loaders_emit_the_same_row_shape():
    """The defect was a copy-paste divergence between two loaders. Pin the
    shape so a third one cannot reintroduce it (engagement §5 — the fix must
    survive the class, not the instance)."""
    live_s3 = _FakeS3({
        "predictor/predictions/2026-08-26.json": _predictions(
            "2026-08-26", champion_version_id="V1"),
    })
    shadow_s3 = _FakeS3({
        "predictor/predictions_shadow/VSHADOW/2026-08-26.json": _predictions("2026-08-26"),
    })
    live = ol._load_live_pairs("bkt", 400, s3_client=live_s3, promotion_history=[])
    shadow = ol._load_shadow_pairs("bkt", "VSHADOW", 400, s3_client=shadow_s3)
    assert live and shadow
    assert set(live[0]) == set(shadow[0])
    # The shadow prefix — never the file's own stamp — names a shadow producer.
    assert shadow[0]["champion_version_id"] == "VSHADOW"
    assert shadow[0]["champion_version_id_source"] == ol.ATTR_SHADOW_PREFIX


def test_shadow_rows_ignore_the_files_live_champion_stamp():
    """policy §7.5: provenance true by construction. A shadow file carries the
    LIVE champion's id at write time; reading it would attribute every shadow
    row to the champion."""
    s3 = _FakeS3({
        "predictor/predictions_shadow/VSHADOW/2026-08-26.json": _predictions(
            "2026-08-26", champion_version_id="THE-LIVE-CHAMPION"),
    })
    rows = ol._load_shadow_pairs("bkt", "VSHADOW", 400, s3_client=s3)
    assert {r["champion_version_id"] for r in rows} == {"VSHADOW"}


# ── the detection blindness, which outranks the defect ─────────────────────

def test_attribution_coverage_separates_no_stamp_from_a_dropped_field():
    attributed = [
        {"date": "2026-08-26", "champion_version_id": "V1",
         "champion_version_id_source": ol.ATTR_STAMPED},
        {"date": "2026-08-26", "champion_version_id": "V1",
         "champion_version_id_source": ol.ATTR_STAMPED},
        {"date": "2026-08-25", "champion_version_id": None,
         "champion_version_id_source": ol.ATTR_UNATTRIBUTED},
    ]
    cov = ol.attribution_coverage(attributed)
    assert cov["n_pairs"] == 3 and cov["n_pairs_attributed"] == 2
    assert cov["n_dates"] == 2 and cov["n_dates_attributed"] == 1
    assert cov["by_source"] == {ol.ATTR_STAMPED: 2, ol.ATTR_UNATTRIBUTED: 1}
    assert cov["fully_unattributed"] is False

    none_of_it = [{"date": "2026-08-26", "champion_version_id": None,
                   "champion_version_id_source": ol.ATTR_UNATTRIBUTED}]
    assert ol.attribution_coverage(none_of_it)["fully_unattributed"] is True
    assert ol.attribution_coverage([])["fully_unattributed"] is False


def test_per_date_rank_ic_buckets_live_rows_by_version_not_none():
    """The consumer the arena reads. Pre-fix every live row landed in the None
    bucket, which `arena_model_slot.build_series` drops entirely."""
    pairs = []
    for i, tkr in enumerate(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]):
        pairs.append({"date": "2026-08-26", "ticker": tkr,
                      "predicted_alpha": 0.01 * i, "realized_alpha": 0.02 * i,
                      "champion_version_id": "V1",
                      "champion_version_id_source": ol.ATTR_STAMPED})
    out = ol.per_date_rank_ic_by_version(pairs)
    assert set(out) == {"V1"}
    assert out["V1"]["2026-08-26"] == pytest.approx(1.0)


# ── the arena consumer: the live record must reach the ladder ──────────────

def test_build_series_scores_the_champion_arm_from_live_predictions():
    """RED pre-fix. `build_series` maps version -> label -> arm; an unattributed
    live row has version None, maps to no arm, and is DISCARDED — so the M slot
    ranked its arms on shadow output alone while emitting a well-formed cycle
    (champion-challenger-policy §7.2, §11)."""
    from training import arena_model_slot as ams

    tickers = list("ABCDEFGHIJK")
    pairs = [
        {"date": "2026-08-26", "ticker": t,
         "predicted_alpha": 0.01 * i, "realized_alpha": 0.02 * i,
         "champion_version_id": "v3.0-meta-2026-08-21-7d3d1cce",
         "champion_version_id_source": ol.ATTR_STAMPED}
        for i, t in enumerate(tickers)
    ]
    attribution: dict = {}
    series = ams.build_series(
        "bkt",
        arm_id_by_label={"v3.0-meta": "arm-champ"},
        version_labels={"v3.0-meta-2026-08-21-7d3d1cce": "v3.0-meta"},
        pairs=pairs, attribution_out=attribution,
    )
    assert series["arm-champ"].scores["2026-08-26"] == pytest.approx(1.0)
    assert attribution["n_pairs_attributed"] == len(tickers)
    assert attribution["unclaimed_version_ids"] == []
    assert attribution["fully_unattributed"] is False


def test_build_series_reports_a_wholly_unattributed_live_record():
    """The guard for the defect, not the defect: a cycle scored on nothing must
    say so in its own artifact rather than reading as coverage (§7.4)."""
    from training import arena_model_slot as ams

    pairs = [
        {"date": "2026-08-26", "ticker": t,
         "predicted_alpha": 0.01 * i, "realized_alpha": 0.02 * i,
         "champion_version_id": None,
         "champion_version_id_source": ol.ATTR_UNATTRIBUTED}
        for i, t in enumerate(list("ABCDEFGHIJK"))
    ]
    attribution: dict = {}
    series = ams.build_series(
        "bkt", arm_id_by_label={"v3.0-meta": "arm-champ"},
        version_labels={}, pairs=pairs, attribution_out=attribution,
    )
    assert series["arm-champ"].scores == {}
    assert attribution["fully_unattributed"] is True
    assert attribution["n_versions_scored"] == 0
    assert attribution["unclaimed_version_ids"] == ["None"]
