"""Operator pointer moves are part of the champion history.

alpha-engine-config-I11479. The rehearsal of 2026-09-23 logged seven dates
where the prediction stamp and the promotion history disagreed:

    predictions/2026-09-10..11 stamped v3.0-meta-2026-08-14-119e069b,
        history resolved v3.0-meta-2026-09-04-cc3271ea
    predictions/2026-09-14..18 stamped v3.0-meta-2026-08-14-119e069b,
        history resolved spec-sota-combine-2026-09-11-753cfbed

Measured live: the 09-04 promotion served 09-08 and 09-09 (both stamped
cc3271ea), then both promotions were rolled back to the 08-14 champion by hand
(the live manifest was last written 2026-09-12 15:45Z, after the 09-11
rotation's 12:08Z marker). The stamps were right. The history had no record of
either rollback, because a move through ``python -m model.registry --promote``
wrote nothing to it.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.observe_leaderboard as ol  # noqa: E402
import model.registry as reg  # noqa: E402

CHAMP_0814 = "v3.0-meta-2026-08-14-119e069b"
PROMO_0904 = "v3.0-meta-2026-09-04-cc3271ea"
PROMO_0911 = "spec-sota-combine-2026-09-11-753cfbed"

# The live stamps, read from predictions/{date}.json on 2026-09-24.
LIVE_STAMPS = {
    "2026-09-08": PROMO_0904, "2026-09-09": PROMO_0904,
    "2026-09-10": CHAMP_0814, "2026-09-11": CHAMP_0814,
    "2026-09-14": CHAMP_0814, "2026-09-15": CHAMP_0814,
    "2026-09-16": CHAMP_0814, "2026-09-17": CHAMP_0814,
    "2026-09-18": CHAMP_0814, "2026-09-21": CHAMP_0814,
    "2026-09-22": CHAMP_0814, "2026-09-23": CHAMP_0814,
}


class _FakeS3:
    def __init__(self, objects: dict | None = None):
        self.objects: dict[str, bytes] = {
            k: json.dumps(v).encode() for k, v in (objects or {}).items()
        }

    def put_object(self, Bucket, Key, Body, **_kw):  # noqa: N803
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        body = self.objects[Key]

        class _B:
            @staticmethod
            def read():
                return body

        return {"Body": _B()}

    def get_paginator(self, _name):
        objects = self.objects

        class _P:
            @staticmethod
            def paginate(Bucket, Prefix):  # noqa: N803
                yield {"Contents": [{"Key": k} for k in sorted(objects) if k.startswith(Prefix)]}

        return _P()


def _rotation_marker(run_date, after, written):
    return {"run_date": run_date, "mode": "cutover",
            "champion_version_id_after": after, "written_at_utc": written}


def _live_rotation_markers() -> dict:
    """The four rotation markers that exist live for September."""
    p = "predictor/model_zoo/promotions/"
    return {
        f"{p}2026-08-28.json": _rotation_marker("2026-08-28", CHAMP_0814, "2026-08-29T11:53:56+00:00"),
        f"{p}2026-09-04.json": _rotation_marker("2026-09-04", PROMO_0904, "2026-09-05T19:06:04+00:00"),
        f"{p}2026-09-11.json": _rotation_marker("2026-09-11", PROMO_0911, "2026-09-12T12:08:26+00:00"),
        f"{p}2026-09-18.json": _rotation_marker("2026-09-18", CHAMP_0814, "2026-09-19T20:38:03+00:00"),
    }


def _disagreements(history) -> dict:
    return {
        d: ol.champion_serving_on(d, history)
        for d, stamp in LIVE_STAMPS.items()
        if ol.champion_serving_on(d, history) != stamp
    }


def test_without_the_rollback_records_the_history_disagrees_on_the_measured_dates():
    """The rehearsal's finding, reproduced from the live markers alone."""
    s3 = _FakeS3(_live_rotation_markers())
    bad = _disagreements(ol.load_promotion_history("bkt", s3_client=s3))
    assert bad == {
        "2026-09-10": PROMO_0904, "2026-09-11": PROMO_0904,
        "2026-09-14": PROMO_0911, "2026-09-15": PROMO_0911,
        "2026-09-16": PROMO_0911, "2026-09-17": PROMO_0911,
        "2026-09-18": PROMO_0911,
    }


def test_recorded_rollbacks_make_the_history_agree_with_every_stamp():
    s3 = _FakeS3(_live_rotation_markers())
    reg.record_operator_promotion(
        s3, "bkt", CHAMP_0814, run_date="2026-09-09",
        reason="rollback of the 09-04 promotion (commitment freeze)",
        now=datetime(2026, 9, 9, 22, 0, tzinfo=timezone.utc),
    )
    # Same run_date as the 09-11 rotation marker, written three hours after it.
    reg.record_operator_promotion(
        s3, "bkt", CHAMP_0814, run_date="2026-09-11",
        reason="rollback of the 09-11 promotion (commitment freeze)",
        now=datetime(2026, 9, 12, 15, 45, 9, tzinfo=timezone.utc),
    )
    history = ol.load_promotion_history("bkt", s3_client=s3)
    assert _disagreements(history) == {}


def test_a_same_day_rollback_orders_after_the_rotation_it_undoes():
    s3 = _FakeS3(_live_rotation_markers())
    reg.record_operator_promotion(
        s3, "bkt", CHAMP_0814, run_date="2026-09-11", reason="rollback",
        now=datetime(2026, 9, 12, 15, 45, 9, tzinfo=timezone.utc),
    )
    same_day = [r for r in ol.load_promotion_history("bkt", s3_client=s3)
                if r["run_date"] == "2026-09-11"]
    assert [r["champion_version_id"] for r in same_day] == [PROMO_0911, CHAMP_0814]


def test_operator_markers_never_touch_the_rotations_dated_key():
    s3 = _FakeS3()
    key = reg.record_operator_promotion(
        s3, "bkt", CHAMP_0814, run_date="2026-09-11", reason="rollback",
        now=datetime(2026, 9, 12, 15, 45, 9, tzinfo=timezone.utc),
    )
    assert key == "predictor/model_zoo/promotions/operator/2026-09-11T154509Z.json"
    marker = json.loads(s3.objects[key])
    assert marker["champion_version_id_after"] == CHAMP_0814
    assert marker["mode"] == "operator"
    assert "predictor/model_zoo/promotions/2026-09-11.json" not in s3.objects


@pytest.mark.parametrize("kw", [
    {"run_date": "2026-09-11", "reason": ""},
    {"run_date": "", "reason": "rollback"},
])
def test_an_unplaceable_or_unexplained_record_is_refused(kw):
    with pytest.raises(reg.RegistryError):
        reg.record_operator_promotion(_FakeS3(), "bkt", CHAMP_0814, **kw)


# ── the CLI path a hand rollback takes ──────────────────────────────────────


def _run_cli(monkeypatch, argv, s3):
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: s3)
    monkeypatch.setattr(sys, "argv", ["model.registry", *argv])
    reg._cli()


def test_cli_promote_records_the_move_in_the_history(monkeypatch):
    s3 = _FakeS3()
    promoted = []
    monkeypatch.setattr(
        reg, "promote_to_champion",
        lambda _s3, _b, vid: promoted.append(vid) or {
            "version_id": vid, "files_copied": ["manifest.json"], "live_prefix": "p/"},
    )
    monkeypatch.setattr(reg, "_current_champion_version_id", lambda _s3, _b: PROMO_0911)
    monkeypatch.setattr(reg, "_operator_run_date", lambda now=None: "2026-09-11")
    _run_cli(monkeypatch, ["--bucket", "bkt", "--promote", CHAMP_0814,
                           "--reason", "commitment freeze rollback"], s3)
    assert promoted == [CHAMP_0814]
    keys = [k for k in s3.objects if k.startswith(reg.OPERATOR_PROMOTIONS_PREFIX)]
    assert len(keys) == 1
    marker = json.loads(s3.objects[keys[0]])
    assert marker["run_date"] == "2026-09-11"
    assert marker["prior_champion_version_id"] == PROMO_0911
    assert marker["champion_version_id_after"] == CHAMP_0814


def test_cli_promote_without_a_reason_moves_nothing(monkeypatch):
    promoted = []
    monkeypatch.setattr(reg, "promote_to_champion", lambda *a, **k: promoted.append(a))
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, ["--bucket", "bkt", "--promote", CHAMP_0814], _FakeS3())
    assert promoted == []


def test_cli_record_promotion_backfills_without_copying(monkeypatch):
    s3 = _FakeS3()
    monkeypatch.setattr(
        reg, "promote_to_champion",
        lambda *a, **k: pytest.fail("a backfill must not move the pointer"),
    )
    _run_cli(monkeypatch, ["--bucket", "bkt", "--record-promotion", CHAMP_0814,
                           "--run-date", "2026-09-09", "--reason", "backfill"], s3)
    (key,) = s3.objects
    assert key.startswith(f"{reg.OPERATOR_PROMOTIONS_PREFIX}2026-09-09T")


def test_operator_run_date_is_the_last_closed_session():
    # Thursday 2026-09-10, 06:00Z: before that day's preopen, so the move
    # applies to 09-10's predictions and must carry run_date 09-09.
    assert reg._operator_run_date(datetime(2026, 9, 10, 6, 0, tzinfo=timezone.utc)) == "2026-09-09"
    # Saturday 2026-09-12: last closed session is Friday 09-11.
    assert reg._operator_run_date(datetime(2026, 9, 12, 15, 45, tzinfo=timezone.utc)) == "2026-09-11"
