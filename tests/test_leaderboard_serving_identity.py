"""The leaderboard must name the model that is SERVING, not its predecessor.

alpha-engine-config-I9260 · parent -I9255.

WHY THIS FILE EXISTS. ``predictor/model_zoo/leaderboard/2026-08-28.json``
carried, inside one block:

    "incumbent_version_id": "v3.0-meta-2026-08-14-119e069b"
    "served_version":       "v3.0-meta-2026-08-12-7d0d9328"
    "served_date":          "2026-08-12"

Two fields disagreeing about what is serving, and the wrong one is the
human-readable one. Verified against S3: the live pointer
``predictor/weights/meta/manifest.json`` names 08-14, and
``predictor/weights/meta/meta_model.pkl`` is ETag-identical to
``predictor/registry/v3.0-meta-2026-08-14-119e069b/meta_model.pkl``. So 08-14 was
serving and the veto's dispersion reference — which uses ``incumbent_version_id``
— was correct throughout. This was a REPORTING defect, not a veto defect.

Root cause: a registry bundle manifest's OWN ``served_version`` is a PREDECESSOR
STAMP, recording who was serving when that bundle trained. The chain on the real
artifacts is unambiguous:

    registry/v3.0-meta-2026-08-14-119e069b → served_version 2026-08-12-7d0d9328
    registry/v3.0-meta-2026-08-12-7d0d9328 → served_version 2026-08-07-22830c0d

``select_winner`` read that nested field as the current answer, chaining one hop
backwards. champion-challenger-policy §7.5 — provenance must be true by
construction.

RED pre-fix (§7.4): ``serving_version`` was
``serving_manifest.get("served_version") or ...``, so the assertion below
reported PREDECESSOR_VID.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training import model_zoo as mz
from training import served_slice_dispersion as ssd
from tests.test_model_zoo import _FakeS3
from tests.test_promotion_served_slice_veto import _manifest

SERVING_VID = "v3.0-meta-2026-08-14-119e069b"
PREDECESSOR_VID = "v3.0-meta-2026-08-12-7d0d9328"
CANDIDATE_VID = "v3.0-meta-2026-08-28-01cf7e1a"


def _s3():
    """The real 2026-08-28 shape: the serving bundle stamps its PREDECESSOR."""
    serving = _manifest(mean_ic=0.131924)
    # The predecessor stamp that made the leaderboard lie.
    serving["served_version"] = PREDECESSOR_VID
    serving["served_date"] = "2026-08-12"
    serving["date"] = "2026-08-14"
    return _FakeS3({
        cfg.META_MANIFEST_KEY: {
            "forward_days": 21,
            "served_version": SERVING_VID,
            "served_date": "2026-08-14",
        },
        f"predictor/registry/{SERVING_VID}/manifest.json": serving,
        f"predictor/registry/{CANDIDATE_VID}/manifest.json": _manifest(mean_ic=0.12303),
    })


@pytest.fixture(autouse=True)
def _stub_served_slice(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(
        ssd, "served_slice_metrics",
        lambda s3, bucket, vids, **kw: {
            "status": "uncomputable",
            "reason": "panel not provided in this test",
            "panel_key": None, "n_panel_rows": None,
            "top_n": 30, "min_confidence": 0.30,
            "metrics": {}, "errors": {},
        },
    )


def _board():
    return mz.select_winner(
        _s3(), "bkt",
        trained=[{"spec_id": "champion-arch", "version_id": CANDIDATE_VID,
                  "model_version": "v3.0-meta"}],
        margin=0.0, date_str="2026-08-28",
    )


def test_served_version_is_the_live_pointer_not_the_bundles_predecessor_stamp():
    serving = _board()["serving_champion"]
    assert serving["incumbent_version_id"] == SERVING_VID
    assert serving["served_version"] == SERVING_VID, (
        "the leaderboard reported the PREDECESSOR of the serving model — the "
        "alpha-engine-config-I9260 defect"
    )
    assert serving["served_version"] != PREDECESSOR_VID
    assert serving["served_date"] == "2026-08-14"


def test_the_two_identity_fields_can_never_disagree():
    serving = _board()["serving_champion"]
    assert serving["served_version"] == serving["incumbent_version_id"]


def test_the_resolver_returns_the_live_manifest_so_callers_need_not_re_read_it():
    vid, bundle, live = mz._resolve_incumbent_from_bundle(_s3(), "bkt")
    assert vid == SERVING_VID
    assert live["served_version"] == SERVING_VID
    # The bundle's own stamp is still there — it is simply not the answer.
    assert bundle["served_version"] == PREDECESSOR_VID
