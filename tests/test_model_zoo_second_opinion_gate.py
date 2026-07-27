"""config#2889 (Brian's 2026-07-18 Decision Queue Option-B ruling) — the
independent second-party IC recomputation must be SURFACED on every
candidate's leaderboard entry always, and BLOCK promotion only when
MODEL_ZOO_SECOND_OPINION_GATE_ENFORCE is True (observe-first rollout, same
pattern as MODEL_ZOO_AUTO_PROMOTE_WINNER).
"""
from __future__ import annotations

import io
import json

import pytest

import config as cfg
from training import model_zoo as mz


class _FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        body = json.dumps(self.objects[Key]).encode()
        return {"Body": io.BytesIO(body), "ContentType": "application/json"}

    def put_object(self, Bucket, Key, Body, ContentType="application/json"):  # noqa: N803
        pass


def _mk_manifest(forward_days, cpcv_ic, *, second_opinion=None):
    return {
        "forward_days": forward_days,
        "meta_model_oos_ic_cpcv": {
            "mean_ic": cpcv_ic,
            "second_opinion": second_opinion or {},
        },
        "meta_model_promotion_stats": {
            "downside": {"passes_downside_gate": True},
            "overfit": {"passes_overfit_gate": True},
        },
    }


def test_diverging_second_opinion_surfaced_but_not_blocking_by_default(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_SECOND_OPINION_GATE_ENFORCE", False, raising=False)
    s3 = _FakeS3({
        "predictor/registry/arch-v/manifest.json": _mk_manifest(21, 0.10),
        "predictor/registry/challenger-v/manifest.json": _mk_manifest(
            21, 0.30,
            second_opinion={"status": "ok", "second_opinion_ic": 0.01, "match_rate": 0.9},
        ),
    })
    trained = [
        {"spec_id": "champion-arch", "version_id": "arch-v", "model_version": "v3.0-meta"},
        {"spec_id": "challenger", "version_id": "challenger-v", "model_version": "spec-challenger"},
    ]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)

    win = next(c for c in board["candidates"] if c["spec_id"] == "challenger")
    assert win["second_opinion_divergence"] is True
    assert win["second_opinion_gate_enforced"] is False
    # Observe-first: divergence is surfaced, but the candidate still wins.
    assert board["winner_version_id"] == "challenger-v"
    assert win["eligible"] is True
    assert win["reason"] == "eligible"


def test_diverging_second_opinion_blocks_when_enforced(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_SECOND_OPINION_GATE_ENFORCE", True, raising=False)
    s3 = _FakeS3({
        "predictor/registry/arch-v/manifest.json": _mk_manifest(21, 0.10),
        "predictor/registry/challenger-v/manifest.json": _mk_manifest(
            21, 0.30,
            second_opinion={"status": "ok", "second_opinion_ic": 0.01, "match_rate": 0.9},
        ),
    })
    trained = [
        {"spec_id": "champion-arch", "version_id": "arch-v", "model_version": "v3.0-meta"},
        {"spec_id": "challenger", "version_id": "challenger-v", "model_version": "spec-challenger"},
    ]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)

    win = next(c for c in board["candidates"] if c["spec_id"] == "challenger")
    assert win["second_opinion_divergence"] is True
    assert win["eligible"] is False
    assert win["reason"] == "second_opinion_diverges"
    # No eligible challenger beats champion-arch -> no winner promoted from
    # this candidate (champion-arch itself never wins as a "challenger").
    assert board["winner_version_id"] is None


def test_corroborating_second_opinion_never_blocks_even_when_enforced(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_SECOND_OPINION_GATE_ENFORCE", True, raising=False)
    s3 = _FakeS3({
        "predictor/registry/arch-v/manifest.json": _mk_manifest(21, 0.10),
        "predictor/registry/challenger-v/manifest.json": _mk_manifest(
            21, 0.30,
            second_opinion={"status": "ok", "second_opinion_ic": 0.28, "match_rate": 0.9},
        ),
    })
    trained = [
        {"spec_id": "champion-arch", "version_id": "arch-v", "model_version": "v3.0-meta"},
        {"spec_id": "challenger", "version_id": "challenger-v", "model_version": "spec-challenger"},
    ]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)

    win = next(c for c in board["candidates"] if c["spec_id"] == "challenger")
    assert win["second_opinion_divergence"] is False
    assert win["eligible"] is True
    assert board["winner_version_id"] == "challenger-v"


def test_missing_second_opinion_on_legacy_manifest_never_blocks(monkeypatch):
    """A manifest written before config#2889 has no `second_opinion` key at
    all — must degrade gracefully (unavailable, non-blocking), not crash."""
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_SECOND_OPINION_GATE_ENFORCE", True, raising=False)
    s3 = _FakeS3({
        "predictor/registry/legacy-v/manifest.json": _mk_manifest(21, 0.15),
    })
    trained = [{"spec_id": "legacy", "version_id": "legacy-v", "model_version": "spec-legacy"}]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)
    c = board["candidates"][0]
    assert c["second_opinion_ic"] is None
    assert c["second_opinion_divergence"] is False
