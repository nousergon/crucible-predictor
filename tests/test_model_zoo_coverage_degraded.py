"""config#2882 — model_zoo.select_winner threads a candidate's ArcticDB
read-coverage (data_coverage_degraded / arctic_coverage_ratio, written by
meta_trainer.run_meta_training into the registry manifest) onto that
candidate's leaderboard entry, so a challenger that wins weekly promotion
after training on a partially-crippled dataset is visible in the leaderboard/
CPCV artifact BEFORE a human or select_winner trusts the win — not silently
absent from the promoted artifact, per the issue's acceptance criteria.
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
        self.puts = {}

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        body = json.dumps(self.objects[Key]).encode()
        return {"Body": io.BytesIO(body), "ContentType": "application/json"}

    def put_object(self, Bucket, Key, Body, ContentType="application/json"):  # noqa: N803
        self.puts[Key] = Body


def _mk_manifest(forward_days, cpcv_ic, gate_pass, *, arctic_coverage=None):
    data_coverage_degraded = (
        bool(arctic_coverage["coverage_ratio"] < cfg.ARCTIC_DEGRADED_COVERAGE_RATIO)
        if arctic_coverage is not None else None
    )
    return {
        "forward_days": forward_days,
        "meta_model_oos_ic_cpcv": (
            {"mean_ic": cpcv_ic} if cpcv_ic is not None else {"status": "error"}
        ),
        "meta_model_promotion_stats": {
            "downside": {"passes_downside_gate": gate_pass},
            "overfit": {"passes_overfit_gate": gate_pass},
        },
        "arctic_coverage": arctic_coverage,
        "data_coverage_degraded": data_coverage_degraded,
    }


def test_select_winner_flags_degraded_coverage_winner(monkeypatch):
    """The exact issue scenario: a challenger trained on a partially-starved
    ArcticDB read still self-reports a plausible CPCV IC and wins the
    rotation. The winner's leaderboard entry must carry
    data_coverage_degraded=True so the promotion isn't silent about it."""
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    degraded_coverage = {
        "n_written": 850, "n_expected": 900, "n_failed": 50,
        "coverage_ratio": 850 / 900, "failed_universe": [], "failed_macro": [],
    }
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),  # stale serving
        "predictor/registry/arch-v/manifest.json": _mk_manifest(21, 0.10, True),
        # This challenger trained on degraded coverage but still beats the
        # champion-arch baseline on its OWN (possibly-inflated) self-reported
        # CPCV IC — the exact silent-promotion scenario from the issue.
        "predictor/registry/crippled-v/manifest.json": _mk_manifest(
            21, 0.20, True, arctic_coverage=degraded_coverage,
        ),
    })
    trained = [
        {"spec_id": "champion-arch", "version_id": "arch-v", "model_version": "v3.0-meta"},
        {"spec_id": "crippled", "version_id": "crippled-v", "model_version": "spec-crippled"},
    ]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)

    assert board["winner_version_id"] == "crippled-v"
    win = next(c for c in board["candidates"] if c["spec_id"] == "crippled")
    assert win["data_coverage_degraded"] is True
    assert win["arctic_coverage_ratio"] == pytest.approx(850 / 900)


def test_select_winner_full_coverage_not_flagged(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    full_coverage = {
        "n_written": 900, "n_expected": 900, "n_failed": 0,
        "coverage_ratio": 1.0, "failed_universe": [], "failed_macro": [],
    }
    s3 = _FakeS3({
        "predictor/registry/healthy-v/manifest.json": _mk_manifest(
            21, 0.15, True, arctic_coverage=full_coverage,
        ),
    })
    trained = [{"spec_id": "healthy", "version_id": "healthy-v", "model_version": "spec-healthy"}]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)
    c = board["candidates"][0]
    assert c["data_coverage_degraded"] is False
    assert c["arctic_coverage_ratio"] == 1.0


def test_select_winner_missing_coverage_field_degrades_to_none(monkeypatch):
    """A manifest written before config#2882 (no arctic_coverage/
    data_coverage_degraded keys at all) must not blow up select_winner — the
    candidate's coverage fields are simply None, not a crash."""
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    s3 = _FakeS3({
        "predictor/registry/legacy-v/manifest.json": _mk_manifest(21, 0.15, True),
    })
    trained = [{"spec_id": "legacy", "version_id": "legacy-v", "model_version": "spec-legacy"}]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)
    c = board["candidates"][0]
    assert c["data_coverage_degraded"] is None
    assert c["arctic_coverage_ratio"] is None


def test_alert_promotion_message_includes_degraded_coverage_warning(monkeypatch):
    """config#2882 — the promotion alert (the completion-notification
    chokepoint) must say so IN THE MESSAGE when the promoted winner trained
    on degraded coverage; this is the 'nothing in the ... completion email
    indicates' gap the issue calls out."""
    leaderboard = {
        "champion": {"cpcv_mean_ic": 0.10},
        "margin": 0.01,
        "promote_min_ic": 0.0,
        "candidates": [
            {
                "spec_id": "crippled", "version_id": "crippled-v",
                "cpcv_mean_ic": 0.20, "dsr_selection": None, "dsr_training": 0.9,
                "dsr_gate_pass": True,
                "data_coverage_degraded": True,
                "arctic_coverage_ratio": 850 / 900,
            },
        ],
    }
    captured = {}

    def _fake_publish(*, message, **kwargs):
        captured["message"] = message

    monkeypatch.setattr("ops_alerts.publish_ops_alert", _fake_publish, raising=False)
    import sys
    import types
    fake_mod = types.ModuleType("ops_alerts")
    fake_mod.publish_ops_alert = _fake_publish
    monkeypatch.setitem(sys.modules, "ops_alerts", fake_mod)

    mz._alert_promotion("bkt", "2026-07-18", leaderboard, "crippled-v", "prior-v")

    assert "DEGRADED" in captured["message"]
    assert "0.944" in captured["message"] or str(850 / 900) in captured["message"]
