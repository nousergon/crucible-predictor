"""Tests for the champion/challenger Phase 1 shadow runner (L4469).

Covers the safety-critical contract: no-op when disabled / dry-run / supplemental
/ no-challengers; clone reuses shared data + swaps the weights prefix; each
challenger is written to its own shadow key; the time-guard stops mid-run; and a
single-challenger failure never aborts the others (the live path is untouched).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from inference.pipeline import PipelineContext
from inference.stages import shadow_versions as sv


class _FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, *, Bucket, Key, Body, ContentType):
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body})


def _base_ctx(**over):
    ctx = PipelineContext()
    ctx.date_str = "2026-06-02"
    ctx.bucket = "bkt"
    ctx.dry_run = False
    ctx.start_ts = time.monotonic()  # else near_timeout() trips on start_ts=0
    ctx.soft_timeout_s = 780
    ctx.tickers = ["AAA", "BBB"]
    ctx.price_data = {"AAA": object(), "BBB": object()}
    ctx.macro = {"vix": 1.0}
    for k, v in over.items():
        setattr(ctx, k, v)
    return ctx


def test_clone_reuses_shared_and_swaps_prefix():
    ctx = _base_ctx()
    ctx.predictions = [{"ticker": "AAA"}]  # live results — must NOT carry over
    shadow = sv._clone_for_shadow(ctx, weights_prefix="predictor/registry/V/")
    assert shadow.weights_prefix_override == "predictor/registry/V/"
    assert shadow.tickers == ["AAA", "BBB"]
    assert shadow.price_data["AAA"] is ctx.price_data["AAA"]  # frames shared
    assert shadow.price_data is not ctx.price_data  # container copied
    assert shadow.predictions == []  # fresh result state


def test_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", False, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    sv.run(_base_ctx())
    assert fake.puts == []


def test_noop_in_dry_run(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    sv.run(_base_ctx(dry_run=True))
    assert fake.puts == []


def test_noop_on_supplemental_run(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    sv.run(_base_ctx(explicit_tickers=["AAA"]))
    assert fake.puts == []


def test_noop_when_no_challengers(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr("model.registry.list_versions", lambda *a, **k: [])
    sv.run(_base_ctx())
    assert fake.puts == []


def _patch_stages(monkeypatch, *, fail_on=None):
    """Patch load_model.run + run_inference.run; run_inference stamps preds."""
    import importlib

    lm = importlib.import_module("inference.stages.load_model")
    ri = importlib.import_module("inference.stages.run_inference")

    def _lm_run(ctx):
        if fail_on and ctx.weights_prefix_override and fail_on in ctx.weights_prefix_override:
            raise RuntimeError("boom")

    def _ri_run(ctx):
        ctx.predictions = [{"ticker": "AAA", "predicted_alpha": 0.01}]

    monkeypatch.setattr(lm, "run", _lm_run)
    monkeypatch.setattr(ri, "run", _ri_run)


def test_shadows_each_challenger(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_MAX_N", 3, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr(
        "model.registry.list_versions",
        lambda *a, **k: [{"version_id": "V1"}, {"version_id": "V2"}],
    )
    _patch_stages(monkeypatch)
    sv.run(_base_ctx())

    keys = sorted(p["Key"] for p in fake.puts)
    assert keys == [
        "predictor/predictions_shadow/V1/2026-06-02.json",
        "predictor/predictions_shadow/V2/2026-06-02.json",
    ]
    body = json.loads(fake.puts[0]["Body"])
    assert body["shadow"] is True and body["n_predictions"] == 1


def test_max_n_caps_challengers(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_MAX_N", 1, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr(
        "model.registry.list_versions",
        lambda *a, **k: [{"version_id": "V1"}, {"version_id": "V2"}],
    )
    _patch_stages(monkeypatch)
    sv.run(_base_ctx())
    assert len(fake.puts) == 1  # capped to max_n=1


def test_failure_continues_to_next_challenger(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_MAX_N", 3, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr(
        "model.registry.list_versions",
        lambda *a, **k: [{"version_id": "V1"}, {"version_id": "V2"}],
    )
    _patch_stages(monkeypatch, fail_on="V1")  # V1 errors in load_model
    sv.run(_base_ctx())
    keys = [p["Key"] for p in fake.puts]
    assert keys == ["predictor/predictions_shadow/V2/2026-06-02.json"]


def test_time_guard_stops_run(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_MAX_N", 3, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr(
        "model.registry.list_versions",
        lambda *a, **k: [{"version_id": "V1"}, {"version_id": "V2"}],
    )
    _patch_stages(monkeypatch)
    ctx = _base_ctx()
    monkeypatch.setattr(ctx, "near_timeout", lambda: True)  # already over budget
    sv.run(ctx)
    assert fake.puts == []  # nothing shadowed — guard fired before the first


# ── alpha-engine-config-I9336: rotation fixes the fixed-tail starvation ────
# The measured live signature: training emits the zoo specs in a fixed order
# every week, so `list_versions`' newest-first sort was effectively constant
# and a plain `[:MAX_N]` truncation always cut the SAME trailing two arms
# (sota-directional-combine, the champion-arch candidate) — 16 days of zero
# shadow coverage for one of them, structurally unrecoverable under the old
# code.

_LIVE_EMISSION_ORDER = [
    {"version_id": "spec-horizon-60d"},
    {"version_id": "spec-horizon-90d"},
    {"version_id": "spec-residual-momentum"},
    {"version_id": "spec-sota-combine"},
    {"version_id": "spec-champion-arch"},
]


class TestSelectChallengersForCycle:
    """_select_challengers_for_cycle — pure function, no S3."""

    def test_guard_red_without_the_fix_same_two_arms_starved_every_day(self):
        """Champion-challenger policy 7.4 guard-red-before-fix: the OLD plain
        positional truncation starves the SAME two arms on every consecutive
        date — this is the measured live defect (16 days of zero coverage)."""
        max_n = 3
        for date_str in (
            "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28",
        ):
            old_selection = {v["version_id"] for v in _LIVE_EMISSION_ORDER[:max_n]}
            assert old_selection == {
                "spec-horizon-60d", "spec-horizon-90d", "spec-residual-momentum",
            }
            assert "spec-sota-combine" not in old_selection  # RED, every single day
            assert "spec-champion-arch" not in old_selection  # RED, every single day

    def test_every_registered_arm_shadowed_within_one_rotation_cycle(self):
        """Deliverable 3's regression test: with the specs in their live
        emission order, every registered challenger receives a shadow slot
        within one full rotation cycle (``ceil(n / max_n)`` consecutive
        trading days — 2, here). MUST fail against the pre-fix
        `challengers[:max_n]` (proven starved above)."""
        max_n = 3
        seen: set[str] = set()
        for date_str in ("2026-08-07", "2026-08-10"):  # ceil(5/3) = 2 trading days (Fri->Mon)
            for v in sv._select_challengers_for_cycle(_LIVE_EMISSION_ORDER, max_n, date_str):
                seen.add(v["version_id"])
        assert seen == {v["version_id"] for v in _LIVE_EMISSION_ORDER}

    def test_deterministic_per_date(self):
        a = sv._select_challengers_for_cycle(_LIVE_EMISSION_ORDER, 3, "2026-08-31")
        b = sv._select_challengers_for_cycle(_LIVE_EMISSION_ORDER, 3, "2026-08-31")
        assert a == b

    def test_under_the_cap_returns_everything_unchanged(self):
        challengers = [{"version_id": "V1"}, {"version_id": "V2"}]
        assert sv._select_challengers_for_cycle(challengers, 3, "2026-08-31") == challengers

    def test_no_challengers_returns_empty(self):
        assert sv._select_challengers_for_cycle([], 3, "2026-08-31") == []

    def test_bad_date_string_falls_back_to_offset_zero(self):
        result = sv._select_challengers_for_cycle(_LIVE_EMISSION_ORDER, 3, "not-a-date")
        ordered = sorted(_LIVE_EMISSION_ORDER, key=lambda v: v["version_id"])
        assert result == ordered[:3]


def test_run_rotates_across_two_dates_end_to_end(monkeypatch):
    """Integration: sv.run(), called on two consecutive dates with the same 5
    registered challengers, shadows every one of them across the two runs —
    not just the same 3."""
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_MAX_N", 3, raising=False)
    monkeypatch.setattr(
        "model.registry.list_versions", lambda *a, **k: _LIVE_EMISSION_ORDER,
    )
    _patch_stages(monkeypatch)

    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    sv.run(_base_ctx(date_str="2026-08-07"))
    sv.run(_base_ctx(date_str="2026-08-10"))

    shadowed_versions = {p["Key"].split("/")[2] for p in fake.puts}
    assert shadowed_versions == {v["version_id"] for v in _LIVE_EMISSION_ORDER}
