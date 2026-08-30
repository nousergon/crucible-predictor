"""Shared ``arena.run_slot`` mocks for model-zoo integration tests (I9319).

``select_and_finalize`` delegates the pointer decision to
``training.arena_model_slot.run_slot``, which needs a full S3 arena state.
Integration tests that exercise selection/promotion mock the adapter here
instead of standing up predictions, pairs, and register history.
"""
from __future__ import annotations

from nousergon_lib.arena.engine import (
    ARENA_CYCLE_SCHEMA_VERSION,
    ArenaCycle,
    PointerDecision,
)


def mock_arena_run_slot(
    monkeypatch,
    *,
    pointer_version_id: str | None,
    moved: bool = False,
    status: str | None = None,
):
    """Patch ``arena_model_slot.run_slot`` to return a controlled pointer."""
    from training import arena_model_slot as arena_mod

    def _run_slot(s3, bucket, **kwargs):
        as_of = kwargs.get("as_of", "")
        _status = status or ("decided" if pointer_version_id else "held")
        decision = PointerDecision(
            slot=arena_mod.SLOT,
            as_of=as_of,
            incumbent="M:test:inc",
            champion="M:test:champ",
            moved=moved,
            status=_status,
            reason="test mock",
            comparisons=(),
            ineligible={},
        )
        cycle = ArenaCycle(
            schema_version=ARENA_CYCLE_SCHEMA_VERSION,
            slot=arena_mod.SLOT,
            slot_kind=arena_mod.SLOT_KIND,
            benchmark=arena_mod.BENCHMARK,
            as_of=as_of,
            ladders=(),
            ranking=None,
            decision=decision,
            retirements=(),
            scored_arms=(),
            active_arms=(),
        )
        return {
            "cycle": cycle,
            "doc": {
                **cycle.to_dict(),
                "arena_config": {"ranking_statistic": "realized-market-relative-rank-ic"},
            },
            "register": None,
            "arm_id_by_label": {},
            "label_by_arm": {},
            "bundle_by_arm": {},
            "pointer_arm": decision.champion,
            "pointer_version_id": pointer_version_id,
            "keys": [f"arena/model/{as_of}.json", "arena/model/latest.json"],
        }

    monkeypatch.setattr(arena_mod, "run_slot", _run_slot)
    return _run_slot


def mock_model_zoo_post_select_reads(monkeypatch):
    """Stub S3-heavy reads that run after arena selection in integration tests."""
    from training import model_zoo as mz

    monkeypatch.setattr(mz, "_latest_attributed_slot_read", lambda *a, **k: {})
    monkeypatch.setattr(mz, "_read_monitor_history", lambda *a, **k: [])
