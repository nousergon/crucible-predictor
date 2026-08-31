"""L4571 — the base champion-architecture competes in the model-zoo pool, and
an auto-promotion announces itself (Telegram/SNS) with an exact revert command.

Why: auto-promote is meaningless unless the base champion-arch retrain (which
runs first on Saturday and registers as a `stage="challenger"` `v3.0-meta`
version) competes against the rotated variants in the SAME pool — else
`select_winner` only ever ranks one variant against a stale serving manifest and
a fresh-data retrain can never win. And because the realized-edge auto-demote
(L4539) is not live, revert is MANUAL — so a promotion MUST fire a loud alert
carrying the revert command, or "revert if it misbehaves" is not actionable.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training import model_zoo as mz
from tests.test_model_zoo import _FakeS3, _mk_manifest, _SPECS


def _stage_versions(challengers, champions):
    """A `list_versions` stub that honours the `stage` filter."""
    return lambda s3c, b, stage=None: {
        "challenger": challengers, "champion": champions
    }.get(stage, challengers + champions)


# ── _resolve_base_champion_version ─────────────────────────────────────────


class TestResolveBaseChampionVersion:
    def test_picks_todays_v3_meta_challenger(self, monkeypatch):
        import model.registry as reg
        monkeypatch.setattr(cfg, "MODEL_VERSION_LABEL", "v3.0-meta", raising=False)
        monkeypatch.setattr(reg, "list_versions", _stage_versions([
            {"version_id": "base-old", "model_version": "v3.0-meta", "date": "2026-06-06"},
            {"version_id": "base-today", "model_version": "v3.0-meta", "date": "2026-06-13"},
            {"version_id": "resid-v", "model_version": "spec-resid", "date": "2026-06-13"},
        ], []))
        out = mz._resolve_base_champion_version(None, "bkt", "2026-06-13")
        assert out == {"spec_id": "champion-arch", "model_version": "v3.0-meta",
                       "version_id": "base-today"}

    def test_none_when_no_base_registered(self, monkeypatch):
        import model.registry as reg
        monkeypatch.setattr(reg, "list_versions", _stage_versions([
            {"version_id": "resid-v", "model_version": "spec-resid", "date": "2026-06-13"},
        ], []))
        assert mz._resolve_base_champion_version(None, "bkt", "2026-06-13") is None

    def test_read_failure_is_none(self, monkeypatch):
        import model.registry as reg
        def _boom(*a, **k):
            raise RuntimeError("registry down")
        monkeypatch.setattr(reg, "list_versions", _boom)
        assert mz._resolve_base_champion_version(None, "bkt", "2026-06-13") is None


# ── base competes in the pool + alerts on promote/observe ──────────────────


def _pool_fixture(monkeypatch, *, auto_promote, base_ic, variant_ic,
                  arena_pointer=None, arena_moved=False):
    """Champion (serving) CPCV 0.10; a base-arch retrain + one variant compete."""
    from tests.arena_test_helpers import mock_arena_run_slot, mock_model_zoo_post_select_reads

    import model.registry as reg
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_VERSION_LABEL", "v3.0-meta", raising=False)
    s3 = _FakeS3({
        # I9018: the live manifest is a POINTER; _FakeS3 materialises the matching
        # registry bundle, which is where the incumbent CPCV is actually read from.
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),          # serving champion
        cfg.META_FEATURE_LIST_KEY: {"features": ["a"]},
        "predictor/registry/base-today/manifest.json": _mk_manifest(21, base_ic, True),
        "predictor/registry/resid-v/manifest.json": _mk_manifest(21, variant_ic, True),
    })
    monkeypatch.setattr(reg, "list_versions", _stage_versions(
        challengers=[
            {"version_id": "base-today", "model_version": "v3.0-meta", "date": "2026-06-13"},
            {"version_id": "resid-v", "model_version": "spec-resid", "date": "2026-06-13"},
        ],
        champions=[{"version_id": "old-champ-v", "model_version": "v3.0-meta", "date": "2026-06-06"}],
    ))
    promotes = []
    monkeypatch.setattr(reg, "promote_to_champion",
                        lambda s3c, b, vid, **k: promotes.append(vid))
    alerts_sent = []
    import krepis.alerts as _alerts
    monkeypatch.setattr(_alerts, "publish",
                        lambda **kw: alerts_sent.append(kw) or None)

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        return {"status": "ok"}

    margin = float(getattr(cfg, "MODEL_ZOO_PROMOTE_MARGIN", 0.01))
    if arena_pointer is None:
        if variant_ic >= base_ic + margin:
            arena_pointer, arena_moved = "resid-v", True
        elif base_ic >= 0.10 + margin:
            arena_pointer, arena_moved = "base-today", False
        else:
            arena_pointer, arena_moved = "old-champ-v", False
    mock_arena_run_slot(monkeypatch, pointer_version_id=arena_pointer, moved=arena_moved)
    mock_model_zoo_post_select_reads(monkeypatch)

    board = mz.run_rotation_and_select(
        "bkt", budget=5, specs=_SPECS, train_fn=_fake_train,
        registered_versions=[], s3=s3, date_str="2026-06-13",
        auto_promote_winner=auto_promote,
    )
    return board, promotes, alerts_sent


class TestBaseInPool:
    def test_base_arch_is_never_a_challenger_winner_but_CAN_refresh(self, monkeypatch):
        """alpha-engine-config-I9061 — the refresh is back, behind a real test.

        #679(ii) still holds: champion-arch (0.30) is the vintage-consistent
        BASELINE and the variant (0.15) does not clear baseline+margin, so no
        CHALLENGER wins and champion-arch is never `winner_version_id` — it
        cannot beat itself.

        What I9024 s1 deleted, and what Brian's 2026-08-28 ruling restores, is
        the SEPARATE refresh path: champion-arch may take the serving slot when
        it beats the SERVING incumbent (0.10 here, read from that incumbent's own
        immutable registry bundle) by margin. The old version of this test
        asserted `champion_arch_refresh_version_id not in board`, which is the
        state that froze the deployed model — no challenger has ever beaten
        champion-arch, so nothing could ever displace a champion.

        The tautology I9024 s1 actually killed is NOT restored: the comparison
        is against the incumbent's bundle CPCV (I9018), re-scored on this
        vintage where possible (I9024 s2), never against a number this run wrote.
        """
        board, promotes, _ = _pool_fixture(monkeypatch, auto_promote=True,
                                           base_ic=0.30, variant_ic=0.15)
        ids = {c["spec_id"] for c in board["candidates"]}
        assert "champion-arch" in ids            # base IS in the pool
        # champion-arch is the BASELINE — never a CHALLENGER winner.
        assert board["winner_version_id"] is None
        assert board["promotion_baseline_source"] == "champion_arch_fresh"
        arch_cand = next(c for c in board["candidates"] if c["spec_id"] == "champion-arch")
        assert arch_cand["reason"] == "champion_arch_baseline"
        assert arch_cand["eligible"] is False
        # …but it DOES refresh the live model, on a comparison it actually won.
        assert board["champion_arch_refresh_version_id"] == "base-today"
        assert board["champion_arch_refresh_refused_reason"] is None
        assert board["promoted"] == "base-today"
        assert board["promoted_kind"] == "refit"
        assert promotes == ["base-today"]

    def test_a_refresh_that_does_not_beat_the_incumbent_is_refused(self, monkeypatch):
        """RED against the PRE-I9024 tree, where `x >= x + 0` promoted this.

        champion-arch scores 0.05 against a serving incumbent at 0.10. Before
        I9018 the incumbent's number was read back off the prefix this run had
        already overwritten, so the comparison was the candidate against itself
        and this promoted. Now it is refused, and the leaderboard says why.
        """
        board, promotes, _ = _pool_fixture(monkeypatch, auto_promote=True,
                                           base_ic=0.05, variant_ic=0.03)
        assert board["winner_version_id"] is None
        assert board["champion_arch_refresh_version_id"] is None
        assert (board["champion_arch_refresh_refused_reason"]
                == "below_serving_incumbent_plus_margin")
        assert board["promoted"] is None
        assert board.get("promoted_kind") is None
        assert promotes == []                    # the live model is UNCHANGED

    def test_variant_can_beat_the_base(self, monkeypatch):
        # A challenger (0.25) that clears the champion-arch baseline (0.12) + margin
        # WINS as a true challenger — the only remaining way to become champion.
        board, _, _ = _pool_fixture(monkeypatch, auto_promote=True,
                                    base_ic=0.12, variant_ic=0.25)
        assert board["winner_version_id"] == "resid-v"
        assert board["promoted_kind"] == "arena-pointer"

    def test_the_refresh_fires_exactly_when_it_beats_the_serving_incumbent(self, monkeypatch):
        """The sweep that used to prove "nothing ever promotes" now proves the
        refresh tracks the SERVING incumbent's score rather than its own.

        The serving incumbent is at CPCV 0.10 throughout; the variant never
        clears the baseline, so `winner_version_id` is None in every case and the
        only thing that can move the pointer is the refresh. Pre-I9024 EVERY one
        of these promoted, including 0.05, because the test was `x >= x`.
        """
        margin = float(getattr(cfg, "MODEL_ZOO_PROMOTE_MARGIN", 0.01))
        for base_ic in (0.05, 0.15, 0.30, 0.90):
            should_refresh = base_ic >= 0.10 + margin
            board, promotes, _ = _pool_fixture(
                monkeypatch, auto_promote=True, base_ic=base_ic,
                variant_ic=base_ic - 0.02,
                arena_pointer="base-today" if should_refresh else "old-champ-v",
                arena_moved=False,
            )
            assert board["winner_version_id"] is None, base_ic
            assert promotes == (["base-today"] if should_refresh else []), base_ic
            if should_refresh:
                assert board["promoted_kind"] == "refit"
            else:
                assert board.get("promoted_kind") is None
            if not should_refresh:
                assert (board["champion_arch_refresh_refused_reason"]
                        == "below_serving_incumbent_plus_margin"), base_ic


class TestPromotionAlert:
    def test_cutover_fires_alert_with_revert_command(self, monkeypatch):
        # A CHALLENGER win is now the only cutover, so the alert fixture is one.
        board, promotes, alerts_sent = _pool_fixture(
            monkeypatch, auto_promote=True, base_ic=0.12, variant_ic=0.25)
        assert promotes == ["resid-v"]
        assert board["reverted_from"] == "old-champ-v"
        assert len(alerts_sent) == 1
        a = alerts_sent[0]
        assert a["severity"] == "warning"
        # The revert command targets the PRIOR champion's version_id, exactly.
        assert "--promote old-champ-v" in a["message"]
        assert "resid-v" in a["message"]         # the new champion
        assert a["dedup_key"] == "model_zoo_promote_2026-06-13"

    def test_observe_fires_info_alert_no_promote(self, monkeypatch):
        board, promotes, alerts_sent = _pool_fixture(
            monkeypatch, auto_promote=False, base_ic=0.12, variant_ic=0.25)
        assert promotes == []                    # observe → never promotes
        assert board["promoted"] is None
        assert len(alerts_sent) == 1
        assert alerts_sent[0]["severity"] == "info"
        assert "--promote resid-v" in alerts_sent[0]["message"]  # manual-promote hint

    def test_no_winner_no_alert(self, monkeypatch):
        # Genuine no-winner case (config#1175). With model_zoo_promote_margin=0.0
        # eligibility is `ic >= baseline` (fresh-best-wins), so a TIE PROMOTES —
        # the former base_ic=variant_ic=0.10 inputs here actually cut over resid-v,
        # making the "no winner / no alert" assertion stale. The variant (0.03) is
        # below the champion-arch baseline (0.05) so no challenger wins, and since
        # I9024 s1 the champion-arch has no self-promotion path → no promotion,
        # no alert.
        board, promotes, alerts_sent = _pool_fixture(
            monkeypatch, auto_promote=True, base_ic=0.05, variant_ic=0.03)
        assert board["winner_version_id"] is None
        assert promotes == []
        assert alerts_sent == []
