"""The 2026-08-21 rotation, replayed, against every guard added for it.

alpha-engine-config-I9018 (tautological gate) · -I9024 (one-way ratchet +
behavioral veto) · -I9028 (corrupt legacy archive) · -I9030 (second-opinion
authority split).

champion-challenger-policy section 7.4: a guard must be verified to FAIL without
the fix. Every test here was run against the pre-fix tree and shown red; the
docstring of each names what the old code did instead.

The numbers are the real ones, read from
``predictor/model_zoo/leaderboard/2026-08-21.json`` and the registry bundles in
``s3://alpha-engine-research/predictor/registry/``:

  incumbent  v3.0-meta-2026-08-14-119e069b  cpcv_mean_ic 0.131924  n_combos 44
  candidate  v3.0-meta-2026-08-21-7d3d1cce  cpcv_mean_ic 0.305594  n_combos 44
             second_opinion_ic 0.284521, status low_match_rate, match_rate 0.08
             dsr_training 1.0, dsr_selection 0.996315
             stdev_p_up 0.060 against the incumbent's 0.191
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training import model_zoo as mz
from tests.test_model_zoo import _FakeS3

INCUMBENT_VID = "v3.0-meta-2026-08-14-119e069b"
CANDIDATE_VID = "v3.0-meta-2026-08-21-7d3d1cce"
INCUMBENT_IC = 0.131924
CANDIDATE_IC = 0.305594


def _manifest(*, mean_ic, n_combos=44, stdev_p_up=None, second_opinion=None,
              stale=False, forward_days=21):
    cpcv = {"mean_ic": mean_ic, "n_combos": n_combos}
    if second_opinion is not None:
        cpcv["second_opinion"] = second_opinion
    if stale:
        cpcv["cpcv_is_stale_snapshot"] = True
    m = {
        "forward_days": forward_days,
        "version": "v3.0-meta",
        "meta_model_oos_ic_cpcv": cpcv,
        "meta_model_promotion_stats": {
            "downside": {"passes_downside_gate": True},
            "overfit": {"passes_overfit_gate": True, "dsr": 1.0},
        },
    }
    if stdev_p_up is not None:
        m["output_distribution_gate"] = {"metrics": {"stdev_p_up": stdev_p_up}}
    return m


LOW_MATCH_SECOND_OPINION = {
    "status": "low_match_rate",
    "second_opinion_ic": 0.284521,
    "n_oos_rows": 1000,
    "n_matched": 80,
    "match_rate": 0.08,
}


def _s3(*, incumbent, candidates):
    """A fake bucket whose live manifest POINTS at the incumbent's bundle."""
    objects = {
        cfg.META_MANIFEST_KEY: {
            "forward_days": 21,
            "served_version": INCUMBENT_VID,
            "served_date": "2026-08-14",
        },
        f"predictor/registry/{INCUMBENT_VID}/manifest.json": incumbent,
    }
    for vid, manifest in candidates.items():
        objects[f"predictor/registry/{vid}/manifest.json"] = manifest
    return _FakeS3(objects)


def _trained(*pairs):
    return [{"spec_id": sid, "version_id": vid, "model_version": "v3.0-meta"}
            for sid, vid in pairs]


# ── I9018: the incumbent's score comes from an artifact this run cannot write ──


def test_serving_ic_is_the_incumbent_bundles_cpcv_not_the_candidates(monkeypatch):
    """RED pre-fix: `serving_ic` was `_cpcv_mean(_read_live_manifest(...))`,
    reading `predictor/weights/meta/manifest.json` — which every spec of the
    rotation had already overwritten. It reported the CANDIDATE'S number as the
    incumbent's, bit-identically, in 4/4 rotations.
    """
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _s3(
        incumbent=_manifest(mean_ic=INCUMBENT_IC),
        candidates={CANDIDATE_VID: _manifest(mean_ic=CANDIDATE_IC)},
    )
    # The live manifest ALSO carries a (candidate-shaped) cpcv block, exactly as
    # the real one did. Reading it is the bug; the fix must ignore it.
    s3.objects[cfg.META_MANIFEST_KEY]["meta_model_oos_ic_cpcv"] = {
        "mean_ic": CANDIDATE_IC, "n_combos": 44,
    }
    board = mz.select_winner(
        s3, "bkt", trained=_trained(("champion-arch", CANDIDATE_VID)), margin=0.0)
    serving = board["serving_champion"]
    assert serving["cpcv_mean_ic"] == INCUMBENT_IC
    assert serving["cpcv_mean_ic"] != CANDIDATE_IC
    assert serving["incumbent_version_id"] == INCUMBENT_VID
    assert serving["cpcv_source"] == (
        f"predictor/registry/{INCUMBENT_VID}/manifest.json"
    )
    assert serving["cpcv_is_stale_snapshot"] is False


def test_identical_incumbent_and_candidate_ic_raises(monkeypatch):
    """RED pre-fix: this was the NORMAL case — `x >= x + 0` promoted every week.

    Two models on different data vintages do not tie to full float precision.
    """
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _s3(
        incumbent=_manifest(mean_ic=CANDIDATE_IC),
        candidates={CANDIDATE_VID: _manifest(mean_ic=CANDIDATE_IC)},
    )
    with pytest.raises(mz.PromotionInputIntegrityError, match="IDENTICAL"):
        mz.select_winner(
            s3, "bkt", trained=_trained(("champion-arch", CANDIDATE_VID)))


def test_a_content_addressed_reregistration_of_the_incumbent_is_not_a_tie(monkeypatch):
    """The one legitimate equality: the same bundle id on both sides.

    A content-addressed bundle that collapses to the incumbent's own version_id
    IS the incumbent — an identical contract, so an identical CPCV is correct
    and must not raise.
    """
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _s3(incumbent=_manifest(mean_ic=INCUMBENT_IC), candidates={})
    board = mz.select_winner(
        s3, "bkt", trained=_trained(("champion-arch", INCUMBENT_VID)), margin=0.0)
    assert board["serving_champion"]["cpcv_mean_ic"] == INCUMBENT_IC


def test_stale_snapshot_manifest_raises(monkeypatch):
    """RED pre-fix: `cpcv_is_stale_snapshot: true` was set on the leaderboard in
    all four rotations and blocked nothing — the promotion went through on it.
    """
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _s3(
        incumbent=_manifest(mean_ic=INCUMBENT_IC),
        candidates={CANDIDATE_VID: _manifest(mean_ic=CANDIDATE_IC, stale=True)},
    )
    with pytest.raises(mz.PromotionInputIntegrityError, match="stale_snapshot"):
        mz.select_winner(
            s3, "bkt", trained=_trained(("champion-arch", CANDIDATE_VID)))


def test_missing_served_version_raises_rather_than_falling_back(monkeypatch):
    """The fallback IS the bug. A live manifest that cannot name the incumbent
    must fail the rotation, not quietly hand over its own cpcv fields.
    """
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _s3(
        incumbent=_manifest(mean_ic=INCUMBENT_IC),
        candidates={CANDIDATE_VID: _manifest(mean_ic=CANDIDATE_IC)},
    )
    s3.objects[cfg.META_MANIFEST_KEY].pop("served_version")
    with pytest.raises(mz.PromotionInputIntegrityError, match="served_version"):
        mz.select_winner(
            s3, "bkt", trained=_trained(("champion-arch", CANDIDATE_VID)))


def test_served_version_naming_a_missing_bundle_raises(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _s3(
        incumbent=_manifest(mean_ic=INCUMBENT_IC),
        candidates={CANDIDATE_VID: _manifest(mean_ic=CANDIDATE_IC)},
    )
    s3.objects[cfg.META_MANIFEST_KEY]["served_version"] = "v-does-not-exist"
    with pytest.raises(mz.PromotionInputIntegrityError, match="could not be read"):
        mz.select_winner(
            s3, "bkt", trained=_trained(("champion-arch", CANDIDATE_VID)))


# ── I9024 section 1: the arch-refresh promotion path no longer exists ─────────


def test_the_arch_refresh_path_is_gone_from_the_source():
    """RED pre-fix: every one of these symbols existed and was reachable."""
    import inspect
    src = inspect.getsource(mz)
    assert "champion_arch_refresh_version_id" not in src
    assert "champion-arch-refresh" not in src
    assert not hasattr(mz, "_snapshot_live_contract")
    assert not hasattr(mz, "_restore_live_contract")


def test_a_champion_arch_only_rotation_promotes_nothing(monkeypatch):
    """RED pre-fix: this is the 2026-08-21 rotation exactly — no challenger won,
    and the champion-arch promoted itself anyway on `x >= x + 0`.
    """
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _s3(
        incumbent=_manifest(mean_ic=INCUMBENT_IC),
        candidates={CANDIDATE_VID: _manifest(mean_ic=CANDIDATE_IC)},
    )
    board = mz.select_winner(
        s3, "bkt", trained=_trained(("champion-arch", CANDIDATE_VID)), margin=0.0)
    assert board["winner_version_id"] is None
    assert "champion_arch_refresh_version_id" not in board
    arch = next(c for c in board["candidates"] if c["version_id"] == CANDIDATE_VID)
    assert arch["eligible"] is False
    # …even though it scored 2.3x the incumbent.
    assert arch["cpcv_mean_ic"] > board["serving_champion"]["cpcv_mean_ic"]


# ── I9030: join integrity blocks with the enforce flag OFF ───────────────────


def test_low_match_rate_candidate_is_refused_with_the_enforce_flag_OFF(monkeypatch):
    """RED pre-fix: `second_opinion_gate_enforced: false` in the real 2026-08-21
    leaderboard, and the candidate promoted with `second_opinion_divergence:
    true` and its own reason naming "a label/join integrity problem".
    """
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(
        cfg, "MODEL_ZOO_SECOND_OPINION_GATE_ENFORCE", False, raising=False)
    s3 = _s3(
        incumbent=_manifest(mean_ic=INCUMBENT_IC),
        candidates={
            CANDIDATE_VID: _manifest(mean_ic=CANDIDATE_IC),
            "chal-v": _manifest(
                mean_ic=0.40, second_opinion=LOW_MATCH_SECOND_OPINION),
        },
    )
    board = mz.select_winner(
        s3, "bkt",
        trained=_trained(("champion-arch", CANDIDATE_VID), ("resid", "chal-v")),
        margin=0.0,
    )
    chal = next(c for c in board["candidates"] if c["version_id"] == "chal-v")
    assert chal["second_opinion_gate_enforced"] is False   # the flag is OFF…
    assert chal["second_opinion_verdict_class"] == "join_integrity_failure"
    assert chal["second_opinion_blocks_unconditionally"] is True
    assert chal["eligible"] is False                       # …and it is refused
    assert chal["reason"] == "second_opinion_join_integrity"
    assert board["winner_version_id"] is None


def test_an_uncomputable_second_opinion_stays_non_blocking(monkeypatch):
    """champion-challenger-policy section 5.1 — a gate that could not RUN is
    `insufficient` and non-blocking. It must not share a code path with a gate
    that ran and returned a data-integrity failure.
    """
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(
        cfg, "MODEL_ZOO_SECOND_OPINION_GATE_ENFORCE", False, raising=False)
    s3 = _s3(
        incumbent=_manifest(mean_ic=INCUMBENT_IC),
        candidates={
            CANDIDATE_VID: _manifest(mean_ic=0.20),
            "chal-v": _manifest(mean_ic=0.40, second_opinion={
                "status": "insufficient_matched_rows",
                "second_opinion_ic": None, "n_oos_rows": 1000,
                "n_matched": 4, "match_rate": 0.004,
            }),
        },
    )
    board = mz.select_winner(
        s3, "bkt",
        trained=_trained(("champion-arch", CANDIDATE_VID), ("resid", "chal-v")),
        margin=0.0,
    )
    chal = next(c for c in board["candidates"] if c["version_id"] == "chal-v")
    assert chal["second_opinion_verdict_class"] == "insufficient"
    assert chal["second_opinion_blocks_unconditionally"] is False
    assert chal["eligible"] is True
    assert board["winner_version_id"] == "chal-v"


def test_a_baseline_that_fails_join_integrity_refuses_the_whole_rotation(monkeypatch):
    """A challenger is scored against the champion-arch baseline. If the
    BASELINE's rows do not reconcile with realized outcomes, the comparison is
    meaningless in both directions, so nothing is promotable this rotation.
    """
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _s3(
        incumbent=_manifest(mean_ic=INCUMBENT_IC),
        candidates={
            CANDIDATE_VID: _manifest(
                mean_ic=CANDIDATE_IC, second_opinion=LOW_MATCH_SECOND_OPINION),
            "chal-v": _manifest(mean_ic=0.40),
        },
    )
    board = mz.select_winner(
        s3, "bkt",
        trained=_trained(("champion-arch", CANDIDATE_VID), ("resid", "chal-v")),
        margin=0.0,
    )
    assert board["baseline_join_integrity_failed"] is True
    assert board["baseline_second_opinion_verdict_class"] == "join_integrity_failure"
    chal = next(c for c in board["candidates"] if c["version_id"] == "chal-v")
    assert chal["eligible"] is False
    assert chal["reason"] == "baseline_join_integrity"
    assert board["winner_version_id"] is None


# ── I9024 section 4: the behavioral veto ─────────────────────────────────────


def test_dispersion_collapse_vetoes_the_2026_08_21_candidate(monkeypatch):
    """RED pre-fix: no promotion-side dispersion check existed at all. The
    inference-side one (PR569) is observe-only by design, so refusal has to
    happen here or nowhere.

    The candidate wins on IC by 2.3x and is refused anyway: stdev_p_up 0.060
    against the incumbent's 0.191 is 31% of the spread the executor ranks on.
    """
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _s3(
        incumbent=_manifest(mean_ic=INCUMBENT_IC, stdev_p_up=0.191),
        candidates={
            "arch-v": _manifest(mean_ic=0.10, stdev_p_up=0.190),
            CANDIDATE_VID: _manifest(mean_ic=CANDIDATE_IC, stdev_p_up=0.060),
        },
    )
    board = mz.select_winner(
        s3, "bkt",
        trained=_trained(("champion-arch", "arch-v"), ("resid", CANDIDATE_VID)),
        margin=0.0,
    )
    cand = next(c for c in board["candidates"] if c["version_id"] == CANDIDATE_VID)
    assert cand["cpcv_mean_ic"] > board["promotion_baseline_ic"]   # it WON on IC
    assert cand["behavioral_veto_status"] == "veto"
    assert cand["eligible"] is False
    assert cand["reason"] == "behavioral_veto"
    assert any("stdev_p_up" in r for r in cand["behavioral_veto_reasons"])
    assert board["winner_version_id"] is None


def test_a_healthy_dispersion_candidate_is_not_vetoed(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _s3(
        incumbent=_manifest(mean_ic=INCUMBENT_IC, stdev_p_up=0.191),
        candidates={
            "arch-v": _manifest(mean_ic=0.10, stdev_p_up=0.190),
            "chal-v": _manifest(mean_ic=0.40, stdev_p_up=0.180),
        },
    )
    board = mz.select_winner(
        s3, "bkt",
        trained=_trained(("champion-arch", "arch-v"), ("resid", "chal-v")),
        margin=0.0,
    )
    chal = next(c for c in board["candidates"] if c["version_id"] == "chal-v")
    assert chal["behavioral_veto_status"] == "pass"
    assert chal["eligible"] is True
    assert board["winner_version_id"] == "chal-v"


def test_uncomputable_behavioral_metrics_are_named_not_silent(monkeypatch):
    """The three served-side metrics I9024 section 4 names are not on the
    training manifest yet. They must be reported `insufficient` and NAMED — an
    unmeasured rule rendered as a pass is the failure mode this whole arc is
    about.
    """
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _s3(
        incumbent=_manifest(mean_ic=INCUMBENT_IC),
        candidates={CANDIDATE_VID: _manifest(mean_ic=CANDIDATE_IC)},
    )
    board = mz.select_winner(
        s3, "bkt", trained=_trained(("champion-arch", CANDIDATE_VID)), margin=0.0)
    cand = board["candidates"][0]
    assert cand["behavioral_veto_status"] == "insufficient"
    assert set(cand["behavioral_veto_uncomputable"]) == {
        "alpha_stdev", "model_hit_rate_30d", "n_high_confidence", "stdev_p_up",
    }


def test_zero_high_confidence_and_sub_coinflip_hit_rate_veto():
    """The other two I9024 section 4 rules, exercised directly on the evaluator
    so they are proven live the moment a producer emits the metrics.
    """
    from training.promotion_behavioral_veto import evaluate_behavioral_veto

    verdict = evaluate_behavioral_veto(
        {"behavioral_metrics": {
            "n_high_confidence": 0, "model_hit_rate_30d": 0.4826,
            "alpha_stdev": 0.0104, "stdev_p_up": 0.060,
        }},
        {"behavioral_metrics": {
            "n_high_confidence": 5, "model_hit_rate_30d": 0.5047,
            "alpha_stdev": 0.0434, "stdev_p_up": 0.191,
        }},
    )
    assert verdict["status"] == "veto"
    vetoed = {v["metric"] for v in verdict["vetoes"]}
    # The 2026-08-21 candidate trips all four.
    assert vetoed == {
        "alpha_stdev", "stdev_p_up", "n_high_confidence", "model_hit_rate_30d",
    }
    assert verdict["uncomputable"] == []


# ── the raises must reach the caller, not be logged and forgotten ────────────


def test_promotion_input_integrity_error_propagates_out_of_the_rotation(monkeypatch):
    """A fail-loud guard that the rotation swallows is not a guard.

    `run_rotation_and_select` wraps selection in a try/finally whose finally
    sends the digest email, and it catches broad exceptions around the realized-
    edge monitor. This pins that none of that absorbs the integrity error — the
    Step Functions state must go red.
    """
    def _boom(*a, **k):
        raise mz.PromotionInputIntegrityError("read bug")

    monkeypatch.setattr(mz, "select_winner", _boom)
    monkeypatch.setattr(mz, "send_zoo_digest_email", lambda *a, **k: False)
    monkeypatch.setattr(mz, "_record_trials", lambda *a, **k: (1, "ok"))
    monkeypatch.setattr(mz, "_current_champion_version_id", lambda *a, **k: None)
    monkeypatch.setattr(mz, "_resolve_base_champion_version", lambda *a, **k: None)
    monkeypatch.setattr(mz, "_read_promotion_marker", lambda *a, **k: None)

    with pytest.raises(mz.PromotionInputIntegrityError, match="read bug"):
        mz.run_rotation_and_select(
            "bkt", budget=0, specs=[], train_fn=lambda *a, **k: {"status": "ok"},
            registered_versions=[], s3=_FakeS3({}), date_str="2026-08-29",
            auto_promote_winner=False,
        )
