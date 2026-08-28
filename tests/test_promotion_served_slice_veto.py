"""The 2026-08-21 candidate, refused on the batch it would actually have TRADED.

alpha-engine-config-I9061 (served-slice dispersion + the restored champion-arch
refresh) · -I9024 s2 (the incumbent re-scored on the current vintage) · -I9024 s4
(the behavioral veto these feed).

champion-challenger-policy section 7.4: a guard must be verified to FAIL without
the fix. Each test below names what the pre-fix tree did instead.

WHY THIS FILE EXISTS. The behavioral veto shipped in crucible-predictor-PR570
reads ``output_distribution_gate.metrics`` off the training manifest. Measured on
the two real registry bundles:

    v3.0-meta-2026-08-14-119e069b (incumbent)  stdev_p_up 0.113644
    v3.0-meta-2026-08-21-7d3d1cce (candidate)  stdev_p_up 0.130132

ratio 1.145 — a comfortable PASS. That number is a 25-point SYNTHETIC alpha sweep
through the calibrator alone; it never touches the meta-model, the universe, or
the selection rule, so it could not have seen what happened next:

    2026-08-21 (incumbent serving)  stdev_p_up 0.191162  alpha_stdev 0.043427  n_high_confidence 5
    2026-08-24 (candidate serving)  stdev_p_up 0.060170  alpha_stdev 0.010437  n_high_confidence 0

THE MEASUREMENT. ``training/served_slice_dispersion.py`` scores both versions'
own bundles (meta_model.pkl + isotonic_calibrator.pkl — exactly the pair
``promote_to_champion`` copies live) over ONE common real feature panel, slices
to the top-N by predicted alpha per date, and measures dispersion inside that
slice. Run against the real artifacts — the 2026-08-21 OOS panel, 167 dates,
top_n=30, min_confidence=0.30 — it produces the constants below.

The bundles and the panel are NOT committed: ``*.pkl`` and ``*.parquet`` are
gitignored in this public repo, and fitted champion weights are private-tier
under repository-tiering-policy. So the file is split: the MEASUREMENT functions
are tested on constructed frames, and the VETO is tested on the measured values.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training import model_zoo as mz
from training import served_slice_dispersion as ssd
from training.promotion_behavioral_veto import (
    MIN_DISPERSION_RATIO,
    evaluate_behavioral_veto,
)
from tests.test_model_zoo import _FakeS3

INCUMBENT_VID = "v3.0-meta-2026-08-14-119e069b"
CANDIDATE_VID = "v3.0-meta-2026-08-21-7d3d1cce"
INCUMBENT_IC = 0.131924
CANDIDATE_IC = 0.305594

# ── The MEASURED served-slice values, from the real artifacts ────────────────
# Reproduce with:
#   AWS_PROFILE=ne-admin, the two registry bundles, and
#   s3://alpha-engine-research/predictor/diagnostics/oos_rows/2026-08-21.parquet
# through ssd.load_scoring_head / ssd.score_panel / ssd.selected_slice_metrics.
SERVED_INCUMBENT = {
    "alpha_stdev": 0.01662409, "stdev_p_up": 0.071064,
    "n_high_confidence": 30, "n_dates": 167,
}
SERVED_CANDIDATE = {
    "alpha_stdev": 0.00542870, "stdev_p_up": 0.041531,
    "n_high_confidence": 0, "n_dates": 167,
}
# What the manifests carry for the same two versions — the sweep numbers.
MANIFEST_INCUMBENT_STDEV_P_UP = 0.113644
MANIFEST_CANDIDATE_STDEV_P_UP = 0.130132


def _manifest(*, mean_ic, stdev_p_up=None, forward_days=21, second_opinion=None,
              incumbent_rescore=None):
    cpcv = {"mean_ic": mean_ic, "n_combos": 44}
    if second_opinion is not None:
        cpcv["second_opinion"] = second_opinion
    m = {
        "forward_days": forward_days,
        "meta_model_oos_ic_cpcv": cpcv,
        "meta_model_promotion_stats": {
            "downside": {"passes_downside_gate": True},
            "overfit": {"passes_overfit_gate": True, "dsr": 1.0},
        },
    }
    if stdev_p_up is not None:
        m["output_distribution_gate"] = {"metrics": {"stdev_p_up": stdev_p_up}}
    if incumbent_rescore is not None:
        m["incumbent_rescore"] = incumbent_rescore
    return m


# ═════════════════════════════════════════════════════════════════════════════
# THE ACCEPTANCE TEST — the one that decides the design
# ═════════════════════════════════════════════════════════════════════════════

def test_the_served_slice_refuses_the_2026_08_21_candidate():
    """THE acceptance test. RED against origin/main, twice over: the pre-fix
    ``evaluate_behavioral_veto`` had no served-metric parameters at all, and the
    manifest metrics it did read PASS this candidate (see the test below).

    Two independent refusals, neither of which required moving a threshold:
      * alpha_stdev at 33% of the incumbent's, against a 0.5 floor;
      * zero high-confidence names, against a > 0 rule.
    """
    verdict = evaluate_behavioral_veto(
        _manifest(mean_ic=CANDIDATE_IC, stdev_p_up=MANIFEST_CANDIDATE_STDEV_P_UP),
        _manifest(mean_ic=INCUMBENT_IC, stdev_p_up=MANIFEST_INCUMBENT_STDEV_P_UP),
        candidate_served_metrics=SERVED_CANDIDATE,
        incumbent_served_metrics=SERVED_INCUMBENT,
    )
    assert verdict["status"] == "veto"
    vetoed = {v["metric"] for v in verdict["vetoes"]}
    assert "alpha_stdev" in vetoed
    assert "n_high_confidence" in vetoed
    ratio = verdict["measured"]["alpha_stdev"]["ratio"]
    assert ratio < MIN_DISPERSION_RATIO
    assert 0.32 < ratio < 0.34, ratio
    # The served slice, not the manifest, is what decided it.
    assert "alpha_stdev" in verdict["served_slice_metrics"]
    assert "stdev_p_up" in verdict["served_slice_metrics"]


def test_the_manifest_metrics_alone_do_NOT_refuse_it():
    """The guard on the guard: this is the state of origin/main, and it PASSES.

    If this test ever starts failing, the served-slice work above has been made
    redundant by a manifest producer — check before deleting anything.
    """
    verdict = evaluate_behavioral_veto(
        _manifest(mean_ic=CANDIDATE_IC, stdev_p_up=MANIFEST_CANDIDATE_STDEV_P_UP),
        _manifest(mean_ic=INCUMBENT_IC, stdev_p_up=MANIFEST_INCUMBENT_STDEV_P_UP),
    )
    assert verdict["status"] == "pass"
    ratio = verdict["measured"]["stdev_p_up"]["ratio"]
    assert ratio > 1.0, ratio          # the candidate looks BETTER on the sweep
    assert verdict["served_slice_metrics"] == []


def test_select_winner_refuses_the_candidate_end_to_end(monkeypatch):
    """The whole rotation, with the served-slice measurement stubbed to the
    values measured off the real bundles. RED pre-fix: no served-slice
    measurement existed, the veto passed, and this promoted.
    """
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(
        ssd, "served_slice_metrics",
        lambda s3, bucket, vids, **kw: {
            "status": "measured", "reason": None,
            "panel_key": "predictor/diagnostics/oos_rows/2026-08-21.parquet",
            "n_panel_rows": 150000, "top_n": 30, "min_confidence": 0.30,
            "metrics": {CANDIDATE_VID: SERVED_CANDIDATE,
                        INCUMBENT_VID: SERVED_INCUMBENT},
            "errors": {},
        },
    )
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: {"forward_days": 21, "served_version": INCUMBENT_VID,
                                "served_date": "2026-08-14"},
        f"predictor/registry/{INCUMBENT_VID}/manifest.json": _manifest(
            mean_ic=INCUMBENT_IC, stdev_p_up=MANIFEST_INCUMBENT_STDEV_P_UP),
        f"predictor/registry/{CANDIDATE_VID}/manifest.json": _manifest(
            mean_ic=CANDIDATE_IC, stdev_p_up=MANIFEST_CANDIDATE_STDEV_P_UP),
    })
    board = mz.select_winner(
        s3, "bkt",
        trained=[{"spec_id": "champion-arch", "version_id": CANDIDATE_VID,
                  "model_version": "v3.0-meta"}],
        margin=0.0, date_str="2026-08-21",
    )
    cand = next(c for c in board["candidates"] if c["version_id"] == CANDIDATE_VID)
    # It WON on IC — 2.3x the incumbent — and is refused anyway.
    assert cand["cpcv_mean_ic"] > board["serving_champion"]["cpcv_mean_ic"]
    assert cand["behavioral_veto_status"] == "veto"
    assert cand["reason"] == "behavioral_veto"
    assert cand["served_slice"] == SERVED_CANDIDATE
    # …and the restored refresh path is refused for the SAME reason, so the
    # candidate has no second route to the serving slot.
    assert board["winner_version_id"] is None
    assert board["champion_arch_refresh_version_id"] is None
    assert board["champion_arch_refresh_refused_reason"] == "behavioral_veto"
    assert board["served_slice_dispersion"]["status"] == "measured"


# ═════════════════════════════════════════════════════════════════════════════
# The MEASUREMENT functions, on constructed frames
# ═════════════════════════════════════════════════════════════════════════════

def _frame(n_dates=40, n_names=200, spread=1.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat([f"2026-0{1 + d // 28}-{1 + d % 28:02d}" for d in range(n_dates)],
                      n_names)
    alpha = rng.normal(0.0, 0.02 * spread, size=n_dates * n_names)
    return alpha, dates


def test_dispersion_is_measured_INSIDE_the_top_n_slice_not_over_the_universe():
    """The whole point: a model can look healthy over the universe and collapse
    over the ~30 names the executor ranks. RED pre-fix — nothing measured the
    slice at all.
    """
    alpha, dates = _frame(seed=1)
    p_up = np.clip(0.5 + alpha * 5, 0.01, 0.99)
    universe_stdev = float(np.std(alpha))
    m = ssd.selected_slice_metrics(alpha, p_up, dates, top_n=30, min_confidence=0.30)
    # The top-30 tail of a normal is far tighter than the whole draw.
    assert m["alpha_stdev"] < universe_stdev / 2
    assert m["n_dates"] == 40
    assert m["top_n"] == 30


def test_a_collapsed_model_reproduces_the_08_21_shape():
    """A model whose spread is a third of another's, over the same rows, is
    measured as such and its high-confidence count goes to zero."""
    alpha, dates = _frame(spread=1.0, seed=2)
    wide_p = np.clip(0.5 + alpha * 8, 0.01, 0.99)
    tight_alpha = alpha / 3.0
    tight_p = np.clip(0.5 + tight_alpha * 1.0, 0.01, 0.99)
    wide = ssd.selected_slice_metrics(alpha, wide_p, dates, top_n=30, min_confidence=0.30)
    tight = ssd.selected_slice_metrics(
        tight_alpha, tight_p, dates, top_n=30, min_confidence=0.30)
    assert tight["alpha_stdev"] / wide["alpha_stdev"] < MIN_DISPERSION_RATIO
    assert tight["n_high_confidence"] == 0
    assert wide["n_high_confidence"] > 0
    verdict = evaluate_behavioral_veto(
        None, None,
        candidate_served_metrics=tight, incumbent_served_metrics=wide)
    assert verdict["status"] == "veto"


def test_n_high_confidence_uses_the_serving_definition():
    """``|p_up - 0.5| * 2 >= MIN_CONFIDENCE`` — identical to
    ``inference/stages/write_output.py``. A different definition here would make
    the veto's count incomparable to the live one it is standing in for."""
    dates = np.array(["2026-01-01"] * 10 + ["2026-01-02"] * 10)
    p_up = np.array([0.5, 0.6, 0.65, 0.7, 0.8] * 4)   # conf 0.0/0.2/0.3/0.4/0.6
    alpha = np.linspace(-0.01, 0.01, 20)
    m = ssd.selected_slice_metrics(
        alpha, p_up, dates, top_n=10, min_confidence=0.30, min_dates=2)
    # Per date: 3 of 10 rows are at conf >= 0.30 for each of the two p_up cycles.
    assert m["n_high_confidence"] == 6


def test_too_few_dates_is_uncomputable_not_a_pass():
    """champion-challenger-policy 7.2 — an unmeasurable result fails LOUD."""
    dates = np.array(["2026-01-01"] * 50)
    with pytest.raises(RuntimeError, match="not a distribution"):
        ssd.selected_slice_metrics(
            np.linspace(-1, 1, 50), np.linspace(0.2, 0.8, 50), dates,
            top_n=30, min_confidence=0.30)


def test_a_missing_feature_is_refused_never_zero_filled():
    """alpha-engine-config-I5949 — a model silently scored on zeros is the
    defect this repo already paid for once."""
    import pandas as pd

    class _M:
        _feature_names = ["a", "b", "c"]
        def predict(self, X):  # pragma: no cover — never reached
            return np.zeros(len(X))

    with pytest.raises(RuntimeError, match="refusing to zero-fill"):
        ssd.score_panel(_M(), None, pd.DataFrame({"a": [1.0], "b": [2.0]}))


# ═════════════════════════════════════════════════════════════════════════════
# Failure posture — uncomputable is reported, never a pass
# ═════════════════════════════════════════════════════════════════════════════

def test_an_unavailable_panel_is_uncomputable_and_non_blocking(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: {"forward_days": 21, "served_version": INCUMBENT_VID},
        f"predictor/registry/{INCUMBENT_VID}/manifest.json": _manifest(mean_ic=INCUMBENT_IC),
        "predictor/registry/arch-v/manifest.json": _manifest(mean_ic=0.20),
    })
    board = mz.select_winner(
        s3, "bkt",
        trained=[{"spec_id": "champion-arch", "version_id": "arch-v",
                  "model_version": "v3.0-meta"}],
        margin=0.0, date_str="2026-08-21",
    )
    block = board["served_slice_dispersion"]
    assert block["status"] == "uncomputable"
    assert "panel" in (block["reason"] or "")
    # Non-blocking: the rotation still ran and the refresh still decided.
    assert board["champion_arch_refresh_version_id"] == "arch-v"


def test_a_served_metric_is_never_ranked_against_a_manifest_metric():
    """Like-for-like or not at all. If only the CANDIDATE has a served-slice
    measurement, the dispersion rules must report uncomputable rather than
    compare 0.0054 (a traded-slice stdev) against 0.1136 (a synthetic sweep) and
    'discover' a 95% collapse."""
    verdict = evaluate_behavioral_veto(
        _manifest(mean_ic=CANDIDATE_IC, stdev_p_up=MANIFEST_CANDIDATE_STDEV_P_UP),
        _manifest(mean_ic=INCUMBENT_IC, stdev_p_up=MANIFEST_INCUMBENT_STDEV_P_UP),
        candidate_served_metrics={"alpha_stdev": SERVED_CANDIDATE["alpha_stdev"],
                                  "stdev_p_up": SERVED_CANDIDATE["stdev_p_up"]},
        incumbent_served_metrics=None,
    )
    # stdev_p_up falls back to the manifest pair on BOTH sides — not mixed.
    assert verdict["measured"]["stdev_p_up"]["candidate"] == MANIFEST_CANDIDATE_STDEV_P_UP
    assert verdict["served_slice_metrics"] == []


def test_a_zero_high_confidence_candidate_is_vetoed_without_an_incumbent_value():
    """The zero rule needs no counterpart, so a candidate that emits no
    actionable names is refused even on a rotation where the incumbent could not
    be measured."""
    verdict = evaluate_behavioral_veto(
        None, None,
        candidate_served_metrics={"n_high_confidence": 0},
        incumbent_served_metrics=None,
    )
    assert verdict["status"] == "veto"
    assert [v["metric"] for v in verdict["vetoes"]] == ["n_high_confidence"]
