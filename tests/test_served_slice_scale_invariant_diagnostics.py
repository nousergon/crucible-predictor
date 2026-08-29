"""Observe-only diagnostics that separate a SCALE collapse from a RANKING collapse.

alpha-engine-config-I9257 · sibling reporting fix -I9260 · parent -I9255.

WHY THIS FILE EXISTS. Every metric the behavioral veto decides on — ``alpha_stdev``,
``stdev_p_up``, ``n_high_confidence`` — is a RAW-MAGNITUDE quantity. A uniform
rescaling of the meta-model's linear predictor drives all three toward zero with
no loss of ranking information whatsoever. On 2026-08-28 that is exactly what
happened: all five pool members were refused, INCLUDING the champion-architecture
control retrained on the same week's data, because the vintage's coefficients had
shrunk ~2.9x.

Measured off the real artifacts (167 dates, top_n=30):

    08-28 champion-arch vs 08-14 incumbent
        alpha_stdev ratio           0.347   → VETO
        standardized ratio          0.943
        top-30 rank corr            0.932
        top-30 name overlap         0.788
        top-30 realized lift       +0.0323 (t +7.25)  vs  -0.0037 (t -0.70)

    08-21 candidate — the one that promoted and produced five consecutive live
    sessions at n_high_confidence 0
        alpha_stdev ratio           0.327   → VETO
        standardized ratio          0.973   ← a scale-invariant RULE passes it
        top-30 realized lift       -0.0353 (t -7.44)

The last line is the whole reason these are DIAGNOSTICS and not a replacement
rule: a scale-invariant dispersion test would have admitted the 2026-08-21
failure. So no threshold moves here, and the tests below assert that the veto's
verdict is byte-identical with and without the new block.

RED pre-fix (champion-challenger-policy §7.4): ``selected_slice_metrics``
returned neither ``scale_invariant`` nor ``realized``, so every assertion on
those keys raises ``KeyError``, and ``_cross_version_diagnostics`` did not exist.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import served_slice_dispersion as ssd
from training.promotion_behavioral_veto import evaluate_behavioral_veto

N_DATES, N_NAMES = 40, 120


def _panel(scale: float = 1.0, seed: int = 0, informative: bool = True):
    """Alpha, p_up, dates and a realized forward return, at a chosen alpha SCALE.

    ``scale`` multiplies alpha only. Ranking, selection and every scale-free
    statistic are therefore identical across scales by construction.
    """
    rng = np.random.default_rng(seed)
    dates = np.repeat([f"2026-01-{1 + d:02d}" for d in range(N_DATES)], N_NAMES)
    base = rng.normal(0.0, 0.02, N_DATES * N_NAMES)
    alpha = base * scale
    p_up = np.round(0.5 + np.clip(alpha, -0.15, 0.15) * 2.0, 4)
    noise = rng.normal(0.0, 0.02, N_DATES * N_NAMES)
    realized = (base * 1.5 + noise) if informative else (-base * 1.5 + noise)
    return alpha, p_up, dates, realized


def _metrics(scale=1.0, seed=0, informative=True, realized=True):
    a, p, d, r = _panel(scale=scale, seed=seed, informative=informative)
    return ssd.selected_slice_metrics(
        a, p, d, top_n=30, min_confidence=0.30, min_dates=20,
        realized=(r if realized else None),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Scale versus information
# ═════════════════════════════════════════════════════════════════════════════

def test_a_uniform_rescale_collapses_the_verdict_metric_and_not_the_diagnostic():
    """The exact shape of the 2026-08-28 refusal, reproduced on constructed data.

    A 0.347x rescale — the measured coefficient ratio — takes the VERDICT metric
    below the 0.5 veto floor while the standardized diagnostic is unmoved.
    """
    full, shrunk = _metrics(scale=1.0), _metrics(scale=0.347)

    ratio_raw = shrunk["alpha_stdev"] / full["alpha_stdev"]
    assert ratio_raw == pytest.approx(0.347, abs=0.01)
    assert ratio_raw < ssd_min_ratio(), "the raw metric must still VETO"

    z_full = full["scale_invariant"]["alpha_stdev_standardized"]
    z_shrunk = shrunk["scale_invariant"]["alpha_stdev_standardized"]
    assert z_full == pytest.approx(z_shrunk, rel=1e-6), (
        "per-date cross-sectional standardization must be exactly invariant to a "
        "uniform rescale — if it is not, it cannot separate scale from shape"
    )


def ssd_min_ratio() -> float:
    from training.promotion_behavioral_veto import MIN_DISPERSION_RATIO
    return MIN_DISPERSION_RATIO


def test_the_realized_block_separates_a_good_ranker_from_an_inverted_one():
    """What the dispersion metrics cannot see, and the 08-21 vs 08-28 difference.

    Both models below have IDENTICAL alpha dispersion and identical selections
    up to sign, so every dispersion statistic agrees about them. Only the
    realized block tells them apart.
    """
    good = _metrics(informative=True)["realized"]
    bad = _metrics(informative=False)["realized"]

    assert good["rank_ic"] > 0 and good["rank_ic_t"] > 2
    assert bad["rank_ic"] < 0 and bad["rank_ic_t"] < -2
    assert good["top_n_lift"] > 0 > bad["top_n_lift"]
    assert good["column"] is None, (
        "the column name is stamped by served_slice_metrics, which is the layer "
        "that knows which column it read"
    )


def test_a_panel_without_realized_returns_is_UNMEASURED_and_says_so():
    """champion-challenger-policy §7.2 — never a silent zero."""
    block = _metrics(realized=False)["realized"]
    assert block["rank_ic"] is None and block["top_n_lift"] is None
    assert block["n_dates"] == 0
    assert "UNMEASURED" in block["reason"]
    assert "actual_fwd_canonical" in block["reason"]


# ═════════════════════════════════════════════════════════════════════════════
# The diagnostics can never arm a rule
# ═════════════════════════════════════════════════════════════════════════════

def test_the_veto_verdict_is_identical_with_and_without_the_new_blocks():
    """The load-bearing safety property of alpha-engine-config-I9257.

    ``_SERVED_METRIC_NAMES`` filters the veto's merge to the four flat rule
    metrics by name, so a nested block is structurally unreachable. Asserted
    rather than assumed, because the cost of being wrong is a promotion gate
    silently deciding on an observability field.
    """
    cand_full, inc_full = _metrics(scale=0.347), _metrics(scale=1.0)
    strip = lambda m: {k: v for k, v in m.items()
                       if k not in ("scale_invariant", "realized")}

    with_blocks = evaluate_behavioral_veto(
        {}, {}, candidate_served_metrics=cand_full,
        incumbent_served_metrics=inc_full)
    without = evaluate_behavioral_veto(
        {}, {}, candidate_served_metrics=strip(cand_full),
        incumbent_served_metrics=strip(inc_full))

    assert with_blocks == without
    assert with_blocks["status"] == "veto"
    assert "scale_invariant" not in with_blocks["measured"]
    assert "realized" not in with_blocks["measured"]


# ═════════════════════════════════════════════════════════════════════════════
# Cross-version agreement
# ═════════════════════════════════════════════════════════════════════════════

def test_a_pure_rescale_of_the_champion_reads_as_a_perfect_clone():
    """Which is the POINT, and simultaneously why this is not a veto metric.

    A candidate that is the incumbent times 0.347 picks the same names in the
    same order. Overlap 1.0 and rank correlation 1.0 say "same ranking, different
    scale" — the reading the raw dispersion ratio of 0.347 cannot give.
    """
    alpha, _, dates, _ = _panel(scale=1.0)
    out = ssd._cross_version_diagnostics(
        {"ref": alpha, "rescaled": alpha * 0.347}, dates, "ref", top_n=30)
    assert out["rescaled"]["top_n_overlap_vs_reference"] == pytest.approx(1.0)
    assert out["rescaled"]["rank_corr_vs_reference"] == pytest.approx(1.0)
    assert out["rescaled"]["reference_version_id"] == "ref"
    assert "ref" not in out, "the reference is not compared against itself"


def test_an_unrelated_ranker_does_not_read_as_a_clone():
    alpha, _, dates, _ = _panel(scale=1.0, seed=1)
    other, _, _, _ = _panel(scale=1.0, seed=2)
    out = ssd._cross_version_diagnostics(
        {"ref": alpha, "other": other}, dates, "ref", top_n=30)
    assert out["other"]["top_n_overlap_vs_reference"] < 0.5
    assert abs(out["other"]["rank_corr_vs_reference"]) < 0.2


def test_overlap_divides_by_the_WIDER_set():
    """champion-challenger-policy §4 — a narrow set inside a wide one is a
    DIFFERENT selection, not a perfect clone. Both selections here are top-30 of
    the same rows, so equal width; the guard is that the denominator is max(),
    which a contained-set construction would expose.
    """
    import inspect
    src = inspect.getsource(ssd._cross_version_diagnostics)
    assert "max(len(cand_top), len(ref_top))" in src


def test_a_missing_reference_yields_no_cross_version_block_rather_than_zeros():
    alpha, _, dates, _ = _panel()
    assert ssd._cross_version_diagnostics({"a": alpha}, dates, None, top_n=30) == {}
    assert ssd._cross_version_diagnostics(
        {"a": alpha}, dates, "absent", top_n=30) == {}
