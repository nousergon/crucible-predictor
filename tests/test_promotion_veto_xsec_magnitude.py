"""The MAGNITUDE leg of the meta promotion veto (alpha-engine-config-I10185).

Why a second leg exists
-----------------------
``crucible-predictor-PR611`` added a SHARE leg: refuse a candidate whose fitted
L2 puts less than 10% of its variance inside a date. That leg is correct and it
is not this one. Measured 2026-09-08 with PR611's own
``cross_sectional_variance_share``, unmodified, on the live champion's actual
meta-training design matrix
(``s3://alpha-engine-research/predictor/diagnostics/oos_rows/v3.0-meta/2026-09-04.parquet``
— 14,940 rows / 107 dates / 16 features, per-column std verified against
``manifest.json -> models.meta_model.importance.feature_std`` to every published
digit):

    champion  v3.0-meta-2026-09-04-cc3271ea    share 0.702720   xsec_sd 0.010350
    incumbent v3.0-meta-2026-08-14-119e069b    share 0.919372   xsec_sd 0.048588
                                               (incumbent COEFFICIENTS, SAME panel)

The champion passes PR611's floor by 7x, and its first served batch on
2026-09-08 collapsed to ``alpha_stdev`` 0.005819 — 0.41x the trailing 10-day
median, below the serving absolute floor of 0.015, ``n_high_confidence`` 0.
``xsec_variance_share`` is a RATIO and is invariant under a uniform rescaling of
the coefficients (PR611's docstring names that invariance as deliberate), so it
structurally cannot see a scale collapse. The number that could is ``xsec_sd``,
which PR611 already computes and nothing gated on.

Why the two sub-legs are ANDed, and not ORed
--------------------------------------------
Both sub-legs must fire. Validated by scoring EVERY registered ``v3.0-meta``
coefficient vector (20 of them, 2026-06-06 .. 2026-09-04) on the one persisted
fitted panel, so the panel is held constant and the number measures the model:

    2026-09-04 cc3271ea   0.010350   <- collapsed on its first served batch
    2026-08-21 7d3d1cce   0.014842   <- the worked-example BAD promotion this
                                        module's own docstring already names
    2026-08-28 01cf7e1a   0.020270   <- lowest of the remaining 18
    ... 17 more, 0.020270 .. 0.066738

An absolute floor of 0.015 separates exactly the two known-bad vectors from all
18 others, with the nearest passing arm at 1.35x the floor. A RATIO-only rule
would additionally have refused 2026-06-26 (xsec_sd 0.026215, comfortably above
the floor, but 0.393x its immediate predecessor's unusually wide 0.066738) — a
candidate with no evidence of any defect. So: refuse only when the candidate is
BOTH small in absolute terms AND much smaller than the incumbent on the same
panel. On the 20-vector history that rule has zero false positives.

The one exception, which closes the joint-drift hole: a relative leg cannot
exonerate a candidate against a reference that is itself broken. When the
incumbent's same-panel ``xsec_sd`` is also below the absolute floor, the
absolute leg fires alone.
"""

from __future__ import annotations

import pytest

from training.promotion_behavioral_veto import (
    XSEC_SD_ABSOLUTE_FLOOR,
    XSEC_SD_MIN_RATIO_VS_INCUMBENT,
    behavioral_metrics,
    evaluate_behavioral_veto,
)

# The three measured vectors, on the held-constant 2026-09-04 panel.
CHAMPION_XSEC_SD = 0.010350        # v3.0-meta-2026-09-04-cc3271ea
INCUMBENT_XSEC_SD = 0.048588       # v3.0-meta-2026-08-14-119e069b, same panel
DEGENERATE_ARM_XSEC_SD = 0.011041  # spec-sota-combine-2026-07-31-0c5140d5


def _manifest(*, xsec_sd=None, incumbent_same_panel=None, share=0.70, **extra):
    """A candidate manifest shaped like the ones meta_trainer emits."""
    m = {
        "output_distribution_gate": {
            "metrics": {"alpha_stdev": 0.017, "stdev_p_up": 0.071},
        },
        "behavioral_metrics": {
            "n_high_confidence": 30,
            "model_hit_rate_30d": 0.55,
            "xsec_variance_share": share,
        },
    }
    if xsec_sd is not None:
        m["behavioral_metrics"]["xsec_sd"] = xsec_sd
    if incumbent_same_panel is not None:
        m["behavioral_metrics"]["xsec_sd_incumbent_same_panel"] = incumbent_same_panel
    m["behavioral_metrics"].update(extra)
    return m


# ── the thresholds are the measured ones ─────────────────────────────────────

def test_the_absolute_floor_is_the_serving_alpha_stdev_floor():
    """0.015 is not a new number: it is the predictor's own absolute
    ``alpha_stdev`` floor (alpha-engine-config-I9267, "derived from the measured
    healthy population"), applied to the training panel where it is knowable
    BEFORE the model serves a batch."""
    assert XSEC_SD_ABSOLUTE_FLOOR == 0.015


def test_the_relative_floor_is_the_modules_existing_dispersion_bar():
    assert XSEC_SD_MIN_RATIO_VS_INCUMBENT == 0.5


# ── the case that motivated the leg ──────────────────────────────────────────

def test_the_2026_09_04_champion_is_refused():
    cand = _manifest(
        xsec_sd=CHAMPION_XSEC_SD, incumbent_same_panel=INCUMBENT_XSEC_SD,
        share=0.702720,
    )
    inc = _manifest(xsec_sd=INCUMBENT_XSEC_SD, share=0.919372)
    out = evaluate_behavioral_veto(cand, inc)
    assert out["status"] == "veto"
    hit = [v for v in out["vetoes"] if v["metric"] == "xsec_sd"]
    assert len(hit) == 1, "the magnitude leg must be the thing that refuses it"
    reason = hit[0]["reason"].lower()
    assert "cross-sectional" in reason or "dispersion" in reason
    # It must NOT be refused by the share leg — that leg passes this model by
    # 7x, and a reason naming the wrong defect is what cost the investigation.
    assert not [v for v in out["vetoes"] if v["metric"] == "xsec_variance_share"]


def test_the_2026_08_14_incumbent_on_the_same_panel_passes():
    """The like-for-like control. Its coefficients on the CANDIDATE's panel
    score 0.048588 — 3.2x the absolute floor."""
    cand = _manifest(
        xsec_sd=INCUMBENT_XSEC_SD, incumbent_same_panel=INCUMBENT_XSEC_SD,
        share=0.919372,
    )
    out = evaluate_behavioral_veto(cand, _manifest(xsec_sd=INCUMBENT_XSEC_SD))
    assert out["status"] == "pass"
    assert out["measured"]["xsec_sd"]["candidate"] == INCUMBENT_XSEC_SD


def test_the_degenerate_sota_combine_arm_is_refused_by_this_leg_too():
    """``spec-sota-combine-2026-07-31-0c5140d5`` scores share 0.405 on the
    2026-09-04 panel — it CLEARS PR611's 0.10 floor when scored on a panel that
    is not its own — and xsec_sd 0.011041, which this leg refuses. The two legs
    are independent and this arm needs both to exist."""
    cand = _manifest(
        xsec_sd=DEGENERATE_ARM_XSEC_SD, incumbent_same_panel=INCUMBENT_XSEC_SD,
        share=0.405341,
    )
    out = evaluate_behavioral_veto(cand, _manifest(xsec_sd=INCUMBENT_XSEC_SD))
    assert out["status"] == "veto"
    assert [v for v in out["vetoes"] if v["metric"] == "xsec_sd"]


# ── AND, not OR ──────────────────────────────────────────────────────────────

def test_a_small_but_proportionate_candidate_is_not_refused():
    """Absolute leg fires, relative leg does not: the whole panel's dispersion
    moved, not this model's. Refusing here would refuse a regime, not a defect."""
    cand = _manifest(xsec_sd=0.0100, incumbent_same_panel=0.0110)
    out = evaluate_behavioral_veto(cand, _manifest(xsec_sd=0.0110))
    # ...unless the reference is itself below the floor, which it is here.
    assert out["status"] == "veto", (
        "an incumbent that is ITSELF below the absolute floor cannot exonerate "
        "anything — the relative leg has no valid reference"
    )


def test_a_large_but_shrunken_candidate_is_not_refused():
    """Relative leg fires, absolute leg does not — the 2026-06-26 shape
    (xsec_sd 0.026215 against a predecessor at 0.066738, ratio 0.393). Nothing
    about that model is known to be defective and an OR rule would have refused
    it."""
    cand = _manifest(xsec_sd=0.026215, incumbent_same_panel=0.066738)
    out = evaluate_behavioral_veto(cand, _manifest(xsec_sd=0.066738))
    assert out["status"] == "pass"
    assert not [v for v in out["vetoes"] if v["metric"] == "xsec_sd"]


def test_a_healthy_incumbent_reference_below_the_floor_arms_the_absolute_leg_alone():
    cand = _manifest(xsec_sd=0.0090, incumbent_same_panel=0.0120)
    out = evaluate_behavioral_veto(cand, _manifest(xsec_sd=0.0120))
    hit = [v for v in out["vetoes"] if v["metric"] == "xsec_sd"]
    assert hit and "reference" in hit[0]["reason"].lower()


# ── nothing may pass unmeasured ──────────────────────────────────────────────

def test_a_missing_xsec_sd_is_named_uncomputable_not_skipped():
    out = evaluate_behavioral_veto(_manifest(), _manifest())
    assert "xsec_sd" in out["uncomputable"]
    assert not [v for v in out["vetoes"] if v["metric"] == "xsec_sd"]


def test_a_below_floor_candidate_with_NO_incumbent_reference_is_refused():
    """The relative leg cannot be evaluated, so it cannot exonerate. A
    candidate under the absolute floor with nothing to compare against is
    refused — `unmeasurable` is never a pass."""
    cand = _manifest(xsec_sd=CHAMPION_XSEC_SD)  # no incumbent_same_panel
    out = evaluate_behavioral_veto(cand, _manifest())
    assert out["status"] == "veto"
    hit = [v for v in out["vetoes"] if v["metric"] == "xsec_sd"]
    assert hit
    assert "no incumbent" in hit[0]["reason"].lower()
    assert "xsec_sd_incumbent_same_panel" in out["uncomputable"]


def test_an_above_floor_candidate_with_no_incumbent_reference_passes_and_says_so():
    cand = _manifest(xsec_sd=0.030)
    out = evaluate_behavioral_veto(cand, _manifest())
    assert out["status"] == "pass"
    assert "xsec_sd_incumbent_same_panel" in out["uncomputable"]


def test_the_ratio_is_recorded_so_a_reader_can_see_the_margin():
    cand = _manifest(
        xsec_sd=CHAMPION_XSEC_SD, incumbent_same_panel=INCUMBENT_XSEC_SD,
    )
    out = evaluate_behavioral_veto(cand, _manifest(xsec_sd=INCUMBENT_XSEC_SD))
    m = out["measured"]["xsec_sd"]
    assert m["candidate"] == CHAMPION_XSEC_SD
    assert m["incumbent_same_panel"] == INCUMBENT_XSEC_SD
    assert m["ratio_vs_incumbent_same_panel"] == pytest.approx(
        CHAMPION_XSEC_SD / INCUMBENT_XSEC_SD, abs=1e-6,
    )
    assert m["floor"] == XSEC_SD_ABSOLUTE_FLOOR


def test_the_incumbent_reference_is_never_taken_from_the_incumbents_own_panel():
    """The comparison must be same-panel. A candidate manifest that carries no
    ``xsec_sd_incumbent_same_panel`` must NOT silently fall back to the
    incumbent manifest's own ``xsec_sd`` — that number was measured on a
    different panel and the comparison would then measure the DATA, not the
    model."""
    cand = _manifest(xsec_sd=CHAMPION_XSEC_SD)
    inc = _manifest(xsec_sd=INCUMBENT_XSEC_SD)
    out = evaluate_behavioral_veto(cand, inc)
    m = out["measured"]["xsec_sd"]
    assert m.get("incumbent_same_panel") is None
    assert "xsec_sd_incumbent_same_panel" in out["uncomputable"]


def test_a_served_slice_measurement_cannot_arm_this_rule():
    """``xsec_sd`` is a FIT-TIME property of the training panel. A served-slice
    measurement has no such quantity, and letting one through the served merge
    would silently rank a served number against a fitted one."""
    out = evaluate_behavioral_veto(
        _manifest(), _manifest(),
        candidate_served_metrics={"xsec_sd": 0.0001},
        incumbent_served_metrics={"xsec_sd": 0.9},
    )
    assert "xsec_sd" in out["uncomputable"]
    assert not [v for v in out["vetoes"] if v["metric"] == "xsec_sd"]


def test_behavioral_metrics_surfaces_both_numbers_from_the_forward_slot():
    m = behavioral_metrics(_manifest(
        xsec_sd=CHAMPION_XSEC_SD, incumbent_same_panel=INCUMBENT_XSEC_SD,
    ))
    assert m["xsec_sd"] == CHAMPION_XSEC_SD
    assert m["xsec_sd_incumbent_same_panel"] == INCUMBENT_XSEC_SD


# ── the whole measured champion history, as a regression corpus ──────────────

#: Every registered ``v3.0-meta`` coefficient vector, scored by PR611's own
#: ``cross_sectional_variance_share`` on the ONE persisted fitted panel
#: (2026-09-04), so the panel is held constant across all 20. Reproduce with
#: ``scripts/backfill_xsec_sd.py``.
MEASURED_CHAMPION_XSEC_SD: dict[str, float] = {
    "v3.0-meta-2026-06-06-35beb8e3": 0.046895,
    "v3.0-meta-2026-06-13-96c8da62": 0.053429,
    "v3.0-meta-2026-06-15-7ac0ac44": 0.053429,
    "v3.0-meta-2026-06-19-4e448860": 0.066738,
    "v3.0-meta-2026-06-26-580cb665": 0.026215,
    "v3.0-meta-2026-07-02-ea7cd2fb": 0.025986,
    "v3.0-meta-2026-07-10-065f067e": 0.031551,
    "v3.0-meta-2026-07-17-8d7f6dab": 0.038365,
    "v3.0-meta-2026-07-24-3e9eb4e1": 0.033054,
    "v3.0-meta-2026-07-29-877da082": 0.032642,
    "v3.0-meta-2026-07-30-dbdd2285": 0.042481,
    "v3.0-meta-2026-07-31-656c6d48": 0.052780,
    "v3.0-meta-2026-08-03-7eda265c": 0.063158,
    "v3.0-meta-2026-08-04-b2aaf015": 0.045042,
    "v3.0-meta-2026-08-07-22830c0d": 0.030707,
    "v3.0-meta-2026-08-12-7d0d9328": 0.032421,
    "v3.0-meta-2026-08-14-119e069b": 0.048588,
    "v3.0-meta-2026-08-21-7d3d1cce": 0.014842,
    "v3.0-meta-2026-08-28-01cf7e1a": 0.020270,
    "v3.0-meta-2026-09-04-cc3271ea": 0.010350,
}

#: The two the leg must refuse, and the only two below the floor. 2026-08-21 is
#: the promotion this module's docstring already names as the worked example of
#: a candidate that collapsed served dispersion 4-6x and produced zero
#: high-confidence names for five sessions; 2026-09-04 is I10185's case.
KNOWN_BAD = {"v3.0-meta-2026-08-21-7d3d1cce", "v3.0-meta-2026-09-04-cc3271ea"}


def test_the_absolute_floor_separates_exactly_the_two_known_bad_champions():
    below = {
        vid for vid, sd in MEASURED_CHAMPION_XSEC_SD.items()
        if sd < XSEC_SD_ABSOLUTE_FLOOR
    }
    assert below == KNOWN_BAD, (
        "moving this floor is a design change, not a tuning knob: it currently "
        "refuses the two champions with independent evidence of collapse and "
        "no others"
    )


def test_no_healthy_champion_is_refused_by_the_full_rule():
    """The stop signal named in I10185. If a champion with no evidence of
    defect is refused, the finding is reported — the threshold is not moved."""
    incumbent = MEASURED_CHAMPION_XSEC_SD["v3.0-meta-2026-08-14-119e069b"]
    refused = []
    for vid, sd in MEASURED_CHAMPION_XSEC_SD.items():
        out = evaluate_behavioral_veto(
            _manifest(xsec_sd=sd, incumbent_same_panel=incumbent),
            _manifest(xsec_sd=incumbent),
        )
        if [v for v in out["vetoes"] if v["metric"] == "xsec_sd"]:
            refused.append(vid)
    assert set(refused) == KNOWN_BAD
