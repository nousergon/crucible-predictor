"""Tests for cross-sectional level-neutralization of predicted_alpha (L4487).

Covers the observe-before-cutover contract:
  * flag OFF → live fields (predicted_alpha + direction/confidence/p_up) are
    byte-identical to the input; only the additive observe fields are stamped;
  * flag ON → predicted_alpha becomes the centered value and direction is
    re-derived from it (the gbm_veto + executor optimizer inherit it);
  * the common-mode skew→flush regression: an all-negative batch with rank
    spread flips ~half UP after centering;
  * mean-removal arithmetic + the n<2 degenerate guard.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.level_neutralization import (
    apply_cross_sectional_neutralization,
    cross_sectional_mean,
)


def _batch(alphas: list[float]) -> list[dict]:
    """Build a predictions batch with direction/p_up derived from raw alpha
    sign the same way the inference loop's linear fallback would (calibrator
    omitted in these tests → _derive uses the same pure-linear formula, so
    flag-OFF vs the input is an exact identity check)."""
    lc = 0.15
    out = []
    for i, a in enumerate(alphas):
        import numpy as np

        p_up = float(np.clip(0.5 + a / (2.0 * lc), 0.0, 1.0))
        out.append({
            "ticker": f"T{i}",
            "predicted_alpha": round(a, 6),
            "predicted_direction": "UP" if p_up >= 0.5 else "DOWN",
            "prediction_confidence": round(abs(p_up - 0.5) * 2.0, 4),
            "p_up": round(p_up, 4),
            "p_down": round(1.0 - p_up, 4),
            "combined_rank": i + 1,
        })
    return out


def test_cross_sectional_mean():
    assert cross_sectional_mean(_batch([0.02, -0.04, 0.0])) == \
        (0.02 - 0.04 + 0.0) / 3


def test_flag_off_is_byte_identical_live_fields():
    alphas = [-0.08, -0.06, -0.05, -0.03, 0.01]
    batch = _batch(alphas)
    snapshot = [dict(p) for p in batch]

    block = apply_cross_sectional_neutralization(
        batch, enabled=False, calibrator=None, label_clip=0.15,
    )

    # Live fields untouched.
    for orig, p in zip(snapshot, batch):
        assert p["predicted_alpha"] == orig["predicted_alpha"]
        assert p["predicted_direction"] == orig["predicted_direction"]
        assert p["prediction_confidence"] == orig["prediction_confidence"]
        assert p["p_up"] == orig["p_up"]
        # Additive observe fields appear; raw == live alpha when off.
        assert p["predicted_alpha_raw"] == orig["predicted_alpha"]
        assert "predicted_alpha_centered" in p
    assert block["enabled"] is False
    assert block["applied"] is False


def test_flag_on_centers_and_rederives():
    alphas = [-0.08, -0.06, -0.05, -0.03, 0.01]
    mean = sum(alphas) / len(alphas)  # = -0.042
    batch = _batch(alphas)

    block = apply_cross_sectional_neutralization(
        batch, enabled=True, calibrator=None, label_clip=0.15,
    )

    assert block["applied"] is True
    assert abs(block["xsec_mean_removed"] - mean) < 1e-9
    # predicted_alpha is now the centered value; cross-section sums to ~0.
    assert abs(sum(p["predicted_alpha"] for p in batch)) < 1e-6
    for a, p in zip(alphas, batch):
        assert abs(p["predicted_alpha"] - (a - mean)) < 1e-6
        # direction re-derived from centered sign.
        assert p["predicted_direction"] == ("UP" if (a - mean) >= 0 else "DOWN")


def test_skew_flush_regression():
    # The 2026-06-02 pathology in miniature: every name negative (common-mode
    # macro) but with genuine cross-sectional spread → raw skew ~100% DOWN.
    alphas = [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.10]
    raw_block = apply_cross_sectional_neutralization(
        _batch(alphas), enabled=False, calibrator=None, label_clip=0.15,
    )
    # Raw: all DOWN (would flush the book to SPY).
    assert raw_block["direction_skew_raw"] == 1.0
    # Centering flips the top half UP → skew ~0.5, optimizer holds the winners.
    assert raw_block["direction_skew_centered"] <= 0.5
    assert raw_block["n_direction_flips"] >= 4


def test_n_lt_2_is_skipped():
    batch = _batch([-0.05])
    block = apply_cross_sectional_neutralization(
        batch, enabled=True, calibrator=None, label_clip=0.15,
    )
    assert block["applied"] is False
    assert block["reason"] == "n<2"
    # The single name's live alpha is NOT zeroed.
    assert batch[0]["predicted_alpha"] == -0.05
    assert batch[0]["predicted_alpha_raw"] == -0.05


def test_empty_batch():
    block = apply_cross_sectional_neutralization(
        [], enabled=True, calibrator=None, label_clip=0.15,
    )
    assert block["applied"] is False


# ── config-I7337 layer 3: the anchor is declared by the step that establishes
# it ────────────────────────────────────────────────────────────────────────
#
# The executor's optimizer compares every alpha against a SPY = 0.0 anchor,
# which is only meaningful for a market-relative alpha. `assert_optimizer_
# anchor` (crucible-executor/executor/alpha_contract.py) REFUSES a batch whose
# numeric `predicted_alpha` carries no `alpha_anchor`, so an unstamped batch
# fails the whole solve closed — measured 2026-08-14: 22 of 23 live records
# raised AlphaAnchorError.


def test_enabled_stamps_the_optimizer_anchor_on_every_record():
    batch = _batch([0.02, -0.04, 0.01])
    block = apply_cross_sectional_neutralization(
        batch, enabled=True, calibrator=None, label_clip=0.15,
    )
    assert all(p["alpha_anchor"] == "market_relative_21d_log" for p in batch)
    assert block["alpha_anchor"] == "market_relative_21d_log"


def test_disabled_stamps_NOTHING_because_the_anchor_does_not_hold():
    """Load-bearing. With the transform off, `predicted_alpha` is the model's
    raw output carrying the batch's common-mode level — the optimizer's
    SPY = 0.0 anchor does not apply to it. An unstamped batch is REFUSED
    downstream, which is the correct outcome; stamping here would assert a
    property the number does not have and hand the solver a silently wrong
    alpha."""
    batch = _batch([0.02, -0.04, 0.01])
    block = apply_cross_sectional_neutralization(
        batch, enabled=False, calibrator=None, label_clip=0.15,
    )
    assert all("alpha_anchor" not in p for p in batch)
    assert block["alpha_anchor"] is None


def test_degenerate_cross_section_stamps_nothing():
    """n < 2 skips the transform entirely — centering one name against its own
    mean would zero it. No transform, no anchor."""
    batch = _batch([0.02])
    block = apply_cross_sectional_neutralization(
        batch, enabled=True, calibrator=None, label_clip=0.15,
    )
    assert "alpha_anchor" not in batch[0]
    assert block.get("alpha_anchor") is None


def test_anchor_string_matches_the_executors_contract():
    """Consumer contract. The literal is duplicated in
    `crucible-executor/executor/alpha_contract.py::OPTIMIZER_ALPHA_ANCHOR`
    deliberately — the two repos deploy independently, so a shared import
    would make the executor's contract wait on a predictor release. This pin
    is what makes the duplication safe: a rename on either side fails here,
    not at 05:15 PT on a trading morning."""
    from inference.level_neutralization import OPTIMIZER_ALPHA_ANCHOR

    assert OPTIMIZER_ALPHA_ANCHOR == "market_relative_21d_log"
