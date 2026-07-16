"""W4.2 dead-L1 pruning monitor — OBSERVE-ONLY (config#1994).

Ratified observe-only by Brian 2026-07-10 via the console Decision Queue
(config#1926, "Approve as scoped, reversible-de-weight posture"). Split from the
CF5 research tail (config#707) per the 2026-07-08 operator decision.

**What this is.** The Phase-0 diagnostic of the generalized dead-L1 prune: it
fuses the two *free* per-L1 reads already produced at meta-training time —

  1. the shipped leave-one-model-out (LOMO) leak-free meta ΔIC
     (``meta_oos_ic_leakfree_per_l1_dropout``, predictor#222 / W4), i.e. how much
     leak-free cross-sectional IC the L2 loses when a given L1/context feature is
     dropped from the stack; and
  2. the fitted meta-Ridge's *scale-free standardized coefficient magnitude*
     (``MetaModel._importance["standardized_coef"]``) — a truly dead learner is
     one the Ridge has already shrunk toward zero.

— into a per-feature ``is_dead_candidate`` flag and a summary, so per-L1 marginal
contribution surfaces on the report card / console every weekly training run.

**What this is NOT — the load-bearing safety contract.** This module is a pure,
side-effect-free *diagnostic*. It NEVER de-weights, drops, or deletes an L1; the
serving L2 stack is untouched. Per the ratified scope, the *retire ACTION*
(reversible de-weight toward zero in a shadow lane, mirroring the champion/
challenger OBSERVE substrate, ARCHITECTURE.md §37) is a SEPARATE, later step that
fires ONLY when a candidate is persistently dead — the ratified retirement bar is
non-degrading LOMO ΔIC across ≥ N consecutive weekly walk-forward cohorts AND a
non-positive monthly Shapley value, with the selection surviving CSCV/PBO
(config#1995). That persistence is read DOWNSTREAM across the accumulated weekly
OBSERVE cohorts (this run contributes one cohort), and choosing *which* L1 to
retire spends lifetime multiple-testing budget charged to config#624. None of
that happens here.

The two caveats that forced the reversible-de-weight posture — the *diversity
premium* (a near-zero-weighted L1 can still add uncorrelated perspective) and
*cyclical, not monotonic, degradation* (an L1 dead this regime may revive) — are
exactly why a single-run ``is_dead_candidate`` is a CANDIDATE, never a verdict.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# A feature carries no leak-free meta IC when dropping it does NOT reduce the
# leak-free cross-sectional IC — i.e. ``delta_vs_full = full_ic - ic_without <= 0``
# (dropping it leaves IC flat or improves it). A tiny positive tolerance absorbs
# LOMO estimation noise so a feature that is dead-to-the-noise-floor still flags.
DEAD_L1_LOMO_DELTA_EPS: float = 1e-4

# A feature the meta-Ridge has shrunk toward zero has |standardized coef| ~ 0.
# Standardized (scale-free) coefficients are compared, not raw ones, so features
# on different scales are directly comparable (model/meta_model.py::_compute_importance).
DEAD_L1_COEF_MAG_EPS: float = 1e-3


def summarize_dead_l1(
    per_l1_dropout: dict | None,
    importance: dict | None,
    coefficients: dict | None = None,
) -> dict:
    """Fuse the LOMO leak-free ΔIC read with the Ridge coefficient magnitude
    into a per-feature dead-L1 CANDIDATE summary. OBSERVE-ONLY: pure, no side
    effects, never raises for the caller (returns a ``status`` sentinel instead).

    Parameters
    ----------
    per_l1_dropout : dict
        The ``meta_oos_ic_leakfree_per_l1_dropout`` structure (W4): expects
        ``{"status": "ok", "full_xsec_ic": float, "per_feature": {name: {
        "delta_vs_full": float|None, ...}}}``. Any other status short-circuits to
        a ``status`` sentinel — the diagnostic simply does not run this week.
    importance : dict
        ``MetaModel._importance`` — read for ``standardized_coef`` (scale-free).
    coefficients : dict, optional
        ``MetaModel._coefficients`` (raw, incl. ``intercept``) — surfaced for
        context only; classification uses the standardized magnitude.

    Returns
    -------
    dict
        ``{"status": "ok", "full_xsec_ic": float, "n_dead_candidates": int,
        "dead_candidates": [name, ...], "per_feature": {name: {lomo_delta_ic,
        coef_magnitude, raw_coef, is_dead_candidate, ...}}, "thresholds": {...},
        "contract": "observe-only; ..."}`` — or ``{"status": <sentinel>}``.
    """
    contract = (
        "OBSERVE-ONLY diagnostic (config#1994): flags dead-L1 CANDIDATES only. "
        "No de-weight, no deletion; serving L2 untouched. Retirement requires "
        "persistence across >=N weekly cohorts + non-positive Shapley surviving "
        "CSCV/PBO, charged to config#624 — evaluated downstream, not here."
    )

    if not isinstance(per_l1_dropout, dict):
        return {"status": "not_run", "reason": "per_l1_dropout missing"}
    status = per_l1_dropout.get("status")
    if status != "ok":
        # Mirror the upstream W4 status (not_run / error / no_leakfree_preds) so
        # the report card shows WHY the monitor produced nothing this week.
        return {"status": status or "not_run", "reason": "LOMO read not ok"}

    per_feature = per_l1_dropout.get("per_feature")
    if not isinstance(per_feature, dict) or not per_feature:
        return {"status": "not_run", "reason": "no per-feature LOMO rows"}

    std_coef = {}
    if isinstance(importance, dict):
        std_coef = importance.get("standardized_coef") or {}
    coefficients = coefficients if isinstance(coefficients, dict) else {}

    out: dict = {}
    dead: list[str] = []
    for name, row in per_feature.items():
        if not isinstance(row, dict):
            continue
        delta = row.get("delta_vs_full")
        coef_mag = abs(float(std_coef[name])) if name in std_coef else None
        raw_coef = coefficients.get(name)

        carries_no_ic = isinstance(delta, (int, float)) and delta <= DEAD_L1_LOMO_DELTA_EPS
        near_zero_weight = coef_mag is not None and coef_mag <= DEAD_L1_COEF_MAG_EPS
        # A candidate must satisfy BOTH lenses; when the coefficient is
        # unavailable we cannot confirm near-zero weight, so we do not flag.
        is_dead = bool(carries_no_ic and near_zero_weight)

        out[name] = {
            "lomo_delta_ic": (round(float(delta), 6) if isinstance(delta, (int, float)) else None),
            "coef_magnitude": (round(coef_mag, 6) if coef_mag is not None else None),
            "raw_coef": (round(float(raw_coef), 6) if isinstance(raw_coef, (int, float)) else None),
            "carries_no_leakfree_ic": carries_no_ic,
            "near_zero_weight": near_zero_weight,
            "is_dead_candidate": is_dead,
        }
        if is_dead:
            dead.append(name)

    result = {
        "status": "ok",
        "full_xsec_ic": per_l1_dropout.get("full_xsec_ic"),
        "n_dead_candidates": len(dead),
        "dead_candidates": sorted(dead),
        "per_feature": out,
        "thresholds": {
            "lomo_delta_eps": DEAD_L1_LOMO_DELTA_EPS,
            "coef_magnitude_eps": DEAD_L1_COEF_MAG_EPS,
        },
        "contract": contract,
    }
    log.info(
        "W4.2 dead-L1 monitor (OBSERVE, NOT gated): %d dead candidate(s)=%s "
        "over %d meta features. Candidates carry ~0 leak-free ΔIC AND ~0 "
        "standardized Ridge weight — this run is ONE weekly cohort; retirement "
        "needs persistence + Shapley (config#624), never taken here.",
        len(dead), sorted(dead), len(out),
    )
    return result
