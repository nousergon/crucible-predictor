"""ic_reliability.py — classify a predictor IC field's methodological reliability.

alpha-engine-config#969 residual: the meta-training manifest emits ~a dozen
"IC" numbers that are NOT comparable — some are genuinely leak-free (per-fold
purged/embargoed walk-forward, CPCV over purged paths), others are in-sample at
the meta level (the L2 read back on its own fit pool), overlapping-pooled, or
sub-horizon proxies. The manifest distinguishes them ONLY by name suffix, so a
downstream consumer (the report-card evaluator) that grades an IC has no
structured way to know whether the number is trustworthy or inflated.

This module is the SINGLE source of the reliability taxonomy on the producer
side. It is a pure name→classification map (no I/O, no numpy) so it is trivially
unit-testable and importable from the manifest-assembly step. The evaluator
consumes the emitted ``ic_reliability`` map; it does NOT re-derive the rule
(the classification is a producer contract — the producer knows how each number
was computed).

Rule (based on the field-name / suffix taxonomy in ``meta_trainer.py``):

  HIGH — the number was produced by a leak-free methodology:
    * ``*_leakfree*``        leak-free per-fold refit walk-forward
    * ``*_cpcv*``            combinatorial purged cross-validation
    * walk-forward OOS median ICs (``*_median_ic`` under ``walk_forward``)
    * standalone leak-free cross-sectional alpha IC (``*_standalone_alpha_ic``,
      ``residual_momentum_leakfree_oos_ic``, ``*_leakfree_oos_ic``)

  LOW — the number has a known validity caveat:
    * ``*_in_sample*``       fit IC read on the same rows the model was fit on
    * ``meta_model_ic``      the L2's in-sample Pearson fit IC (unsuffixed alias)
    * ``meta_model_oos_ic``  OOS at the L1 level but IN-SAMPLE at the meta level,
                             and a *pooled overlapping* Spearman (overlapping
                             21d labels → inflated) — see meta_trainer.py
    * ``*_test_ic``          single train/val/test split IC (in-sample-prone,
                             not walk-forward / not purged)
    * ``*_val_ic`` / ``*_train_ic`` single early-stop-holdout split reads
    * anything not recognised as leak-free (fail-safe to LOW so an UNKNOWN new
      IC field is never silently trusted as high-reliability).
"""
from __future__ import annotations

from typing import Literal

Reliability = Literal["high", "low"]

# Substrings that mark a genuinely leak-free methodology. Checked first; the
# most specific / trustworthy wins.
_HIGH_MARKERS: tuple[str, ...] = (
    "leakfree",
    "leak_free",
    "cpcv",
    "median_ic",              # walk-forward per-fold median (OOS)
    "standalone_alpha_ic",    # standalone leak-free cross-sectional alpha IC
)

# Substrings that mark a KNOWN-caveat (in-sample / overlapping-pooled /
# split-only / sub-horizon-proxy) methodology.
_LOW_MARKERS: tuple[str, ...] = (
    "in_sample",
    "insample",
    "test_ic",
    "val_ic",
    "train_ic",
)

# Exact field names whose reliability is not inferable from a generic suffix.
# ``meta_model_ic`` is the L2 in-sample Pearson fit; ``meta_model_oos_ic`` is
# OOS at L1 but in-sample + overlapping-pooled at the meta level.
_EXPLICIT_LOW: frozenset[str] = frozenset(
    {"meta_model_ic", "meta_model_oos_ic"}
)


def classify_ic_reliability(field_name: str) -> Reliability:
    """Classify an IC field as ``"high"`` (leak-free) or ``"low"`` (caveated).

    Pure function of the field name. Fail-safe: an unrecognised field is LOW so
    a newly-added IC is never silently graded as high-reliability before its
    methodology is audited and added here.
    """
    name = field_name.lower()

    if name in _EXPLICIT_LOW:
        return "low"

    # A high marker wins UNLESS an explicit-low exact match already fired above.
    for marker in _HIGH_MARKERS:
        if marker in name:
            return "high"

    for marker in _LOW_MARKERS:
        if marker in name:
            return "low"

    # Unknown IC field → conservative LOW.
    return "low"


def build_ic_reliability_map(ic_field_names: list[str]) -> dict[str, Reliability]:
    """Classify every field name in ``ic_field_names`` → the manifest's
    ``ic_reliability`` block. Order-stable, one entry per input name."""
    return {name: classify_ic_reliability(name) for name in ic_field_names}
