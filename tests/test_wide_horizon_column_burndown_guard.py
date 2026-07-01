"""Wide-horizon burn-down guard (EPIC config#1483 Phase 3, seeded by the
config#1527 calibrator-label cutover).

Consumes the shared ratchet primitive ``nousergon_lib.quant.horizon_guard``
(lifted on this second adoption from the backtester's original guard — see
crucible-backtester/tests/test_wide_horizon_column_burndown_guard.py for the
full rationale). Rules:

  * a production read of a wide horizon-suffixed ``score_performance`` outcome
    column fails CI — consumers read the long-format
    ``score_performance_outcomes`` store via
    ``nousergon_lib.quant.horizons.HorizonPolicy`` instead;
  * ``_MIGRATING`` is EMPTY: config#1527 migrated this repo's only
    score_performance consumer (the research-calibrator label in
    ``training/meta_trainer.py``) in the same PR that seeded this guard, so
    the silent-starvation bug class is already structurally dead here;
  * ``_EXEMPT`` holds the predictor's OWN feature/label columns whose names
    collide with the wide score_performance literals (the ``return_5d`` /
    ``forward_return_5d`` label lineage, ``spy_30d_return`` regime features).
    Each exemption is pinned to the EXACT literal set it may contain — a new
    wide-column literal appearing in an exempt file fails the honesty test
    below, so the exemption can never hide a real score_performance read.
"""

from __future__ import annotations

from pathlib import Path

from nousergon_lib.quant.horizon_guard import assert_burndown, wide_columns_in

_REPO = Path(__file__).resolve().parent.parent

# Burned down to {} at seed time — config#1527 migrated the only consumer.
_MIGRATING: frozenset[str] = frozenset()

# The predictor's own feature/label columns (NOT score_performance outcome
# reads), pinned per file to exactly the literals they legitimately contain:
# ``return_5d`` is the L1 label/feature lineage (meta_trainer's hit is the
# substring inside its own ``forward_return_5d`` label column);
# ``spy_30d_return`` is a regime feature computed from SPY prices.
_EXEMPT_PINNED: dict[str, set[str]] = {
    "config.py": {"return_5d"},
    "data/label_generator.py": {"return_5d"},
    "model/calibrator.py": {"return_5d"},
    "model/gbm_scorer.py": {"return_5d"},
    "regime/features.py": {"spy_30d_return"},
    "regime/substrate.py": {"spy_30d_return"},
    "training/meta_trainer.py": {"return_5d"},
}
_EXEMPT = frozenset(_EXEMPT_PINNED)


def test_wide_horizon_column_burndown():
    assert_burndown(_REPO, migrating=_MIGRATING, exempt=_EXEMPT)


def test_exempt_literal_sets_are_pinned_exactly():
    """Honesty check: each exempt file may contain ONLY its pinned own-column
    literals. Any additional wide-column literal (e.g. a new beat_spy_* /
    spy_*_return / log_alpha_21d read) means a real score_performance outcome
    read snuck into an exempt file — migrate it to the long store instead."""
    drift = {}
    for rel, allowed in sorted(_EXEMPT_PINNED.items()):
        hits = set(wide_columns_in(_REPO / rel))
        if hits != allowed:
            drift[rel] = {"pinned": sorted(allowed), "actual": sorted(hits)}
    assert not drift, (
        f"exempt file literal drift — a new wide-column literal must migrate "
        f"to score_performance_outcomes (config#1483), a vanished one means "
        f"the pin should shrink: {drift}"
    )
