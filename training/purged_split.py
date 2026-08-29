"""training/purged_split.py — the one early-stopping split every L1 arm uses.

alpha-engine-config-I9333. Companion to ``training/l1_fit_validity.py``
(I9271), which refuses to SHIP a degenerate L1; this module refuses to
PRODUCE one, by making the split the arm early-stops on a purged, embargoed,
multi-date block instead of a bare index cut.

## What was wrong

``research_gbm``'s early-stop split was a contiguous row-index tail::

    split_idx = int(n_finite * 0.8)
    fit(X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:])

on a date-ordered panel whose rows-per-date is ~899 after the 2026-07-18
research-universe expansion (27 tickers -> 902, measured over every
``s3://alpha-engine-research/signals/<date>/signals.json``). Three defects,
all measured:

1. **The validation block was one to two cross-sections wide.** The val
   blocks across three consecutive weekly retrains were 386 / 961 / 1675 rows
   = ~15 sparse-era dates / **1.07** dates / **1.87** dates. ``val_ic`` was
   one date's cross-sectional accident, and ``best_iteration`` was decided by
   it: **3 -> 500 -> 2**, ``val_ic`` 0.0430 -> 0.2137 -> 0.0140.
2. **No purge and no embargo against the label horizon.** The label is
   ``actual_fwd_10d > 0``; adjacent train and val rows share ~90% of their
   forward window. On 2026-08-21 the fit ran all 500 rounds —
   ``best_iteration == n_estimators``, so early stopping never fired at all.
   That is a leak signature, not health, and it was the vintage every
   downstream reader treated as the healthy baseline.
3. **The design matrix had one varying column of nine.** ``conviction`` is
   ``stable`` for all 903 tickers in every snapshot since 2026-07-18,
   ``sector_modifiers`` holds a single value across all 11 sectors, and the
   six macro columns were zero-filled (``-I9256``, ``-I9324``). Only
   ``research_composite_score`` varies, and it is a near-static per-ticker
   percentile rank (std exactly 28.868 in every snapshot, week-over-week
   Spearman 0.94-1.00). Upstream cause: ``-I9307``.

## The split this builds

López de Prado, *Advances in Financial Machine Learning*, Ch. 7: purge every
training observation whose label window overlaps the validation window, then
embargo a buffer against the serial dependence the purge does not reach. The
repo already implements exactly these semantics for the L2 in
``training/leakfree_meta_ic.expanding_wf_folds`` (purge of ``forward_days``
dates before the test block, embargo of ``embargo_days`` after); this is the
same rule applied to the L1 early-stop split rather than a second one
(``shared-code-policy.md``).

Blocks are cut on the **date axis**, never on the row axis, so a cut can
never land inside a cross-section. Between train and val sits a gap of
``label_horizon_days`` (the purge) plus ``embargo_days`` (the buffer). For a
validation block with training data only before it, LdP's post-test embargo
has nothing to remove, so the buffer is applied on the edge that exists — the
leading one — and ``embargo_edges`` records which edges were used. Where a
test block follows the validation block, the same gap is applied at that
boundary too and the trailing embargo is load-bearing.

## Where the floors come from

Measured on the fleet's own per-date cross-sectional IC history (167 trading
dates, ~898 names per date, 2025-07-10 to 2026-03-09): per-date IC standard
deviation **0.0581** (incumbent) / **0.0528** (champion architecture), lag-1
autocorrelation **0.90 / 0.94** — the autocorrelation being the overlapping
forward window, which is what makes a contiguous block of ``D`` dates worth
roughly ``D / label_horizon_days`` independent observations rather than ``D``.

``MIN_VAL_DATES = 10``
    The failure was a **1.07-date** validation block. Ten equally-sized dates
    cap any single cross-section's leverage on ``val_ic`` at 10%.

``MAX_VAL_DATE_WEIGHT = 0.20``
    A date count alone does not bound leverage on THIS panel, and that is
    measured rather than hypothetical: the sparse era carries ~26 rows per
    date against the dense era's ~899, a **34x** imbalance, so ten dates can
    still be one date wearing a date count. The property the floor actually
    needs is enforced directly.

``MIN_TRAIN_DATE_MULTIPLE = 3``
    Training must span at least three label horizons of dates, i.e. at least
    three non-overlapping label windows. Below that the training labels are
    one overlapping window and the booster has one effective observation to
    early-stop against.

``MAX_VAL_ROW_FRACTION = 0.40``
    A validation block that grew past 40% of the rows to reach its date floor
    has starved training instead of validating it; the split reports
    ``insufficient`` rather than fitting the remainder.

``MIN_VARYING_FEATURES = 2``
    Structural, not a quality judgment: with one varying column a gradient
    booster is a one-dimensional step function, its validation curve is a
    one-dimensional curve, and ``best_iteration`` is decided by label noise.
    Quality floors are ``l1_fit_validity``'s job; this is the floor below
    which the fit cannot mean anything at all.

## What ``insufficient`` does

It does not fit. The arm is registered with ``fitted=False`` and a reason,
``l1_fit_validity``'s gate fails the run, and the live champion keeps serving
(training writes a staging prefix; ``model.registry.promote_to_champion`` is
the only writer of the serving prefix). **A split whose own properties were
not measured cannot be gated on** (``champion-challenger-policy.md`` §5.1) —
and fitting on a one-date holdout to avoid reporting that is precisely the
coin flip this module exists to remove.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

__all__ = [
    "DegenerateDesignMatrixError",
    "PurgedSplit",
    "MIN_VAL_DATES",
    "MAX_VAL_DATE_WEIGHT",
    "MAX_VAL_ROW_FRACTION",
    "MIN_TRAIN_DATE_MULTIPLE",
    "MIN_TRAIN_ROWS",
    "MIN_VARYING_FEATURES",
    "MIN_VAL_DATES_RESEARCH_GBM",
    "TARGET_VAL_IC_SE",
    "min_val_dates_for_target_se",
    "build_purged_split",
    "describe_design_matrix",
    "assert_design_matrix_supports_fit",
    "val_ic_precision",
]

MIN_VAL_DATES = 10
MAX_VAL_DATE_WEIGHT = 0.20
MAX_VAL_ROW_FRACTION = 0.40
MIN_TRAIN_DATE_MULTIPLE = 3
MIN_TRAIN_ROWS = 500
MIN_VARYING_FEATURES = 2


def min_val_dates_for_target_se(
    per_date_ic_std: float, target_se: float, label_horizon_days: int,
) -> int:
    """Derive the date-count floor that makes ``SE(val_ic) <= target_se``.

    alpha-engine-config-I9376. ``val_ic`` is a per-date cross-sectional IC
    averaged over ``D`` dates; overlapping ``label_horizon_days``-forward
    labels make it worth ``n_eff = D / label_horizon_days`` independent
    observations rather than ``D`` (measured lag-1 autocorrelation of the
    fleet's own per-date IC series: 0.90-0.94 over 167 dates). So
    ``SE(val_ic) = per_date_ic_std / sqrt(n_eff)``, and solving for the
    date count that bounds it at ``target_se``::

        n_eff_required = ceil((per_date_ic_std / target_se) ** 2)
        D_required = n_eff_required * label_horizon_days

    This is the derivation, not a fixed constant: call it fresh against the
    THEN-measured ``per_date_ic_std`` rather than carrying a prior
    measurement forward (the issue's explicit instruction — the panel's
    measured IC std moves as the research-feature panel composition
    changes, most recently 0.0581 -> 0.0528 between the incumbent and
    champion architectures).
    """
    import math

    if per_date_ic_std <= 0 or target_se <= 0 or label_horizon_days <= 0:
        raise ValueError(
            "min_val_dates_for_target_se requires strictly positive "
            f"per_date_ic_std ({per_date_ic_std}), target_se ({target_se}) "
            f"and label_horizon_days ({label_horizon_days})."
        )
    n_eff_required = math.ceil((per_date_ic_std / target_se) ** 2)
    return int(n_eff_required * label_horizon_days)


# alpha-engine-config-I9376 — `research_gbm`'s own floor, derived (not
# hardcoded as a bare literal) from the champion-architecture measurement in
# `purged_split`'s own module docstring: per-date cross-sectional IC std
# 0.0528 (167 dates, 2025-07-10..2026-03-09, champion arch), a target SE of
# half of `l1_fit_validity.L1FitSpec.min_abs_val_ic` (0.05) so a `val_ic` at
# the floor sits >= 2 SE from zero, and the arm's own 10-trading-day label
# horizon. Re-derive this against a freshly measured `per_date_ic_std` when
# the research-feature panel composition next changes materially (I9307) —
# do not carry 0.0528 forward without re-measuring it.
_RESEARCH_GBM_MEASURED_PER_DATE_IC_STD = 0.0528
TARGET_VAL_IC_SE = 0.025
MIN_VAL_DATES_RESEARCH_GBM = min_val_dates_for_target_se(
    _RESEARCH_GBM_MEASURED_PER_DATE_IC_STD, TARGET_VAL_IC_SE, 10,
)


class DegenerateDesignMatrixError(RuntimeError):
    """Too few varying columns for the fit to mean anything.

    Raised BEFORE a booster is built, so the run never produces a model whose
    ``best_iteration`` is a coin flip. Callers record it against the arm and
    let ``l1_fit_validity`` fail the run, so every arm is graded before the
    first raise rather than the run dying on whichever arm fits first.
    """


@dataclass(frozen=True)
class PurgedSplit:
    """A purged, embargoed, date-blocked split and every property of it.

    ``status`` is ``ok`` or ``insufficient``. ``insufficient`` carries a
    ``reason`` naming the floor that was not met, the measured value and the
    required one, and the index arrays are empty — there is no partial split
    to accidentally fit on.
    """

    status: str
    reason: str
    label_horizon_days: int
    embargo_days: int
    embargo_edges: tuple
    train_idx: object = None
    val_idx: object = None
    test_idx: object = None
    n_train_rows: int = 0
    n_val_rows: int = 0
    n_test_rows: int = 0
    n_train_dates: int = 0
    n_val_dates: int = 0
    n_test_dates: int = 0
    n_purged_rows: int = 0
    n_purged_dates: int = 0
    n_embargoed_rows: int = 0
    n_embargoed_dates: int = 0
    val_row_fraction: float = 0.0
    max_val_date_weight: float = 0.0
    train_end_date: "str | None" = None
    val_start_date: "str | None" = None
    val_end_date: "str | None" = None
    test_start_date: "str | None" = None
    floors: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_manifest_block(self) -> dict:
        """The block written to ``models.<arm>.split``.

        Every number a reader needs to decide whether ``val_ic`` was a
        statistic or an accident, without re-deriving it from the panel.
        """
        return {
            "status": self.status,
            "reason": self.reason or None,
            "label_horizon_days": self.label_horizon_days,
            "embargo_days": self.embargo_days,
            "embargo_edges": list(self.embargo_edges),
            "n_train_dates": self.n_train_dates,
            "n_val_dates": self.n_val_dates,
            "n_test_dates": self.n_test_dates,
            "n_train_rows": self.n_train_rows,
            "n_val_rows": self.n_val_rows,
            "n_test_rows": self.n_test_rows,
            "n_purged_rows": self.n_purged_rows,
            "n_purged_dates": self.n_purged_dates,
            "n_embargoed_rows": self.n_embargoed_rows,
            "n_embargoed_dates": self.n_embargoed_dates,
            "val_row_fraction": round(self.val_row_fraction, 6),
            "max_val_date_weight": round(self.max_val_date_weight, 6),
            "train_end_date": self.train_end_date,
            "val_start_date": self.val_start_date,
            "val_end_date": self.val_end_date,
            "test_start_date": self.test_start_date,
            "floors": dict(self.floors),
        }


def _insufficient(reason: str, *, horizon: int, embargo: int,
                  edges: tuple, floors: dict, **measured) -> PurgedSplit:
    import numpy as np

    log.error("purged_split: INSUFFICIENT — %s", reason)
    return PurgedSplit(
        status="insufficient", reason=reason,
        label_horizon_days=horizon, embargo_days=embargo, embargo_edges=edges,
        train_idx=np.asarray([], dtype=int), val_idx=np.asarray([], dtype=int),
        test_idx=np.asarray([], dtype=int), floors=floors, **measured,
    )


def build_purged_split(
    dates,
    *,
    label_horizon_days: int,
    train_frac: float,
    val_frac: "float | None" = None,
    embargo_days: "int | None" = None,
    min_val_dates: int = MIN_VAL_DATES,
    max_val_date_weight: float = MAX_VAL_DATE_WEIGHT,
    max_val_row_fraction: float = MAX_VAL_ROW_FRACTION,
    min_train_date_multiple: int = MIN_TRAIN_DATE_MULTIPLE,
    min_train_rows: int = MIN_TRAIN_ROWS,
    test_gap: bool = True,
) -> PurgedSplit:
    """Cut a purged, embargoed early-stopping split on the DATE axis.

    Parameters
    ----------
    dates : sequence
        One entry per row, in non-decreasing order. Row order is asserted, not
        assumed: an unsorted panel makes every temporal slice in the caller
        wrong, and silently producing a split from it is the failure mode this
        check exists for.
    label_horizon_days : int
        Trading days of forward window in the label (``10`` for
        ``actual_fwd_10d``, ``cfg.FORWARD_DAYS`` for the volatility arms).
        This is the purge width.
    train_frac, val_frac
        Target row fractions for the train and validation blocks. ``val_frac``
        of ``None`` means the validation block is the tail and there is no
        test block. The validation block is then GROWN, if needed, until it
        spans ``min_val_dates`` distinct dates — and reports ``insufficient``
        if growing it breaches ``max_val_row_fraction``.
    embargo_days : int, optional
        ``None`` (default) resolves to ``label_horizon_days`` — the
        overlapping-label embargo, matching the repo's existing L4488a
        convention in ``leakfree_meta_ic``.
    test_gap : bool
        Whether a purge+embargo gap is also cut between the validation and
        test blocks. ``True`` is correct and is the default. It is set
        ``False`` for the volatility arms and the reason is recorded rather
        than hidden: their panel is 167 trading dates, a 15% test block is ~25
        dates, and a trailing gap of ``forward_days + embargo_days`` = 42
        dates would consume the whole test block. The LEADING gap — the one
        that decides ``best_iteration``, which is what this issue is about —
        is applied either way; the trailing gap is gated on panel length and
        tracked separately. Ignored when ``val_frac`` is ``None``.

    Returns
    -------
    PurgedSplit
        ``status == "ok"`` with index arrays, or ``status == "insufficient"``
        with a reason and empty arrays. Never a partial split.
    """
    import numpy as np

    if embargo_days is None:
        embargo_days = int(label_horizon_days)
    horizon = int(label_horizon_days)
    embargo = int(embargo_days)
    if horizon < 0 or embargo < 0:
        raise ValueError(
            f"label_horizon_days ({horizon}) and embargo_days ({embargo}) must "
            "both be non-negative — a negative gap would ADD leakage."
        )
    carve_test_gap = bool(test_gap) and val_frac is not None
    edges = ("leading", "trailing") if carve_test_gap else ("leading",)
    floors = {
        "min_val_dates": min_val_dates,
        "max_val_date_weight": max_val_date_weight,
        "max_val_row_fraction": max_val_row_fraction,
        "min_train_dates": min_train_date_multiple * horizon,
        "min_train_rows": min_train_rows,
    }

    keys = [str(d) for d in dates]
    n_rows = len(keys)
    if n_rows == 0:
        return _insufficient(
            "the panel is empty: measured 0 rows, required >= "
            f"{min_train_rows} training rows.",
            horizon=horizon, embargo=embargo, edges=edges, floors=floors,
        )
    if any(keys[i] > keys[i + 1] for i in range(n_rows - 1)):
        raise ValueError(
            "build_purged_split received a panel whose rows are not in "
            "non-decreasing date order. Every temporal slice the caller takes "
            "off this panel is then wrong, so this raises rather than "
            "returning a split that reads as valid."
        )

    uniq: list[str] = []
    counts: list[int] = []
    for k in keys:
        if not uniq or uniq[-1] != k:
            uniq.append(k)
            counts.append(0)
        counts[-1] += 1
    n_dates = len(uniq)
    counts_arr = np.asarray(counts, dtype=int)
    # Row index of the first row of each date, plus a sentinel end.
    starts = np.concatenate([[0], np.cumsum(counts_arr)])

    def rows(lo_date: int, hi_date: int) -> int:
        return int(starts[hi_date] - starts[lo_date])

    # ── Choose the validation block on the date axis ──────────────────────
    cum = np.cumsum(counts_arr) / float(n_rows)
    val_start = int(np.searchsorted(cum, train_frac, side="left"))
    val_start = min(max(val_start, 0), n_dates - 1)
    if val_frac is None:
        val_end = n_dates
    else:
        val_end = int(np.searchsorted(cum, train_frac + val_frac, side="left")) + 1
        val_end = min(max(val_end, val_start + 1), n_dates)

    # Grow the validation block backwards until it spans min_val_dates.
    if (val_end - val_start) < min_val_dates:
        val_start = max(0, val_end - min_val_dates)

    n_val_dates = val_end - val_start
    n_val_rows = rows(val_start, val_end)
    val_row_fraction = n_val_rows / float(n_rows)
    max_weight = (
        float(counts_arr[val_start:val_end].max()) / n_val_rows
        if n_val_rows else 1.0
    )

    # ── Purge + embargo, on the date axis ────────────────────────────────
    gap = horizon + embargo
    train_end = val_start - gap
    if val_frac is None:
        test_start = n_dates
    elif carve_test_gap:
        test_start = min(n_dates, val_end + gap)
    else:
        test_start = val_end

    n_purged_dates = min(horizon, max(0, val_start - max(train_end, 0)))
    n_embargoed_dates = min(embargo, max(0, val_start - max(train_end, 0)) - n_purged_dates)
    gap_lo = max(0, min(train_end, val_start))
    n_gap_rows = rows(gap_lo, val_start)
    # Rows in the leading gap, split by which control removed them: the purge
    # covers the `horizon` dates nearest the validation block, the embargo the
    # `embargo` dates before those.
    purge_lo = max(gap_lo, val_start - horizon)
    n_purged_rows = rows(purge_lo, val_start)
    n_embargoed_rows = n_gap_rows - n_purged_rows
    if carve_test_gap:
        trail_hi = min(n_dates, test_start)
        n_embargoed_rows += rows(val_end, trail_hi)
        n_embargoed_dates += max(0, trail_hi - val_end - horizon)
        n_purged_rows += rows(val_end, min(trail_hi, val_end + horizon))
        n_purged_dates += min(horizon, max(0, trail_hi - val_end))

    measured = {
        "n_train_rows": rows(0, max(train_end, 0)),
        "n_val_rows": n_val_rows,
        "n_test_rows": rows(test_start, n_dates) if val_frac is not None else 0,
        "n_train_dates": max(train_end, 0),
        "n_val_dates": n_val_dates,
        "n_test_dates": (n_dates - test_start) if val_frac is not None else 0,
        "n_purged_rows": n_purged_rows,
        "n_purged_dates": n_purged_dates,
        "n_embargoed_rows": n_embargoed_rows,
        "n_embargoed_dates": n_embargoed_dates,
        "val_row_fraction": val_row_fraction,
        "max_val_date_weight": max_weight,
        "train_end_date": uniq[train_end - 1] if train_end > 0 else None,
        "val_start_date": uniq[val_start],
        "val_end_date": uniq[val_end - 1],
        "test_start_date": (
            uniq[test_start] if val_frac is not None and test_start < n_dates
            else None
        ),
    }

    # ── Floors ───────────────────────────────────────────────────────────
    if n_val_dates < min_val_dates:
        return _insufficient(
            f"validation block spans {n_val_dates} distinct date(s); measured "
            f"{n_val_dates}, required >= {min_val_dates}. The panel holds "
            f"{n_dates} distinct dates in total, so no block can meet the "
            "floor: val_ic would be one cross-section's accident and "
            "best_iteration would be decided by it (I9333).",
            horizon=horizon, embargo=embargo, edges=edges, floors=floors,
            **measured)
    if val_row_fraction > max_val_row_fraction:
        return _insufficient(
            f"the validation block had to take {val_row_fraction:.1%} of the "
            f"panel's rows to span {min_val_dates} dates; measured "
            f"{val_row_fraction:.4f}, required <= {max_val_row_fraction}. The "
            "panel's rows are concentrated in too few recent dates for a "
            "multi-date holdout to leave a trainable remainder — measured "
            f"rows-per-date range {int(counts_arr.min())}..{int(counts_arr.max())} "
            f"over {n_dates} dates.",
            horizon=horizon, embargo=embargo, edges=edges, floors=floors,
            **measured)
    if max_weight > max_val_date_weight:
        return _insufficient(
            f"one date carries {max_weight:.1%} of the validation rows; "
            f"measured {max_weight:.4f}, required <= {max_val_date_weight}. A "
            "date COUNT does not bound a single cross-section's leverage when "
            "rows-per-date is uneven, and on this panel it is uneven by "
            f"{int(counts_arr.max()) / max(1, int(counts_arr.min())):.0f}x.",
            horizon=horizon, embargo=embargo, edges=edges, floors=floors,
            **measured)
    min_train_dates = min_train_date_multiple * horizon
    if measured["n_train_dates"] < min_train_dates:
        return _insufficient(
            f"after a purge of {horizon} and an embargo of {embargo} trading "
            f"dates the training block spans {measured['n_train_dates']} "
            f"date(s); measured {measured['n_train_dates']}, required >= "
            f"{min_train_dates} ({min_train_date_multiple} x the "
            f"{horizon}-day label horizon, so the training labels span at "
            "least three non-overlapping windows).",
            horizon=horizon, embargo=embargo, edges=edges, floors=floors,
            **measured)
    if measured["n_train_rows"] < min_train_rows:
        return _insufficient(
            f"the training block holds {measured['n_train_rows']} rows; "
            f"measured {measured['n_train_rows']}, required >= "
            f"{min_train_rows}.",
            horizon=horizon, embargo=embargo, edges=edges, floors=floors,
            **measured)

    train_idx = np.arange(0, starts[max(train_end, 0)], dtype=int)
    val_idx = np.arange(starts[val_start], starts[val_end], dtype=int)
    test_idx = (
        np.arange(starts[test_start], n_rows, dtype=int)
        if val_frac is not None else np.asarray([], dtype=int)
    )
    log.info(
        "purged_split: train %d rows / %d dates (<= %s) | GAP purge=%d "
        "embargo=%d (%d rows dropped) | val %d rows / %d dates (%s..%s, "
        "max date weight %.1f%%) | test %d rows / %d dates",
        measured["n_train_rows"], measured["n_train_dates"],
        measured["train_end_date"], horizon, embargo,
        n_purged_rows + n_embargoed_rows,
        measured["n_val_rows"], n_val_dates,
        measured["val_start_date"], measured["val_end_date"],
        100.0 * max_weight, measured["n_test_rows"], measured["n_test_dates"],
    )
    return PurgedSplit(
        status="ok", reason="", label_horizon_days=horizon,
        embargo_days=embargo, embargo_edges=edges, train_idx=train_idx,
        val_idx=val_idx, test_idx=test_idx, floors=floors, **measured,
    )


def describe_design_matrix(X, feature_names) -> dict:
    """Name the constant columns of a design matrix.

    Reuses ``arm_validity.constant_input_columns`` — the measurement already
    written for the L2 panel — rather than a second implementation of it
    (``shared-code-policy.md``; issue deliverable 3).
    """
    from training.arm_validity import constant_input_columns

    names = list(feature_names or [])
    constant = constant_input_columns(X, names)
    varying = [f for f in names if f not in set(constant)]
    return {
        "n_features": len(names),
        "n_varying": len(varying),
        "n_constant": len(constant),
        "constant_columns": constant,
        "varying_columns": varying,
    }


def assert_design_matrix_supports_fit(
    arm: str, X, feature_names,
    *, min_varying_features: int = MIN_VARYING_FEATURES,
) -> dict:
    """Refuse to fit an L1 whose design matrix has too few varying columns.

    Raises ``DegenerateDesignMatrixError`` naming the constant columns. This is
    a STRUCTURAL floor, not a quality judgment: with fewer than two varying
    columns the booster is a one-dimensional step function, its validation
    curve is one-dimensional, and ``best_iteration`` is decided by label noise
    — which is what produced 3 -> 500 -> 2 on ``research_gbm``.
    """
    report = describe_design_matrix(X, feature_names)
    report["min_varying_features"] = min_varying_features
    if report["n_varying"] < min_varying_features:
        raise DegenerateDesignMatrixError(
            f"{arm}: refusing to fit — the design matrix has "
            f"{report['n_varying']} varying column(s) of "
            f"{report['n_features']} declared; measured "
            f"{report['n_varying']}, required >= {min_varying_features}. "
            f"Constant across the whole panel: "
            f"{', '.join(report['constant_columns']) or '(none)'}. Varying: "
            f"{', '.join(report['varying_columns']) or '(none)'}. A booster "
            "fitted on this cannot early-stop on anything but label noise, so "
            "its best_iteration is a coin flip that ships as a model "
            "(alpha-engine-config-I9333; upstream cause of the constant "
            "research columns is -I9307)."
        )
    return report


def val_ic_precision(preds, y, dates, *, label_horizon_days: int) -> dict:
    """How precise the validation IC actually is, measured — never assumed.

    ``val_ic`` is a per-date cross-sectional IC averaged over the validation
    block. Overlapping labels make adjacent dates near-duplicates — measured
    lag-1 autocorrelation of the fleet's own per-date IC series is 0.90-0.94
    over 167 dates — so a block of ``D`` dates carries roughly
    ``D / label_horizon_days`` independent observations.

    Recorded, never gated on here. ``l1_fit_validity``'s ``min_abs_val_ic``
    floor is the gate; this is the number that says whether that floor is
    being read against a statistic or against noise, which is the difference
    between a measurement and a well-formed artifact containing nothing
    (``champion-challenger-policy.md`` §7.2).
    """
    import numpy as np

    from training.leakfree_meta_ic import cross_sectional_ic_series

    series = [
        float(v) for v in cross_sectional_ic_series(preds, y, dates)
        if v is not None and np.isfinite(v)
    ]
    n_dates = len(series)
    if n_dates == 0:
        return {"n_val_dates_scored": 0, "mean_per_date_ic": None,
                "per_date_ic_std": None, "n_eff": None, "val_ic_se": None,
                "label_horizon_days": int(label_horizon_days)}
    n_eff = max(1.0, n_dates / float(max(1, int(label_horizon_days))))
    std = float(np.std(series, ddof=1)) if n_dates > 1 else None
    return {
        "n_val_dates_scored": n_dates,
        "mean_per_date_ic": round(float(np.mean(series)), 6),
        "per_date_ic_std": round(std, 6) if std is not None else None,
        "n_eff": round(n_eff, 3),
        "val_ic_se": round(std / (n_eff ** 0.5), 6) if std is not None else None,
        "label_horizon_days": int(label_horizon_days),
    }
