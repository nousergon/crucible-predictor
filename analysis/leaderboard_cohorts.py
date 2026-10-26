"""
analysis/leaderboard_cohorts.py — independent realized cohorts from a leaderboard.

alpha-engine-config-I12662 (steps 1-2, ruled 2026-10-25). The config#1524
kill-gate's e-values assume each paired cohort difference is INDEPENDENT of the
next. The scanner and producer leaderboards score WEEKLY dates against a
21-trading-day forward horizon, so consecutive dates share about three weeks of
the same forward returns: ``n_dates`` counts overlapping windows, not cohorts.

This module turns a leaderboard into the harness's realized-outcome records
(``{challenger_id: [{date, realized_alpha, champion_realized_alpha}, ...]}``,
the shape ``cohorts_from_realized_records`` reads) using only dates whose
forward windows do not overlap, and reports each challenger's independent
depth. ``independent_depth_gate`` is the data gate the real run must pass:
every challenger needs at least ``MIN_INDEPENDENT_COHORTS`` non-overlapping
cohorts, and a record set whose rows overlap or carry no date is refused.

Two cohorts dated ``d1 < d2`` are independent iff their forward windows
``(d1, d1+H]`` and ``(d2, d2+H]`` (H = horizon in NYSE trading days) do not
overlap, i.e. ``count_trading_days(d1, d2) >= H``. Greedy earliest-first
selection is optimal for equal-length windows (interval scheduling), so the
depth reported is the maximum attainable on the common dates.

Pure functions; the S3 read lives in ``scripts/low_n_estimator_kill_gate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from nousergon_lib.trading_calendar import count_trading_days

#: The ruled data gate (alpha-engine-config-I12662 Q1, 2026-10-25): at least
#: this many independent 21d cohorts PER challenger before the real run.
MIN_INDEPENDENT_COHORTS = 4

#: The leaderboards' primary horizon, in NYSE trading days.
DEFAULT_HORIZON_DAYS = 21

SERIES_FIELD = "topn_alpha_vs_population_by_date"


@dataclass(frozen=True)
class GateResult:
    """Outcome of the independent-depth data gate (fail-loud)."""

    passed: bool
    min_independent_cohorts: int
    depths: dict[str, int]
    reasons: tuple[str, ...]


def _as_date(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def non_overlapping_dates(dates: list[str], *, horizon_days: int) -> list[str]:
    """Greedy earliest-first subsample of ``dates`` whose forward windows of
    ``horizon_days`` trading days do not overlap. Input order is irrelevant."""
    kept: list[str] = []
    last: date | None = None
    for d in sorted(set(dates), key=_as_date):
        cur = _as_date(d)
        if last is None or count_trading_days(last, cur) >= horizon_days:
            kept.append(d)
            last = cur
    return kept


def _block_for_horizon(lb: Mapping[str, Any], horizon_days: int) -> Mapping[str, Any]:
    """The leaderboard block (top level or ``horizons[]``) for ``horizon_days``.

    Refuses rather than falling back: scoring a 21d gate on another horizon's
    series would silently change what a cohort is."""
    if int(lb.get("horizon_days") or 0) == horizon_days and lb.get("specs") is not None:
        return lb
    for block in lb.get("horizons") or []:
        if int(block.get("horizon_days") or 0) == horizon_days:
            return block
    raise ValueError(
        f"leaderboard {lb.get('leaderboard_id')!r} carries no {horizon_days}-day "
        f"block (top-level horizon_days={lb.get('horizon_days')!r})"
    )


def records_from_leaderboard(
    lb: Mapping[str, Any],
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Paired, non-overlapping realized cohorts for every live challenger.

    For each spec that is neither the champion nor ``kind == "retired"``, pair
    its ``topn_alpha_vs_population_by_date[d]`` with the champion's on their
    common dates, keep only dates whose ``horizon_days`` forward windows do not
    overlap, and emit one record per kept date.

    Returns ``(records, depths)``. A challenger with no common dates is kept
    with depth 0 so the data gate sees it (and refuses) rather than the
    challenger silently leaving the family.
    """
    champion = lb.get("champion")
    if not champion:
        raise ValueError("leaderboard names no champion")
    block = _block_for_horizon(lb, horizon_days)
    specs = [s for s in (block.get("specs") or []) if isinstance(s, dict)]
    by_name = {str(s.get("name")): s for s in specs if s.get("name")}
    if champion not in by_name:
        raise ValueError(f"champion {champion!r} has no spec row in the leaderboard")
    champ_series = by_name[champion].get(SERIES_FIELD) or {}

    records: dict[str, list[dict]] = {}
    depths: dict[str, int] = {}
    for name, spec in by_name.items():
        if name == champion or spec.get("kind") == "retired":
            continue
        series = spec.get(SERIES_FIELD) or {}
        common = [
            d for d in series
            if d in champ_series and series[d] is not None and champ_series[d] is not None
        ]
        kept = non_overlapping_dates(common, horizon_days=horizon_days)
        records[name] = [
            {
                "date": d,
                "realized_alpha": float(series[d]),
                "champion_realized_alpha": float(champ_series[d]),
            }
            for d in kept
        ]
        depths[name] = len(kept)
    return records, depths


def independent_depth_gate(
    records: Mapping[str, list[dict]],
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    min_cohorts: int = MIN_INDEPENDENT_COHORTS,
) -> GateResult:
    """The ruled data gate: every challenger has >= ``min_cohorts`` cohorts,
    every row is dated, and no two of a challenger's cohorts overlap.

    Applies to ANY record set the real run is given (leaderboard-derived or a
    ``--cohorts`` file), so overlapping windows cannot enter by another door.
    """
    reasons: list[str] = []
    depths: dict[str, int] = {}
    if not records:
        reasons.append("no challengers in the record set")
    for cid, rows in records.items():
        depths[cid] = len(rows)
        if any("date" not in r for r in rows):
            reasons.append(f"{cid}: rows carry no 'date'; independence cannot be checked")
            continue
        dates = [str(r["date"]) for r in rows]
        if len(non_overlapping_dates(dates, horizon_days=horizon_days)) != len(dates):
            reasons.append(
                f"{cid}: cohorts overlap (forward windows closer than "
                f"{horizon_days} trading days, or a repeated date)"
            )
        if len(rows) < min_cohorts:
            reasons.append(
                f"{cid}: {len(rows)} independent cohort(s) < {min_cohorts} required"
            )
    return GateResult(
        passed=not reasons,
        min_independent_cohorts=min_cohorts,
        depths=depths,
        reasons=tuple(reasons),
    )
