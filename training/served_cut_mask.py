"""training/served_cut_mask.py — restrict a fitted panel to the rows the model
will actually SERVE.

alpha-engine-config-I11106.

Why this exists
---------------
``training/xsec_variance_share.py`` measures the cross-sectional dispersion of
the fitted L2. ``training/promotion_behavioral_veto.py`` refuses a candidate
whose ``xsec_sd`` falls below ``XSEC_SD_ABSOLUTE_FLOOR`` (0.015).

That floor is not a fit-time number. It is the predictor's own absolute serving
``alpha_stdev`` floor, and alpha-engine-config-I9267 states its provenance
verbatim: *"an absolute floor alongside the relative one, derived from the
healthy population (2026-08-10..08-21 ranged 0.0200-0.0434)"*. Those values are
``alpha_stdev`` over a SERVED batch — 26-30 names, the predictor cut plus the
book (``inference/stages/write_output.py``).

It was being compared against a number pooled over the whole L2 training design
matrix — ~896 names per date since 2026-07-18. Measured on the serving champion
``v3.0-meta-2026-08-14-119e069b``, the same model's own cross-sectional output::

    served cut (attractiveness_top_20)   0.007408   0.49x floor   FAILS
    whole served batch (35 names)        0.016768   1.12x floor
    fit-time panel (what the veto read)  0.048588   3.24x floor   passes by 3x

                          0.048588 / 0.007408 = 6.56x overstatement

A threshold calibrated on a ~28-name served batch, applied to a ~896-name
fitted panel, is **6.6x too LENIENT** — not too strict. A model clearing the
floor by 3x at fit time has half the required dispersion on the names the
executor can actually enter. This module supplies the row mask that puts the
measurement back on the population the threshold was derived from
(champion-challenger-policy.md §7.3: gate on the invariant the actor consumes,
never on a downstream transform of it).

What the served cut IS
----------------------
For a given panel date, ``cuts[predictor_universe_cut] ∪ held`` — exactly what
``inference/stages/load_universe.py::resolve_universe_from_membership`` scores:

* the cut, read from the dated, immutable
  ``universe_membership/{date}/membership.json`` artifact, resolved through
  ``nousergon_lib.decision_set`` (the declared resolver — this module never
  re-derives a ranking and never hardcodes a cut name);
* the book, read from ``trades/eod_pnl.csv`` by
  ``inference.stages.load_universe._read_holdings`` — imported, never
  reimplemented, so there is one implementation of that read in the tree.

**Stated assumption (alpha-engine-config-I11106):** the held set is the CURRENT
book applied to every panel date, because ``_read_holdings`` reads the last row
of the EOD surface and that is the one read the inference path performs. It is
a union term, so it can only WIDEN a date's population; the cut — the
load-bearing part — is resolved per date from that date's own artifact.

Resolution is DATED-ONLY, deliberately
--------------------------------------
``load_universe._read_membership`` consults ``universe_membership/latest.json``
first. That is right for a live run and wrong for reconstructing history: it is
a mutable pointer, so a historical date would resolve to today's cut and the
measurement would not be reproducible. Here the available dated prefixes are
listed once and each panel date takes the greatest membership date at or before
it, within ``MEMBERSHIP_MAX_AGE_DAYS`` — the same staleness rule the inference
path applies, imported from it rather than restated.

Posture — every degraded date is a NAMED state, never a fallback
----------------------------------------------------------------
This module NEVER raises: a diagnostic must not take the weekly training run
down. But it also never falls back to the full panel for a date it could not
resolve, because that fallback is precisely the defect being fixed — it would
silently re-admit the ~896-name population the floor cannot judge.

A date resolves to exactly one of:

``ok``
    a dated artifact within the age limit, a cut of at least
    ``MIN_CUT_TICKERS`` names, and at least one panel row inside
    ``cut ∪ held``.
``no_membership_artifact``
    no dated artifact exists at or before this date at all (every panel date
    older than the producer's first publication is here).
``membership_stale``
    the nearest dated artifact at or before this date is older than
    ``MEMBERSHIP_MAX_AGE_DAYS`` — a real gap in the weekly producer, not a
    pre-history date.
``cut_unresolvable``
    the artifact names no ``predictor_universe_cut``, or names one it does not
    contain (``nousergon_lib.decision_set.DecisionSetContractError``).
``cut_too_narrow``
    the cut resolved to fewer than ``MIN_CUT_TICKERS`` names — a truncated
    artifact, whose dispersion over a handful of names would be noise wearing
    the gate's name.
``no_rows_in_cut``
    artifact and cut are both sound, but none of this date's panel rows are in
    ``cut ∪ held``.

Rows on any non-``ok`` date are EXCLUDED. When that leaves fewer than two
distinct dates, ``cross_sectional_variance_share`` returns ``unmeasurable`` and
the promotion veto refuses the candidate — absence is never a pass
(champion-challenger-policy.md §§7.2, 5.1).
"""

from __future__ import annotations

import logging
from datetime import date as _date, datetime as _datetime

log = logging.getLogger(__name__)

#: Prefix holding the dated, immutable membership artifacts.
MEMBERSHIP_PREFIX = "universe_membership/"

#: Fewer tickers than this in a resolved cut and the artifact is treated as
#: broken rather than as a narrow day. ``PREDICTOR_CUT_TOP_N`` is 20, so half
#: of it missing is a truncation, not a market condition. Declared here so the
#: state is nameable; it is never a reason to widen back to the full panel.
MIN_CUT_TICKERS: int = 10

#: How many excluded dates to enumerate per state in the report. The COUNTS are
#: always complete; the enumeration is a bounded sample so a 900-date panel
#: cannot bloat the manifest.
MAX_REPORTED_DATES_PER_STATE: int = 20

#: The basis label recorded on every number computed through this mask. A
#: reader must never have to infer which population a figure came from —
#: mixing the two bases in one block is the defect I11106 records.
BASIS_SERVED_CUT = "served_cut"

#: The basis label for the whole training design matrix.
BASIS_FIT_PANEL = "fit_panel"


def _max_age_days() -> int:
    """The inference path's staleness limit, imported rather than restated."""
    from inference.stages.load_universe import MEMBERSHIP_MAX_AGE_DAYS

    return int(MEMBERSHIP_MAX_AGE_DAYS)


def as_iso_date(value) -> str | None:
    """``YYYY-MM-DD`` for a panel row's ``date``, or ``None``.

    Panel dates arrive as ``pandas.Timestamp``, ``datetime``, ``date`` or a
    string depending on the fold's source. Normalising here — rather than at
    each comparison — is what makes the membership lookup a dict hit instead of
    a type-dependent guess.
    """
    if value is None:
        return None
    if isinstance(value, _datetime):
        return value.date().isoformat()
    if isinstance(value, _date):
        return value.isoformat()
    # pandas.Timestamp satisfies the datetime check above; anything else is
    # parsed from its string form, and an unparseable value is NOT coerced.
    text = str(value).strip()
    if not text:
        return None
    try:
        return _datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    try:
        return _date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


def list_membership_dates(s3, bucket: str) -> list[str]:
    """Every ``YYYY-MM-DD`` under ``universe_membership/``, ascending.

    One ``ListObjectsV2`` walk instead of up to eleven ``GetObject`` probes per
    panel date. On a multi-hundred-date panel the probe-per-date shape is
    thousands of round trips for the same ~30 distinct answers.
    """
    dates: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=bucket, Prefix=MEMBERSHIP_PREFIX, Delimiter="/"
    ):
        for entry in page.get("CommonPrefixes") or []:
            token = (entry.get("Prefix") or "")[len(MEMBERSHIP_PREFIX):].strip("/")
            iso = as_iso_date(token)
            if iso is not None and iso == token:
                dates.append(iso)
    return sorted(set(dates))


def _membership_key(iso_date: str) -> str:
    return f"{MEMBERSHIP_PREFIX}{iso_date}/membership.json"


def resolve_served_cut_mask(
    s3,
    bucket: str,
    dates,
    tickers,
    *,
    min_cut_tickers: int = MIN_CUT_TICKERS,
    held: set[str] | None = None,
):
    """``(mask, report)`` selecting the panel rows the model would SERVE.

    Parameters
    ----------
    dates, tickers : positionally aligned with the design matrix's rows.
    held : override for the current book; ``None`` reads it through
        ``inference.stages.load_universe._read_holdings``.

    ``mask`` is a boolean ``numpy`` array of ``len(dates)``. Never raises, and
    never returns an all-``True`` mask as a degraded answer: a date this
    function could not resolve contributes NO rows, so an unresolvable panel
    yields an unmeasurable summary rather than a quietly re-widened one.
    """
    import numpy as np

    n = len(list(dates)) if not hasattr(dates, "__len__") else len(dates)
    iso_dates = [as_iso_date(d) for d in dates]
    row_tickers = [
        (str(t).strip().upper() if t is not None else None) for t in tickers
    ]
    mask = np.zeros(n, dtype=bool)

    report: dict = {
        "status": "error",
        "basis": BASIS_SERVED_CUT,
        "reason": None,
        "membership_prefix": MEMBERSHIP_PREFIX,
        "held_source": "trades/eod_pnl.csv",
        "min_cut_tickers": int(min_cut_tickers),
        "max_age_days": None,
        "n_rows_total": int(n),
        "n_rows_selected": 0,
        "n_dates_total": 0,
        "n_dates_resolved": 0,
        "n_held": None,
        "cut_names": [],
        "date_states": {},
        "excluded_dates": {},
    }

    if n == 0 or len(row_tickers) != n:
        report["reason"] = (
            f"dates ({n}) and tickers ({len(row_tickers)}) are not positionally "
            "aligned, or the panel is empty — no served cut can be resolved and "
            "the full panel is NOT substituted (alpha-engine-config-I11106)"
        )
        log.error("served-cut mask: %s", report["reason"])
        return mask, report

    try:
        from nousergon_lib.decision_set import (
            DecisionSetContractError,
            PREDICTOR_CUT_TOP_N,
            cut_tickers,
            predictor_cut_name,
        )
        from nousergon_lib.signals import try_read_s3_json

        report["predictor_cut_top_n"] = int(PREDICTOR_CUT_TOP_N)
        max_age = _max_age_days()
        report["max_age_days"] = max_age

        if held is None:
            from inference.stages.load_universe import _read_holdings

            held_set = {str(t).strip().upper() for t in _read_holdings(s3, bucket)}
        else:
            held_set = {str(t).strip().upper() for t in held}
        held_set.discard("")
        report["n_held"] = len(held_set)

        available = list_membership_dates(s3, bucket)
        report["n_membership_artifacts"] = len(available)
        if available:
            report["membership_date_range"] = [available[0], available[-1]]

        unique_dates = sorted({d for d in iso_dates if d is not None})
        n_unparseable = sum(1 for d in iso_dates if d is None)
        report["n_dates_total"] = len(unique_dates)
        if n_unparseable:
            report["n_rows_unparseable_date"] = int(n_unparseable)

        states: dict[str, str] = {}
        chosen: dict[str, str] = {}
        for d in unique_dates:
            prior = [m for m in available if m <= d]
            if not prior:
                states[d] = "no_membership_artifact"
                continue
            nearest = prior[-1]
            age = (_date.fromisoformat(d) - _date.fromisoformat(nearest)).days
            if age > max_age:
                states[d] = "membership_stale"
                continue
            chosen[d] = nearest

        # Fetch each distinct artifact once; a weekly producer means ~30 reads
        # for a panel of hundreds of dates.
        allowed_by_artifact: dict[str, set[str] | str] = {}
        cut_names: set[str] = set()
        for artifact_date in sorted(set(chosen.values())):
            payload = try_read_s3_json(s3, bucket, _membership_key(artifact_date))
            if not payload:
                allowed_by_artifact[artifact_date] = "no_membership_artifact"
                continue
            try:
                name = predictor_cut_name(payload)
                names = cut_tickers(payload, name)
            except DecisionSetContractError as exc:
                log.warning(
                    "served-cut mask: membership %s is unresolvable (%s)",
                    artifact_date, exc,
                )
                allowed_by_artifact[artifact_date] = "cut_unresolvable"
                continue
            if len(names) < min_cut_tickers:
                log.warning(
                    "served-cut mask: membership %s cut %s has %d tickers, "
                    "below the declared minimum %d — excluded, NOT widened to "
                    "the full panel",
                    artifact_date, name, len(names), min_cut_tickers,
                )
                allowed_by_artifact[artifact_date] = "cut_too_narrow"
                continue
            cut_names.add(name)
            allowed_by_artifact[artifact_date] = {t.upper() for t in names} | held_set
        report["cut_names"] = sorted(cut_names)

        rows_by_date: dict[str, list[int]] = {}
        for i, d in enumerate(iso_dates):
            if d is not None:
                rows_by_date.setdefault(d, []).append(i)

        for d in unique_dates:
            if d in states:
                continue
            resolved = allowed_by_artifact.get(chosen[d])
            if isinstance(resolved, str):
                states[d] = resolved
                continue
            hits = [i for i in rows_by_date.get(d, []) if row_tickers[i] in resolved]
            if not hits:
                states[d] = "no_rows_in_cut"
                continue
            for i in hits:
                mask[i] = True
            states[d] = "ok"

        counts: dict[str, int] = {}
        excluded: dict[str, list[str]] = {}
        for d, state in states.items():
            counts[state] = counts.get(state, 0) + 1
            if state != "ok":
                excluded.setdefault(state, []).append(d)
        report["date_states"] = dict(sorted(counts.items()))
        report["excluded_dates"] = {
            state: sorted(v)[:MAX_REPORTED_DATES_PER_STATE]
            for state, v in sorted(excluded.items())
        }
        report["n_dates_resolved"] = counts.get("ok", 0)
        report["n_rows_selected"] = int(mask.sum())

        if report["n_rows_selected"] == 0:
            report["status"] = "unresolvable"
            report["reason"] = (
                f"no panel row resolved to a served cut over {len(unique_dates)} "
                f"date(s): {report['date_states']}. The full panel is NOT "
                "substituted — a served-cut figure that does not exist is "
                "reported as unmeasurable, and the promotion veto refuses on "
                "it (alpha-engine-config-I11106)"
            )
            log.error("served-cut mask: %s", report["reason"])
            return mask, report

        report["status"] = "ok"
        log.info(
            "served-cut mask (alpha-engine-config-I11106): %d/%d rows on %d/%d "
            "dates resolve to cuts %s ∪ %d held name(s). xsec_sd is measured "
            "HERE — the floor it is compared against was derived from served "
            "batches of this width (alpha-engine-config-I9267), not from the "
            "~900-name fitted panel. Excluded date states: %s",
            report["n_rows_selected"], n, report["n_dates_resolved"],
            report["n_dates_total"], report["cut_names"], len(held_set),
            {k: v for k, v in report["date_states"].items() if k != "ok"},
        )
        return mask, report
    except Exception as exc:  # noqa: BLE001 — a diagnostic never fails training
        report["status"] = "error"
        report["reason"] = f"served-cut mask resolution failed: {exc}"
        # Loud, and the mask stays all-False. Degrading to the full panel here
        # would restore exactly the 6.6x-lenient comparison this exists to end.
        log.error(
            "served-cut mask failed — xsec_sd on the served cut will be "
            "UNMEASURABLE and the promotion veto will refuse. The fit-panel "
            "figure is still recorded, and is NOT substituted for it "
            "(alpha-engine-config-I11106).",
            exc_info=True,
        )
        return mask, report
