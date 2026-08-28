"""
training/realized_ic_second_opinion.py — independent second-party IC
recomputation from realized outcomes (config#2889, Brian's 2026-07-18
Decision Queue Option-B ruling).

The champion's promotion gate has historically read ONE self-reported number
— ``meta_model_oos_ic_cpcv.mean_ic`` — computed by
``training/leakfree_meta_ic.py``'s CPCV pipeline from the trainer's own
in-process ``meta_y`` label array (built off ``actual_fwd_canonical`` in
``training/meta_trainer.py``). A bug anywhere in that computation (a leak, a
stale/mislabeled canonical label, a CV-mechanics defect) propagates uncaught
into BOTH the promotion decision (``model_zoo.py::select_winner``) and the
report-card grade that reads the same manifest field — no independent check
exists today.

This module is that independent check. Per the ruling it does NOT retrain or
replay history (Option D's cost). Instead it takes the CPCV pipeline's own
held-out test predictions (``cpcv_meta_oos_ic``'s ``_oos_preds``/``_oos_ids``/
``_oos_dates``, captured via ``return_preds=True``) and independently
re-derives the ground-truth label for each (ticker, score_date) pair via a
FRESH read of ``universe_returns`` in research.db (alpha-engine-config-I9038;
was ``score_performance_outcomes``, whose seeder died 2026-07-10) — bypassing the
trainer's own ``meta_y``/``actual_fwd_canonical`` array entirely. A
mislabeled/stale/leaked training-side ground truth cannot also poison this
check, since it never touches ``meta_y``.

If the resulting second-opinion IC materially disagrees with the
self-reported CPCV mean IC — or too few held-out rows can even be rejoined to
the outcome store — that is exactly the class of silent computation/labeling
bug this exists to catch.

Usage (from ``training/meta_trainer.py``, after ``cpcv_meta_oos_ic(...,
return_preds=True)``)::

    from training.realized_ic_second_opinion import compute_second_opinion_ic
    second_opinion = compute_second_opinion_ic(
        _cpcv_oos_preds, _cpcv_oos_ids, _cpcv_oos_dates,
        db_path=db_tmp, horizon_days=cfg.FORWARD_DAYS,
    )
    cpcv_meta_ic["second_opinion"] = second_opinion

Usage (from ``training/model_zoo.py::select_winner``, per candidate)::

    from training.realized_ic_second_opinion import evaluate_second_opinion_gate
    verdict = evaluate_second_opinion_gate(ic, second_opinion)
"""
from __future__ import annotations

import logging
import sqlite3

import numpy as np
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

# Below this many rejoined (prediction, realized-outcome) pairs, a verdict
# would be noise — not enough data to trust either "corroborates" or
# "diverges". Mirrors the CPCV/WF modules' own min-sample floors.
MIN_MATCHED_FOR_VERDICT = 30

# Below this fraction of IN-COVERAGE OOS rows successfully rejoined to the
# ground truth, the MATCH ITSELF is suspect — a join/label integrity problem
# (symbol-format mismatch, wrong horizon, a key change), not just "the model is
# weak". Surfaced as its own divergence reason rather than silently folded into
# a low IC.
#
# alpha-engine-config-I9038: the denominator is IN-COVERAGE rows, not all OOS
# rows. Rows predating the ground truth's earliest resolved date cannot match
# no matter how sound the join is, and folding them in turned this threshold
# into an archive-depth check that no rotation could pass — blocking every
# promotion from 2026-07-24 onward. Coverage shortfall is reported separately
# as ``coverage_fraction``.
MIN_MATCH_RATE = 0.5

# A self-reported CPCV mean IC exceeding the independently-sourced
# second-opinion IC by more than this absolute gap (or a sign flip) flags
# divergence. Deliberately loose on day 1 — see
# config.MODEL_ZOO_SECOND_OPINION_GATE_ENFORCE (observe-first rollout); this
# is a starting point for Brian to tighten once real second-opinion numbers
# have been observed for a few weeks.
DEFAULT_MAX_IC_GAP = 0.05


# alpha-engine-config-I9038 — the ground truth is read from ``universe_returns``,
# NOT ``score_performance_outcomes``.
#
# Measured 2026-08-28. ``score_performance_outcomes`` is derived from
# ``score_performance``, whose seeder stopped on 2026-07-10 (the same date the
# retired six-team Research LangGraph's other tables stop — see
# alpha-engine-config-I8757). It holds 534 rows for horizon 21 spanning
# 2026-03-04..2026-07-10, against CPCV OOS sets of ~150k rows spanning
# 2025-07-02..2026-03-02 — a 0-8% match, every rotation since 2026-07-24. The
# outcome producer itself is healthy; its INPUT is dead.
#
# ``universe_returns`` is the live source both stores derive from: 1.6M rows
# carrying ``log_return_{h}d`` and ``log_spy_return_{h}d``, and
# ``log_alpha = log_return - log_spy_return`` reproduces the stored
# ``score_performance_outcomes.log_alpha`` EXACTLY — verified on all 534
# overlapping rows, agreement 534/534 to 1e-9. So this is the same number from
# a source that is still being written, not a looser substitute.
def _horizon_columns(horizon_days: int) -> tuple[str, str]:
    """The (stock, spy) log-return column pair for a horizon."""
    return f"log_return_{int(horizon_days)}d", f"log_spy_return_{int(horizon_days)}d"


def _fetch_realized_outcomes(
    db_path: str, tickers, score_dates, horizon_days: int,
) -> tuple[dict[tuple[str, str], float], tuple[str | None, str | None]]:
    """Fresh, independent read of realized log-alpha keyed by
    ``(ticker, eval_date)``, plus the ground truth's own COVERAGE WINDOW.

    Bypasses the trainer's in-process ``meta_y``/``actual_fwd_canonical`` array
    entirely, so a labeling bug there cannot also poison this check — the whole
    point of the second opinion (config#2889).

    Returns ``(realized, (cov_min, cov_max))``. The window is the min/max
    ``eval_date`` at which this horizon actually resolves, and it is what lets
    the caller tell "the join is broken" apart from "the archive does not reach
    back that far" — two different findings with two different fixes, which the
    pre-I9038 code collapsed into one `low_match_rate` verdict.
    """
    ret_col, spy_col = _horizon_columns(horizon_days)
    keys = {
        (str(t), str(d)[:10])
        for t, d in zip(tickers, score_dates)
        if t is not None and d is not None
    }
    if not keys:
        return {}, (None, None)
    conn = sqlite3.connect(str(db_path))
    try:
        has_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='universe_returns'"
        ).fetchone()
        if not has_table:
            log.warning(
                "second_opinion: universe_returns ABSENT from research.db — "
                "cannot compute an independent IC this cycle"
            )
            return {}, (None, None)

        cols = {r[1] for r in conn.execute("PRAGMA table_info(universe_returns)")}
        missing = [c for c in (ret_col, spy_col) if c not in cols]
        if missing:
            # A horizon the returns collector does not populate. Honest absence,
            # NOT a join failure — the caller degrades to a non-blocking verdict.
            log.warning(
                "second_opinion: universe_returns has no %s for horizon %sd — "
                "cannot compute an independent IC this cycle", missing, horizon_days,
            )
            return {}, (None, None)

        cov = conn.execute(
            f"SELECT MIN(eval_date), MAX(eval_date) FROM universe_returns "
            f"WHERE {ret_col} IS NOT NULL AND {spy_col} IS NOT NULL"
        ).fetchone()
        cov_min = str(cov[0])[:10] if cov and cov[0] else None
        cov_max = str(cov[1])[:10] if cov and cov[1] else None

        rows = conn.execute(
            f"SELECT ticker, eval_date, {ret_col} - {spy_col} FROM universe_returns "
            f"WHERE {ret_col} IS NOT NULL AND {spy_col} IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    out: dict[tuple[str, str], float] = {}
    for ticker, eval_date, log_alpha in rows:
        key = (str(ticker), str(eval_date)[:10])
        if key in keys and log_alpha is not None:
            out[key] = float(log_alpha)
    return out, (cov_min, cov_max)


def compute_second_opinion_ic(
    oos_preds, oos_ids, oos_dates,
    *,
    db_path: str,
    horizon_days: int,
    min_matched: int = MIN_MATCHED_FOR_VERDICT,
    min_match_rate: float = MIN_MATCH_RATE,
) -> dict:
    """Independently recompute an OOS IC for the CPCV pipeline's own
    held-out test predictions, using ground truth read FRESH from
    ``score_performance_outcomes`` instead of the trainer's ``meta_y`` array.

    ``oos_preds`` / ``oos_ids`` / ``oos_dates`` are the ``_oos_preds`` /
    ``_oos_ids`` / ``_oos_dates`` arrays popped off
    ``leakfree_meta_ic.cpcv_meta_oos_ic(..., row_ids=tickers,
    return_preds=True)``.

    Returns a JSON-serializable dict: ``status``, ``second_opinion_ic``,
    ``n_oos_rows``, ``n_matched``, ``match_rate``. Never raises — a DB read
    failure or missing store degrades to ``status="unavailable"`` (the
    caller's gate treats an unavailable second opinion as non-blocking; see
    ``evaluate_second_opinion_gate``).
    """
    try:
        preds = np.asarray(oos_preds, dtype=float) if oos_preds is not None else np.array([])
        ids = list(oos_ids) if oos_ids is not None else []
        dates = list(oos_dates) if oos_dates is not None else []
    except Exception:  # noqa: BLE001 — malformed inputs degrade to unavailable
        log.warning("second_opinion: malformed OOS inputs", exc_info=True)
        return {
            "status": "unavailable", "second_opinion_ic": None,
            "n_oos_rows": 0, "n_matched": 0, "match_rate": None,
        }

    n_oos = len(preds)
    if n_oos == 0 or not ids or len(ids) != n_oos or len(dates) != n_oos:
        return {
            "status": "no_oos_predictions",
            "second_opinion_ic": None,
            "n_oos_rows": n_oos,
            "n_matched": 0,
            "match_rate": None,
        }

    try:
        realized, (cov_min, cov_max) = _fetch_realized_outcomes(
            db_path, ids, dates, horizon_days
        )
    except Exception:  # noqa: BLE001 — best-effort: a DB read failure must not
        # crash training; the gate below treats "unavailable" as non-blocking.
        log.warning("second_opinion: research.db read failed", exc_info=True)
        return {
            "status": "unavailable", "second_opinion_ic": None,
            "n_oos_rows": n_oos, "n_matched": 0, "match_rate": None,
        }

    # alpha-engine-config-I9038 — match_rate is measured over the OOS rows the
    # ground truth COULD cover, not over every OOS row.
    #
    # An OOS row dated before the archive's earliest resolved date can never
    # match, however healthy the join is. Counting it as a miss makes
    # ``match_rate`` a measure of ARCHIVE DEPTH rather than of join integrity —
    # and since ``low_match_rate`` blocks promotion unconditionally (I9030),
    # that conflation makes a short archive permanently indistinguishable from
    # a broken key. Measured 2026-08-28: 34,901 of 35,178 in-coverage OOS pairs
    # matched (99.2%), against 23.3% of all 150,000 OOS pairs — the join is
    # sound and the archive is short. Different findings, different fixes; the
    # horizon-vs-retention one is its own class (champion-challenger-policy 7.1)
    # and is reported through ``coverage_fraction`` rather than folded in here.
    in_coverage = 0
    matched_pred: list[float] = []
    matched_real: list[float] = []
    for pred, tid, d in zip(preds, ids, dates):
        day = str(d)[:10]
        if cov_min is not None and cov_max is not None and cov_min <= day <= cov_max:
            in_coverage += 1
        val = realized.get((str(tid), day))
        if val is not None and np.isfinite(pred):
            matched_pred.append(float(pred))
            matched_real.append(val)

    n_matched = len(matched_pred)
    # Denominator is the coverable rows. Falls back to n_oos when the window is
    # unknown (no ground truth at all), preserving the old, stricter behaviour
    # for the genuinely-unavailable case rather than dividing by zero.
    denom = in_coverage if in_coverage > 0 else n_oos
    match_rate = (n_matched / denom) if denom else 0.0
    coverage_fraction = (in_coverage / n_oos) if n_oos else 0.0

    if coverage_fraction < 1.0:
        log.warning(
            "second_opinion: ground truth covers %.1f%% of OOS rows (%d/%d, "
            "window %s..%s) — the rest predate the archive and are excluded "
            "from match_rate; see alpha-engine-config-I9038",
            100 * coverage_fraction, in_coverage, n_oos, cov_min, cov_max,
        )

    coverage_fields = {
        "n_in_coverage": in_coverage,
        "coverage_fraction": round(coverage_fraction, 4),
        "coverage_window": [cov_min, cov_max],
        "ground_truth_source": "universe_returns",
    }

    if n_matched < min_matched:
        return {
            "status": "insufficient_matched_rows",
            "second_opinion_ic": None,
            "n_oos_rows": n_oos,
            "n_matched": n_matched,
            "match_rate": round(match_rate, 4),
            **coverage_fields,
        }

    sp = spearmanr(matched_pred, matched_real).correlation
    ic = float(sp) if np.isfinite(sp) else None
    status = "ok" if match_rate >= min_match_rate else "low_match_rate"

    log.info(
        "second_opinion: IC=%s over %d/%d matched in-coverage OOS rows "
        "(match_rate=%.2f, coverage=%.2f of %d total, status=%s)",
        ic, n_matched, denom, match_rate, coverage_fraction, n_oos, status,
    )
    return {
        "status": status,
        "second_opinion_ic": round(ic, 6) if ic is not None else None,
        "n_oos_rows": n_oos,
        "n_matched": n_matched,
        "match_rate": round(match_rate, 4),
        **coverage_fields,
    }


def evaluate_second_opinion_gate(
    cpcv_mean_ic: float | None,
    second_opinion: dict,
    *,
    max_ic_gap: float = DEFAULT_MAX_IC_GAP,
    min_match_rate: float = MIN_MATCH_RATE,
) -> dict:
    """Compare the self-reported CPCV mean IC against the
    independently-sourced second-opinion IC.

    Returns ``{"divergence_detected": bool, "reason": str, "cpcv_mean_ic":
    ..., "second_opinion_ic": ...}``. This function only JUDGES — the caller
    (``model_zoo.py::select_winner``) decides whether a divergence verdict
    actually blocks promotion, gated on
    ``config.MODEL_ZOO_SECOND_OPINION_GATE_ENFORCE`` (observe-first rollout,
    config#2889).
    """
    status = second_opinion.get("status")
    so_ic = second_opinion.get("second_opinion_ic")
    match_rate = second_opinion.get("match_rate")

    if status in (None, "unavailable", "no_oos_predictions", "insufficient_matched_rows") or so_ic is None:
        return {
            "divergence_detected": False,
            "reason": f"second opinion unavailable ({status}) — not held against the candidate",
            "cpcv_mean_ic": cpcv_mean_ic,
            "second_opinion_ic": so_ic,
        }

    if match_rate is not None and match_rate < min_match_rate:
        return {
            "divergence_detected": True,
            "reason": (
                f"only {match_rate:.0%} of CPCV OOS rows rejoined to "
                f"score_performance_outcomes (need >= {min_match_rate:.0%}) "
                f"— a label/join integrity problem, not just a weak model"
            ),
            "cpcv_mean_ic": cpcv_mean_ic,
            "second_opinion_ic": so_ic,
        }

    if cpcv_mean_ic is None:
        return {
            "divergence_detected": False,
            "reason": "no self-reported CPCV mean IC to compare against",
            "cpcv_mean_ic": cpcv_mean_ic,
            "second_opinion_ic": so_ic,
        }

    sign_flip = (cpcv_mean_ic > 0 and so_ic < 0) or (cpcv_mean_ic < 0 and so_ic > 0)
    gap = cpcv_mean_ic - so_ic

    if sign_flip and abs(so_ic) > 0.01:
        return {
            "divergence_detected": True,
            "reason": (
                f"sign flip: self-reported CPCV mean IC {cpcv_mean_ic:.4f} "
                f"vs independently-sourced second-opinion IC {so_ic:.4f}"
            ),
            "cpcv_mean_ic": cpcv_mean_ic,
            "second_opinion_ic": so_ic,
        }

    if gap > max_ic_gap:
        return {
            "divergence_detected": True,
            "reason": (
                f"self-reported CPCV mean IC {cpcv_mean_ic:.4f} exceeds the "
                f"independently-sourced second-opinion IC {so_ic:.4f} by "
                f"{gap:.4f} (> {max_ic_gap:.4f} floor)"
            ),
            "cpcv_mean_ic": cpcv_mean_ic,
            "second_opinion_ic": so_ic,
        }

    return {
        "divergence_detected": False,
        "reason": (
            f"second opinion corroborates: self-reported {cpcv_mean_ic:.4f} "
            f"vs independent {so_ic:.4f} (gap {gap:.4f} <= {max_ic_gap:.4f})"
        ),
        "cpcv_mean_ic": cpcv_mean_ic,
        "second_opinion_ic": so_ic,
    }
