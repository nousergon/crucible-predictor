"""
analysis/horizon_battery.py — offline horizon-IC measurement battery.

Reads the OOS rows persisted by meta_trainer at
``s3://{bucket}/predictor/diagnostics/oos_rows/{latest|YYYY-MM-DD}.parquet``
and computes the same overlap / non-overlap / per-regime / bootstrap-CI
breakdown that the training pipeline emits. Lets operators iterate on
horizon decisions without waiting for the next Saturday training cycle:
vary regime thresholds, horizon sets, bootstrap iteration counts, etc.

Usage:
    python -m analysis.horizon_battery
    python -m analysis.horizon_battery --date 2026-05-09 --output report.json
    python -m analysis.horizon_battery --bootstrap-iter 5000

Per the 2026-05-07 predictor audit Track B (PR 4/N): the audit's
"Phase 1 deliverable B" requires multi-regime IC ratios at 5d / 10d /
21d / 60d / 90d using non-overlapping windows + bootstrap CIs. This
module is the offline harness that makes those measurements iteratable
on demand.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from training.meta_trainer import (
    _DIAGNOSTIC_HORIZONS,
    _ENSEMBLE_HORIZONS,
    _bootstrap_ic_ci_by_date,
    _classify_regime,
    _nonoverlapping_date_mask,
)

OOS_ROWS_PREFIX = "predictor/diagnostics/oos_rows/"


def load_oos_rows(bucket: str, date: str | None = None):
    """Download the OOS rows parquet from S3 and return a DataFrame.

    ``date`` of ``None`` reads ``latest.parquet``. Otherwise reads
    ``{date}.parquet``. Returns ``None`` if the object doesn't exist —
    caller decides whether that's a hard fail or a "training hasn't run
    yet" warning.
    """
    import boto3
    import io
    import pandas as pd

    key = f"{OOS_ROWS_PREFIX}{'latest' if date is None else date}.parquet"
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        log.info("Loaded %d OOS rows from s3://%s/%s", len(df), bucket, key)
        return df
    except s3.exceptions.NoSuchKey:
        log.error("No OOS rows at s3://%s/%s — has training run since the persistence PR landed?", bucket, key)
        return None


def compute_horizon_battery(
    df,
    horizons: list[int] | None = None,
    bootstrap_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute the full horizon IC battery against the loaded OOS rows.

    Mirrors the diagnostic computed during training in meta_trainer's
    Step 7.1, but operates on the persisted rows so no training rerun
    is needed. Useful for iterating on horizon parameters: bootstrap
    iterations, regime thresholds (via custom classifier), etc.

    Args:
        df: DataFrame from ``load_oos_rows``. Must have columns
            ``date``, ``actual_fwd_{h}d`` for each horizon h, and the
            META_FEATURES used by the meta-model. The meta-model
            predictions are not in the rows, so this function fits a
            fresh Ridge to the rows and computes IC against that — same
            pattern as training, just standalone.
        horizons: list of horizon ints. Default: _DIAGNOSTIC_HORIZONS.
        bootstrap_iter: number of bootstrap iterations per horizon.
        seed: RNG seed for bootstrap reproducibility.

    Returns:
        dict with shape:
            {
                "horizons": [...],
                "regime_distribution": {"bull": N, "neutral": N, "bear": N},
                "curve": {
                    "5d": {"spearman": ..., "n": ..., "spearman_ci_lo": ...,
                           "spearman_ci_hi": ..., "spearman_nonoverlap": ...,
                           "n_nonoverlap": ..., "spearman_nonoverlap_ci_lo": ...,
                           "spearman_nonoverlap_ci_hi": ...,
                           "by_regime": {"bull": {...}, "neutral": {...}, "bear": {...}}},
                    ...
                },
            }
    """
    import numpy as np
    from scipy.stats import spearmanr
    from model.meta_model import MetaModel, META_FEATURES

    horizons = horizons or list(_DIAGNOSTIC_HORIZONS)

    # Refit the meta-model on the OOS rows so we get predictions to
    # correlate against. The training pipeline does this same step at
    # meta_trainer.py:866 — same Ridge alpha=1.0, same META_FEATURES.
    log.info("Fitting Ridge on %d OOS rows × %d META_FEATURES", len(df), len(META_FEATURES))
    meta_X = df[META_FEATURES].to_numpy(dtype=np.float32)
    meta_y = df["actual_fwd"].to_numpy(dtype=np.float32)
    model = MetaModel(alpha=1.0)
    model.fit(meta_X, meta_y, feature_names=META_FEATURES)
    preds = model.predict(meta_X).ravel()

    # Classify regime per row using the same heuristic as training.
    regimes = [_classify_regime(r) for r in df.to_dict(orient="records")]
    regime_dist = {
        "bull": regimes.count("bull"),
        "neutral": regimes.count("neutral"),
        "bear": regimes.count("bear"),
    }
    log.info("Regime distribution: %s", regime_dist)

    dates = df["date"].tolist()

    curve: dict = {}
    for h in horizons:
        col = f"actual_fwd_{h}d"
        if col not in df.columns:
            log.warning("Column %s not in OOS rows — skipping horizon %dd", col, h)
            continue
        y_h = df[col].to_numpy()
        mask = np.isfinite(y_h)

        # Overall overlapping IC.
        if mask.sum() >= 100:
            sp = spearmanr(preds[mask], y_h[mask])
            ic = float(sp.correlation) if np.isfinite(sp.correlation) else 0.0
        else:
            ic = float("nan")

        # Bootstrap CI on overlapping IC.
        ic_ci_lo, ic_ci_hi = float("nan"), float("nan")
        if mask.sum() >= 100:
            ic_ci_lo, ic_ci_hi = _bootstrap_ic_ci_by_date(
                preds[mask], y_h[mask],
                [d for d, m in zip(dates, mask) if m],
                n_iter=bootstrap_iter, ci=0.95, seed=seed + h,
            )

        # Non-overlapping subsample IC + CI.
        nonoverlap_ic, nonoverlap_n = float("nan"), 0
        nonoverlap_ci_lo, nonoverlap_ci_hi = float("nan"), float("nan")
        nonoverlap_mask = _nonoverlapping_date_mask(dates, h)
        combined_mask = np.array(nonoverlap_mask) & mask
        if combined_mask.sum() >= 10:
            sp_no = spearmanr(preds[combined_mask], y_h[combined_mask])
            nonoverlap_ic = (
                float(sp_no.correlation)
                if np.isfinite(sp_no.correlation) else float("nan")
            )
            nonoverlap_n = int(combined_mask.sum())
            nonoverlap_ci_lo, nonoverlap_ci_hi = _bootstrap_ic_ci_by_date(
                preds[combined_mask], y_h[combined_mask],
                [d for d, m in zip(dates, combined_mask) if m],
                n_iter=bootstrap_iter, ci=0.95, seed=seed + h + 1000,
            )

        # Per-regime IC slice.
        regime_ics: dict = {}
        for regime in ("bull", "neutral", "bear"):
            r_mask = np.array([rr == regime for rr in regimes]) & mask
            if r_mask.sum() >= 30:
                sp_r = spearmanr(preds[r_mask], y_h[r_mask])
                ic_r = (
                    float(sp_r.correlation)
                    if np.isfinite(sp_r.correlation) else float("nan")
                )
            else:
                ic_r = float("nan")
            regime_ics[regime] = {
                "spearman": _round_or_none(ic_r),
                "n": int(r_mask.sum()),
            }

        curve[f"{h}d"] = {
            "spearman": _round_or_none(ic),
            "n": int(mask.sum()),
            "spearman_ci_lo": _round_or_none(ic_ci_lo),
            "spearman_ci_hi": _round_or_none(ic_ci_hi),
            "spearman_nonoverlap": _round_or_none(nonoverlap_ic),
            "n_nonoverlap": nonoverlap_n,
            "spearman_nonoverlap_ci_lo": _round_or_none(nonoverlap_ci_lo),
            "spearman_nonoverlap_ci_hi": _round_or_none(nonoverlap_ci_hi),
            "by_regime": regime_ics,
        }

    return {
        "horizons": [f"{h}d" for h in horizons],
        "regime_distribution": regime_dist,
        "curve": curve,
        "n_rows": len(df),
        "bootstrap_iter": bootstrap_iter,
    }


def _round_or_none(x: float, ndigits: int = 6):
    """Round if finite, else None (JSON-friendly)."""
    import math
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), ndigits)


def format_report(report: dict) -> str:
    """Human-readable summary for console output."""
    lines = [
        f"Horizon battery — {report['n_rows']} OOS rows, "
        f"{report['bootstrap_iter']}-iter bootstrap CI",
        f"Regime distribution: {report['regime_distribution']}",
        "",
        f"{'Horizon':<8}{'IC':<10}{'CI':<22}{'IC (no-ovr)':<14}{'CI (no-ovr)':<22}"
        f"{'IC bull':<10}{'IC neutral':<12}{'IC bear':<10}",
    ]
    for hkey, h in report["curve"].items():
        ic = _fmt(h.get("spearman"))
        ci = _fmt_ci(h.get("spearman_ci_lo"), h.get("spearman_ci_hi"))
        ic_no = _fmt(h.get("spearman_nonoverlap"))
        ci_no = _fmt_ci(h.get("spearman_nonoverlap_ci_lo"), h.get("spearman_nonoverlap_ci_hi"))
        ic_bull = _fmt(h["by_regime"]["bull"]["spearman"])
        ic_neutral = _fmt(h["by_regime"]["neutral"]["spearman"])
        ic_bear = _fmt(h["by_regime"]["bear"]["spearman"])
        lines.append(
            f"{hkey:<8}{ic:<10}{ci:<22}{ic_no:<14}{ci_no:<22}"
            f"{ic_bull:<10}{ic_neutral:<12}{ic_bear:<10}"
        )
    return "\n".join(lines)


def _fmt(x):
    return f"{x:+.4f}" if isinstance(x, (int, float)) else "—"


def _fmt_ci(lo, hi):
    if lo is None or hi is None:
        return "[—]"
    return f"[{lo:+.3f},{hi:+.3f}]"


# ---------------------------------------------------------------------------
# Phase-1 offline temporal-ensemble blend (config#937, operator-ratified
# 2026-07-09 / restated 2026-07-10).
#
# Question the ratified experiment answers: is the single-21d serving horizon
# near-optimal, or does blending members across the ratified ladder
# (10/21/42/63/126d) predict the canonical 21d realized alpha materially
# better? This is OBSERVE-ONLY and runs entirely on the persisted OOS rows —
# it fits nothing into the live serving slot and never touches the
# regression-locked multi-horizon inference path.
#
# Method (all offline, on the OOS-row cross-section):
#   * per-horizon member = a fresh Ridge (same MetaModel as training) fit
#     META_FEATURES -> actual_fwd_{h}d, giving a signal vector p_h;
#   * baseline = the single-target-horizon member (default 21d), scored by
#     its Spearman IC vs the target realized return;
#   * IC-weighted blend = sum_h w_h * zscore(p_h), w_h ∝ max(IC_h-vs-target, 0);
#   * Ridge-over-horizons blend = Ridge([zscore(p_h) for h]) -> target return;
#   * uplift = blend IC − baseline IC, with a paired bootstrap-by-date CI and a
#     non-overlapping-subsample point estimate (the mask is trading-day-exact
#     since config#937 Step 0 / PR crucible-predictor#353);
#   * CSCV/PBO (#1995, nousergon_lib.quant.stats.cscv_pbo) over the trials
#     {single-21d, IC-weighted, Ridge} guards the "pick the best blend"
#     selection against backtest overfitting.
#
# NOT computed here (the downstream ratified gate): the after-TC Sharpe / turnover
# / drawdown projection through the crucible-backtester turnover model, and the
# ≥0.15 after-TC-Sharpe success bar + BH-FDR (#624) charge. Those consume this
# harness's IC/blend evidence; the IC uplift below is the offline precondition,
# not the final go/no-go.
# ---------------------------------------------------------------------------


def _zscore(v):
    """Standardize a signal vector to zero mean / unit std (NaN-safe).

    A degenerate (all-equal / zero-variance) vector returns all-zeros so it
    contributes nothing to a blend rather than exploding it.
    """
    import numpy as np

    v = np.asarray(v, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.zeros_like(v)
    mu = float(finite.mean())
    sd = float(finite.std())
    if not np.isfinite(sd) or sd == 0.0:
        return np.zeros_like(v)
    return (v - mu) / sd


def _spearman_ic(pred, actual, mask):
    """Spearman IC of ``pred`` vs ``actual`` over ``mask`` (NaN → nan)."""
    import numpy as np
    from scipy.stats import spearmanr

    if mask.sum() < 30:
        return float("nan")
    sp = spearmanr(pred[mask], actual[mask])
    return float(sp.correlation) if np.isfinite(sp.correlation) else float("nan")


def _paired_uplift_ci_by_date(blend, baseline, target, mask, dates,
                              n_iter=1000, ci=0.95, seed=42):
    """Bootstrap-by-date CI of the PAIRED uplift (blend IC − baseline IC).

    Resamples unique dates with replacement (the same independence unit as
    ``_bootstrap_ic_ci_by_date``), recomputing BOTH ICs on each resample so the
    uplift CI honours the shared cross-section — a per-IC CI would overstate the
    uplift's uncertainty. Returns ``(lo, hi)`` or ``(nan, nan)`` on <3 dates.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    idx = np.where(mask)[0]
    if idx.size == 0:
        return float("nan"), float("nan")
    d_ts = [pd.Timestamp(dates[i]) for i in idx]
    by_date: dict = {}
    for j, d in zip(idx, d_ts):
        by_date.setdefault(d, []).append(j)
    unique_dates = sorted(by_date)
    if len(unique_dates) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(unique_dates)
    diffs: list[float] = []
    for _ in range(n_iter):
        pick = rng.integers(0, n, size=n)
        rows: list[int] = []
        for p in pick:
            rows.extend(by_date[unique_dates[p]])
        rows = np.asarray(rows)
        sb = spearmanr(blend[rows], target[rows]).correlation
        s0 = spearmanr(baseline[rows], target[rows]).correlation
        if np.isfinite(sb) and np.isfinite(s0):
            diffs.append(float(sb - s0))
    if not diffs:
        return float("nan"), float("nan")
    lo = float(np.percentile(diffs, 100 * (1 - ci) / 2))
    hi = float(np.percentile(diffs, 100 * (1 + ci) / 2))
    return lo, hi


def compute_horizon_blend(
    df,
    horizons: list[int] | None = None,
    target_horizon: int = 21,
    bootstrap_iter: int = 1000,
    n_cscv_blocks: int = 8,
    seed: int = 42,
) -> dict:
    """Offline temporal-ensemble blend battery (config#937 Phase 1).

    Fits a per-horizon Ridge member over the OOS rows, blends the members two
    ways (IC-weighted and Ridge-over-horizons), and scores each vs the single-
    target-horizon baseline on Spearman IC against the target realized return.
    In-sample point estimates (matching this module's existing iterate-fast
    contract — the leak-free curves come from training); the CSCV/PBO trial
    matrix is the out-of-sample selection-overfitting guard.

    Args:
        df: OOS rows (``META_FEATURES`` + ``actual_fwd_{h}d`` columns + ``date``).
        horizons: ensemble ladder. Default: ``_ENSEMBLE_HORIZONS`` (10/21/42/63/126).
        target_horizon: the horizon whose realized return the blend must beat
            the single-horizon baseline on. Default 21 (the serving canonical).
        bootstrap_iter: bootstrap iterations for the uplift CI.
        n_cscv_blocks: contiguous date blocks for the CSCV/PBO matrix.
        seed: RNG seed.

    Returns a JSON-friendly dict; see ``format_blend_report`` for the layout.
    """
    import numpy as np
    from model.meta_model import MetaModel, META_FEATURES
    from nousergon_lib.quant.stats.pbo import cscv_pbo

    horizons = horizons or list(_ENSEMBLE_HORIZONS)
    if target_horizon not in horizons:
        # The baseline member must exist in the ladder.
        horizons = sorted({*horizons, target_horizon})

    target_col = f"actual_fwd_{target_horizon}d"
    if target_col not in df.columns:
        raise ValueError(
            f"target column {target_col} absent — OOS rows predate the ratified "
            f"ladder (train must re-persist with _ENSEMBLE_HORIZONS present)."
        )

    X = df[META_FEATURES].to_numpy(dtype=np.float32)
    y_target = df[target_col].to_numpy(dtype=np.float64)
    tmask = np.isfinite(y_target)
    dates = df["date"].tolist()

    # Per-horizon member signals (z-scored so IC-weighting is scale-free).
    members: dict[int, np.ndarray] = {}
    member_ic: dict[int, float] = {}
    usable: list[int] = []
    for h in horizons:
        col = f"actual_fwd_{h}d"
        if col not in df.columns:
            log.warning("Column %s absent — dropping horizon %dd from blend", col, h)
            continue
        y_h = df[col].to_numpy(dtype=np.float64)
        if np.isfinite(y_h).sum() < 50:
            log.warning("Horizon %dd has <50 finite labels — dropping from blend", h)
            continue
        model = MetaModel(alpha=1.0)
        model.fit(X, y_h, feature_names=META_FEATURES)
        p = _zscore(model.predict(X).ravel())
        members[h] = p
        member_ic[h] = _spearman_ic(p, y_target, tmask)
        usable.append(h)

    if target_horizon not in members:
        raise ValueError(
            f"target horizon {target_horizon}d produced no usable member "
            f"(insufficient finite labels) — cannot form the baseline."
        )

    baseline = members[target_horizon]
    baseline_ic = member_ic[target_horizon]

    # IC-weighted blend: weights ∝ max(member IC vs target, 0); equal-weight
    # fallback when no member has positive IC.
    weights = {h: max(member_ic[h], 0.0) for h in usable}
    wsum = sum(weights.values())
    if wsum <= 0:
        weights = {h: 1.0 / len(usable) for h in usable}
    else:
        weights = {h: w / wsum for h, w in weights.items()}
    icw_signal = np.zeros(len(df), dtype=np.float64)
    for h in usable:
        icw_signal += weights[h] * members[h]
    icw_ic = _spearman_ic(icw_signal, y_target, tmask)

    # Ridge-over-horizons blend: Ridge([member signals]) -> target return,
    # fit on finite-target rows only.
    from sklearn.linear_model import Ridge

    stack = np.column_stack([members[h] for h in usable])
    ridge = Ridge(alpha=1.0)
    ridge.fit(stack[tmask], y_target[tmask])
    ridge_signal = ridge.predict(stack)
    ridge_ic = _spearman_ic(ridge_signal, y_target, tmask)

    # Non-overlapping subsample (trading-day-exact mask, config#937 Step 0).
    no_mask = np.array(_nonoverlapping_date_mask(dates, target_horizon)) & tmask
    baseline_ic_no = _spearman_ic(baseline, y_target, no_mask)
    icw_ic_no = _spearman_ic(icw_signal, y_target, no_mask)
    ridge_ic_no = _spearman_ic(ridge_signal, y_target, no_mask)

    # Paired uplift CIs (blend − baseline) by date.
    icw_ci = _paired_uplift_ci_by_date(icw_signal, baseline, y_target, tmask,
                                       dates, n_iter=bootstrap_iter, seed=seed)
    ridge_ci = _paired_uplift_ci_by_date(ridge_signal, baseline, y_target, tmask,
                                         dates, n_iter=bootstrap_iter, seed=seed + 1)

    # CSCV/PBO over the trials {single-target, IC-weighted, Ridge}: per-block IC
    # matrix (n_blocks × n_trials), then the selection-overfit probability.
    trials = {
        f"single_{target_horizon}d": baseline,
        "ic_weighted": icw_signal,
        "ridge_blend": ridge_signal,
    }
    pbo = _cscv_pbo_over_trials(trials, y_target, dates, tmask,
                                n_blocks=n_cscv_blocks, cscv_pbo=cscv_pbo)

    def _u(ic):
        return None if (ic is None or np.isnan(ic)) else round(ic - baseline_ic, 6)

    return {
        "target_horizon": f"{target_horizon}d",
        "ladder": [f"{h}d" for h in usable],
        "n_rows": int(len(df)),
        "n_target_finite": int(tmask.sum()),
        "baseline": {
            "horizon": f"{target_horizon}d",
            "ic": _round_or_none(baseline_ic),
            "ic_nonoverlap": _round_or_none(baseline_ic_no),
        },
        "members": {
            f"{h}d": {"ic": _round_or_none(member_ic[h]),
                      "blend_weight": round(weights[h], 6)}
            for h in usable
        },
        "blends": {
            "ic_weighted": {
                "ic": _round_or_none(icw_ic),
                "ic_nonoverlap": _round_or_none(icw_ic_no),
                "uplift_vs_baseline": _u(icw_ic),
                "uplift_ci_lo": _round_or_none(icw_ci[0]),
                "uplift_ci_hi": _round_or_none(icw_ci[1]),
            },
            "ridge_blend": {
                "ic": _round_or_none(ridge_ic),
                "ic_nonoverlap": _round_or_none(ridge_ic_no),
                "uplift_vs_baseline": _u(ridge_ic),
                "uplift_ci_lo": _round_or_none(ridge_ci[0]),
                "uplift_ci_hi": _round_or_none(ridge_ci[1]),
            },
        },
        "cscv_pbo": pbo,
        "note": (
            "Observe-only IC evidence. IN-SAMPLE point estimates; CSCV/PBO is the "
            "OOS selection guard. The ratified go/no-go bar (after-TC Sharpe uplift "
            "≥0.15 + BH-FDR #624) is the downstream crucible-backtester projection, "
            "not computed here."
        ),
    }


def _cscv_pbo_over_trials(trials, y_target, dates, tmask, n_blocks, cscv_pbo):
    """Build the (n_blocks × n_trials) per-block IC matrix and run CSCV/PBO.

    Blocks are contiguous slices of the sorted unique dates (chronological CSCV
    blocks). Each cell is a trial's Spearman IC on that block's finite-target
    rows. Falls back to ``{"pbo": None, ...}`` when there aren't enough blocks
    with usable rows for the primitive's ``min_splits``.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    idx = np.where(tmask)[0]
    d_ts = np.array([pd.Timestamp(dates[i]) for i in idx])
    order = np.argsort(d_ts)
    idx_sorted = idx[order]
    uniq = sorted(set(d_ts))
    if len(uniq) < n_blocks:
        return {"pbo": None, "n_splits": 0,
                "reason": f"only {len(uniq)} dates for {n_blocks} blocks"}

    # Assign each sorted date to one of n_blocks contiguous groups.
    block_of_date = {}
    for bi, chunk in enumerate(np.array_split(uniq, n_blocks)):
        for d in chunk:
            block_of_date[pd.Timestamp(d)] = bi

    spec_ids = list(trials)
    rows_by_block: dict[int, list[int]] = {b: [] for b in range(n_blocks)}
    for i in idx_sorted:
        rows_by_block[block_of_date[pd.Timestamp(dates[i])]].append(i)

    matrix = []
    kept_blocks = 0
    for b in range(n_blocks):
        rows = np.asarray(rows_by_block[b])
        if rows.size < 30:
            continue
        rowvals = []
        for name in spec_ids:
            sig = trials[name][rows]
            sp = spearmanr(sig, y_target[rows]).correlation
            rowvals.append(float(sp) if np.isfinite(sp) else 0.0)
        matrix.append(rowvals)
        kept_blocks += 1

    if kept_blocks < 4:
        return {"pbo": None, "n_splits": kept_blocks,
                "reason": f"only {kept_blocks} usable blocks (need >=4)"}

    result = cscv_pbo(np.asarray(matrix), spec_ids=spec_ids, min_splits=4)
    selected = result.get("selected_counts") or {}
    most_selected = max(selected, key=selected.get) if selected else None
    return {
        "status": result.get("status"),
        "n_splits": int(result.get("n_splits", kept_blocks)),
        "trials": spec_ids,
        "pbo": _round_or_none(result.get("pbo")),
        "mean_logit": _round_or_none(result.get("mean_logit")),
        "most_selected": most_selected,
        "selected_counts": selected,
    }


def format_blend_report(report: dict) -> str:
    """Human-readable summary of the offline blend battery."""
    b = report["baseline"]
    lines = [
        f"Temporal-ensemble blend — target {report['target_horizon']}, "
        f"ladder {report['ladder']}, {report['n_target_finite']} finite rows",
        f"Baseline (single {b['horizon']}): IC {_fmt(b['ic'])} "
        f"(non-overlap {_fmt(b['ic_nonoverlap'])})",
        "",
        f"{'Member':<10}{'IC':<12}{'blend wt':<10}",
    ]
    for hk, m in report["members"].items():
        lines.append(f"{hk:<10}{_fmt(m['ic']):<12}{m['blend_weight']:<10.3f}")
    lines.append("")
    lines.append(f"{'Blend':<14}{'IC':<12}{'uplift':<12}{'uplift 95% CI':<24}{'IC no-ovr':<12}")
    for name, bl in report["blends"].items():
        ci = _fmt_ci(bl.get("uplift_ci_lo"), bl.get("uplift_ci_hi"))
        lines.append(
            f"{name:<14}{_fmt(bl['ic']):<12}{_fmt(bl['uplift_vs_baseline']):<12}"
            f"{ci:<24}{_fmt(bl['ic_nonoverlap']):<12}"
        )
    pbo = report.get("cscv_pbo", {})
    lines.append("")
    lines.append(f"CSCV/PBO (selection overfit, {pbo.get('n_splits', 0)} splits): "
                 f"{_fmt(pbo.get('pbo'))}  most-selected={pbo.get('most_selected', '—')}")
    lines.append("")
    lines.append(report.get("note", ""))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--date", default=None,
        help="OOS rows date (YYYY-MM-DD). Default: latest.",
    )
    parser.add_argument(
        "--bucket", default=cfg.S3_BUCKET,
        help=f"S3 bucket. Default: {cfg.S3_BUCKET}",
    )
    parser.add_argument(
        "--bootstrap-iter", type=int, default=1000,
        help="Bootstrap iterations per IC. Default: 1000.",
    )
    parser.add_argument(
        "--horizons", default=None,
        help="Comma-separated horizons (e.g. 5,10,21,60). Default: all diagnostic horizons.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional JSON output path. Default: print summary only.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Bootstrap RNG seed. Default: 42.",
    )
    parser.add_argument(
        "--blend", action="store_true",
        help="Run the config#937 offline temporal-ensemble blend battery "
             "(per-horizon members + IC-weighted/Ridge blends vs the single-"
             "target-horizon baseline, with CSCV/PBO) instead of the per-horizon "
             "IC curve. Default ladder: the ratified 10/21/42/63/126d.",
    )
    parser.add_argument(
        "--target-horizon", type=int, default=21,
        help="Blend target horizon (the serving canonical the blend must beat). "
             "Default: 21. Only used with --blend.",
    )
    args = parser.parse_args()

    horizons = None
    if args.horizons:
        horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    df = load_oos_rows(args.bucket, date=args.date)
    if df is None or len(df) == 0:
        log.error("No OOS rows available — exiting with nonzero status.")
        sys.exit(1)

    if args.blend:
        report = compute_horizon_blend(
            df, horizons=horizons,
            target_horizon=args.target_horizon,
            bootstrap_iter=args.bootstrap_iter,
            seed=args.seed,
        )
        print(format_blend_report(report))
    else:
        report = compute_horizon_battery(
            df, horizons=horizons,
            bootstrap_iter=args.bootstrap_iter,
            seed=args.seed,
        )
        print(format_report(report))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        log.info("JSON report written to %s", args.output)


if __name__ == "__main__":
    main()
