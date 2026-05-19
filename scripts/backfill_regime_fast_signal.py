#!/usr/bin/env python3
"""
scripts/backfill_regime_fast_signal.py — Stage F1 calibration backfill.

Recalibrates the **daily** BOCPD hazard + run-length window against a 10y
daily ``intensity_z`` reconstruction, and reports the F1 gate metrics:

  • T1-analog timing  — median lead time (trading days) of forced_bear
    onset vs the realized SPY peak-to-trough drawdown onset, plus the
    episode-level false-positive rate. Positive lead = the fast leg
    fired *before* the drawdown (the entire point); ≤0 = no protective
    value.
  • Whipsaw / churn   — toggles/yr, mean forced_bear dwell (days),
    fraction of episodes shorter than ``min_clear_days``. A combo that
    toggles > ~6×/yr or dwells < ~15 trading days is miscalibrated
    *regardless of downstream Sortino* — fail F1, do not promote.

Design notes
------------
- Daily ``intensity_z`` is reconstructed from the SAME price-cache
  parquets the live path uses, via ``fetch_macro_feature_history_daily``
  (the weekly fetcher already builds a daily frame then downsamples;
  this skips the downsample). No new data plumbing.
- Ground truth is the **SPY peak-to-trough drawdown** bear stretch, NOT
  a weekly HMM argmax. The 2026-05-15 prod backfill proved a fresh
  3-feature weekly HMM produces a biweekly sawtooth of 0-day "stretches"
  (state-label instability over a secular bull — regime-v3 §5.5 #3/#4),
  making lead/FPR degenerate. A drawdown reference is the institutional
  norm for a de-risking overlay anyway: forced_bear exists to dodge
  drawdowns, so lead vs the realized drawdown ONSET is the directly
  meaningful metric and carries zero estimation risk. Hysteresis
  (enter -10%, exit -5%) keeps the reference itself from sawtoothing.
- Bounded compute (per feedback_no_resource_intensive_local_scripts):
  one SPY cummax/drawdown pass + a microsecond-per-step daily replay
  over ~2500 obs × a small sweep grid. No model fit, no torch. Ground
  truth is best-effort — whipsaw/episode stats still print if the SPY
  parquet can't be read (e.g. offline).

Usage
-----
    python scripts/backfill_regime_fast_signal.py \
        [--bucket alpha-engine-research] \
        [--from-parquet-dir /local/dir/with/SPY.parquet/...] \
        [--out report.md]
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from regime.bocpd import BOCPDDetector  # noqa: E402
from regime.composite import compute_composite_intensity  # noqa: E402
from regime.fast_signal import FastSignalState, FastSignalTunables, step  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("backfill_fast_signal")
log.setLevel(logging.INFO)

# Sweep grid. Daily hazard ≈ 1/N trading days mean regime; the daily
# run-length window in trading days.
HAZARD_GRID = (1 / 250.0, 1 / 500.0, 1 / 750.0, 1 / 1000.0)
MIN_RUNLENGTH_GRID = (5, 10, 15)
TRADING_DAYS_PER_YEAR = 252
# Daily z-score window for the composite — daily analog of the live
# path's 260-week (~5y) AQR convention.
DAILY_ZSCORE_WINDOW = 1260


# ── Local-parquet stub so the script runs without S3/network ─────────────
class _LocalParquetS3:
    """Minimal ``get_object`` stub reading ``{dir}/{TICKER}.parquet``."""

    def __init__(self, directory: str) -> None:
        self._dir = directory.rstrip("/")

    class exceptions:  # noqa: N801 — mimic botocore client.exceptions
        class NoSuchKey(Exception):
            pass

    def get_object(self, Bucket: str, Key: str):  # noqa: N803
        path = f"{self._dir}/{Key.split('/')[-1]}"
        try:
            with open(path, "rb") as fh:
                return {"Body": io.BytesIO(fh.read())}
        except FileNotFoundError as e:  # pragma: no cover - dev convenience
            raise _LocalParquetS3.exceptions.NoSuchKey(str(e))


def _daily_intensity_series(daily_features: pd.DataFrame) -> pd.Series:
    """Per-day composite intensity_z over a rolling 5y window. Replays
    the same ``compute_composite_intensity`` the live path uses so the
    backfill is apples-to-apples with production."""
    out: list[float] = []
    idx: list[pd.Timestamp] = []
    cols = list(daily_features.columns)
    values = daily_features.to_dict("records")
    stamps = list(daily_features.index)
    for i, (ts, row) in enumerate(zip(stamps, values)):
        lo = max(0, i - DAILY_ZSCORE_WINDOW)
        hist = daily_features.iloc[lo : i + 1]
        if len(hist) < 60:  # not enough history for a stable z-score yet
            continue
        res = compute_composite_intensity(
            current={c: row[c] for c in cols},
            history=hist,
            rolling_window_weeks=None,  # already windowed by the slice
        )
        # Live inference convention: negative = risk-off. The composite
        # returns stress orientation (positive = risk-off); invert to
        # match ctx.regime_intensity_z exactly.
        out.append(-float(res["intensity_z"]))
        idx.append(ts)
    return pd.Series(out, index=pd.DatetimeIndex(idx), name="intensity_z")


@dataclass
class ComboResult:
    hazard: float
    min_runlength: int
    n_days: int
    n_forced_bear_days: int
    forced_bear_fraction: float
    n_episodes: int
    toggles_per_year: float
    mean_dwell_days: float
    frac_episodes_below_clear: float
    median_lead_days: float | None
    false_positive_rate: float | None


def _episodes(forced: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    eps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    for ts, on in forced.items():
        if on and start is None:
            start = ts
        elif not on and start is not None:
            eps.append((start, prev))
            start = None
        prev = ts
    if start is not None:
        eps.append((start, forced.index[-1]))
    return eps


def _replay(iz: pd.Series, hazard: float, min_rl: int) -> pd.Series:
    det = BOCPDDetector.for_daily(hazard=hazard)
    tun = FastSignalTunables(min_runlength_for_change=min_rl)
    state: FastSignalState | None = None
    forced: list[bool] = []
    for ts, val in iz.items():
        d = ts.strftime("%Y-%m-%d")
        state, art = step(
            state, intensity_z=float(val), trading_day=d,
            calendar_date=d, run_id="backfill", detector=det, tunables=tun,
        )
        forced.append(bool(art["forced_bear"]))
    return pd.Series(forced, index=iz.index, name="forced_bear")


# SPY peak-to-trough drawdown bear reference. A fresh weekly HMM
# argmax==bear was tried first (2026-05-15 prod backfill) and produced
# a biweekly sawtooth of 0-day "stretches" — HMM state-label
# instability on a 3-feature weekly fit over a secular bull (the exact
# failure regime-v3 §5.5 #3/#4 warns about). A drawdown reference is
# the institutional norm for evaluating a de-risking overlay anyway:
# forced_bear exists to dodge drawdowns, so lead vs the realized
# drawdown ONSET is the directly meaningful metric, and it carries no
# estimation risk. Hysteresis (enter at -10%, exit when recovered to
# within -5%) keeps the REFERENCE itself from sawtoothing.
BEAR_DD_ENTER: float = 0.10   # SPY ≥10% below trailing peak ⇒ bear onset
BEAR_DD_EXIT: float = 0.05    # recovered to within 5% of peak ⇒ bear ends
# Lead-pairing windows (trading days → calendar approx). A forced_bear
# episode counts as "leading" a drawdown onset if it starts within
# LEAD_WINDOW before it; episodes starting up to LAG_WINDOW after still
# pair (late but related). Bounded so a far-away episode can't fabricate
# a multi-year "lead".
_LEAD_WINDOW = pd.Timedelta(days=90)   # ≈ 60 trading days before onset
_LAG_WINDOW = pd.Timedelta(days=31)    # ≈ 21 trading days after onset


def _bear_stretches(s3, bucket: str) -> list[tuple[pd.Timestamp, pd.Timestamp]] | None:
    """SPY peak-to-trough drawdown bear stretches (ground truth).

    Bear onset = SPY closes ≥ ``BEAR_DD_ENTER`` below its trailing
    all-time high; the stretch ends when SPY recovers to within
    ``BEAR_DD_EXIT`` of the running peak. Hysteresis between the two
    thresholds prevents reference sawtooth.
    """
    try:
        from regime.features import (
            DEFAULT_PRICE_CACHE_PREFIX,
            _read_parquet_close,
        )

        from regime.drawdown import bear_stretches as _shared_bear_stretches

        spy = _read_parquet_close(
            "SPY", s3_client=s3, bucket=bucket, prefix=DEFAULT_PRICE_CACHE_PREFIX,
        ).dropna()
        if spy.empty:
            log.warning("SPY parquet empty — drawdown reference unavailable.")
            return None
        # Single source of truth: the production drawdown leg and this
        # eval reference are now the SAME code (regime.drawdown). Keeping
        # the BEAR_DD_ENTER/EXIT constants here as the explicit reference
        # thresholds; raising them re-opens the F1 calibration (plan §3
        # single-source-of-truth invariant).
        stretches = _shared_bear_stretches(
            spy, enter=BEAR_DD_ENTER, exit=BEAR_DD_EXIT,
        )
        log.info(
            "drawdown bear reference: %d stretches (enter -%.0f%%, exit -%.0f%%)",
            len(stretches), BEAR_DD_ENTER * 100, BEAR_DD_EXIT * 100,
        )
        return stretches
    except Exception as exc:  # noqa: BLE001 — best-effort
        log.warning("Drawdown reference unavailable (%s) — lead-time omitted.", exc)
        return None


def _overlaps(ep: tuple, st: tuple) -> bool:
    return ep[0] <= st[1] and st[0] <= ep[1]


def _lead_metrics(
    forced: pd.Series,
    stretches: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[float | None, float]:
    """Median lead (trading days) of forced_bear vs drawdown onset, and
    the episode-level false-positive rate.

    Lead: for each bear-stretch onset ``s``, the nearest forced_bear
    episode whose start falls in ``[s - LEAD_WINDOW, s + LAG_WINDOW]``;
    lead = trading days from episode start to ``s`` (positive = fired
    BEFORE the drawdown — the whole point). Onsets with no episode in
    window are misses (no lead contribution).

    FPR: fraction of forced_bear EPISODES that overlap no bear stretch
    (episode-level, not the prior day-level reindex which silently
    degenerated to ~1.0).
    """
    eps = _episodes(forced)
    leads: list[float] = []
    for s, _ in stretches:
        cand = [e for e in eps if (s - _LEAD_WINDOW) <= e[0] <= (s + _LAG_WINDOW)]
        if not cand:
            continue
        nearest = min(cand, key=lambda e: abs((s - e[0]).days))
        leads.append((s - nearest[0]).days * (TRADING_DAYS_PER_YEAR / 365.0))
    if not eps:
        fpr = 0.0
    else:
        fp = sum(1 for e in eps if not any(_overlaps(e, st) for st in stretches))
        fpr = fp / len(eps)
    median_lead = float(np.median(leads)) if leads else None
    return median_lead, fpr


def _evaluate(iz: pd.Series, stretches) -> list[ComboResult]:
    results: list[ComboResult] = []
    years = max(len(iz) / TRADING_DAYS_PER_YEAR, 1e-9)
    for hz in HAZARD_GRID:
        for mr in MIN_RUNLENGTH_GRID:
            forced = _replay(iz, hz, mr)
            eps = _episodes(forced)
            dwell = [
                (e[1] - e[0]).days * (TRADING_DAYS_PER_YEAR / 365.0)
                for e in eps
            ]
            below = sum(1 for d in dwell if d < FastSignalTunables().min_clear_days)
            if stretches is not None:
                lead, fpr = _lead_metrics(forced, stretches)
            else:
                lead, fpr = None, None
            results.append(ComboResult(
                hazard=hz, min_runlength=mr, n_days=len(iz),
                n_forced_bear_days=int(forced.sum()),
                forced_bear_fraction=float(forced.mean()),
                n_episodes=len(eps),
                toggles_per_year=len(eps) / years,
                mean_dwell_days=float(np.mean(dwell)) if dwell else 0.0,
                frac_episodes_below_clear=(below / len(eps)) if eps else 0.0,
                median_lead_days=lead, false_positive_rate=fpr,
            ))
    return results


def _report(iz: pd.Series, results: list[ComboResult]) -> str:
    lines = [
        "# Stage F1 — Daily BOCPD fast-signal calibration backfill",
        "",
        f"Daily intensity_z series: **{len(iz)} obs**, "
        f"{iz.index.min().date()} → {iz.index.max().date()} "
        f"(~{len(iz)/TRADING_DAYS_PER_YEAR:.1f}y).",
        "",
        "Ground truth: SPY peak-to-trough drawdown (enter -10%, exit -5%). "
        "Gate reminders (regime-fast-signal-260515.md §5): promote a combo "
        "only if **median lead > 0** (fires before the drawdown onset) "
        "*and* whipsaw is sane (toggles/yr ≲ 6, mean dwell ≳ 15d, few "
        "sub-clear episodes). Lead/FPR omitted ⇒ SPY parquet unreadable.",
        "",
        "| hazard | min_rl | fb_days | fb_frac | episodes | toggles/yr | "
        "mean_dwell_d | %ep<clear | median_lead_d | FPR |",
        "|--------|--------|---------|---------|----------|------------|"
        "--------------|-----------|---------------|-----|",
    ]
    for r in results:
        lead = "—" if r.median_lead_days is None else f"{r.median_lead_days:+.1f}"
        fpr = "—" if r.false_positive_rate is None else f"{r.false_positive_rate:.2f}"
        lines.append(
            f"| 1/{1/r.hazard:.0f} | {r.min_runlength} | "
            f"{r.n_forced_bear_days} | {r.forced_bear_fraction:.3f} | "
            f"{r.n_episodes} | {r.toggles_per_year:.2f} | "
            f"{r.mean_dwell_days:.1f} | {r.frac_episodes_below_clear:.2f} | "
            f"{lead} | {fpr} |"
        )
    lines += [
        "",
        "**Interpretation.** Pick the combo with positive median lead and "
        "the lowest whipsaw that still fires on the known bear stretches. "
        "If every combo whipsaws, the composite intensity_z is too noisy "
        "for a daily detector — escalate before F2 (do not relax the "
        "confirmation gate to force a pass).",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="alpha-engine-research")
    ap.add_argument("--from-parquet-dir", default=None,
                    help="Local dir with {TICKER}.parquet to skip S3.")
    ap.add_argument("--out", default=None, help="Write the report here too.")
    args = ap.parse_args()

    if args.from_parquet_dir:
        s3 = _LocalParquetS3(args.from_parquet_dir)
    else:
        import boto3
        s3 = boto3.client("s3")

    from regime.features import fetch_macro_feature_history_daily

    daily = fetch_macro_feature_history_daily(s3_client=s3, bucket=args.bucket)
    if daily.empty:
        log.error("No daily feature history — cannot calibrate. Check the "
                  "price-cache parquets.")
        return 1
    iz = _daily_intensity_series(daily)
    if len(iz) < 252:
        log.error("Only %d daily intensity_z obs — need ≥1y to calibrate.",
                  len(iz))
        return 1

    stretches = _bear_stretches(s3, args.bucket)
    results = _evaluate(iz, stretches)
    report = _report(iz, results)
    print(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")
        log.info("Report written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
