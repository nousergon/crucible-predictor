"""
scripts/low_n_estimator_kill_gate.py — the config#1524 KILL-GATE runner.

Runs the falsifiable experiment from config#1524 (§0.6b): does the low-n
selection-bias estimator (``analysis/low_n_selection_estimator``) SEPARATE a
known-overfit challenger from a real one BETTER than the current fresh-best-wins
punt (relative-best, margin=0), at OUR cohort depths?

Two modes
---------
1. ``--synthetic`` (default; runnable ANYWHERE, no data) — Monte-Carlo the exact
   config#1524 adversary at a SWEEP of low cohort depths: a **known-overfit**
   challenger (the garden-of-forking-paths winner — the best-sample-mean of
   ``K`` pure-noise streams, so its in-sample edge is 100% selection artefact)
   head-to-head vs a **real** challenger (genuine per-cohort edge). For each
   depth it reports:
     * baseline overfit-promotion rate  P(fresh-best-wins promotes the OVERFIT)
     * estimator overfit-promotion rate P(estimator promotes the OVERFIT) [≤α]
     * estimator real-power             P(estimator promotes the REAL one)
   plus ``d_star`` — the cohort depth at which the estimator's power first
   crosses a usable level (the actionable "how many cohorts before enforce is
   defensible" number).
   KILL-GATE PASS iff, versus the fresh-best-wins punt: (i) the estimator's
   overfit-promotion rate is FDR-controlled (≤ α) at EVERY depth; (ii) where the
   low-n selection bias actually bites (the shallow end) the estimator promotes
   the overfit STRICTLY LESS than the baseline does; and (iii) power EMERGES —
   estimator real-power ≥ floor by the deepest swept depth (not a trivial
   reject-all). If any leg fails, the run prints ``FALSIFIED`` and exits non-zero
   (log the negative result to EXPERIMENTS.md and close the EPIC per §0.6).

2. ``--cohorts PATH`` (the config#1524 closes-when artifact; DATA-GATED to
   ~2026-07-20 when ≥4 realized 21d champion/challenger cohorts have accrued on
   ≥1 substrate — predictor model-zoo (M) / scanner (A) / producer (R) OBSERVE
   shadows) — loads REAL realized-outcome records (JSON: {challenger_id: [
   {realized_alpha, champion_realized_alpha}, ...]}) and reports the SAME
   separation verdict on the actual accrued cohorts.

Both modes write a result artifact and append a dated entry to ``EXPERIMENTS.md``
(the kill-gate's required log, incl. a FALSIFIED negative result if it fails).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.low_n_selection_estimator import (  # noqa: E402
    ChallengerCohorts,
    cohorts_from_realized_records,
    estimator_promote,
    fresh_best_wins_promote,
)

REAL_ID = "real"
OVERFIT_ID = "overfit"


def _forking_paths_overfit(
    rng: np.random.Generator, *, n_cohorts: int, k_forks: int, noise_sd: float
) -> np.ndarray:
    """The known-overfit challenger: best-sample-mean of ``k_forks`` NULL streams.

    True edge is exactly zero; the winner only *looks* good because it was
    selected — the garden of forking paths that fresh-best-wins cannot see
    through (its inflated in-sample mean tops the ranking)."""
    candidates = rng.normal(0.0, noise_sd, size=(k_forks, n_cohorts))
    winner = int(np.argmax(candidates.mean(axis=1)))
    return candidates[winner]


def _depth_row(
    *,
    n_cohorts: int,
    real_effect: float,
    noise_sd: float,
    k_forks: int,
    alpha: float,
    trials: int,
    seed: int,
) -> dict:
    """One depth of the sweep: overfit-vs-real head-to-head over ``trials``."""
    rng = np.random.default_rng(seed)
    base_overfit = est_overfit = est_power = est_blocked = 0
    for _ in range(trials):
        real = ChallengerCohorts(
            REAL_ID, rng.normal(real_effect, noise_sd, size=n_cohorts)
        )
        overfit = ChallengerCohorts(
            OVERFIT_ID,
            _forking_paths_overfit(
                rng, n_cohorts=n_cohorts, k_forks=k_forks, noise_sd=noise_sd
            ),
        )
        fam = [real, overfit]
        if fresh_best_wins_promote(fam).promoted_id == OVERFIT_ID:
            base_overfit += 1
        est = estimator_promote(fam, alpha=alpha)
        if est.blocked:
            est_blocked += 1
        elif est.promoted_id == OVERFIT_ID:
            est_overfit += 1
        else:
            est_power += 1
    return {
        "n_cohorts": n_cohorts,
        "baseline_overfit_rate": round(base_overfit / trials, 4),
        "estimator_overfit_rate": round(est_overfit / trials, 4),
        "estimator_power": round(est_power / trials, 4),
        "estimator_block_rate": round(est_blocked / trials, 4),
    }


def run_synthetic(
    *,
    depths: tuple[int, ...] = (6, 8, 12, 16, 20, 26),
    real_effect: float = 0.8,
    noise_sd: float = 1.0,
    k_forks: int = 8,
    alpha: float = 0.10,
    trials: int = 2000,
    power_floor: float = 0.5,
    shallow_baseline_floor: float = 0.15,
    seed: int = 1524,
) -> dict:
    """Depth-sweep the overfit-vs-real separation and evaluate the kill-gate."""
    rows = [
        _depth_row(
            n_cohorts=n,
            real_effect=real_effect,
            noise_sd=noise_sd,
            k_forks=k_forks,
            alpha=alpha,
            trials=trials,
            seed=seed + n,  # decorrelate depths, keep determinism
        )
        for n in depths
    ]

    fdr_controlled = all(r["estimator_overfit_rate"] <= alpha for r in rows)
    shallow = rows[0]
    separates_where_it_bites = (
        shallow["baseline_overfit_rate"] >= shallow_baseline_floor
        and shallow["estimator_overfit_rate"] < shallow["baseline_overfit_rate"]
    )
    power_emerges = rows[-1]["estimator_power"] >= power_floor
    passed = bool(fdr_controlled and separates_where_it_bites and power_emerges)

    d_star = next(
        (r["n_cohorts"] for r in rows if r["estimator_power"] >= power_floor), None
    )

    return {
        "mode": "synthetic",
        "params": {
            "depths": list(depths),
            "real_effect": real_effect,
            "noise_sd": noise_sd,
            "k_forks": k_forks,
            "alpha": alpha,
            "trials": trials,
            "power_floor": power_floor,
            "shallow_baseline_floor": shallow_baseline_floor,
            "seed": seed,
        },
        "sweep": rows,
        "d_star_cohorts_for_power": d_star,
        "checks": {
            "fdr_controlled_all_depths": fdr_controlled,
            "separates_where_low_n_bias_bites": separates_where_it_bites,
            "power_emerges_by_max_depth": power_emerges,
        },
        "kill_gate": "PASS" if passed else "FALSIFIED",
    }


def run_real(path: str, *, alpha: float = 0.10) -> dict:
    """Run the kill-gate against REAL accrued-cohort records (data-gated)."""
    records = json.loads(Path(path).read_text())
    cohorts = cohorts_from_realized_records(records)
    depths = [c.n_cohorts for c in cohorts]
    base = fresh_best_wins_promote(cohorts)
    est = estimator_promote(cohorts, alpha=alpha)
    return {
        "mode": "real",
        "source": path,
        "n_challengers": len(cohorts),
        "cohort_depths": depths,
        "min_depth": min(depths) if depths else 0,
        "baseline_promoted": base.promoted_id,
        "estimator_promoted": est.promoted_id,
        "estimator_blocked": est.blocked,
        "estimator_rejected": list(est.rejected_ids),
        "e_values": {k: round(v, 4) for k, v in est.e_values.items()},
        "alpha": alpha,
        "note": (
            "Interpretation vs the labelled overfit/real challengers is the "
            "kill-gate verdict; record PASS/FALSIFIED in EXPERIMENTS.md."
        ),
    }


def _append_experiments_log(result: dict, run_date: str) -> None:
    root = Path(__file__).resolve().parents[1]
    log = root / "EXPERIMENTS.md"
    verdict = result.get("kill_gate", "(real-cohort run — record verdict)")
    entry = (
        f"\n## {run_date} — config#1524 low-n selection-bias estimator kill-gate "
        f"({result['mode']})\n\n"
        f"- Verdict: **{verdict}**\n"
        f"- Result: `{json.dumps(result, sort_keys=True)}`\n"
    )
    if not log.exists():
        log.write_text(
            "# EXPERIMENTS\n\nFalsifiable-experiment log (config#1522/#1524 "
            "kill-gate + future negative results).\n"
        )
    with log.open("a") as fh:
        fh.write(entry)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohorts", help="path to real realized-outcome records JSON")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--out", help="write the result artifact JSON to this path")
    ap.add_argument("--log", action="store_true", help="append to EXPERIMENTS.md")
    args = ap.parse_args(argv)

    if args.cohorts:
        result = run_real(args.cohorts, alpha=args.alpha)
    else:
        result = run_synthetic(alpha=args.alpha, trials=args.trials)

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    if args.log:
        _append_experiments_log(result, date.today().isoformat())
    # Non-zero exit on a falsified synthetic gate so CI/operators notice.
    return 1 if result.get("kill_gate") == "FALSIFIED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
