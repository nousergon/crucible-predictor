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

2. ``--leaderboard URI`` / ``--cohorts PATH`` — the REAL run
   (alpha-engine-config-I12662, ruled 2026-10-25). ``--leaderboard`` reads a
   scanner or producer leaderboard (local path or ``s3://``) and builds the
   records with ``analysis/leaderboard_cohorts.records_from_leaderboard``;
   ``--cohorts`` loads records JSON ({challenger_id: [{date, realized_alpha,
   champion_realized_alpha}, ...]}) directly. Either way:

   * **Data gate (Q1).** Every challenger needs >= 4 INDEPENDENT cohorts:
     dated, with non-overlapping 21-trading-day forward windows. Raw
     ``n_dates`` is not a depth (weekly dates on a 21d horizon overlap by
     ~3 weeks, and the e-values assume independence). Below the gate the run
     REFUSES (exit 2) and logs nothing.
   * **Verdict (Q2): the planted-overfit control.** Real challengers carry no
     overfit/real label, so the verdict is a semi-synthetic control at the
     REAL depth and noise. Every diff is standardised by the real family's
     pooled within-challenger SD; then, per trial, a known-overfit challenger
     (best-sample-mean of ``K`` null streams, unit SD = the real noise) is
     injected into the real family. Run at every distinct measured depth.
     Three outcomes (amended 2026-10-28, rulings 1b/2a/3b):
       - FALSIFIED if the estimator promotes the planted overfit at a rate
         > alpha at ANY measured depth (the safety leg failed);
       - else INCONCLUSIVE if fresh-best-wins promotes it at < the shallow
         floor at the shallowest measured depth: the control has no contrast,
         so it cannot falsify the estimator (1b). Never read as PASS;
       - else PASS iff the estimator promotes it strictly less than
         fresh-best-wins there, otherwise FALSIFIED.
     The data gate stays over the WHOLE leaderboard: an arm below depth
     stays in the record set and refuses the run, so the e-BH family is never
     shrunk to the arms that have data (2a). A power leg (a planted true-edge
     arm at +``CONTROL_POWER_EFFECT_SD`` pooled SD) is REPORTED at every
     measured depth and never gates the verdict; a PASS whose power is below
     ``CONTROL_POWER_REPORT_FLOOR`` at any measured depth is labelled
     ``PASS (safety only)`` (3b). The parameters are the ``CONTROL_*``
     constants below, pre-registered in the PR and in
     ``alpha-engine-config/private-docs/EXPERIMENTS.md`` before any run, and
     not overridable from the CLI.

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

from analysis.leaderboard_cohorts import (  # noqa: E402
    DEFAULT_HORIZON_DAYS,
    MIN_INDEPENDENT_COHORTS,
    independent_depth_gate,
    records_from_leaderboard,
)
from analysis.low_n_selection_estimator import (  # noqa: E402
    ChallengerCohorts,
    cohorts_from_realized_records,
    estimator_promote,
    fresh_best_wins_promote,
)

REAL_ID = "real"
OVERFIT_ID = "overfit"
PLANTED_ID = "__planted_overfit__"

# Pre-registered verdict rule for the REAL run (alpha-engine-config-I12662 Q2,
# ruled 2026-10-25; amended 2026-10-28 by rulings 1b/2a/3b, before any real
# run). Changing any of these after the first real run voids the
# pre-registration; a change needs a new EXPERIMENTS.md entry first.
CONTROL_ALPHA = 0.10
CONTROL_TRIALS = 2000
CONTROL_K_FORKS = 8
CONTROL_SHALLOW_BASELINE_FLOOR = 0.15
CONTROL_SEED = 12662

#: 1b: what a control with no contrast reports. FDR failure still FALSIFIES.
CONTROL_NO_CONTRAST_VERDICT = "INCONCLUSIVE"
#: 2a: the data gate covers every live challenger on the leaderboard.
CONTROL_GATE_SCOPE = "whole_leaderboard"
#: 3b: the power leg — planted true edge, in pooled-SD units; reported only.
CONTROL_POWER_EFFECT_SD = 1.0
CONTROL_POWER_REPORT_FLOOR = 0.20
CONTROL_POWER_GATES_VERDICT = False
CONTROL_POWER_SEED_OFFSET = 100_000

PLANTED_EDGE_ID = "__planted_true_edge__"
SAFETY_ONLY_LABEL = "PASS (safety only)"

EXIT_PASS, EXIT_FALSIFIED, EXIT_REFUSED, EXIT_INCONCLUSIVE = 0, 1, 2, 3


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


def pooled_within_sd(cohorts: list[ChallengerCohorts]) -> float:
    """Pooled within-challenger SD of the paired diffs: the real noise level.

    Deviations are taken from each challenger's OWN mean, so a genuine edge
    difference between challengers does not inflate the noise estimate."""
    ss = 0.0
    dof = 0
    for c in cohorts:
        if c.n_cohorts >= 2:
            ss += float(((c.diffs - c.diffs.mean()) ** 2).sum())
            dof += c.n_cohorts - 1
    return float(np.sqrt(ss / dof)) if dof > 0 else 0.0


def planted_overfit_control(
    cohorts: list[ChallengerCohorts],
    *,
    alpha: float = CONTROL_ALPHA,
    trials: int = CONTROL_TRIALS,
    k_forks: int = CONTROL_K_FORKS,
    shallow_baseline_floor: float = CONTROL_SHALLOW_BASELINE_FLOOR,
    seed: int = CONTROL_SEED,
) -> dict:
    """The pre-registered verdict for unlabeled real challengers (I12662 Q2).

    ``cohorts`` must already be standardised to unit noise. At each distinct
    real depth, inject a planted overfit of that depth into the REAL family
    ``trials`` times and count who promotes it."""
    depths = sorted({c.n_cohorts for c in cohorts})
    rows = []
    for n in depths:
        rng = np.random.default_rng(seed + n)
        base_planted = est_planted = est_real = est_blocked = 0
        for _ in range(trials):
            planted = ChallengerCohorts(
                PLANTED_ID,
                _forking_paths_overfit(
                    rng, n_cohorts=n, k_forks=k_forks, noise_sd=1.0
                ),
            )
            fam = [*cohorts, planted]
            if fresh_best_wins_promote(fam).promoted_id == PLANTED_ID:
                base_planted += 1
            est = estimator_promote(fam, alpha=alpha)
            if est.blocked:
                est_blocked += 1
            elif est.promoted_id == PLANTED_ID:
                est_planted += 1
            else:
                est_real += 1
        rows.append(
            {
                "n_cohorts": n,
                "baseline_planted_rate": round(base_planted / trials, 4),
                "estimator_planted_rate": round(est_planted / trials, 4),
                "estimator_real_promote_rate": round(est_real / trials, 4),
                "estimator_block_rate": round(est_blocked / trials, 4),
            }
        )
    fdr_controlled = bool(rows) and all(
        r["estimator_planted_rate"] <= alpha for r in rows
    )
    shallow = rows[0] if rows else None
    has_contrast = bool(
        shallow and shallow["baseline_planted_rate"] >= shallow_baseline_floor
    )
    separates = bool(
        has_contrast
        and shallow["estimator_planted_rate"] < shallow["baseline_planted_rate"]
    )
    if not fdr_controlled:
        verdict = "FALSIFIED"
    elif not has_contrast:
        verdict = CONTROL_NO_CONTRAST_VERDICT
    else:
        verdict = "PASS" if separates else "FALSIFIED"

    power = power_leg(cohorts, depths=depths, alpha=alpha, trials=trials, seed=seed)
    label = verdict
    if verdict == "PASS" and not power["meets_report_floor_all_measured_depths"]:
        label = SAFETY_ONLY_LABEL
    return {
        "params": {
            "alpha": alpha,
            "trials": trials,
            "k_forks": k_forks,
            "shallow_baseline_floor": shallow_baseline_floor,
            "seed": seed,
            "no_contrast_verdict": CONTROL_NO_CONTRAST_VERDICT,
            "gate_scope": CONTROL_GATE_SCOPE,
        },
        "sweep": rows,
        "checks": {
            "fdr_controlled_all_measured_depths": fdr_controlled,
            "contrast_at_shallowest_measured_depth": has_contrast,
            "separates_at_shallowest_measured_depth": separates,
        },
        "power": power,
        "kill_gate": verdict,
        "kill_gate_label": label,
    }


def power_leg(
    cohorts: list[ChallengerCohorts],
    *,
    depths: list[int],
    alpha: float = CONTROL_ALPHA,
    trials: int = CONTROL_TRIALS,
    effect_sd: float = CONTROL_POWER_EFFECT_SD,
    report_floor: float = CONTROL_POWER_REPORT_FLOOR,
    seed: int = CONTROL_SEED,
) -> dict:
    """The pre-registered power leg (I12662 ruling 3b): REPORTED, never gating.

    At each measured depth, plant a true-edge arm (``effect_sd`` pooled SDs,
    unit noise) into the real family and count how often the estimator
    promotes it. Its own seed stream, so the safety legs above draw exactly
    what they drew before this leg existed."""
    rows = []
    for n in depths:
        rng = np.random.default_rng(seed + CONTROL_POWER_SEED_OFFSET + n)
        promoted = 0
        for _ in range(trials):
            planted = ChallengerCohorts(
                PLANTED_EDGE_ID, rng.normal(effect_sd, 1.0, size=n)
            )
            est = estimator_promote([*cohorts, planted], alpha=alpha)
            if not est.blocked and est.promoted_id == PLANTED_EDGE_ID:
                promoted += 1
        rows.append(
            {"n_cohorts": n, "estimator_true_edge_rate": round(promoted / trials, 4)}
        )
    return {
        "effect_sd": effect_sd,
        "report_floor": report_floor,
        "gates_verdict": CONTROL_POWER_GATES_VERDICT,
        "sweep": rows,
        "meets_report_floor_all_measured_depths": bool(rows)
        and all(r["estimator_true_edge_rate"] >= report_floor for r in rows),
    }


def run_real(
    records: dict[str, list[dict]],
    *,
    source: str,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    min_cohorts: int = MIN_INDEPENDENT_COHORTS,
) -> dict:
    """The real-cohort kill-gate run: data gate first, then the control.

    Refuses (``refused: True``, no verdict) unless every challenger has
    ``min_cohorts`` independent cohorts. Uses only the pre-registered
    ``CONTROL_*`` parameters."""
    gate = independent_depth_gate(
        records, horizon_days=horizon_days, min_cohorts=min_cohorts
    )
    out: dict = {
        "mode": "real",
        "source": source,
        "horizon_days": horizon_days,
        "gate": {
            "passed": gate.passed,
            "min_independent_cohorts": gate.min_independent_cohorts,
            "independent_depths": gate.depths,
            "reasons": list(gate.reasons),
        },
    }
    if not gate.passed:
        return {**out, "refused": True, "kill_gate": None}

    raw = cohorts_from_realized_records(records)
    sigma = pooled_within_sd(raw)
    if not np.isfinite(sigma) or sigma <= 0.0:
        out["gate"]["reasons"].append("pooled within-challenger SD is zero")
        return {**out, "refused": True, "kill_gate": None}
    cohorts = [ChallengerCohorts(c.challenger_id, c.diffs / sigma) for c in raw]

    base = fresh_best_wins_promote(cohorts)
    est = estimator_promote(cohorts, alpha=CONTROL_ALPHA)
    control = planted_overfit_control(cohorts)
    return {
        **out,
        "refused": False,
        "noise_sd_pooled_within": sigma,
        "observed": {
            "baseline_promoted": base.promoted_id,
            "estimator_promoted": est.promoted_id,
            "estimator_blocked": est.blocked,
            "estimator_rejected": list(est.rejected_ids),
            "e_values": {k: round(v, 4) for k, v in est.e_values.items()},
        },
        "control": control,
        "kill_gate": control["kill_gate"],
        "kill_gate_label": control["kill_gate_label"],
    }


def load_leaderboard(uri: str) -> dict:
    """Read a leaderboard JSON from a local path or ``s3://bucket/key``."""
    if uri.startswith("s3://"):
        import boto3

        bucket, _, key = uri[len("s3://"):].partition("/")
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    return json.loads(Path(uri).read_text())


def _append_experiments_log(result: dict, run_date: str) -> None:
    root = Path(__file__).resolve().parents[1]
    log = root / "EXPERIMENTS.md"
    verdict = result.get("kill_gate")
    entry = (
        f"\n## {run_date} — config#1524 low-n selection-bias estimator kill-gate "
        f"({result['mode']})\n\n"
        f"- Verdict: **{result.get('kill_gate_label') or verdict}**\n"
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
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--cohorts", help="path to real realized-outcome records JSON")
    src.add_argument(
        "--leaderboard",
        help="scanner/producer leaderboard JSON (local path or s3://bucket/key)",
    )
    ap.add_argument("--alpha", type=float, default=0.10, help="synthetic mode only")
    ap.add_argument("--trials", type=int, default=3000, help="synthetic mode only")
    ap.add_argument("--out", help="write the result artifact JSON to this path")
    ap.add_argument("--log", action="store_true", help="append to EXPERIMENTS.md")
    args = ap.parse_args(argv)

    if args.leaderboard:
        records, _ = records_from_leaderboard(load_leaderboard(args.leaderboard))
        result = run_real(records, source=args.leaderboard)
    elif args.cohorts:
        result = run_real(
            json.loads(Path(args.cohorts).read_text()), source=args.cohorts
        )
    else:
        result = run_synthetic(alpha=args.alpha, trials=args.trials)

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    if result.get("refused"):
        print(
            "REFUSED: the independent-cohort data gate is not met "
            f"(alpha-engine-config-I12662): {'; '.join(result['gate']['reasons'])}",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if args.log:
        _append_experiments_log(result, date.today().isoformat())
    # Non-zero exit on anything but a PASS so CI/operators notice; INCONCLUSIVE
    # never exits like a PASS (I12662 ruling 1b). The power leg never changes it.
    verdict = result.get("kill_gate")
    if verdict == "FALSIFIED":
        return EXIT_FALSIFIED
    if verdict == CONTROL_NO_CONTRAST_VERDICT:
        return EXIT_INCONCLUSIVE
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
