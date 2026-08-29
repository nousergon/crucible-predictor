# EXPERIMENTS

Falsifiable-experiment log (config#1522/#1524 kill-gate + future negative results).

## 2026-07-03 — config#1524 low-n selection-bias estimator kill-gate (synthetic)

- Verdict: **PASS**
- Result: `{"checks": {"fdr_controlled_all_depths": true, "power_emerges_by_max_depth": true, "separates_where_low_n_bias_bites": true}, "d_star_cohorts_for_power": 20, "kill_gate": "PASS", "mode": "synthetic", "params": {"alpha": 0.1, "depths": [6, 8, 12, 16, 20, 26], "k_forks": 8, "noise_sd": 1.0, "power_floor": 0.5, "real_effect": 0.8, "seed": 1524, "shallow_baseline_floor": 0.15, "trials": 3000}, "sweep": [{"baseline_overfit_rate": 0.322, "estimator_block_rate": 1.0, "estimator_overfit_rate": 0.0, "estimator_power": 0.0, "n_cohorts": 6}, {"baseline_overfit_rate": 0.239, "estimator_block_rate": 0.9983, "estimator_overfit_rate": 0.0, "estimator_power": 0.0017, "n_cohorts": 8}, {"baseline_overfit_rate": 0.1303, "estimator_block_rate": 0.893, "estimator_overfit_rate": 0.0003, "estimator_power": 0.1067, "n_cohorts": 12}, {"baseline_overfit_rate": 0.0627, "estimator_block_rate": 0.6177, "estimator_overfit_rate": 0.003, "estimator_power": 0.3793, "n_cohorts": 16}, {"baseline_overfit_rate": 0.036, "estimator_block_rate": 0.3753, "estimator_overfit_rate": 0.0063, "estimator_power": 0.6183, "n_cohorts": 20}, {"baseline_overfit_rate": 0.013, "estimator_block_rate": 0.1833, "estimator_overfit_rate": 0.0073, "estimator_power": 0.8093, "n_cohorts": 26}]}`

## 2026-08-29 — M slot: the multi-horizon arms are declared OUT of the slot (alpha-engine-config-I9313)

- Verdict: **NEGATIVE RESULT — arms withdrawn from the slot, not promoted, not normalized.**
- Evidence: leaderboard `predictor/model_zoo/leaderboard/2026-08-28.json`. `horizon-60d`
  scored CPCV mean IC 0.046929, passed the DSR gate and the registry bar, and was
  refused `non_canonical_horizon`. `horizon-90d` scored 0.024067 and was
  separately dropped by `selection_pbo` as a `dropped_misaligned_spec`. Two of
  four challengers were structurally unable to win a promotion regardless of
  score, at the cost of one full weekly training run each — and no artifact
  anywhere recorded why.
- Promotion history: `predictor/model_zoo/promotions/` shows exactly ONE
  challenger promotion ever (2026-07-17, `spec-residual-mom-2026-07-17-f478ece3`).
  Every promotion 07-24 through 08-21 was `promoted_kind: champion-arch-refresh`
  with `winner_version_id: null` — the champion architecture retraining itself.
- Decision, and why NOT horizon normalization: a normalized statistic
  (IC/sqrt(h), annualized IC-IR) would make the numbers comparable but not the
  arms. The M-slot champion is what `model.registry.promote_to_champion`
  publishes, and the executor implements a 21-day hold; promoting a 60d arm on a
  normalized score changes the contract the slot serves rather than winning the
  slot. `champion-challenger-policy.md` §2: a different decision is a different
  slot with its own scoring path, never a borrowed one.
- Where the horizon question goes instead: `analysis/horizon_battery.py` already
  computes 5d/10d/21d/60d/90d IC with non-overlapping windows and bootstrap CIs
  from the OOS rows the champion run persists, at ~zero marginal cost. The
  question keeps being measured; it stops costing two training runs a week.
- Effect: the arm register (`training/model_zoo_registry.py`) resolves both arms
  `inapplicable` and refuses to schedule them, freeing 2 of the 4
  `model_zoo_weekly_budget` slots for 21-day challengers. Retirement proper —
  flipping `status` and deleting the spec per §6 — is an operator decision and
  is recorded in `config/predictor.sample.yaml` awaiting a ruling on the live
  `predictor.yaml`.
