# EXPERIMENTS

Falsifiable-experiment log (config#1522/#1524 kill-gate + future negative results).

## 2026-07-03 — config#1524 low-n selection-bias estimator kill-gate (synthetic)

- Verdict: **PASS**
- Result: `{"checks": {"fdr_controlled_all_depths": true, "power_emerges_by_max_depth": true, "separates_where_low_n_bias_bites": true}, "d_star_cohorts_for_power": 20, "kill_gate": "PASS", "mode": "synthetic", "params": {"alpha": 0.1, "depths": [6, 8, 12, 16, 20, 26], "k_forks": 8, "noise_sd": 1.0, "power_floor": 0.5, "real_effect": 0.8, "seed": 1524, "shallow_baseline_floor": 0.15, "trials": 3000}, "sweep": [{"baseline_overfit_rate": 0.322, "estimator_block_rate": 1.0, "estimator_overfit_rate": 0.0, "estimator_power": 0.0, "n_cohorts": 6}, {"baseline_overfit_rate": 0.239, "estimator_block_rate": 0.9983, "estimator_overfit_rate": 0.0, "estimator_power": 0.0017, "n_cohorts": 8}, {"baseline_overfit_rate": 0.1303, "estimator_block_rate": 0.893, "estimator_overfit_rate": 0.0003, "estimator_power": 0.1067, "n_cohorts": 12}, {"baseline_overfit_rate": 0.0627, "estimator_block_rate": 0.6177, "estimator_overfit_rate": 0.003, "estimator_power": 0.3793, "n_cohorts": 16}, {"baseline_overfit_rate": 0.036, "estimator_block_rate": 0.3753, "estimator_overfit_rate": 0.0063, "estimator_power": 0.6183, "n_cohorts": 20}, {"baseline_overfit_rate": 0.013, "estimator_block_rate": 0.1833, "estimator_overfit_rate": 0.0073, "estimator_power": 0.8093, "n_cohorts": 26}]}`
