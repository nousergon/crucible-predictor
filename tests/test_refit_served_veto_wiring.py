"""alpha-engine-config-I10290 — the behavioural veto must be able to bite on a REFIT.

The 2026-09-04 rotation, measured from the real artifacts:

* `predictor/model_zoo/leaderboard/2026-09-04.json` — the arena HELD
  (`arena_decision.moved: false`, `status: held`) on `M:champion-arch:4db81ad0d630`,
  and `promoted_kind: "refit"` replaced the serving `v3.0-meta-2026-08-14-119e069b`
  with `v3.0-meta-2026-09-04-cc3271ea`.
* `served_slice_dispersion.status: "uncomputable"` (no OOS panel — -I10173), so the
  three scale-dependent metrics were `behavioral_veto_uncomputable:
  ["alpha_stdev", "model_hit_rate_30d", "n_high_confidence"]`.
* The only metric that DID compute was the manifest's synthetic 25-point
  calibrator sweep, and it read BETTER: `stdev_p_up` 0.191475 vs 0.113644,
  ratio 1.685 — a comfortable pass.
* The served book on the refit's first day, `predictor/predictions/2026-09-08.json`
  vs `2026-09-04.json`: mean confidence 0.1619 -> 0.0593, cross-sectional `p_up`
  stdev 0.0907 -> 0.0217 (ratio 0.24), `predicted_alpha` stdev 0.01898 -> 0.00582
  (ratio 0.31), names clearing MIN_CONFIDENCE 2 -> 0 (max confidence 0.107).

Two structural defects, both fixed here:

1. `serving_preconditions` took the incumbent side of every ratio from
   `manifests_by_arm[incumbent_arm]` — the incumbent arm's NEWEST bundle. On a
   refit that IS the candidate bundle, so the veto compared a manifest against
   itself: every ratio is exactly 1.0, and the guard cannot refuse a refit
   however far the weights have collapsed.
2. `run_slot`'s `served_metrics_by_arm` had no caller. The served-slice
   measurement — the only scale-DEPENDENT one, and the only one measuring what
   the executor consumes (policy §7.3) — was computed in `select_winner`,
   written to the leaderboard, then dropped before the precondition ran.
"""
from training import arena_model_slot as ams


def _manifest(stdev_p_up, *, alpha_stdev=None):
    m = {"output_distribution_gate": {"metrics": {"stdev_p_up": stdev_p_up}}}
    if alpha_stdev is not None:
        m["behavioral_metrics"] = {"alpha_stdev": alpha_stdev}
    return m


def _veto_check(preconditions, arm_id):
    for c in preconditions[arm_id]:
        if c.name == "behavioral_veto":
            return c
    raise AssertionError("no behavioral_veto precondition emitted")


ARM = "M:champion-arch:4db81ad0d630"


def test_a_refit_is_compared_against_the_SERVING_bundle_not_itself():
    """The defect: on a refit the arm's newest bundle is the candidate, so
    taking the incumbent side from `manifests_by_arm[incumbent_arm]` compares
    the candidate to itself. Fails on the pre-fix code — the collapsed refit
    passed because 0.02/0.02 == 1.0.
    """
    refit = _manifest(0.02, alpha_stdev=0.00582)      # the collapsed new bundle
    serving = _manifest(0.11, alpha_stdev=0.01898)    # what is actually serving

    pre = ams.serving_preconditions(
        arm_ids=[ARM],
        manifests_by_arm={ARM: refit},   # the arm's NEWEST bundle IS the refit
        incumbent_arm=ARM,
        serving_manifest=serving,
    )
    check = _veto_check(pre, ARM)
    assert not check.passed, (
        "a refit collapsing alpha_stdev to 0.31x and stdev_p_up to 0.18x of the "
        "SERVING bundle must be refused — this is the 2026-09-04 shape"
    )


def test_without_the_serving_manifest_the_refit_self_comparison_still_passes():
    """Pins WHY the fix is the parameter and not a threshold: with the old
    incumbent basis the very same collapsed bundle reads as healthy, because
    every ratio is 1.0 by construction."""
    refit = _manifest(0.02, alpha_stdev=0.00582)
    pre = ams.serving_preconditions(
        arm_ids=[ARM], manifests_by_arm={ARM: refit}, incumbent_arm=ARM,
    )
    assert _veto_check(pre, ARM).passed


def test_a_pointer_move_to_another_arm_is_unaffected():
    """Strict widening: when the pointer moves to a DIFFERENT arm, the serving
    bundle is the incumbent arm's own newest bundle and nothing changes."""
    challenger = "M:sota-directional-combine:94ed0db42137"
    manifests = {
        challenger: _manifest(0.12, alpha_stdev=0.019),
        ARM: _manifest(0.11, alpha_stdev=0.018),
    }
    with_serving = ams.serving_preconditions(
        arm_ids=[challenger], manifests_by_arm=manifests, incumbent_arm=ARM,
        serving_manifest=manifests[ARM],
    )
    without = ams.serving_preconditions(
        arm_ids=[challenger], manifests_by_arm=manifests, incumbent_arm=ARM,
    )
    assert _veto_check(with_serving, challenger).passed
    assert _veto_check(without, challenger).passed


def test_served_slice_metrics_reach_the_precondition_and_win_over_the_manifest():
    """The served slice is the highest-precedence source (policy §7.3). The
    2026-09-04 numbers exactly: the SYNTHETIC manifest metric improves 1.685x
    while the SERVED spread falls to 0.31x. Only the served measurement can
    refuse it, and before this change it never reached here.
    """
    refit = _manifest(0.191475)     # synthetic sweep: BETTER than the incumbent
    serving = _manifest(0.113644)
    pre = ams.serving_preconditions(
        arm_ids=[ARM],
        manifests_by_arm={ARM: refit},
        incumbent_arm=ARM,
        serving_manifest=serving,
        served_metrics_by_arm={ARM: {"alpha_stdev": 0.005429,
                                     "n_high_confidence": 0}},
        serving_served_metrics={"alpha_stdev": 0.016624,
                                "n_high_confidence": 30},
    )
    check = _veto_check(pre, ARM)
    assert not check.passed
    assert "n_high_confidence" in check.reason or "alpha_stdev" in check.reason


def test_the_manifest_only_verdict_would_have_PASSED_the_same_bundle():
    """The counterfactual that makes the point: with only the manifest numbers,
    the 2026-09-04 refit passes — stdev_p_up ratio 1.685. This is why wiring
    the served metrics through is the fix, not a tighter manifest threshold."""
    pre = ams.serving_preconditions(
        arm_ids=[ARM],
        manifests_by_arm={ARM: _manifest(0.191475)},
        incumbent_arm=ARM,
        serving_manifest=_manifest(0.113644),
    )
    assert _veto_check(pre, ARM).passed
