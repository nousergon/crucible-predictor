"""Post-fit arm validity — Brian's ruling of 2026-08-29 (I9290, layer 2).

    "it sounds like predictor should have failed this week due to the outage
    not properly training any of the models. really if any of the arms is not
    trained properly then the predictor module should fail the task"

The fixture throughout is the live outage: the macro block plus
``regime_intensity_z`` hard-zeroed, and the arm's standardized coefficient
norm running 0.3142 -> 0.2701 -> 0.1214 across three vintages.
"""
from __future__ import annotations

import numpy as np
import pytest

from training.arm_validity import (
    FEATURE_BLOCKS,
    ArmValidityError,
    assert_arm_valid,
    coef_norm,
    constant_input_columns,
    evaluate_arm_validity,
)

_MACRO = list(FEATURE_BLOCKS["macro"]) + list(FEATURE_BLOCKS["regime_derived"])
_LIVE = ["momentum_score", "expected_move", "research_calibrator_prob"]
_ALL = _LIVE + _MACRO


def _panel(*, macro_dead: bool, n: int = 200, seed: int = 3):
    rng = np.random.default_rng(seed)
    cols = []
    for name in _ALL:
        if macro_dead and name in _MACRO:
            cols.append(np.zeros(n))
        else:
            cols.append(rng.normal(size=n))
    return np.column_stack(cols)


def _coefs(*, macro_dead: bool, scale: float = 1.0):
    out = {n: scale * 0.10 for n in _LIVE}
    out.update({n: (0.0 if macro_dead else scale * 0.05) for n in _MACRO})
    return out


class TestTheWeekThatShouldHaveFailed:
    def test_a_hard_zeroed_macro_block_fails_the_arm(self):
        v = evaluate_arm_validity(
            arm="v3.0-meta",
            standardized_coef=_coefs(macro_dead=True),
            meta_X=_panel(macro_dead=True),
            feature_names=_ALL,
            prior_standardized_coef=_coefs(macro_dead=False),
            prior_coef_norms=[0.3142, 0.2701],
        )
        assert v["status"] == "invalid"
        names = {f["check"] for f in v["failures"]}
        assert "constant_input_column" in names
        assert "dead_feature_block" in names

    def test_the_task_actually_fails_and_the_message_names_everything(self):
        v = evaluate_arm_validity(
            arm="spec-residual-mom",
            standardized_coef=_coefs(macro_dead=True),
            meta_X=_panel(macro_dead=True),
            feature_names=_ALL,
            prior_standardized_coef=_coefs(macro_dead=False),
            prior_coef_norms=[0.3142, 0.2701],
        )
        with pytest.raises(ArmValidityError) as exc:
            assert_arm_valid(v)
        msg = str(exc.value)
        # the arm
        assert "spec-residual-mom" in msg
        # the assertion
        assert "constant_input_column" in msg
        # the input
        assert "macro_vix_term_slope" in msg
        # measured vs required
        assert "measured" in msg and "required" in msg
        # and the reader is told serving is unaffected
        assert "promote_to_champion" in msg
        assert "training failed" not in msg.lower()

    def test_the_coefficient_norm_collapse_of_2026_08_28_is_refused(self):
        v = evaluate_arm_validity(
            arm="v3.0-meta",
            standardized_coef={"a": 0.1214},          # norm 0.1214
            meta_X=_panel(macro_dead=False),
            feature_names=_ALL,
            prior_standardized_coef=_coefs(macro_dead=False),
            prior_coef_norms=[0.3142, 0.3142, 0.2701],  # median 0.3142
        )
        collapse = next(c for c in v["checks"] if c["check"] == "coef_norm_collapse")
        assert collapse["status"] == "fail"
        assert "0.3865" in collapse["reason"] or "0.386" in collapse["reason"]
        assert v["status"] == "invalid"

    def test_the_2026_08_21_vintage_is_NOT_refused_by_this_gate(self):
        """0.2701 / 0.3142 = 0.860. That week is the BEHAVIORAL VETO's case;
        this gate must not be tuned to duplicate a guard that already works."""
        v = evaluate_arm_validity(
            arm="v3.0-meta",
            standardized_coef={"a": 0.2701},
            meta_X=_panel(macro_dead=False),
            feature_names=_ALL,
            prior_standardized_coef=_coefs(macro_dead=False),
            prior_coef_norms=[0.3142, 0.3142],
        )
        collapse = next(c for c in v["checks"] if c["check"] == "coef_norm_collapse")
        assert collapse["status"] == "pass"


class TestAHealthyArm:
    def test_a_healthy_fit_is_valid_and_does_not_raise(self):
        v = evaluate_arm_validity(
            arm="v3.0-meta",
            standardized_coef=_coefs(macro_dead=False),
            meta_X=_panel(macro_dead=False),
            feature_names=_ALL,
            prior_standardized_coef=_coefs(macro_dead=False),
            prior_coef_norms=[0.30, 0.31, 0.32],
        )
        assert v["status"] == "valid"
        assert v["failures"] == []
        assert_arm_valid(v)


class TestUncomputableIsNeverAPass:
    def test_a_first_vintage_is_degraded_not_valid_and_does_not_raise(self):
        v = evaluate_arm_validity(
            arm="brand-new-arm",
            standardized_coef=_coefs(macro_dead=False),
            meta_X=_panel(macro_dead=False),
            feature_names=_ALL,
            prior_standardized_coef=None,
            prior_coef_norms=[],
        )
        assert v["status"] == "degraded"
        statuses = {c["check"]: c["status"] for c in v["checks"]}
        assert statuses["dead_feature_block"] == "insufficient"
        assert statuses["coef_norm_collapse"] == "insufficient"
        # the history-free check is never insufficient
        assert statuses["constant_input_column"] == "pass"
        assert_arm_valid(v)  # insufficient does not block (policy §5.1)

    def test_a_first_vintage_with_a_dead_column_still_FAILS(self):
        """The history-free assertion has no excuse. This is the case that
        makes 'no history' safe to treat as non-blocking."""
        v = evaluate_arm_validity(
            arm="brand-new-arm",
            standardized_coef=_coefs(macro_dead=True),
            meta_X=_panel(macro_dead=True),
            feature_names=_ALL,
            prior_standardized_coef=None,
            prior_coef_norms=[],
        )
        assert v["status"] == "invalid"
        with pytest.raises(ArmValidityError, match="constant_input_column"):
            assert_arm_valid(v)

    def test_a_missing_panel_is_insufficient_never_a_pass(self):
        v = evaluate_arm_validity(
            arm="a", standardized_coef={"x": 0.2},
            meta_X=None, feature_names=None,
            prior_coef_norms=[0.2],
        )
        statuses = {c["check"]: c["status"] for c in v["checks"]}
        assert statuses["constant_input_column"] == "insufficient"

    def test_an_empty_coefficient_block_is_insufficient_not_zero(self):
        assert coef_norm(None) is None
        assert coef_norm({}) is None
        assert coef_norm({"a": float("nan")}) is None


class TestTheHistoryFreeCheck:
    def test_constant_and_all_nan_columns_are_both_named(self):
        X = np.column_stack([
            np.arange(10.0), np.zeros(10), np.full(10, np.nan), np.full(10, 4.2),
        ])
        assert constant_input_columns(X, ["live", "zeros", "nans", "const"]) == [
            "zeros", "nans", "const",
        ]

    def test_a_live_panel_names_nothing(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 3))
        assert constant_input_columns(X, ["a", "b", "c"]) == []


class TestWiring:
    def test_meta_trainer_gates_on_arm_validity_after_the_fit(self):
        import inspect

        from training import meta_trainer

        src = inspect.getsource(meta_trainer.run_meta_training)
        assert "assert_arm_valid(arm_validity)" in src
        # after the fit, not before it
        assert src.index("meta_model.fit(") < src.index("assert_arm_valid(")

    def test_the_verdict_reaches_the_manifest(self):
        import inspect

        from training import meta_trainer

        src = inspect.getsource(meta_trainer.run_meta_training)
        assert src.count('"arm_validity": arm_validity') >= 2

    def test_a_failed_training_cannot_write_the_live_serving_prefix(self):
        """Ruling constraint A. The invariant that makes a hard failure safe:
        `promote_to_champion` is the only writer of the live prefix, so a raise
        in training leaves the serving champion exactly where it was.

        The fleet-wide guard is tests/test_live_prefix_single_writer.py; this
        asserts the property arm_validity's failure message relies on.
        """
        import inspect

        from training import arm_validity as av

        assert "promote_to_champion" in inspect.getsource(av.assert_arm_valid)
        import model.registry as reg

        assert hasattr(reg, "promote_to_champion")
