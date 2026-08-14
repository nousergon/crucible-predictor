"""self_test_cases.py — the TRAINING-scope half of the predictor's numeric
self-test (alpha-engine-config-I7262).

WHY THIS IS A SEPARATE MODULE FROM ``inference/self_test.py``
--------------------------------------------------------------
``training/`` is deliberately NOT copied into the predictor's Lambda image
(``tests/test_dockerfile_import_closure.py`` pins this, and its
``test_training_package_is_not_reachable_from_any_lambda_entrypoint`` fails the
build if anything reachable from a Lambda entrypoint imports it). Training runs
on an EC2 spot instance via ``infrastructure/spot_train.sh``; inference runs in
the Lambda. They are two deployment targets with two different closures.

So the battery is split along that seam rather than against it:

===================================  ==============================  ===========
Module                               Covers                          Runs in
===================================  ==============================  ===========
``inference/self_test.py``           regime scoring, calibration,    the Lambda
                                     veto thresholds, ranking        (+ CI)
``training/self_test_cases.py``      IC, deflated Sharpe,            spot (+ CI)
(this file)                          downside statistics
===================================  ==============================  ===========

The alternative — reaching ``training/`` from ``inference/self_test.py`` behind a
string-based ``importlib`` call the AST walker cannot follow — would pass the
guard while reproducing exactly the defect the guard exists to catch: a deferred
import the deploy target cannot resolve at runtime. The guard's own docstring says
"Investigate before adjusting any test", and this is the investigation's outcome.

**The scope is recorded in the artifact.** ``run_self_test`` writes a ``scope``
field naming which batteries ran, so an inference-scope PASS can never be read as
full coverage. A battery that silently asks fewer questions is the "a number
published without its members" failure, and it is the reason the field is not
optional.

WHAT THIS HALF PINS
-------------------
Its headline was a real cross-implementation divergence found on first contact
(**alpha-engine-config-I7271**): ``downside_ic_stats`` computed
``downside = ic[ic < target] - target`` and then ``mean(downside ** 2)``, so its
downside-deviation denominator was ``n_downside``, while
``nousergon_lib.quant.riskstats.sortino_ratio`` uses ``min(0, r)`` over ALL ``n``
— the standard Sortino/Satchell convention, and the one the evaluator's own
self-test documents as "the load-bearing half".

**RULED 2026-08-13 (Brian, "use sota" / standing "ALWAYS USE SOTA UNLESS THERE
IS A CLEAR RATIONALE NOT TO"): aligned to the divide-by-N convention.**
``downside_ic_stats`` now agrees with the lib. On the frozen fixture below the
old (n_downside) convention OVERSTATED the deviation by ``sqrt(10/3) =
1.8257x``, so every historical ``sortino_of_ic`` RISES by that same factor
under the fix, and ``passes_downside_gate`` becomes
MORE PERMISSIVE at any fixed ``sortino_threshold`` (which defaults to 0.0 and
has never been set to a tuned nonzero value — W1.4 was still pending, so this
was free to fix now rather than after a threshold was tuned against the wrong
scale).
"""

from __future__ import annotations

import math

from inference.self_test import TOLERANCE, Case

#: Ten IC observations, three negative — above the n<5 insufficiency floor, and
#: deliberately asymmetric so a sign error cannot cancel.
DOWNSIDE_IC = (0.05, 0.03, -0.02, 0.07, -0.04, 0.01, 0.06, -0.01, 0.02, 0.08)

#: Cross-sectional IC fixture: 24 names on one date, above the lib's default
#: min_samples=20 floor so both implementations are in their measuring regime.
IC_N = 24

SCOPE = "training"


# ── fixtures ────────────────────────────────────────────────────────────────

def ic_fixture() -> tuple[list[float], list[float]]:
    """24 (prediction, realized) pairs on one date, with a known rank structure.

    Predictions are 0..23. Realized returns follow the same order EXCEPT for one
    adjacent transposition near the middle, so the Spearman IC is high but not
    1.0 — a perfect-correlation fixture would pass under any implementation that
    merely preserves order, including a broken one.
    """
    preds = [float(i) for i in range(IC_N)]
    realized = [float(i) * 0.001 for i in range(IC_N)]
    realized[10], realized[11] = realized[11], realized[10]
    return preds, realized


# ── expectations, derived on paper ──────────────────────────────────────────

def expected_spearman_ic() -> float:
    """All ranks distinct, so rho = 1 - 6*sum(d^2)/(n(n^2-1)). One adjacent
    transposition gives two pairs with d = +/-1, so sum(d^2) = 2."""
    n = IC_N
    return 1.0 - 12.0 / (n * (n * n - 1))


def expected_sortino_pre_fix_n_downside() -> float:
    """The PRE-FIX (retired) convention: denominator was the count of NEGATIVES.

    Kept ONLY as a documented historical value — this is what
    ``downside_ic_stats`` returned before alpha-engine-config-I7271 was fixed,
    and it is what ``downside_ic_sortino_closed_form`` would still assert if the
    production code ever regressed back to it. Rounded to 6dp because the
    production path rounds there.
    """
    mean_ic = sum(DOWNSIDE_IC) / len(DOWNSIDE_IC)
    downside = [v for v in DOWNSIDE_IC if v < 0.0]
    dd = math.sqrt(sum(d * d for d in downside) / len(downside))
    return round(mean_ic / dd, 6)


def expected_sortino_full_n() -> float:
    """The FLEET-STANDARD convention (matches
    ``nousergon_lib.quant.riskstats.sortino_ratio``, and now matches
    ``training/deflated_sharpe.py::downside_ic_stats`` post-I7271-fix): RMS of
    ``min(0, r)`` over ALL n."""
    n = len(DOWNSIDE_IC)
    mean_ic = sum(DOWNSIDE_IC) / n
    downside = [min(0.0, v) for v in DOWNSIDE_IC]
    dd = math.sqrt(sum(d * d for d in downside) / n)
    return mean_ic / dd


def expected_sortino_convention_ratio() -> float:
    """POST-FIX: local-over-reference must be 1.0 — both now divide by N.

    Retained for its historical value too: pre-fix this was
    ``sqrt(n_downside/n) = sqrt(3/10) = 0.5477225575051661``, the reciprocal of
    the ``sqrt(10/3)`` inflation the n_downside denominator put on the downside
    deviation. If this number ever moves off 1.0 again, one of the two
    conventions changed — the re-examination trigger for I7271.
    """
    return 1.0


def expected_sortino_pre_fix_convention_ratio() -> float:
    """Historical value of ``expected_sortino_convention_ratio`` before the
    I7271 fix: ``sqrt(n_downside/n) = sqrt(3/10)``."""
    n = len(DOWNSIDE_IC)
    n_down = sum(1 for v in DOWNSIDE_IC if v < 0.0)
    return math.sqrt(n_down / n)


def expected_downside_deviation_ratio() -> float:
    """sqrt(n/n_downside) = sqrt(10/3) — the factor on the DOWNSIDE DEVIATION
    itself, which is the quantity the two conventions actually disagree on."""
    n = len(DOWNSIDE_IC)
    n_down = sum(1 for v in DOWNSIDE_IC if v < 0.0)
    return math.sqrt(n / n_down)


def sample_sd(values) -> float:
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


# ── production drivers ──────────────────────────────────────────────────────

def _downside_stats(ic_series=DOWNSIDE_IC) -> dict:
    import numpy as np

    from training.deflated_sharpe import downside_ic_stats

    return downside_ic_stats(np.array(list(ic_series)))


def _lib_sortino_of_ic(ic_series=DOWNSIDE_IC) -> float:
    """The SAME quantity via ``nousergon_lib.quant.riskstats.sortino_ratio``,
    called UNROUNDED (production rounds ``sortino_of_ic`` to 6dp; this reference
    does not) so the agreement case measures the real cross-implementation gap
    rather than a rounding artifact. ``periods_per_year=1`` disables the lib's
    annualization (irrelevant to an IC series) and ``risk_free_rate=0.0``
    matches ``downside_ic_stats``'s default ``target=0.0``."""
    from nousergon_lib.quant.riskstats import sortino_ratio

    return sortino_ratio(
        list(ic_series), risk_free_rate=0.0, periods_per_year=1,
    )


def _ic_series(preds, realized, dates=None):
    import numpy as np

    from training.leakfree_meta_ic import cross_sectional_ic_series

    return cross_sectional_ic_series(
        np.array(preds), np.array(realized),
        np.array(dates if dates is not None else ["2026-08-15"] * len(preds)),
    )


def _predictor_ic() -> float:
    """The predictor's OWN per-date cross-sectional IC (scipy.stats.spearmanr)."""
    preds, realized = ic_fixture()
    series = _ic_series(preds, realized)
    if len(series) != 1:
        raise AssertionError(
            f"the IC fixture is one date but the production path returned "
            f"{len(series)} values — the self-test measured the wrong thing"
        )
    return float(series[0])


def _lib_ic() -> float:
    """The SAME quantity via ``nousergon_lib.quant.stats.information_coefficient``."""
    from nousergon_lib.quant.stats.information_coefficient import compute_ic

    preds, realized = ic_fixture()
    result = compute_ic(preds, realized)
    if result.get("status") != "ok":
        raise AssertionError(
            f"the lib IC reference returned status={result.get('status')!r} on a "
            f"{IC_N}-pair fixture with variance on both sides — the agreement "
            "case measured nothing"
        )
    return float(result["ic"])


def _negated_ic() -> float:
    preds, realized = ic_fixture()
    return float(_ic_series([-p for p in preds], realized)[0])


def _permuted_ic() -> float:
    preds, realized = ic_fixture()
    return float(_ic_series(list(reversed(preds)), list(reversed(realized)))[0])


def _constant_prediction_ic_length() -> float:
    _, realized = ic_fixture()
    return float(len(_ic_series([1.0] * IC_N, realized)))


def _is_undefined(value) -> float:
    if value is None:
        return 1.0
    try:
        return 0.0 if math.isfinite(float(value)) else 1.0
    except (TypeError, ValueError):
        return 1.0


def _deflated_sharpe_ic_sd() -> float:
    """The ddof=1 standard deviation the deflated-sharpe path itself computes."""
    import numpy as np

    return float(np.array(list(DOWNSIDE_IC)).std(ddof=1))


# ════════════════════════════════════════════════════════════════════════════
# The training-scope battery
# ════════════════════════════════════════════════════════════════════════════

def build_cases() -> list[Case]:
    """The training-scope cases. Shape-compatible with
    ``inference.self_test.build_cases`` so one runner drives both."""
    return [
        # ── closed form ─────────────────────────────────────────────────────
        Case(
            name="predictor_ic_closed_form",
            description=(
                "training/leakfree_meta_ic.py::cross_sectional_ic_series over 24 "
                "names with all-distinct ranks and ONE adjacent transposition: "
                "rho = 1 - 6*sum(d^2)/(n(n^2-1)) = 1 - 12/13800. Derived from "
                "Spearman's definition, not from scipy"
            ),
            inputs={"n": IC_N, "sum_d_squared": 2.0, "ties": "none",
                    "units": "rank correlation [-1,1]"},
            expected=expected_spearman_ic(),
            compute=_predictor_ic,
        ),
        Case(
            name="downside_ic_sortino_closed_form",
            description=(
                "FIXED 2026-08-13 (alpha-engine-config-I7271, Brian ruling "
                "'use sota'). training/deflated_sharpe.py::downside_ic_stats now "
                "divides the downside sum-of-squares by n (10 here), matching "
                "nousergon_lib.quant.riskstats.sortino_ratio's min(0, r)-over-ALL-n "
                "convention — the standard Sortino/Satchell definition and the "
                "evaluator's own documented 'load-bearing half'. PRE-FIX this "
                "denominator was n_DOWNSIDE (3), which overstated the downside "
                "deviation by sqrt(10/3) = 1.8257x and understated Sortino by its "
                "reciprocal sqrt(3/10) = 0.5477x — the pre-fix expected value is "
                "preserved in expected_sortino_pre_fix_n_downside() so a "
                "regression back to it goes red here"
            ),
            inputs={"ic_series": list(DOWNSIDE_IC), "target": 0.0,
                    "downside_denominator_in_use": "n (10)",
                    "pre_fix_denominator": "n_downside (3)",
                    "pre_fix_value_for_reference": expected_sortino_pre_fix_n_downside(),
                    "fleet_reference_denominator":
                        "n (10), nousergon_lib.quant.riskstats.sortino_ratio",
                    "downside_deviation_factor_pre_fix":
                        expected_downside_deviation_ratio(),
                    "sortino_factor_pre_fix":
                        expected_sortino_pre_fix_convention_ratio(),
                    "direction": "the fix RAISES every historical sortino_of_ic "
                                 "by sqrt(10/3) = 1.8257x, making "
                                 "passes_downside_gate MORE PERMISSIVE at any "
                                 "fixed sortino_threshold",
                    "rounding": "production rounds to 6dp", "units": "ratio"},
            expected=round(expected_sortino_full_n(), 6),
            compute=lambda: float(_downside_stats()["sortino_of_ic"]),
        ),

        # ── cross-implementation agreement (the I7236 class) ────────────────
        Case(
            name="ic_cross_implementation_agreement",
            description=(
                "CROSS-IMPLEMENTATION. training/leakfree_meta_ic.py calls "
                "scipy.stats.spearmanr DIRECTLY while "
                "nousergon_lib.quant.stats.information_coefficient.compute_ic "
                "computes the SAME quantity. Nothing asserted they agree — which "
                "is exactly the absence that let a sqrt(365)-vs-sqrt(252) Sharpe "
                "divergence live on one Report Card (I7236). Difference expected 0.0"
            ),
            inputs={"local": "training.leakfree_meta_ic.cross_sectional_ic_series",
                    "reference":
                        "nousergon_lib.quant.stats.information_coefficient.compute_ic",
                    "n": IC_N, "units": "rank correlation (difference)"},
            expected=0.0,
            compute=lambda: _predictor_ic() - _lib_ic(),
        ),
        Case(
            name="sortino_convention_agreement",
            description=(
                "CROSS-IMPLEMENTATION. Post-I7271-fix, both implementations "
                "divide by N, so ratio = local sortino_of_ic / "
                "nousergon_lib.quant.riskstats.sortino_ratio(same series, "
                "periods_per_year=1) must equal 1.0. Renamed from "
                "'sortino_convention_divergence_agreement' — pre-fix this case "
                "measured the SIZE of the divergence (sqrt(3/10) = "
                "0.5477225575051661, preserved in "
                "expected_sortino_pre_fix_convention_ratio()) rather than "
                "asserting agreement, because the two implementations provably "
                "did not agree. If this number ever moves off 1.0, one of the "
                "two conventions changed — the re-examination trigger for I7271"
            ),
            inputs={"local": "training.deflated_sharpe.downside_ic_stats",
                    "reference": "nousergon_lib.quant.riskstats.sortino_ratio "
                                 "(periods_per_year=1, risk_free_rate=0.0)",
                    "n": len(DOWNSIDE_IC),
                    "n_downside": sum(1 for v in DOWNSIDE_IC if v < 0.0),
                    "pre_fix_ratio_for_reference":
                        expected_sortino_pre_fix_convention_ratio(),
                    "units": "ratio"},
            expected=expected_sortino_convention_ratio(),
            compute=lambda: (float(_downside_stats()["sortino_of_ic"])
                             / _lib_sortino_of_ic()),
            # 6dp production rounding on the numerator caps achievable agreement
            # at ~1e-6; tightening below that would assert precision the artifact
            # does not carry.
            tolerance=1e-5,
        ),

        # ── metamorphic ─────────────────────────────────────────────────────
        Case(
            name="ic_sign_flip_metamorphic",
            description=(
                "METAMORPHIC. Negating every prediction reverses every rank, so "
                "the cross-sectional IC must flip sign exactly. Catches a rank "
                "direction inversion — the error that turns a real edge into a "
                "real anti-edge while the magnitude still looks right. Sum expected 0.0"
            ),
            inputs={"n": IC_N, "relation": "IC(-p, r) + IC(p, r) == 0",
                    "units": "rank correlation (sum)"},
            expected=0.0,
            compute=lambda: _predictor_ic() + _negated_ic(),
        ),
        Case(
            name="ic_permutation_invariance_metamorphic",
            description=(
                "METAMORPHIC. A cross-sectional IC is a property of the (pred, "
                "realized) PAIRS, not of the row order they arrive in. Reversing "
                "the row order must leave it unchanged. Catches an implementation "
                "that correlates against position rather than against its pair. "
                "Difference expected 0.0"
            ),
            inputs={"n": IC_N, "permutation": "full reverse",
                    "units": "rank correlation (difference)"},
            expected=0.0,
            compute=lambda: _permuted_ic() - _predictor_ic(),
        ),

        # ── degenerate inputs (the I7237 class) ─────────────────────────────
        Case(
            name="downside_ic_insufficient_is_undefined_degenerate",
            description=(
                "training/deflated_sharpe.py::downside_ic_stats below the n<5 "
                "floor must report sortino_of_ic as None, never +inf and never a "
                "measured-looking 0.0. 1.0 iff undefined. This is the exact shape "
                "I7237 recorded on a flat market (sharpe +inf, sortino 0.0)"
            ),
            inputs={"ic_series": [0.01, 0.02], "n": 2, "min_n": 5,
                    "units": "boolean-encoded contract"},
            expected=1.0,
            compute=lambda: _is_undefined(
                _downside_stats((0.01, 0.02))["sortino_of_ic"]),
            tolerance=0.0,
        ),
        Case(
            name="downside_ic_no_negatives_is_undefined_degenerate",
            description=(
                "An all-positive IC series has NO downside observations, so the "
                "downside deviation is 0 and the Sortino is UNDEFINED. Must report "
                "None, never +inf — the production code guards dd > 1e-12 for "
                "exactly this. 1.0 iff undefined"
            ),
            inputs={"ic_series": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                    "n_negative": 0, "units": "boolean-encoded contract"},
            expected=1.0,
            compute=lambda: _is_undefined(
                _downside_stats((0.01, 0.02, 0.03, 0.04, 0.05, 0.06))["sortino_of_ic"]),
            tolerance=0.0,
        ),
        Case(
            name="ic_zero_variance_date_dropped_degenerate",
            description=(
                "A date on which every prediction is IDENTICAL has an UNDEFINED "
                "IC (correlation of a constant with anything). The production path "
                "DROPS that date from the series rather than contributing a "
                "fabricated 0.0 that would dilute the Fama-MacBeth mean. Series "
                "length expected 0"
            ),
            inputs={"predictions": "constant", "n": IC_N, "units": "series length"},
            expected=0.0,
            compute=_constant_prediction_ic_length,
            tolerance=0.0,
        ),

        # ── convention pinning ──────────────────────────────────────────────
        Case(
            name="deflated_sharpe_ddof_convention",
            description=(
                "CONVENTION. training/deflated_sharpe.py uses SAMPLE std "
                "(ddof=1) on the IC series, while regime/composite.py uses "
                "population std (ddof=0) on the macro series. Both are defensible "
                "in isolation and neither is declared anywhere; this case pins the "
                "deflated-sharpe side against the hand-computed ddof=1 standard "
                "deviation of the fixture"
            ),
            inputs={"ic_series": list(DOWNSIDE_IC), "ddof_in_use": 1,
                    "source_of_truth": "training/deflated_sharpe.py",
                    "units": "standard deviation"},
            expected=sample_sd(DOWNSIDE_IC),
            compute=_deflated_sharpe_ic_sd,
        ),
    ]


TOLERANCE = TOLERANCE  # re-exported for symmetry with inference.self_test
