"""self_test.py — the predictor's published known-answer SELF-TEST (config-I7262).

WHY THIS EXISTS
---------------
The weekly pipeline's reliability work establishes that each stage **runs**. None
of it establishes that any stage's output is **right**. `sf-pipeline-policy.md`
§2.3a names this as an independent axis: a missing *data artifact* makes a
consumer fail visibly, while a missing *correctness verdict* makes every consumer
succeed **as though the check had passed**.

Counted at `origin/main` on 2026-08-13, `crucible-evaluator` carried three modules
with a numeric self-test and `crucible-backtester` one. `crucible-predictor` — the
M slot, the model whose numbers everything downstream grades — carried **zero**.
That is what this module closes.

The instrument the evaluator's equivalent battery pointed at found a real defect
on first contact: the backtester annualising its headline Sharpe by sqrt(365)
while the evaluator and `nousergon_lib.quant.riskstats` use sqrt(252), a 20.3%
divergence between two numbers on one Report Card (`alpha-engine-config-I7236`).
The pre-existing test asserted only `sharpe > 0`.

WHY IT DRIVES THE PRODUCTION FUNCTIONS ON THE DEPLOYED IMAGE
-------------------------------------------------------------
CI proves the code is correct **on a runner**. It does not prove the **deployed
instrument** is. `requirements.txt` pins `nousergon-lib` by commit SHA and
`numpy`/`scipy`/`pandas`/`scikit-learn` by compatible-release range, all resolved
into the Lambda image at **build** time. A changed `ddof`, tanh scale, rank tie
rule or Spearman implementation would move the regime score and every IC the
promotion gate reads — coherently, plausibly, and entirely invisibly, with CI
green throughout.

So every case below calls the **production** function (`regime.composite`,
`regime.drawdown`, `model.calibrator`, `inference.stages.write_output`) on the
image's own site-packages, and the artifact records the resolved version of every numeric
distribution actually loaded. **The library versions in the header are the
point** — they are what makes this an instrument check rather than a code check.

Every `expected` is derived **on paper from the function's definition** and
recomputed here in plain arithmetic — never by calling the code under test, which
would agree with whatever that code ever does.

THE FOUR CASE CLASSES
---------------------
``*_closed_form``
    Inputs whose correct output is analytically derivable, asserted to 1e-9.

``*_metamorphic``
    Relations that must hold regardless of the data — affine invariance of a
    z-score, sign antisymmetry of tanh, permutation invariance of a rank, IC sign
    flip under negated predictions, monotonicity of p_up in alpha. These catch
    errors closed-form cases cannot, because they do not require knowing the right
    answer, only the right relationship.

``*_degenerate``
    Flat series, zero variance, single observation, empty input. The class that
    produced `alpha-engine-config-I7237`, where a flat market yielded
    `sharpe: +inf` and `calmar: nan`.

``*_convention`` / ``*_agreement``
    Every annualisation-adjacent constant, lookback window and ddof asserted
    against the value ACTUALLY IN USE, and cross-implementation agreement where
    two implementations of one quantity exist. `I7236` exists precisely because
    nothing asserted the second kind.

FIXED (config-I7272) — FORMERLY THE PINNED KNOWN GAPS HERE
-------------------------------------------------------------
Two cases below used to pin behaviour this module considered **wrong**, and are
now FIXED:

``regime_zscore_zero_variance_degenerate``
    `regime/composite.py::_zscore` used to return a finite **0.0** when the
    history has zero variance. A z-score against a constant history is
    UNDEFINED, and 0.0 is the exact value a genuinely at-the-mean observation
    produces — so a measured zero and an undefined value were indistinguishable
    downstream (the `I7237` class at the regime layer). Now returns ``None``,
    and every consumer (`compute_composite_intensity`,
    `compute_intensity_z_series`, `regime/substrate.py`) was audited to drop
    the leg / propagate the None rather than fabricate a value.

``regime_score_no_heads_degenerate``
    `regime/drawdown.py::compose_regime_score` used to return `regime_score:
    0.0` when NO head is present, which `regime_score_to_categorical` then
    projected to `"neutral"` — the same output a genuinely neutral market
    produces. Now returns `regime_score: None` /
    `regime_score_categorical: "unknown"`, a fourth outcome distinct from
    bull/neutral/bear.

Both cases now assert the HONEST representation and will go red on a
regression. `regime_score`/`regime_score_categorical` are still an
OBSERVE-ONLY, additive field (per `regime/drawdown.py`'s `compose_effective_
regime` docstring) — no live veto/promotion decision reads them yet, so this
change moves no currently-graded number. Filed as `alpha-engine-config-I7272`.

CONTRACT
--------
``run_self_test()`` **never raises**, and the caller writes its output
unconditionally. A case that DISAGREED is ``FAIL`` (evidence the numbers are
wrong); a case that could not RUN is ``UNKNOWN`` (absence of evidence).
Collapsing the two would make a broken image read as a correctness regression.
Per Brian's ruling 2026-08-13, **a case that exceeds its time budget is FAIL,
never UNKNOWN**.

This module introduces no hard-fail path, no new SF state and no topology change.
An accuracy instrument that can take down the pipeline is a worse defect than the
one it detects.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple

logger = logging.getLogger(__name__)

SCHEMA = "predictor_self_test-1.0.0"
COMPONENT = "predictor"

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

#: ``predictor/{run_date}/self_test.json`` — beside the run's predictions.
_KEY_TEMPLATE = "predictor/{run_date}/self_test.json"

#: The console component id. Renders through the fleet `checks-envelope` adapter
#: (`nous-ergon-ops/nousergon-console/config.d/fleet-checks.yaml`), which
#: discovers producers by S3 prefix — so this row appears on the console on its
#: first successful publish with no console deploy.
CHECK_ID = "ae-predictor-self-test"

#: Weekly cadence: this runs with the weekly SF, so a row older than ~1.5 weeks
#: is the console's honest STALE, not a green. Declared, never guessed.
CADENCE_MINUTES = 7 * 24 * 60

#: Distributions whose resolved version decides what every number here means.
#: Recorded via ``importlib.metadata`` (the DISTRIBUTION version pip actually
#: resolved into the image) rather than ``module.__version__`` — an attribute a
#: package may lack, lie about, or inherit from a vendored copy.
_TRACKED_DISTRIBUTIONS = (
    "nousergon-lib",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "krepis",
    "boto3",
)

#: Per-case wall-clock budget. Each case is one production call over a handful of
#: in-memory rows; anything approaching this is a hang, not a slow machine.
CASE_TIMEOUT_SECONDS = 30.0

#: 1e-9 absolute, per `alpha-engine-config-I7262`. Every expectation is an exact
#: float64 identity and the observed agreement is ~1e-16 — the band is far tighter
#: than any convention change could hide under, and is not tuned to make the
#: battery pass.
TOLERANCE = 1e-9

# ── the frozen fixtures ─────────────────────────────────────────────────────
# Small, hand-derivable, and deliberately NOT round: a fixture of 1.0s hides a
# multiplication bug, and a symmetric fixture hides a sign bug.

#: Six-point macro history for the regime z-score. mean = 3.5, population
#: variance = 35/12, so sigma = sqrt(35/12) exactly — an irrational number, so an
#: accidental ddof change cannot coincide to 1e-9.
_Z_HISTORY = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
_Z_VALUE = 8.0
#: Second feature's current value. Deliberately different from _Z_VALUE so the
#: two-feature composite blend cannot collapse to an identity-independent 0.0.
_Z_VALUE_B = 2.0
#: Affine transform for the metamorphic invariance case. Both non-trivial.
_Z_AFFINE_SCALE = 7.0
_Z_AFFINE_SHIFT = -13.0

#: Veto-adjustment probe. 1.3 * 0.10 = 0.13, inside the 0.20 cap.
_VETO_Z = 1.3
_VETO_Z_OVER_CAP = 5.0

#: HMM staleness probe: exactly one half-life past the grace window, so the decay
#: term is 0.5**1 and the expectation is derivable without a calculator.
_HMM_WEEKS = 20  # 8 grace + 12 halflife
_HMM_P_BULL = 0.70
_HMM_P_BEAR = 0.10

#: Calibrator probe. The linear fallback is p_up = 0.5 + a/(2*clip).
_CAL_ALPHA = 0.03
_CAL_CLIP = 0.15



class _CaseTimeout(Exception):
    """Raised when a case exceeds :data:`CASE_TIMEOUT_SECONDS`."""


class Case(NamedTuple):
    """One known-answer, metamorphic, degenerate or convention check.

    ``expected`` is derived by hand from the function's definition; ``compute``
    drives the production code and returns the comparable observed number. They
    are kept apart — rather than ``compute`` returning a bool — so the artifact
    carries both numbers and a later divergence is diagnosable from the artifact
    alone. ``inputs`` is published verbatim: a reader must be able to re-derive
    ``expected`` on paper without opening this file.

    ``known_gap`` marks a case that pins behaviour believed WRONG, so a reader
    never mistakes its PASS for an endorsement. See the module docstring.
    """

    name: str
    description: str
    inputs: dict
    expected: float
    compute: Callable[[], float]
    tolerance: float = TOLERANCE
    known_gap: bool = False
    gap_issue: str | None = None


# ════════════════════════════════════════════════════════════════════════════
# Fixtures — built here so a case never touches S3, the network or a real model
# ════════════════════════════════════════════════════════════════════════════

def _z_history_series(scale: float = 1.0, shift: float = 0.0):
    import pandas as pd

    return pd.Series([v * scale + shift for v in _Z_HISTORY])


def _macro_history_frame():
    """Two-feature macro panel for ``compute_composite_intensity``.

    ``feat_a`` carries the _Z_HISTORY series; ``feat_b`` is that series reversed,
    so the two z-scores differ and a weight swap cannot pass unnoticed.
    """
    import pandas as pd

    return pd.DataFrame({"feat_a": list(_Z_HISTORY), "feat_b": list(reversed(_Z_HISTORY))})


# ════════════════════════════════════════════════════════════════════════════
# The expectations — plain arithmetic from each function's definition
# ════════════════════════════════════════════════════════════════════════════

def _expected_zscore() -> float:
    """(_Z_VALUE - mean) / population_sd. ddof=0 is the load-bearing half:
    the sample-sd alternative differs by sqrt(6/5) = 1.0954, a 9.5% error."""
    n = len(_Z_HISTORY)
    mean = sum(_Z_HISTORY) / n
    variance = sum((v - mean) ** 2 for v in _Z_HISTORY) / n  # ddof=0
    return (_Z_VALUE - mean) / math.sqrt(variance)


def _expected_zscore_sample_ddof() -> float:
    """The WRONG answer, kept so the convention case can assert it is not this."""
    n = len(_Z_HISTORY)
    mean = sum(_Z_HISTORY) / n
    variance = sum((v - mean) ** 2 for v in _Z_HISTORY) / (n - 1)  # ddof=1
    return (_Z_VALUE - mean) / math.sqrt(variance)


def _expected_composite_intensity() -> float:
    """sum(w*z) / sum(|w|) over the two present features.

    feat_a's history is _Z_HISTORY and feat_b's is its reverse, so both share the
    same mean and sd — but the two CURRENT values differ (_Z_VALUE = 8.0 vs
    _Z_VALUE_B = 2.0), which is what makes the blend non-degenerate. An earlier
    revision of this fixture used the same value for both features at weights
    (+1, -1); the expectation was then exactly 0.0 for every possible pair of
    z-scores, so the case would have passed against an arbitrarily broken
    function. A closed-form case whose expected value is independent of the
    computation is not a test.
    """
    n = len(_Z_HISTORY)
    mean = sum(_Z_HISTORY) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in _Z_HISTORY) / n)
    z_a = (_Z_VALUE - mean) / sd
    z_b = (_Z_VALUE_B - mean) / sd
    w_a, w_b = 1.0, -1.0
    return (w_a * z_a + w_b * z_b) / (abs(w_a) + abs(w_b))


def _expected_hmm_staleness_discount() -> float:
    """floor + (1 - floor) * 0.5**((weeks - grace)/halflife).
    At exactly one half-life past grace: 0.25 + 0.75*0.5 = 0.625."""
    return 0.25 + (1.0 - 0.25) * 0.5 ** ((_HMM_WEEKS - 8) / 12.0)


def _expected_hmm_posterior_expectation() -> float:
    """(P(bull) - P(bear)) * staleness discount = 0.60 * 0.625 = 0.375."""
    return (_HMM_P_BULL - _HMM_P_BEAR) * _expected_hmm_staleness_discount()


def _expected_intensity_z_to_score() -> float:
    """tanh(z / INTENSITY_Z_TANH_SCALE), with the scale pinned at 2.0."""
    return math.tanh(1.0 / 2.0)


def _expected_calibrator_p_up() -> float:
    """Uncalibrated linear fallback: clip(0.5 + alpha/(2*label_clip), 0, 1)."""
    return min(1.0, max(0.0, 0.5 + _CAL_ALPHA / (2.0 * _CAL_CLIP)))


def _expected_calibrator_confidence() -> float:
    """|p_up - 0.5| * 2, rounded to 4dp by the production path."""
    return round(abs(_expected_calibrator_p_up() - 0.5) * 2.0, 4)




# ════════════════════════════════════════════════════════════════════════════
# The battery
# ════════════════════════════════════════════════════════════════════════════

def _regime_zscore(value: float, scale: float = 1.0, shift: float = 0.0) -> float:
    from regime.composite import _zscore

    return float(_zscore(value * scale + shift, _z_history_series(scale, shift)))


def _composite_intensity() -> float:
    from regime.composite import compute_composite_intensity

    result = compute_composite_intensity(
        {"feat_a": _Z_VALUE, "feat_b": _Z_VALUE_B},
        _macro_history_frame(),
        weights={"feat_a": 1.0, "feat_b": -1.0},
        rolling_window_weeks=None,
    )
    return float(result["intensity_z"])


def _calibrator_prediction(alpha: float) -> dict:
    from model.calibrator import PlattCalibrator

    return PlattCalibrator().calibrate_prediction(alpha, label_clip=_CAL_CLIP)


def _combined_ranks(order: list[float]) -> list[int]:
    """Drive the production merge/rank path and return ranks keyed by ticker."""
    from inference.stages.write_output import _merge_predictions

    new = [{"ticker": f"T{i}", "predicted_alpha": a} for i, a in enumerate(order)]
    merged = _merge_predictions(new, [])
    by_ticker = {p["ticker"]: p["combined_rank"] for p in merged}
    return [by_ticker[f"T{i}"] for i in range(len(order))]


def _is_undefined(value: Any) -> float:
    """1.0 iff ``value`` is reported as undefined (``None``/NaN/inf)."""
    if value is None:
        return 1.0
    try:
        return 0.0 if math.isfinite(float(value)) else 1.0
    except (TypeError, ValueError):
        return 1.0


def build_cases() -> list[Case]:
    """The battery. A callable (not a module constant) so no production import
    happens at import time of this module, and so a test can substitute it."""
    from inference.stages.write_output import regime_conditional_veto_adjustment
    from regime import drawdown as dd

    cases: list[Case] = []

    # ── closed form ─────────────────────────────────────────────────────────
    cases += [
        Case(
            name="regime_zscore_closed_form",
            description=(
                "regime/composite.py::_zscore over history 1..6: mean = 3.5, "
                "POPULATION variance (ddof=0) = 35/12, so z = (8 - 3.5)/sqrt(35/12). "
                "The sample-sd alternative differs by sqrt(6/5) = 1.0954 (9.5%)"
            ),
            inputs={"history": list(_Z_HISTORY), "value": _Z_VALUE, "ddof": 0,
                    "units": "z-score (ratio)"},
            expected=_expected_zscore(),
            compute=lambda: _regime_zscore(_Z_VALUE),
        ),
        Case(
            name="composite_intensity_closed_form",
            description=(
                "regime/composite.py::compute_composite_intensity — "
                "sum(w*z)/sum(|w|) over the two present features at weights "
                "(+1, -1). Pins the ABSOLUTE-VALUE denominator: a signed-sum "
                "denominator would divide by zero here"
            ),
            inputs={"weights": {"feat_a": 1.0, "feat_b": -1.0},
                    "feat_a_value": _Z_VALUE, "feat_b_value": _Z_VALUE_B,
                    "rolling_window_weeks": None, "units": "z-score (ratio)"},
            expected=_expected_composite_intensity(),
            compute=_composite_intensity,
        ),
        Case(
            name="veto_adjustment_closed_form",
            description=(
                "write_output.py::regime_conditional_veto_adjustment — "
                "raw = intensity_z * scale = 1.3 * 0.10 = 0.13, inside the 0.20 cap"
            ),
            inputs={"intensity_z": _VETO_Z, "scale": 0.10, "cap": 0.20,
                    "units": "confidence threshold delta"},
            expected=_VETO_Z * 0.10,
            compute=lambda: float(regime_conditional_veto_adjustment(_VETO_Z)),
        ),
        Case(
            name="veto_adjustment_cap_closed_form",
            description=(
                "Same function at intensity_z = 5.0: raw = 0.50, clamped to the "
                "0.20 cap. Pins the clamp, which is the only thing standing "
                "between a macro spike and an unbounded confidence threshold"
            ),
            inputs={"intensity_z": _VETO_Z_OVER_CAP, "scale": 0.10, "cap": 0.20,
                    "units": "confidence threshold delta"},
            expected=0.20,
            compute=lambda: float(regime_conditional_veto_adjustment(_VETO_Z_OVER_CAP)),
        ),
        Case(
            name="hmm_staleness_discount_closed_form",
            description=(
                "regime/drawdown.py::hmm_staleness_discount at exactly one "
                "half-life past grace (20 = 8 + 12 weeks): "
                "floor + (1-floor)*0.5^1 = 0.25 + 0.75*0.5 = 0.625"
            ),
            inputs={"weeks_in_current_state": _HMM_WEEKS, "grace_weeks": 8,
                    "halflife_weeks": 12.0, "floor": 0.25, "units": "multiplier"},
            expected=_expected_hmm_staleness_discount(),
            compute=lambda: float(dd.hmm_staleness_discount(_HMM_WEEKS)),
        ),
        Case(
            name="hmm_posterior_expectation_closed_form",
            description=(
                "regime/drawdown.py::hmm_posterior_expectation — "
                "(P(bull) - P(bear)) * staleness discount = (0.70-0.10)*0.625 = 0.375. "
                "Pins that the HMM enters as a signed EXPECTATION, not an argmax"
            ),
            inputs={"p_bull": _HMM_P_BULL, "p_bear": _HMM_P_BEAR,
                    "weeks_in_current_state": _HMM_WEEKS, "units": "signed score [-1,1]"},
            expected=_expected_hmm_posterior_expectation(),
            compute=lambda: float(dd.hmm_posterior_expectation(
                {"bull": _HMM_P_BULL, "bear": _HMM_P_BEAR, "neutral": 0.20},
                weeks_in_current_state=_HMM_WEEKS)),
        ),
        Case(
            name="intensity_z_to_score_closed_form",
            description=(
                "regime/drawdown.py::intensity_z_to_score — tanh(z/2.0) at z = 1.0. "
                "Pins INTENSITY_Z_TANH_SCALE = 2.0: at scale 1.0 the same 1-sigma "
                "reading would score 0.762 instead of 0.462, a 65% difference"
            ),
            inputs={"intensity_z": 1.0, "tanh_scale": 2.0, "units": "signed score [-1,1]"},
            expected=_expected_intensity_z_to_score(),
            compute=lambda: float(dd.intensity_z_to_score(1.0)),
        ),
        Case(
            name="calibrator_p_up_closed_form",
            description=(
                "model/calibrator.py::calibrate_prediction UNFITTED linear "
                "fallback: p_up = clip(0.5 + alpha/(2*label_clip), 0, 1) = "
                "0.5 + 0.03/0.30 = 0.60"
            ),
            inputs={"raw_alpha": _CAL_ALPHA, "label_clip": _CAL_CLIP, "fitted": False,
                    "units": "probability"},
            expected=_expected_calibrator_p_up(),
            compute=lambda: float(_calibrator_prediction(_CAL_ALPHA)["p_up"]),
        ),
        Case(
            name="calibrator_confidence_closed_form",
            description=(
                "Same call: confidence = |p_up - 0.5| * 2 = 0.20. Pins the "
                "distance-from-coin-flip definition the veto threshold compares "
                "against — a raw p_up would gate at a completely different level"
            ),
            inputs={"raw_alpha": _CAL_ALPHA, "label_clip": _CAL_CLIP,
                    "units": "confidence [0,1]"},
            expected=_expected_calibrator_confidence(),
            compute=lambda: float(_calibrator_prediction(_CAL_ALPHA)["prediction_confidence"]),
        ),
        Case(
            name="combined_rank_closed_form",
            description=(
                "write_output.py::_merge_predictions — combined_rank is a 1-based "
                "ORDINAL rank by predicted_alpha DESCENDING. Input alphas "
                "[0.01, 0.05, 0.03] must rank [3, 1, 2]; sum of |rank - expected| = 0"
            ),
            inputs={"alphas": [0.01, 0.05, 0.03], "expected_ranks": [3, 1, 2],
                    "order": "descending by predicted_alpha", "units": "rank error sum"},
            expected=0.0,
            compute=lambda: float(sum(
                abs(a - b) for a, b in zip(_combined_ranks([0.01, 0.05, 0.03]), [3, 1, 2])
            )),
        ),
    ]

    # ── cross-implementation agreement (the I7236 class) ────────────────────
    cases += [
    ]

    # ── metamorphic ─────────────────────────────────────────────────────────
    cases += [
        Case(
            name="regime_zscore_affine_invariance_metamorphic",
            description=(
                "METAMORPHIC. A z-score is invariant under any positive affine "
                "transform of BOTH the value and its history: z(7x - 13) = z(x). "
                "Catches a lost mean-centering or a units change without needing "
                "to know the right answer. Difference expected 0.0"
            ),
            inputs={"scale": _Z_AFFINE_SCALE, "shift": _Z_AFFINE_SHIFT,
                    "units": "z-score (difference)"},
            expected=0.0,
            compute=lambda: (
                _regime_zscore(_Z_VALUE, _Z_AFFINE_SCALE, _Z_AFFINE_SHIFT)
                - _regime_zscore(_Z_VALUE)
            ),
        ),
        Case(
            name="intensity_z_to_score_antisymmetry_metamorphic",
            description=(
                "METAMORPHIC. tanh is odd, so the regime squash must satisfy "
                "f(-z) = -f(z) for every z — a bull reading and its mirrored bear "
                "reading must be equal and opposite. An asymmetric clamp or an "
                "off-centre shift breaks this. Sum expected 0.0"
            ),
            inputs={"z": 1.7, "relation": "f(-z) + f(z) == 0",
                    "units": "signed score (sum)"},
            expected=0.0,
            compute=lambda: float(dd.intensity_z_to_score(1.7))
                            + float(dd.intensity_z_to_score(-1.7)),
        ),
        Case(
            name="regime_score_antisymmetry_metamorphic",
            description=(
                "METAMORPHIC. compose_regime_score over the two continuous heads "
                "must be sign-honest: mirroring the market (negate intensity_z, "
                "swap P(bull)/P(bear)) must negate the score exactly. This is the "
                "property that stops a bear reading being systematically softer "
                "than the identical bull reading. Sum expected 0.0"
            ),
            inputs={"intensity_z": 1.1, "p_bull": 0.70, "p_bear": 0.10,
                    "mirrored": {"intensity_z": -1.1, "p_bull": 0.10, "p_bear": 0.70},
                    "units": "signed score (sum)"},
            expected=0.0,
            compute=lambda: (
                float(dd.compose_regime_score(
                    hmm_probs={"bull": 0.70, "bear": 0.10, "neutral": 0.20},
                    intensity_z=1.1)["regime_score"])
                + float(dd.compose_regime_score(
                    hmm_probs={"bull": 0.10, "bear": 0.70, "neutral": 0.20},
                    intensity_z=-1.1)["regime_score"])
            ),
        ),
        Case(
            name="combined_rank_permutation_invariance_metamorphic",
            description=(
                "METAMORPHIC. combined_rank must depend only on predicted_alpha, "
                "never on the order the tickers were merged in. The same three "
                "alphas supplied in reverse must produce the same per-ticker ranks. "
                "Sum of |difference| expected 0.0"
            ),
            inputs={"alphas": [0.01, 0.05, 0.03], "permutation": "reverse",
                    "units": "rank difference sum"},
            expected=0.0,
            compute=_rank_permutation_delta,
        ),
        Case(
            name="calibrator_monotonicity_metamorphic",
            description=(
                "METAMORPHIC. p_up must be NON-DECREASING in raw_alpha: a more "
                "bullish alpha can never produce a lower probability of up. "
                "Asserted across 21 alphas spanning the clip band. Count of "
                "monotonicity violations expected 0"
            ),
            inputs={"alpha_range": [-_CAL_CLIP, _CAL_CLIP], "n_points": 21,
                    "relation": "a1 < a2 => p_up(a1) <= p_up(a2)",
                    "units": "violation count"},
            expected=0.0,
            compute=_calibrator_monotonicity_violations,
            tolerance=0.0,
        ),
    ]

    # ── degenerate inputs (the I7237 class) ─────────────────────────────────
    cases += [
        Case(
            name="hmm_posterior_absent_is_undefined_degenerate",
            description=(
                "No HMM posterior supplied => the leg must report UNDEFINED "
                "(None) so it DROPS OUT of the renormalized blend, rather than "
                "voting a fabricated neutral 0. 1.0 iff None. This is the "
                "CORRECT convention and is pinned as the reference the two "
                "known_gap cases below are measured against"
            ),
            inputs={"probs": None, "units": "boolean-encoded contract"},
            expected=1.0,
            compute=lambda: _is_undefined(dd.hmm_posterior_expectation(None)),
            tolerance=0.0,
        ),
        # ── the former two known gaps: FIXED (config-I7272) ────────────────
        # regime/composite.py::_zscore and regime/drawdown.py::compose_
        # regime_score used to return a finite 0.0 on a degenerate input —
        # indistinguishable from a genuinely at-the-mean / neutral reading.
        # Both now report the value as UNDEFINED (None), matching the
        # already-correct hmm_posterior_expectation convention pinned
        # above. These cases now assert the HONEST behaviour and will go
        # red if a regression reintroduces the fabricated 0.0.
        Case(
            name="regime_zscore_zero_variance_degenerate",
            description=(
                "FIXED (alpha-engine-config-I7272). "
                "regime/composite.py::_zscore against a ZERO-VARIANCE history "
                "now reports UNDEFINED (None) rather than a finite 0.0 — 0.0 "
                "is also the exact value a genuinely at-the-mean observation "
                "produces, so the two were indistinguishable downstream (the "
                "I7237 class at the regime layer). 1.0 iff undefined."
            ),
            inputs={"history": [4.0] * 6, "value": 8.0, "units": "z-score"},
            expected=1.0,
            compute=lambda: _is_undefined(_zero_variance_zscore()),
            tolerance=0.0,
        ),
        Case(
            name="regime_score_no_heads_degenerate",
            description=(
                "FIXED (alpha-engine-config-I7272). "
                "regime/drawdown.py::compose_regime_score with NO head present "
                "now reports regime_score=None, which regime_score_to_categorical "
                "projects to 'unknown' — a fourth, distinct outcome from bull/"
                "neutral/bear — rather than the previous 0.0 -> 'neutral', "
                "indistinguishable from a genuinely neutral market. 1.0 iff "
                "regime_score is undefined AND the categorical is 'unknown'."
            ),
            inputs={"hmm_probs": None, "intensity_z": None, "spy_tier": None,
                    "excess_tier": None, "units": "signed score"},
            expected=1.0,
            compute=lambda: (
                _is_undefined(dd.compose_regime_score()["regime_score"])
                if dd.compose_regime_score()["regime_score_categorical"] == "unknown"
                else 0.0
            ),
            tolerance=0.0,
        ),
    ]

    # ── convention pinning ──────────────────────────────────────────────────
    cases += [
        Case(
            name="regime_zscore_ddof_convention",
            description=(
                "CONVENTION. regime/composite.py::_zscore uses POPULATION std "
                "(ddof=0). Source of truth: regime/composite.py, "
                "`sigma = float(history.std(ddof=0))`. Asserted as the DIFFERENCE "
                "from the sample-sd (ddof=1) answer, which must be NON-zero — so "
                "this case fails if the convention ever silently flips. "
                "Expected difference = z_pop - z_sample"
            ),
            inputs={"ddof_in_use": 0, "alternative_ddof": 1,
                    "source_of_truth": "regime/composite.py::_zscore",
                    "note": "training/deflated_sharpe.py uses ddof=1 on its own "
                            "IC series. Two std conventions coexist in this repo "
                            "and both are defensible — cross-sectional macro "
                            "z-scoring vs a time-series sample statistic — so "
                            "this is recorded, not filed. What matters is that "
                            "each is PINNED, so neither can drift into the "
                            "other silently",
                    "units": "z-score (difference)"},
            expected=_expected_zscore() - _expected_zscore_sample_ddof(),
            compute=lambda: _regime_zscore(_Z_VALUE) - _expected_zscore_sample_ddof(),
        ),
        Case(
            name="intensity_z_tanh_scale_convention",
            description=(
                "CONVENTION. INTENSITY_Z_TANH_SCALE = 2.0 "
                "(regime/drawdown.py). It sets how much of [-1,1] a 1-sigma macro "
                "move spans, so every regime-conditional veto threshold moves with "
                "it. Read from the module, not restated"
            ),
            inputs={"constant": "regime.drawdown.INTENSITY_Z_TANH_SCALE",
                    "units": "z-score divisor"},
            expected=2.0,
            compute=lambda: float(dd.INTENSITY_Z_TANH_SCALE),
            tolerance=0.0,
        ),
        Case(
            name="regime_neutral_band_convention",
            description=(
                "CONVENTION. REGIME_SCORE_NEUTRAL_BAND = 0.20 — the symmetric "
                "|score| threshold below which the signed score projects to "
                "'neutral'. Widening it silently converts bull and bear calls into "
                "neutral ones across the whole history"
            ),
            inputs={"constant": "regime.drawdown.REGIME_SCORE_NEUTRAL_BAND",
                    "units": "signed score threshold"},
            expected=0.20,
            compute=lambda: float(dd.REGIME_SCORE_NEUTRAL_BAND),
            tolerance=0.0,
        ),
        Case(
            name="hmm_staleness_constants_convention",
            description=(
                "CONVENTION. HMM staleness ramp: grace 8 weeks, half-life 12.0 "
                "weeks, floor 0.25 (regime/drawdown.py). Encoded as "
                "grace + halflife + 100*floor = 8 + 12 + 25 = 45 so a change to "
                "ANY of the three moves this number"
            ),
            inputs={"grace_weeks": 8, "halflife_weeks": 12.0, "floor": 0.25,
                    "encoding": "grace + halflife + 100*floor",
                    "units": "composite constant"},
            expected=45.0,
            compute=lambda: (float(dd.HMM_STALENESS_GRACE_WEEKS)
                             + float(dd.HMM_STALENESS_HALFLIFE_WEEKS)
                             + 100.0 * float(dd.HMM_STALENESS_FLOOR)),
            tolerance=0.0,
        ),
        Case(
            name="regime_score_weights_sum_convention",
            description=(
                "CONVENTION. REGIME_SCORE_WEIGHTS must sum to 1.0 "
                "(intensity_z 0.40 + hmm 0.35 + drawdown_spy 0.15 + "
                "drawdown_excess 0.10). They are renormalized over PRESENT heads "
                "at runtime, so a set that no longer sums to 1 would still produce "
                "plausible scores — silently reweighted"
            ),
            inputs={"weights": {"intensity_z": 0.40, "hmm": 0.35,
                                "drawdown_spy": 0.15, "drawdown_excess": 0.10},
                    "units": "weight sum"},
            expected=1.0,
            compute=lambda: float(sum(dd.REGIME_SCORE_WEIGHTS.values())),
        ),
    ]

    return cases


# ── compute helpers used by the battery ─────────────────────────────────────

def _zero_variance_zscore() -> Any:
    import pandas as pd

    from regime.composite import _zscore

    return _zscore(8.0, pd.Series([4.0] * 6))


def _rank_permutation_delta() -> float:
    alphas = [0.01, 0.05, 0.03]
    forward = _combined_ranks(alphas)
    reverse = list(reversed(_combined_ranks(list(reversed(alphas)))))
    return float(sum(abs(a - b) for a, b in zip(forward, reverse)))


def _calibrator_monotonicity_violations() -> float:
    step = (2.0 * _CAL_CLIP) / 20.0
    alphas = [-_CAL_CLIP + i * step for i in range(21)]
    p_ups = [float(_calibrator_prediction(a)["p_up"]) for a in alphas]
    return float(sum(1 for a, b in zip(p_ups, p_ups[1:]) if b < a - 1e-12))


# ════════════════════════════════════════════════════════════════════════════
# Provenance header — the reason this is an INSTRUMENT check, not a code check
# ════════════════════════════════════════════════════════════════════════════

def resolved_library_versions(
    distributions: tuple[str, ...] = _TRACKED_DISTRIBUTIONS,
) -> dict[str, str]:
    """The installed version of every numeric distribution loaded at runtime.

    ``importlib.metadata.version`` reads the DISTRIBUTION metadata pip resolved
    into this image — the thing that moves between the CI runner and the deployed
    Lambda. A missing distribution is recorded explicitly, never omitted: an
    absent key and a missing library must not look the same.
    """
    from importlib.metadata import PackageNotFoundError, version

    resolved: dict[str, str] = {}
    for dist in distributions:
        try:
            resolved[dist] = version(dist)
        except PackageNotFoundError:
            resolved[dist] = "<not installed>"
        except Exception as exc:  # noqa: BLE001 — a version probe never blocks
            resolved[dist] = f"<unavailable: {type(exc).__name__}>"
    return resolved


def code_sha() -> str:
    """The SHA of the code that ran, without shelling out.

    Deploy-time stamps first (``GIT_SHA`` env, then ``/var/task/GIT_SHA.txt`` —
    the fleet's Lambda-image convention), then the checkout's own ``.git`` refs
    for a local run. ``unknown`` is a legitimate answer and is recorded as one — a
    fabricated SHA is worse than an absent one.
    """
    for env_key in ("GIT_SHA", "CODE_SHA", "GITHUB_SHA"):
        stamped = os.environ.get(env_key)
        if stamped:
            return stamped.strip()
    try:
        lambda_stamp = Path("/var/task/GIT_SHA.txt")
        if lambda_stamp.is_file():
            stamped = lambda_stamp.read_text().strip()
            if stamped:
                return stamped
    except Exception:  # noqa: BLE001 — provenance never blocks the battery
        pass
    try:
        git_dir = Path(__file__).resolve().parents[1] / ".git"
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:].strip()
        ref_path = git_dir / ref
        if ref_path.is_file():
            return ref_path.read_text().strip()
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0].strip()
        return "unknown"
    except Exception:  # noqa: BLE001 — provenance never blocks the battery
        return "unknown"


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def _call_with_timeout(fn: Callable[[], float], seconds: float) -> float:
    """Run ``fn`` under a wall-clock budget.

    A SIGALRM budget is installed when this is the main thread of a platform that
    has one (a Lambda handler thread qualifies). Where it is not available the
    elapsed time is checked after the call instead — that cannot interrupt a hang,
    but it does catch an overrun, and the caller distinguishes neither: both FAIL.
    """
    can_interrupt = (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not can_interrupt:
        started = time.monotonic()
        value = fn()
        elapsed = time.monotonic() - started
        if elapsed > seconds:
            raise _CaseTimeout(
                f"case exceeded its {seconds:g}s budget "
                f"({elapsed:.1f}s, detected after the fact)"
            )
        return value

    def _fire(_signum, _frame):
        raise _CaseTimeout(f"case exceeded its {seconds:g}s budget")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


#: The battery scope this module can reach on its own. ``training/`` is NOT in
#: the Lambda image (``tests/test_dockerfile_import_closure.py``), so the
#: training-path cases live in ``training/self_test_cases.py`` and are composed
#: in by a caller that can reach them — CI and the spot trainer.
SCOPE_INFERENCE = "inference"


def run_self_test(
    run_date: str | None = None,
    *,
    case_provider: "Callable[[], list[Case]] | None" = None,
    extra_case_providers: "dict[str, Callable[[], list[Case]]] | None" = None,
    component: str = COMPONENT,
    schema: str = SCHEMA,
    case_timeout_seconds: float = CASE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the known-answer battery on the DEPLOYED instrument and return the body.

    ``extra_case_providers`` maps a SCOPE NAME to an additional case builder, so a
    caller that can reach ``training/`` composes the full battery while the Lambda
    runs the inference scope alone. The scopes that actually ran are recorded in
    the artifact's ``scope`` field — an inference-scope PASS must never be
    readable as full coverage, which is the "a number published without its
    members" failure.

    Never raises — see the module docstring's CONTRACT section.
    """
    started = time.monotonic()
    scopes = [SCOPE_INFERENCE] + sorted(extra_case_providers or {})
    header = {
        "schema": schema,
        "component": component,
        "run_date": run_date,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_sha": code_sha(),
        "libraries": resolved_library_versions(),
        "case_timeout_seconds": case_timeout_seconds,
        "scope": scopes,
        "scope_note": (
            "The scopes whose cases ran. `inference` covers regime scoring, "
            "calibration, veto thresholds and ranking (the Lambda's closure); "
            "`training` covers IC, deflated Sharpe and downside statistics and "
            "runs where training runs (EC2 spot + CI), because `training/` is "
            "deliberately absent from the Lambda image. A verdict is a statement "
            "about the scopes listed here and about nothing else."
        ),
    }

    def _provider() -> list[Case]:
        collected = list((case_provider or build_cases)())
        for _scope, extra in sorted((extra_case_providers or {}).items()):
            collected.extend(extra())
        return collected

    provider = _provider
    try:
        # Materialised inside the try: a provider returning a lazy or broken
        # iterable would otherwise raise at the FOR loop below, outside every
        # handler, and take down the stage this must never be able to fail.
        cases = list(provider())
    except Exception as exc:  # noqa: BLE001 — see CONTRACT: this becomes UNKNOWN
        logger.error(
            "self-test: the battery could not be constructed (%s: %s) — verdict "
            "UNKNOWN. No correctness guarantee is granted this cycle.",
            type(exc).__name__, exc, exc_info=True,
        )
        return {**header, "status": "error", "verdict": UNKNOWN, "cases": [],
                "n_cases": 0, "n_failed": 0, "n_errored": 0, "n_known_gaps": 0,
                "error_class": type(exc).__name__, "error_msg": str(exc)[:500],
                "wall_clock_seconds": round(time.monotonic() - started, 3)}

    records: list[dict] = []
    for index, case in enumerate(cases):
        try:
            record: dict[str, Any] = {
                "case": case.name,
                "description": case.description,
                "inputs": case.inputs,
                "expected": case.expected,
                "actual": None,
                "abs_error": None,
                "tolerance": case.tolerance,
                "verdict": UNKNOWN,
            }
        except Exception as exc:  # noqa: BLE001 — see CONTRACT: never raises
            # A provider that returned a well-formed LIST of malformed items
            # gets past the materialisation guard above and would otherwise
            # raise here, outside every handler — taking down the stage this
            # module must never be able to fail. Recorded as UNKNOWN (the
            # battery could not ask the question), never as a pass.
            logger.error(
                "self-test: case %d is not a Case (%s: %s) => UNKNOWN",
                index, type(exc).__name__, exc,
            )
            records.append({
                "case": f"<malformed case {index}>",
                "description": "the case provider returned a non-Case object",
                "inputs": {"units": "n/a"},
                "expected": None, "actual": None, "abs_error": None,
                "tolerance": None, "verdict": UNKNOWN, "errored": True,
                "error_class": type(exc).__name__, "error_msg": str(exc)[:500],
                "wall_clock_seconds": 0.0,
            })
            continue
        if case.known_gap:
            # Stated in the artifact, in words, so a reader never reads this
            # row's PASS as an endorsement of the behaviour it pins.
            record["known_gap"] = True
            record["gap_issue"] = case.gap_issue
            record["known_gap_note"] = (
                "This case PINS behaviour believed incorrect at its MEASURED "
                "value so further drift goes red. PASS means 'unchanged since "
                "recorded', NOT 'this is correct'. Tracked in "
                f"{case.gap_issue}."
            )
        case_started = time.monotonic()
        try:
            actual = float(_call_with_timeout(case.compute, case_timeout_seconds))
            error = abs(actual - case.expected)
            record["actual"] = actual
            record["abs_error"] = error
            record["verdict"] = PASS if error <= case.tolerance else FAIL
            if record["verdict"] == FAIL:
                logger.error(
                    "self-test case FAILED: %s expected=%r actual=%r abs_error=%r "
                    "tolerance=%r", case.name, case.expected, actual, error,
                    case.tolerance,
                )
        except _CaseTimeout as exc:
            # Brian ruling 2026-08-13: a timeout is FAIL, never UNKNOWN. A
            # known-answer case over a handful of in-memory rows that cannot
            # finish in its budget is itself evidence something is wrong with the
            # instrument — it must not buy the benefit of the doubt that "the
            # battery could not be constructed" gets.
            record["verdict"] = FAIL
            record["timed_out"] = True
            record["error_class"] = type(exc).__name__
            record["error_msg"] = str(exc)[:500]
            logger.error("self-test case TIMED OUT (=> FAIL): %s (%s)", case.name, exc)
        except Exception as exc:  # noqa: BLE001 — a case that could not RUN is UNKNOWN
            record["verdict"] = UNKNOWN
            record["errored"] = True
            record["error_class"] = type(exc).__name__
            record["error_msg"] = str(exc)[:500]
            logger.error(
                "self-test case ERRORED (=> UNKNOWN): %s (%s: %s)",
                case.name, type(exc).__name__, exc, exc_info=True,
            )
        record["wall_clock_seconds"] = round(time.monotonic() - case_started, 3)
        records.append(record)

    n_failed = sum(1 for r in records if r["verdict"] == FAIL)
    n_errored = sum(1 for r in records if r["verdict"] == UNKNOWN)
    n_known_gaps = sum(1 for r in records if r.get("known_gap"))
    if n_failed:
        verdict = FAIL
    elif n_errored or not records:
        verdict = UNKNOWN
    else:
        verdict = PASS

    body = {**header, "status": "ok", "verdict": verdict, "cases": records,
            "n_cases": len(records), "n_failed": n_failed, "n_errored": n_errored,
            "n_known_gaps": n_known_gaps,
            "wall_clock_seconds": round(time.monotonic() - started, 3)}

    if verdict == PASS:
        logger.info(
            "self-test PASS — %d/%d known-answer cases agreed on %s (%s); "
            "%d of them PIN a known gap rather than endorse it",
            len(records), len(records), header["python"],
            ", ".join(f"{k} {v}" for k, v in header["libraries"].items()),
            n_known_gaps,
        )
    elif verdict == UNKNOWN:
        logger.error(
            "self-test UNKNOWN — %d/%d cases could not run. The correctness "
            "guarantee is WITHHELD this cycle (never granted by default).",
            n_errored, len(records),
        )
    else:
        logger.error(
            "self-test FAIL — %d/%d known-answer cases DISAGREE with their "
            "hand-derived expectation on the DEPLOYED libraries (%s). THIS "
            "CYCLE'S PREDICTIONS ARE NOT TRUSTWORTHY.",
            n_failed, len(records),
            ", ".join(f"{k} {v}" for k, v in header["libraries"].items()),
        )
    return body


def verdict_is_pass(verdict: str | None) -> bool:
    """True only for an explicit PASS — ``None`` and ``"ok"`` withhold the guarantee."""
    return verdict == PASS


def self_test_key(run_date: str) -> str:
    """S3 key of the predictor's published self-test for ``run_date``."""
    return _KEY_TEMPLATE.format(run_date=run_date)


def write_self_test(bucket: str, run_date: str, body: dict, s3_client=None) -> str:
    """Persist the artifact. Returns the key written.

    Raises on failure: this artifact IS the evidence the stage graded its own
    arithmetic, so a silent write failure would reproduce exactly the absence the
    self-test exists to remove. The caller isolates it so the predictions — the
    primary deliverable — survive regardless.
    """
    import json

    import boto3

    client = s3_client or boto3.client("s3")
    key = self_test_key(run_date)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(body, indent=2, default=str).encode(),
        ContentType="application/json",
    )
    return key


# ════════════════════════════════════════════════════════════════════════════
# Console surface — `principles.md` §2.7: a check that reports nowhere is unobserved
# ════════════════════════════════════════════════════════════════════════════

#: The fleet check-result envelope contract, as published by
#: ``nousergon_lib.fleet_check_result`` and read by the console's
#: ``checks-envelope`` adapter
#: (``nous-ergon-ops/nousergon-console/config.d/fleet-checks.yaml``).
#:
#: WHY THIS IS BUILT HERE RATHER THAN IMPORTED FROM THE LIB
#: --------------------------------------------------------
#: The lib is the right home and ``fleet_check_result`` is already there — but it
#: first ships in **nousergon-lib v0.124.29**, and this repo pins an older commit
#: (``requirements.txt``). Bumping the pin to reach it would pull 26+ lib versions
#: into the predictor's production image inside a PR whose entire purpose is to
#: VERIFY that image's arithmetic — and a lib bump can move the very numbers this
#: battery measures. That makes the bump self-defeating here, not merely large.
#:
#: So this is the same interim duplication the lib's own module docstring records
#: for ``alpha-engine-config`` ("still carries its own copy for now … staged
#: deliberately rather than silently, per ``principles.md`` §2.4, and the interim
#: duplication is held legitimate by a contract test asserting the two modules
#: still agree on the envelope they produce" — ``shared-code-policy.md`` §5.1's
#: fork-detection backstop). ``tests/test_self_test.py`` carries that contract
#: test; migration to the lib import is tracked in **alpha-engine-config-I7274**.
_ENVELOPE_SCHEMA_VERSION = 1

STATUS_OK = "ok"
STATUS_ATTENTION = "attention"
STATUS_ERROR = "error"

#: The bucket the console's checks-envelope adapter reads.
RESEARCH_BUCKET = "alpha-engine-research"
CHECKS_PREFIX = "ops/checks/"


def console_envelope_key(check_id: str = CHECK_ID) -> str:
    return f"{CHECKS_PREFIX}{check_id}/latest.json"


def console_envelope(body: dict, now=None) -> dict:
    """The fleet check-result envelope for this self-test.

    The status mapping is deliberately NOT an identity of the internal verdict —
    the envelope vocabulary is a published contract with the console adapter:

    * ``PASS``    -> ``ok``
    * ``FAIL``    -> ``error``    (evidence the numbers are wrong)
    * ``UNKNOWN`` -> ``attention`` (could not measure — `principles.md` §2.7
      forbids rendering this green, and it is not a defect either)

    ``ran_at`` + ``cadence_minutes`` are what let the console mark this check
    STALE when it stops publishing, whatever status it last wrote — the last
    thing a dying check writes is almost always "ok".
    """
    from datetime import datetime, timezone

    verdict = body.get("verdict")
    status = {PASS: STATUS_OK, FAIL: STATUS_ERROR}.get(verdict, STATUS_ATTENTION)

    n_cases = body.get("n_cases", 0)
    n_failed = body.get("n_failed", 0)
    n_errored = body.get("n_errored", 0)
    n_gaps = body.get("n_known_gaps", 0)

    findings = [
        {"key": c["case"], "detail": (
            f"expected={c.get('expected')!r} actual={c.get('actual')!r} "
            f"abs_error={c.get('abs_error')!r} tolerance={c.get('tolerance')!r}"
            + (f" [{c.get('error_class')}: {c.get('error_msg')}]"
               if c.get("error_class") else "")
        )}
        for c in body.get("cases", [])
        if c.get("verdict") != PASS
    ]

    return {
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "check_id": CHECK_ID,
        "label": "Predictor numeric self-test (M slot)",
        "ran_at": (now or datetime.now(timezone.utc)).isoformat(),
        "status": status,
        "summary": (
            f"predictor known-answer battery: {n_cases - n_failed - n_errored} "
            f"passed, {n_failed} failed, {n_errored} could not run "
            f"({n_gaps} of the passing cases PIN a known gap at its measured "
            f"value rather than endorse it). Scope: CORRECTNESS of the M slot's "
            f"arithmetic on the deployed libraries — it says nothing about "
            f"whether the stage ran or its inputs were fresh, which is the "
            f"preflight sweep's axis (sf-pipeline-policy §2.3a)."
        ),
        "cadence_minutes": CADENCE_MINUTES,
        "deep_link": None,
        "findings": findings,
        # Beyond the base contract, carried so the console row is diagnosable
        # without opening the full artifact.
        "verdict": verdict,
        "n_cases": n_cases,
        "n_failed": n_failed,
        "n_errored": n_errored,
        "n_known_gaps": n_gaps,
        "code_sha": body.get("code_sha"),
        "libraries": body.get("libraries"),
    }


def publish_console_row(body: dict, *, dry_run: bool = False, s3_client=None) -> str | None:
    """Publish the console row. Returns the s3:// URI, or None if nothing written.

    NEVER raises. A check must not go red because its telemetry did, and this
    module must not be able to fail the pipeline at all. A missing envelope
    renders on the console as ``unreadable``, never ``ok``, so a failed publish
    degrades to a visible gap rather than a false all-clear.
    """
    import json

    envelope = console_envelope(body)
    key = console_envelope_key()
    uri = f"s3://{RESEARCH_BUCKET}/{key}"
    if dry_run:
        logger.info("[dry-run] would publish %s (%s)", uri, envelope["status"])
        return None
    try:
        import boto3

        client = s3_client or boto3.client("s3")
        client.put_object(
            Bucket=RESEARCH_BUCKET, Key=key,
            Body=json.dumps(envelope, indent=2, default=str).encode(),
            ContentType="application/json",
        )
    except Exception:  # noqa: BLE001 — a failed publish must never fail the check
        logger.warning(
            "could not publish the self-test console row to %s — the console will "
            "render this check as `unreadable` (never `ok`), so the gap is visible",
            uri, exc_info=True,
        )
        return None
    return uri
