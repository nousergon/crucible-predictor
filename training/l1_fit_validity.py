"""training/l1_fit_validity.py — post-fit validity of every L1 ARM (I9271).

Layer 3 of the ruling of 2026-08-29 (Brian, verbatim):

    "if any of the arms is not trained properly then the predictor module
    should fail the task"

Layer 1 (``training/data_completeness.py``) refuses to START training when an
input is short, frozen or absent. Layer 2 (``training/arm_validity.py``)
refuses to FINISH when the fitted **L2** is degenerate. Neither can see the
case this module exists for: every input arrived, the L2 fitted cleanly, and
an **L1** early-stopped after two boosting rounds into a near-constant
prediction that the L2 then correctly — and invisibly — down-weighted to
nothing.

## The week that forced it (alpha-engine-config-I9271)

``models.research_gbm`` is the canonical producer of
``research_calibrator_prob``, the meta-Ridge's largest cross-sectional
coefficient. Its ``best_iteration`` across three consecutive weekly retrains
went **3 -> 500 -> 2** and its ``val_ic`` **0.0430 -> 0.2137 -> 0.0140**.

Replayed on the identical 2026-08-28 inference inputs (903 tickers, the
snapshot's own ``score``/``conviction``/``sector_modifiers``), the three
shipped boosters produce:

===============  ===========  ===============  ==================
vintage          best_iter    std(prediction)  distinct values
===============  ===========  ===============  ==================
2026-08-14                 3         0.011149                   4
2026-08-21               500         0.107021                 136
2026-08-28                 2         0.003731                   7
===============  ===========  ===============  ==================

A **28.7x** collapse in the served feature's cross-sectional spread between
08-21 and 08-28, which is what carried the served-alpha dispersion down with
it. The 2026-08-28 manifest recorded ``overfit_warn: true`` and
``train_val_ic_ratio: 8.987169`` beside ``best_iteration: 2`` — and **nothing
in the repository read either field**. That detection failure, not the fit
itself, is what this module closes.

## Why the floors are where they are

Each floor is set so that it **independently** blocks 2026-08-28 and passes
2026-08-21 — the replay the issue's ``Closes-when`` requires — with the
separating measurement named:

``min_best_iteration = 10``
    Measured: 3 / 500 / 2. At ``learning_rate=0.05`` and ``num_leaves=8`` a
    two-round booster cannot move a prediction more than a few percent off the
    base rate; the replay above shows it did not (7 distinct values over 903
    names). 10 sits an order of magnitude below the healthy vintage and five
    times above the failing ones, so it is not fitted to either.

``min_abs_val_ic = 0.05``
    Measured: 0.0430 / 0.2137 / 0.0140. Absolute value, because a strongly
    NEGATIVE validation IC is a usable (invertible) learner while a
    near-zero one is noise, and it is the near-zero case that ships a
    constant.

``max_train_val_ic_ratio = 5.0``
    Measured: 6.19 / 1.93 / 8.99. This is the number ``_compute_overfit_signal``
    has been computing since 2026-05-19 and emitting to CloudWatch and the
    manifest with **no consumer**. The advisory warn level stays at 3.0; this
    is where it becomes load-bearing, set above the warn level deliberately so
    the gate bites strictly later than the alert.

**2026-08-14 fails these floors too, and that is the correct verdict**, not
an over-tight threshold: its replayed dispersion (0.011149, four distinct
values over 903 names) is nearer the collapsed vintage than the healthy one.
Two of the last three weekly vintages shipped a degenerate canonical L1.

## Two rules this module obeys

1. **A check that cannot be computed is reported ``insufficient`` and does not
   block** (``champion-challenger-policy.md`` §5.1). But an arm that is
   *declared* and simply **absent** is a FAILURE, not an insufficiency — a
   missing canonical producer means something silently substituted for it,
   which is the bug class the gate exists to catch.
2. **Nothing here is scale-invariant.** ``output_dispersion`` is recorded as a
   raw standard deviation on the arm's own output scale. The M-slot behavioural
   veto is measured to be the only serving guard that works precisely because
   it is scale-DEPENDENT (``champion-challenger-policy.md`` §5.3); a
   standardized version of this number would divide the collapse away.

## Why failing training is safe

Training writes a per-run staging prefix (``TrainingIOSpec``,
alpha-engine-config-I9018) and ``model.registry.promote_to_champion`` is the
only writer of the live serving prefix — asserted by
``tests/test_live_prefix_single_writer.py``. A raise here cannot strand or
blank the serving weights: the existing champion keeps serving preopen
inference and the week produces no new candidate.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

log = logging.getLogger(__name__)

__all__ = [
    "L1FitValidityError",
    "L1FitSpec",
    "L1_FIT_REGISTER",
    "measure_output_dispersion",
    "evaluate_l1_fits",
    "assert_l1_fits_valid",
]


class L1FitValidityError(RuntimeError):
    """A declared L1 arm did not train properly. The run fails.

    Deliberately NOT a subclass of ``Exception``-catching convenience: every
    call site that wraps an L1 fit in a broad ``except Exception`` must
    re-raise this type, or the gate is decorative.
    """


# Headline-status precedence, most-actionable first. ``absent`` outranks the
# fit qualities because a missing arm means a substitution already happened;
# ``underfit_early_stop`` outranks ``no_val_signal`` because the iteration
# count names the mechanism while the IC only names the symptom.
_STATUS_PRECEDENCE = (
    "absent",
    "not_fitted",
    "underfit_early_stop",
    "no_val_signal",
    "train_val_gap",
    "degenerate_output",
    "unmeasured",
)


@dataclass(frozen=True)
class L1FitSpec:
    """One declared L1 arm and the bar its fit must clear.

    Attributes
    ----------
    name : str
        Manifest key under ``models`` — the identity a finding names.
    severity : {"required", "optional"}
        ``required`` — a failure raises ``L1FitValidityError`` and no candidate
        is produced. ``optional`` — a failure is a NAMED degradation on the
        manifest.
    canonical_for : str or None
        The meta-feature this arm produces, so a finding says what went dark
        downstream without the reader tracing the graph.
    may_be_absent : bool
        ``True`` only for arms whose absence is a declared configuration
        state. ``False`` means absence is itself a failure.
    min_best_iteration : int
        Floor on the early-stopped iteration count.
    min_abs_val_ic : float
        Floor on ``abs(val_ic)``.
    max_train_val_ic_ratio : float or None
        Ceiling on ``train_ic / abs(val_ic)``. ``None`` where the arm does not
        compute a train-split IC.
    min_output_dispersion : float or None
        Floor on the raw standard deviation of the arm's predictions over the
        panel it was fitted on. ``None`` records the number without gating on
        it — used where no cross-vintage series exists yet to site a floor
        honestly.
    """

    name: str
    severity: str = "required"
    canonical_for: "str | None" = None
    may_be_absent: bool = False
    min_best_iteration: int = 10
    min_abs_val_ic: float = 0.05
    max_train_val_ic_ratio: "float | None" = 5.0
    min_output_dispersion: "float | None" = None


# The single list of L1 arms whose fit is gated. An L1 that early-stops and is
# NOT in this register is ungated — which is exactly how `research_gbm` shipped
# a two-round booster for a full weekly cycle. A new L1 gets a row here in the
# same PR that fits it.
L1_FIT_REGISTER: tuple[L1FitSpec, ...] = (
    L1FitSpec(
        name="research_gbm",
        severity="required",
        canonical_for="research_calibrator_prob",
        # Absence means the bucket-lookup ResearchCalibrator silently supplies
        # the canonical feature instead. That substitution is the reason this
        # module exists, so it is a failure, not a fallback.
        may_be_absent=False,
        min_best_iteration=10,
        min_abs_val_ic=0.05,
        max_train_val_ic_ratio=5.0,
        # Recorded, not gated: three vintages is not enough history to site a
        # dispersion floor that is not simply fitted to them.
        min_output_dispersion=None,
    ),
    L1FitSpec(
        name="volatility",
        severity="required",
        canonical_for="volatility_score",
        may_be_absent=False,
        min_best_iteration=10,
        min_abs_val_ic=0.05,
        # The plain volatility GBM computes no train-split IC.
        max_train_val_ic_ratio=None,
    ),
    L1FitSpec(
        name="volatility_macro_aug",
        severity="required",
        canonical_for=None,
        # Config-gated parallel observe-only variant.
        may_be_absent=True,
        min_best_iteration=10,
        min_abs_val_ic=0.05,
        max_train_val_ic_ratio=None,
    ),
    L1FitSpec(
        name="volatility_risk_aug",
        severity="required",
        canonical_for=None,
        may_be_absent=True,
        min_best_iteration=10,
        min_abs_val_ic=0.05,
        max_train_val_ic_ratio=None,
    ),
)


def measure_output_dispersion(preds) -> "float | None":
    """Raw standard deviation of an L1's predictions. Never standardized.

    ``None`` when fewer than two finite predictions exist — reported, never
    defaulted to a number that would read as a measurement.
    """
    import numpy as np

    arr = np.asarray(preds, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return None
    return float(np.std(finite))


@dataclass
class _Finding:
    arm: str
    status: str
    severity: str
    reason: str

    def as_dict(self) -> dict:
        return {"arm": self.arm, "status": self.status,
                "severity": self.severity, "reason": self.reason}


def _finite(value) -> "float | None":
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def evaluate_l1_fits(fits: "dict | None",
                     register: "tuple[L1FitSpec, ...] | None" = None) -> dict:
    """Grade every declared L1 fit. Pure — no S3, no config, unit-testable.

    Parameters
    ----------
    fits : dict
        ``{arm_name: {"fitted": bool, "best_iteration": int, "val_ic": float,
        "train_ic": float | None, "n_estimators": int | None,
        "output_dispersion": float | None, "n_samples": int | None}}``.
        An arm absent from this mapping is treated as absent from the run.
    """
    register = register or L1_FIT_REGISTER
    fits = fits or {}
    arms: dict = {}
    findings: list[_Finding] = []
    notes: list[dict] = []

    for spec in register:
        fit = fits.get(spec.name)
        best_it = _finite((fit or {}).get("best_iteration"))
        val_ic = _finite((fit or {}).get("val_ic"))
        train_ic = _finite((fit or {}).get("train_ic"))
        n_est = _finite((fit or {}).get("n_estimators"))
        dispersion = _finite((fit or {}).get("output_dispersion"))
        ratio = None
        if train_ic is not None and val_ic is not None and abs(val_ic) > 1e-12:
            ratio = abs(train_ic) / abs(val_ic)

        issues: list[tuple[str, str]] = []

        if fit is None:
            if spec.may_be_absent:
                arms[spec.name] = {
                    "status": "not_evaluated", "severity": spec.severity,
                    "reason": (
                        f"{spec.name} was not fitted this run and its absence is "
                        "a declared configuration state; no assertion applies."
                    ),
                    "best_iteration": None, "val_ic": None, "train_ic": None,
                    "train_val_ic_ratio": None, "output_dispersion": None,
                    "n_samples": None, "issues": [],
                }
                continue
            issues.append(("absent", (
                f"{spec.name} produced no fit this run. It is the canonical "
                f"producer of {spec.canonical_for or 'a declared L1 score'}, so "
                "its absence means a fallback silently supplied that value — "
                "measured: no fit record; required: a fitted arm."
            )))
        elif not fit.get("fitted"):
            issues.append(("not_fitted", (
                f"{spec.name} recorded fitted=False; measured: no booster, "
                "required: a fitted arm."
            )))
        else:
            if best_it is None:
                issues.append(("unmeasured", (
                    f"{spec.name} reported no best_iteration, so its early stop "
                    "is unverifiable. Reported as a failure rather than a pass: "
                    "an unmeasured gate that reads as green is the defect the "
                    "gate exists to prevent (champion-challenger §5.1)."
                )))
            elif best_it < spec.min_best_iteration:
                issues.append(("underfit_early_stop", (
                    f"{spec.name}: underfit_early_stop — early stopping halted "
                    f"the booster at iteration {int(best_it)}; measured "
                    f"{int(best_it)}, required >= {spec.min_best_iteration}. A "
                    "booster this short emits a near-constant prediction, and "
                    f"{spec.canonical_for or 'its consumer'} loses its "
                    "cross-sectional spread (I9271: best_iteration 3 -> 500 -> 2 "
                    "over three weekly vintages carried the served alpha down "
                    "with it)."
                )))

            if val_ic is None:
                issues.append(("unmeasured", (
                    f"{spec.name} reported no val_ic, so its holdout signal is "
                    "unverifiable. Reported as a failure, not a pass."
                )))
            elif abs(val_ic) < spec.min_abs_val_ic:
                issues.append(("no_val_signal", (
                    f"{spec.name}: no_val_signal — holdout IC {val_ic:.6f}; "
                    f"measured |{val_ic:.6f}|, required >= "
                    f"{spec.min_abs_val_ic}. The arm carries no usable signal "
                    "on data it did not fit."
                )))

            if spec.max_train_val_ic_ratio is not None and ratio is not None \
                    and ratio > spec.max_train_val_ic_ratio:
                issues.append(("train_val_gap", (
                    f"{spec.name}: train_val_gap — train_ic {train_ic:.6f} "
                    f"against val_ic {val_ic:.6f}; measured ratio {ratio:.4f}, "
                    f"required <= {spec.max_train_val_ic_ratio}. This is the "
                    "number `_compute_overfit_signal` has emitted to the "
                    "manifest and to CloudWatch since 2026-05-19 with no "
                    "consumer; it is load-bearing here."
                )))

            if spec.min_output_dispersion is not None and dispersion is not None \
                    and dispersion < spec.min_output_dispersion:
                issues.append(("degenerate_output", (
                    f"{spec.name}: degenerate_output — prediction std "
                    f"{dispersion:.6f} over the fitted panel; measured "
                    f"{dispersion:.6f}, required >= "
                    f"{spec.min_output_dispersion}. Raw, never standardized: a "
                    "scale-invariant form divides the collapse away "
                    "(champion-challenger §5.3)."
                )))

            # Recorded, never gating: early stopping that never fired is not a
            # failure, but it means `val_ic` was still improving at the last
            # round and is therefore an optimistic estimate rather than a
            # converged one. 2026-08-21 hit 500/500 — the only vintage of the
            # three that cleared every floor, and it never early-stopped at all.
            if best_it is not None and n_est is not None and best_it >= n_est:
                notes.append({
                    "arm": spec.name,
                    "note": "early_stopping_never_fired",
                    "reason": (
                        f"{spec.name} used all {int(n_est)} boosting rounds "
                        "without early stopping firing, so val_ic "
                        f"({val_ic if val_ic is None else round(val_ic, 6)}) is "
                        "the value at the round cap rather than at a converged "
                        "optimum."
                    ),
                })

        by_status = dict(issues)
        status, reason = "valid", ""
        for candidate in _STATUS_PRECEDENCE:
            if candidate in by_status:
                status, reason = candidate, by_status[candidate]
                break

        arms[spec.name] = {
            "status": status,
            "severity": spec.severity,
            "canonical_for": spec.canonical_for,
            "best_iteration": int(best_it) if best_it is not None else None,
            "val_ic": round(val_ic, 6) if val_ic is not None else None,
            "train_ic": round(train_ic, 6) if train_ic is not None else None,
            "train_val_ic_ratio": round(ratio, 6) if ratio is not None else None,
            "output_dispersion": (
                round(dispersion, 8) if dispersion is not None else None
            ),
            "n_samples": (fit or {}).get("n_samples"),
            "floors": {
                "min_best_iteration": spec.min_best_iteration,
                "min_abs_val_ic": spec.min_abs_val_ic,
                "max_train_val_ic_ratio": spec.max_train_val_ic_ratio,
                "min_output_dispersion": spec.min_output_dispersion,
            },
            "reason": reason or None,
            "issues": [{"status": st, "reason": rs} for st, rs in issues],
        }
        if status != "valid":
            findings.append(_Finding(spec.name, status, spec.severity, reason))

    failures = [f for f in findings if f.severity == "required"]
    degradations = [f for f in findings if f.severity != "required"]
    return {
        "status": "failed" if failures else ("degraded" if degradations else "ok"),
        "arms": arms,
        "failures": [f.as_dict() for f in failures],
        "degradations": [f.as_dict() for f in degradations],
        "notes": notes,
    }


def assert_l1_fits_valid(block: dict) -> None:
    """Raise ``L1FitValidityError`` when a REQUIRED L1 arm did not train.

    Brian's ruling of 2026-08-29: one arm that did not train properly fails the
    predictor task. The message names the arm, the assertion, and measured
    versus required.
    """
    block = block or {}
    for note in block.get("notes") or []:
        log.warning("l1_fit_validity: %s", note.get("reason"))
    for d in block.get("degradations") or []:
        log.warning(
            "l1_fit_validity: DEGRADED arm %s (%s) — %s",
            d.get("arm"), d.get("status"), d.get("reason"),
        )
    failures = block.get("failures") or []
    if not failures:
        log.info(
            "l1_fit_validity: all declared L1 arms cleared their fit floors "
            "(%d arms graded)", len(block.get("arms") or {}),
        )
        return
    detail = " | ".join(f.get("reason", "") for f in failures)
    raise L1FitValidityError(
        f"Refusing to produce a candidate: {len(failures)} L1 arm(s) FAILED the "
        f"fit-validity gate — {detail} "
        "(alpha-engine-config-I9271; Brian ruling 2026-08-29: if any arm is not "
        "trained properly the predictor task fails. The live champion is "
        "untouched — training writes a staging prefix and "
        "model.registry.promote_to_champion is the only writer of the serving "
        "prefix, so preopen inference continues on the existing champion and "
        "this week simply produces no new candidate.)"
    )
