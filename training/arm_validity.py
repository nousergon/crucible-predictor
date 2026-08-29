"""training/arm_validity.py — post-fit validity of ONE trained arm (I9290).

Layer 2 of the ruling of 2026-08-29 (Brian, verbatim):

    "it sounds like predictor should have failed this week due to the outage
    not properly training any of the models. really if any of the arms is not
    trained properly then the predictor module should fail the task"

Layer 1 (``training/data_completeness.py``) refuses to START training when an
input is short, frozen or absent. This module refuses to FINISH training when
the fitted model is degenerate — the cases an input gate cannot see, because
the inputs arrived and the fit still produced nothing.

The week that forced it: ``PredictorTraining`` returned SSM ``Status:
Success``, all four zoo specs reported OK, ``ModelZooSelect`` wrote a full
leaderboard, and ``branch_outcomes.branch_b_degraded`` was ``false`` — over a
training run in which every model was fitted with seven features hard-zeroed
by the VIX3M outage. The predictor reported unqualified success over a
structurally void run. `champion-challenger-policy.md` §7.2 names this as the
fleet's dominant bug class: **a record asserting an action that never
happened**.

## The three assertions

``constant_input_column``
    A meta-feature column with no variance anywhere in the training panel. A
    linear L2 with a fitted intercept hands it an exactly-zero coefficient, so
    the model is structurally smaller than its declared feature list claims.
    Needs no history, is therefore ALWAYS computable, and therefore ALWAYS
    gates.

``dead_feature_block``
    A whole declared block (the macro block, the regime-derived block, the L1
    scores) whose standardized coefficients are all exactly zero *while the
    arm's own prior vintage had them non-zero*. One dead feature can be a
    regime; a whole block going from live to dead between two vintages of the
    same arm is a producer failure.

``coef_norm_collapse``
    The L2 norm of the arm's standardized coefficients **restricted to the
    cross-sectionally varying features** (``XSEC_FEATURES``), against the arm's
    own trailing history. Measured across the outage:
    ``0.3142 -> 0.2701 -> 0.1214``. The macro death cost 14%; the following
    week cost 55% more. A floor at ``ARM_COEF_NORM_MIN_RATIO`` (0.50) of the
    trailing reference refuses 2026-08-28 (ratio 0.386) and passes 2026-08-21
    (0.860) — deliberately, because 08-21 is the case the BEHAVIORAL VETO
    already catches and this gate must not be tuned to duplicate it.

    **Corrected 2026-08-29 (alpha-engine-config-I9271).** Those three numbers
    are the cross-sectional norm; the check as first written took the norm over
    the WHOLE coefficient vector, which measured ``0.9676 -> 0.2701 -> 0.1214``
    and would have refused 2026-08-21 at a ratio of ``0.279`` — the opposite of
    the verdict its own justification claims, because the market-wide macro
    block dying moves the full-vector norm hard while costing the served
    within-date spread nothing. The gated quantity is now the one the floor was
    sited on.

## Two rules this module obeys

1. **A check that cannot be computed is reported ``insufficient`` and does not
   block** (`champion-challenger-policy.md` §5.1: "You cannot gate on a
   statistic you did not measure... an uncomputed gate reported as a *pass* is
   the defect it is designed to prevent"). The two history-dependent checks are
   insufficient on an arm's first vintage. The history-free one never is.
2. **The message names the arm, the assertion, the input, and measured versus
   required, on one line.** One broken challenger now halts the weekly
   pipeline; that trade is only tolerable if the cause is obvious in five
   seconds.

## Why failing training is safe

Training writes a per-run staging prefix (``TrainingIOSpec``,
alpha-engine-config-I9018) and ``model.registry.promote_to_champion`` is the
only writer of the live serving prefix — asserted by
``tests/test_live_prefix_single_writer.py``. A raise here therefore cannot
strand or blank the serving weights: the existing champion keeps serving
preopen inference and the week simply produces no new candidate, which is the
safe outcome.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

log = logging.getLogger(__name__)

__all__ = [
    "ArmValidityError",
    "FEATURE_BLOCKS",
    "XSEC_FEATURES",
    "coef_norm",
    "xsec_coef_norm",
    "constant_input_columns",
    "evaluate_arm_validity",
    "assert_arm_valid",
]


class ArmValidityError(RuntimeError):
    """A trained arm is degenerate. The run fails; no candidate is produced."""


# Declared feature blocks. A block is a set of columns that arrive together
# from one producer, so all of them going dead at once is a producer failure
# rather than the model's judgement about a feature.
FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    # The six raw market-wide macros plus the derived regime composite — the
    # exact seven columns the 2026-08-22 and 2026-08-29 vintages hard-zeroed.
    "macro": (
        "macro_spy_20d_return", "macro_spy_20d_vol", "macro_vix_level",
        "macro_vix_term_slope", "macro_yield_curve_slope", "macro_market_breadth",
    ),
    "regime_derived": ("regime_intensity_z",),
}

# Exact-zero tolerance. A BayesianRidge shrinks toward but rarely exactly to
# zero on a live feature; a constant column gets a coefficient that is zero to
# floating point. The bar is deliberately tight so this cannot fire on
# shrinkage.
_ZERO = 1e-12


def coef_norm(standardized_coef: "dict | None") -> "float | None":
    """L2 norm of an arm's standardized coefficient vector.

    ``None`` when the importance block is absent or carries no finite values —
    reported, never defaulted to a number that would read as a measurement.
    """
    if not isinstance(standardized_coef, dict) or not standardized_coef:
        return None
    vals = [
        float(v) for v in standardized_coef.values()
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    if not vals:
        return None
    return float(math.sqrt(sum(v * v for v in vals)))


# The meta-features that vary WITHIN a date. Everything else the L2 consumes
# is market-wide — one value broadcast to every ticker on the date — and by
# construction contributes exactly zero within-date prediction dispersion,
# however large its coefficient. A norm taken over the whole vector therefore
# measures something the served alpha's spread does not depend on
# (alpha-engine-config-I9271).
XSEC_FEATURES: tuple[str, ...] = (
    "research_calibrator_prob",
    "momentum_score",
    "expected_move",
    "research_composite_score",
    "research_conviction",
    "sector_macro_modifier",
)


def xsec_coef_norm(standardized_coef: "dict | None") -> "float | None":
    """L2 norm restricted to the cross-sectionally VARYING features.

    This is the quantity that governs a linear L2's within-date prediction
    spread: a feature contributes ``coef_k * std(x_k) = standardized_coef_k *
    std(y)``, and a market-wide feature has ``std(x_k) == 0`` within a date.

    Measured across the 2026-08 collapse: ``0.3142 -> 0.2701 -> 0.1214``,
    step ratios ``0.860`` then ``0.449``, against a served-slice ``alpha_stdev``
    ratio of ``0.347`` — the restricted norm tracks the served dispersion and
    the full-vector norm does not (the full norm read ``0.9676 -> 0.2701 ->
    0.1214``, a first step of ``0.279`` driven entirely by the market-wide
    macro block dying, which cost the served spread nothing).

    ``None`` when no declared cross-sectional feature carries a finite
    coefficient — reported, never defaulted to a number.
    """
    if not isinstance(standardized_coef, dict) or not standardized_coef:
        return None
    vals = [
        float(standardized_coef[k]) for k in XSEC_FEATURES
        if k in standardized_coef
        and isinstance(standardized_coef[k], (int, float))
        and not isinstance(standardized_coef[k], bool)
        and math.isfinite(float(standardized_coef[k]))
    ]
    if not vals:
        return None
    return float(math.sqrt(sum(v * v for v in vals)))


def constant_input_columns(meta_X, feature_names) -> list[str]:
    """Feature columns with no variance anywhere in the training panel.

    Computed directly off the fitted panel rather than inferred from a
    downstream diagnostic, so it is available on every run with no dependency
    on any other observability block.
    """
    import numpy as np

    X = np.asarray(meta_X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return []
    dead: list[str] = []
    for j, name in enumerate(feature_names or []):
        if j >= X.shape[1]:
            break
        col = X[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            dead.append(name)
            continue
        if float(np.std(col[finite])) <= _ZERO:
            dead.append(name)
    return dead


@dataclass
class _Check:
    name: str
    status: str          # pass | fail | insufficient
    reason: str

    def as_dict(self) -> dict:
        return {"check": self.name, "status": self.status, "reason": self.reason}


def _block_of(feature: str) -> "str | None":
    for block, cols in FEATURE_BLOCKS.items():
        if feature in cols:
            return block
    return None


def evaluate_arm_validity(
    *,
    arm: str,
    standardized_coef: "dict | None",
    meta_X=None,
    feature_names: "list | None" = None,
    prior_standardized_coef: "dict | None" = None,
    prior_coef_norms: "list | None" = None,
    min_norm_ratio: float = 0.50,
) -> dict:
    """Grade one freshly-fitted arm. Pure — no S3, no config, unit-testable.

    Parameters
    ----------
    arm : str
        The arm's identity for the message (``MODEL_VERSION_LABEL``).
    standardized_coef : dict
        ``MetaModel._importance["standardized_coef"]`` for this fit.
    meta_X, feature_names
        The fitted panel and its column names, for the history-free check.
    prior_standardized_coef : dict, optional
        The same block from the arm's previous registered vintage. Absent on a
        first vintage, or when the registry read failed — the block check is
        then ``insufficient``, never a pass.
    prior_coef_norms : list[float], optional
        Trailing CROSS-SECTIONAL coefficient norms for THIS arm, recomputed
        from each prior manifest's own ``standardized_coef``. The reference is
        their median,
        so one bad week does not move the bar it will be judged against next
        week.
    """
    checks: list[_Check] = []
    # Both are recorded; only the cross-sectional one gates. The full-vector
    # norm is retained on the block because it is the series eight prior
    # vintages carry, and dropping it would silently reset the history.
    full_norm = coef_norm(standardized_coef)
    norm = xsec_coef_norm(standardized_coef)

    # ── 1. constant input columns — history-free, always gating ─────────────
    if meta_X is None or not feature_names:
        checks.append(_Check(
            "constant_input_column", "insufficient",
            f"{arm}: the fitted panel was not supplied, so column variance "
            "could not be measured. Reported insufficient, NOT a pass "
            "(champion-challenger-policy §5.1).",
        ))
    else:
        dead_cols = constant_input_columns(meta_X, feature_names)
        if dead_cols:
            blocks = sorted({_block_of(c) or "ungrouped" for c in dead_cols})
            checks.append(_Check(
                "constant_input_column", "fail",
                f"{arm}: constant_input_column — {len(dead_cols)} of "
                f"{len(feature_names)} meta features have ZERO variance across "
                f"the whole training panel: {dead_cols} (blocks: {blocks}); "
                f"measured panel std <= {_ZERO}, required > {_ZERO}. A linear "
                "L2 gives each an exactly-zero coefficient, so this arm is "
                "structurally smaller than its declared feature list.",
            ))
        else:
            checks.append(_Check("constant_input_column", "pass", ""))

    # ── 2. a whole declared block dead that WAS live last vintage ───────────
    if not isinstance(prior_standardized_coef, dict) or not prior_standardized_coef:
        checks.append(_Check(
            "dead_feature_block", "insufficient",
            f"{arm}: no prior vintage coefficients available (first vintage, "
            "or the registry read failed), so live->dead cannot be measured. "
            "Reported insufficient, NOT a pass.",
        ))
    else:
        failed: list[str] = []
        for block, cols in FEATURE_BLOCKS.items():
            present = [c for c in cols if c in (standardized_coef or {})]
            if not present:
                continue
            now_dead = all(
                abs(float((standardized_coef or {}).get(c) or 0.0)) <= _ZERO
                for c in present
            )
            was_live = any(
                abs(float(prior_standardized_coef.get(c) or 0.0)) > _ZERO
                for c in present
            )
            if now_dead and was_live:
                failed.append(
                    f"{block} ({len(present)} cols: {present}) — every "
                    f"coefficient is 0 this vintage, non-zero last vintage"
                )
        if failed:
            checks.append(_Check(
                "dead_feature_block", "fail",
                f"{arm}: dead_feature_block — {'; '.join(failed)}. A whole "
                "producer's block going live->dead between two vintages of the "
                "same arm is an input failure, not the model's judgement; "
                "measured max|coef| <= "
                f"{_ZERO}, required > {_ZERO} for at least one column.",
            ))
        else:
            checks.append(_Check("dead_feature_block", "pass", ""))

    # ── 3. coefficient-norm collapse against the arm's OWN history ──────────
    ref_norms = [
        float(n) for n in (prior_coef_norms or [])
        if isinstance(n, (int, float)) and math.isfinite(float(n)) and float(n) > 0
    ]
    if norm is None:
        checks.append(_Check(
            "coef_norm_collapse", "insufficient",
            f"{arm}: this fit carries no finite standardized coefficient on "
            f"any of the {len(XSEC_FEATURES)} declared cross-sectional "
            "features, so its cross-sectional norm is unmeasurable. Reported "
            "insufficient, NOT a pass.",
        ))
    elif not ref_norms:
        checks.append(_Check(
            "coef_norm_collapse", "insufficient",
            f"{arm}: no trailing cross-sectional coefficient-norm history for "
            f"this arm (current xsec norm {norm:.6f}), so a collapse cannot be "
            "measured. Reported insufficient, NOT a pass.",
        ))
    else:
        ordered = sorted(ref_norms)
        reference = ordered[len(ordered) // 2]  # median of the arm's own history
        ratio = norm / reference if reference else None
        if ratio is not None and ratio < min_norm_ratio:
            checks.append(_Check(
                "coef_norm_collapse", "fail",
                f"{arm}: coef_norm_collapse — CROSS-SECTIONAL standardized "
                f"coefficient norm {norm:.6f} against this arm's trailing median "
                f"{reference:.6f} over {len(ref_norms)} vintage(s); measured "
                f"ratio {ratio:.4f}, required >= {min_norm_ratio}. The model "
                "shrank toward the intercept, which is what a starved or dead "
                "feature block does to a Ridge.",
            ))
        else:
            checks.append(_Check("coef_norm_collapse", "pass", ""))

    failures = [c for c in checks if c.status == "fail"]
    insufficient = [c for c in checks if c.status == "insufficient"]
    return {
        "arm": arm,
        "status": "invalid" if failures else ("degraded" if insufficient else "valid"),
        # The gating quantity. Kept under the historical key so eight
        # vintages of `prior_coef_norms` stay comparable with what is written
        # from here on, and named explicitly beside it.
        "coef_norm": round(norm, 6) if norm is not None else None,
        "xsec_coef_norm": round(norm, 6) if norm is not None else None,
        "full_coef_norm": round(full_norm, 6) if full_norm is not None else None,
        "checks": [c.as_dict() for c in checks],
        "failures": [c.as_dict() for c in failures],
        "insufficient": [c.as_dict() for c in insufficient],
    }


def assert_arm_valid(block: dict) -> None:
    """Raise ``ArmValidityError`` when any assertion failed.

    Brian's ruling of 2026-08-29: one arm that did not train properly fails the
    predictor task. The message names the arm, the assertion, the input, and
    measured versus required — nothing here says "training failed".
    """
    block = block or {}
    for c in block.get("insufficient") or []:
        log.warning("arm_validity: %s", c.get("reason"))
    failures = block.get("failures") or []
    if not failures:
        log.info(
            "arm_validity: %s is VALID (coef_norm=%s; %d checks)",
            block.get("arm"), block.get("coef_norm"), len(block.get("checks") or []),
        )
        return
    detail = " | ".join(f.get("reason", "") for f in failures)
    raise ArmValidityError(
        f"Refusing to produce a candidate for arm {block.get('arm')!r}: "
        f"{len(failures)} post-fit validity assertion(s) FAILED — {detail} "
        "(Brian ruling 2026-08-29: if any arm is not trained properly the "
        "predictor task fails. The live champion is untouched — training writes "
        "a staging prefix and model.registry.promote_to_champion is the only "
        "writer of the serving prefix, so preopen inference continues on the "
        "existing champion and this week simply produces no new candidate.)"
    )


def load_arm_history(bucket: str, arm: str, *, s3=None, max_vintages: int = 8) -> dict:
    """The arm's OWN prior vintages: last standardized coefs + trailing
    CROSS-SECTIONAL norms.

    Reads the immutable registry bundles for ``model_version == arm``, newest
    first. Best-effort by DESIGN and honest about it: a read failure returns
    empty history, which makes the two history-dependent checks report
    ``insufficient`` rather than pass — you cannot gate on a statistic you did
    not measure (champion-challenger-policy §5.1), and an uncomputed gate
    reported as a pass is the defect the gate exists to prevent.

    Returns ``{"prior_standardized_coef", "prior_coef_norms", "n_vintages",
    "status", "reason"}``.
    """
    import json

    out: dict = {
        "prior_standardized_coef": None,
        "prior_coef_norms": [],
        "n_vintages": 0,
        "status": "unavailable",
        "reason": None,
    }
    try:
        if s3 is None:
            import boto3
            s3 = boto3.client("s3")
        from model.registry import list_versions
        versions = [
            v for v in (list_versions(s3, bucket) or [])
            if v.get("model_version") == arm and v.get("version_id")
        ]
    except Exception as exc:  # noqa: BLE001 — see the docstring: this degrades
        # to `insufficient`, never to a pass. The reason is carried onto the
        # manifest so "we did not check" is visible, not inferred.
        out["reason"] = f"registry enumeration failed: {type(exc).__name__}: {exc}"
        log.warning(
            "arm_validity: could not enumerate prior vintages for %s — the "
            "history-dependent checks will report INSUFFICIENT, not pass. %s",
            arm, out["reason"], exc_info=True,
        )
        return out

    versions.sort(key=lambda v: str(v.get("date") or ""), reverse=True)
    norms: list[float] = []
    latest_coef: "dict | None" = None
    for v in versions[:max_vintages]:
        key = f"predictor/registry/{v['version_id']}/manifest.json"
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            manifest = json.loads(body)
        except Exception as exc:  # noqa: BLE001 — one unreadable bundle is not
            # fatal to the history; it is named and the rest still count.
            log.warning(
                "arm_validity: prior vintage %s unreadable (%s) — excluded "
                "from %s's history.", key, exc, arm,
            )
            continue
        coef = (
            ((manifest.get("models") or {}).get("meta_model") or {})
            .get("importance", {}) or {}
        ).get("standardized_coef")
        # Recomputed from each prior manifest's own `standardized_coef`, so
        # switching the gated quantity to the cross-sectional norm
        # (alpha-engine-config-I9271) applies retroactively to the whole
        # history rather than resetting it.
        n = xsec_coef_norm(coef)
        if n is not None:
            norms.append(n)
        if latest_coef is None and isinstance(coef, dict) and coef:
            latest_coef = coef

    out["prior_standardized_coef"] = latest_coef
    out["prior_coef_norms"] = norms
    out["n_vintages"] = len(norms)
    out["status"] = "ok" if norms else "unavailable"
    if not norms:
        out["reason"] = (
            f"no prior registered vintage of {arm!r} carried a readable "
            "standardized_coef block"
        )
    return out
