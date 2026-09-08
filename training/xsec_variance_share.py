"""Fit-time cross-sectional variance-share diagnostic for the meta-model (L2).

**Why this exists.** The predictor's task is CROSS-SECTIONAL: rank today's
names against each other. A linear L2 fitted on a stacked panel can score well
on pooled metrics while carrying essentially all of its variance in features
that are CONSTANT within a date (macro level, VIX, yield-curve slope, regime
intensity). Such a model moves every ticker together and separates none. Its
served alphas collapse onto one or two values, and any calibrator applied to
them plateaus — which is how the defect surfaces downstream: as a *calibrator*
alarm.

That misattribution is the reason this module is at fit time. Measured on the
2026-09-08 pre-open batch: three shadow arms
(``spec-sota-combine-2026-07-{24,29,30}``) emitted 6 distinct ``predicted_alpha``
values across 29 tickers — 24 of them identical — while ``momentum_20d`` and
``expected_move`` spanned 24 distinct values each. Their manifests put every
coefficient above 0.10 on a date-constant macro feature and left 0.0027–0.0133
on the cross-sectionally varying ones. The champion of the same day was healthy
(29 distinct alphas, 8 unique p_up bins). The inference-time variance gate could
only report the symptom, on 29 live tickers, hours after the fit.

**The metric.** Decompose the fitted linear predictor
``y_hat = intercept + X @ c`` into its within-date and total variance:

    xsec_variance_share = Var_within_date(y_hat) / Var_total(y_hat)

Within-date variance is pooled: the mean over dates of each date's variance
about that date's own mean. The share is invariant under a uniform rescaling of
the coefficients, so a SCALE collapse (a separate defect — see
alpha-engine-config-I9255) does not masquerade as a cross-section fix here, and
vice versa. It is also invariant to the intercept.

**Posture.** ``cross_sectional_variance_share`` is a pure measurement and never
raises on a healthy-or-not verdict; ``assert_cross_sectionally_live`` is the
loud gate a producer calls. An unmeasurable summary NEVER passes: a fit that
could not be evaluated is unobserved, not healthy.

**What this does not claim.** Per-feature ``xsec_sd`` and ``abs_contrib`` are
MARGINAL (correlation-blind) reads used for attribution only; the headline
``xsec_sd`` / ``xsec_variance_share`` are computed on the assembled predictor
and are therefore correlation-aware.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

#: A feature whose pooled within-date standard deviation is at or below this is
#: cross-sectionally dead — it cannot separate two names on the same day.
XSEC_DEAD_SD_EPS: float = 1e-9

#: Default floor for ``xsec_variance_share``. A model below this carries less
#: than a tenth of its variance in the axis it is asked to predict. Chosen as
#: an order-of-magnitude bar, not a tuned threshold: the measured healthy
#: champion sits far above it and the measured degenerate arms sit near zero.
#: Overridable per call so the gate can be tightened without a code change.
XSEC_VARIANCE_SHARE_FLOOR: float = 0.10

#: Minimum distinct dates before a within/total split means anything. With one
#: date every observation shares its date, the share is 1.0 by construction and
#: carries no information.
MIN_DATES: int = 2


class CrossSectionDegenerate(RuntimeError):
    """The fitted L2 does not separate names within a date (or could not be
    measured). Raised by :func:`assert_cross_sectionally_live`."""


def cross_sectional_variance_share(
    meta_X,
    dates,
    feature_names,
    coefficients: dict,
    *,
    intercept: float = 0.0,
    floor: float = XSEC_VARIANCE_SHARE_FLOOR,
    dead_sd_eps: float = XSEC_DEAD_SD_EPS,
) -> dict:
    """Measure how much of a fitted linear L2's variance is cross-sectional.

    Parameters
    ----------
    meta_X : (n_rows, n_features) array of the training design matrix.
    dates : length-n_rows sequence of the date each row belongs to.
    feature_names : length-n_features names, positionally aligned with ``meta_X``.
    coefficients : ``{feature_name: coef}``; every name in ``feature_names``
        must be present. A missing name RAISES rather than defaulting to 0.0 —
        a wiring bug that silently zeroes a live coefficient would understate
        the model's cross-sectional variance and turn a defect into a PASS.
    intercept : fitted intercept. Shifts ``y_hat`` and cannot change either
        variance; accepted so callers can pass the whole fit without thinking.

    Returns a summary dict; never raises for an unhealthy model (that is
    :func:`assert_cross_sectionally_live`'s job) but DOES raise ``ValueError``
    on malformed input.
    """
    X = np.asarray(meta_X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"meta_X must be 2-D, got shape {X.shape}")
    names = list(feature_names)
    if X.shape[1] != len(names):
        raise ValueError(
            f"feature_names has {len(names)} entries for {X.shape[1]} columns"
        )
    date_arr = np.asarray(list(dates), dtype=object)
    if date_arr.shape[0] != X.shape[0]:
        raise ValueError(
            f"dates has {date_arr.shape[0]} entries for {X.shape[0]} meta_X rows"
        )
    if not np.isfinite(X).all():
        raise ValueError(
            "meta_X contains non-finite values; the caller must decide how to "
            "handle them — dropping them here would silently change which "
            "cross-section is being measured"
        )

    missing = [n for n in names if n not in coefficients]
    if missing:
        raise ValueError(
            f"coefficients missing for fitted feature(s): {sorted(missing)}"
        )
    coefs = np.array([float(coefficients[n]) for n in names], dtype=float)
    if not np.isfinite(coefs).all():
        raise ValueError("coefficients contain non-finite values")

    uniq_dates, date_idx = np.unique(date_arr, return_inverse=True)
    n_dates = int(uniq_dates.shape[0])
    n_rows = int(X.shape[0])

    base = {
        "status": "ok",
        "n_rows": n_rows,
        "n_dates": n_dates,
        "n_features": len(names),
        "floor": float(floor),
    }

    if n_dates < MIN_DATES:
        return {
            **base,
            "status": "insufficient_data",
            "verdict": "unmeasurable",
            "reason": (
                f"n_dates={n_dates} < {MIN_DATES}: with a single date every row "
                "shares its date, so the within/total split is 1.0 by "
                "construction and measures nothing"
            ),
            "xsec_sd": None,
            "total_sd": None,
            "xsec_variance_share": None,
            "per_feature": {},
            "n_xsec_dead_features": None,
            "coef_mass_on_xsec_dead": None,
        }

    y_hat = float(intercept) + X @ coefs

    # Pooled within-date variance of the assembled predictor: the mean over
    # dates of each date's variance about its OWN mean. Dates with a single row
    # contribute 0 variance and are counted — a day on which the model scores
    # one name genuinely separates nothing.
    def _pooled_within_var(v: np.ndarray) -> float:
        centered = v - np.bincount(date_idx, weights=v)[date_idx] / np.bincount(date_idx)[date_idx]
        return float(np.mean(centered ** 2))

    xsec_var = _pooled_within_var(y_hat)
    total_var = float(np.var(y_hat))

    per_feature: dict = {}
    dead: list[str] = []
    total_mass = 0.0
    dead_mass = 0.0
    for j, name in enumerate(names):
        col = X[:, j]
        f_xsec_sd = float(np.sqrt(_pooled_within_var(col)))
        f_total_sd = float(np.std(col))
        is_dead = f_xsec_sd <= dead_sd_eps
        mass = abs(float(coefs[j])) * f_total_sd
        total_mass += mass
        if is_dead:
            dead.append(name)
            dead_mass += mass
        per_feature[name] = {
            "coef": float(coefs[j]),
            "xsec_sd": f_xsec_sd,
            "total_sd": f_total_sd,
            "abs_contrib": abs(float(coefs[j])) * f_xsec_sd,
            "is_xsec_dead": bool(is_dead),
        }

    out = {
        **base,
        "xsec_sd": float(np.sqrt(xsec_var)),
        "total_sd": float(np.sqrt(total_var)),
        "per_feature": per_feature,
        "n_xsec_dead_features": len(dead),
        "xsec_dead_features": sorted(dead),
        "coef_mass_on_xsec_dead": (
            float(dead_mass / total_mass) if total_mass > 0 else None
        ),
    }

    if total_var <= 0.0:
        out.update({
            "xsec_variance_share": 0.0,
            "verdict": "degenerate_cross_section",
            "reason": (
                "the fitted linear predictor is constant over the whole "
                "training panel (total variance 0) — it separates nothing, "
                "cross-sectionally or otherwise"
            ),
        })
        return out

    share = float(xsec_var / total_var)
    out["xsec_variance_share"] = share
    if share < floor:
        out["verdict"] = "degenerate_cross_section"
        out["reason"] = (
            f"xsec_variance_share={share:.6f} < floor={floor}: "
            f"{len(dead)} of {len(names)} features are cross-sectionally dead "
            f"and carry {out['coef_mass_on_xsec_dead']:.1%} of the coefficient "
            "mass; this model moves every name together and separates none"
        )
    else:
        out["verdict"] = "ok"
        out["reason"] = None
    return out


def assert_cross_sectionally_live(summary: dict, *, context: str = "meta-model") -> None:
    """Raise :class:`CrossSectionDegenerate` unless ``summary`` says ``ok``.

    FAIL LOUD (fleet default). A degenerate fit that reaches serving cannot be
    distinguished from a healthy one by any downstream surface except the
    inference-time p_up variance gate — which fires on live tickers, names the
    calibrator, and is therefore a false lead. Refusing here is the only place
    the true cause is nameable.

    ``unmeasurable`` is NOT a pass: a fit that could not be evaluated is
    unobserved, never healthy.
    """
    verdict = (summary or {}).get("verdict")
    if verdict == "ok":
        return
    raise CrossSectionDegenerate(
        f"{context} is not cross-sectionally live "
        f"(verdict={verdict!r}, status={(summary or {}).get('status')!r}, "
        f"xsec_variance_share={(summary or {}).get('xsec_variance_share')!r}, "
        f"floor={(summary or {}).get('floor')!r}): "
        f"{(summary or {}).get('reason') or 'no summary produced'}"
    )
