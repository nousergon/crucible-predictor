"""training/xsec_magnitude.py — the incumbent's cross-sectional dispersion,
recomputed on the CANDIDATE's own panel.

alpha-engine-config-I10185.

Why this exists
---------------
``training/xsec_variance_share.py`` (crucible-predictor-PR611) measures what
FRACTION of a fitted L2's variance is cross-sectional. That is a ratio, and its
module docstring names its invariance under a uniform rescaling of the
coefficients as a deliberate design property — so it cannot see a model that
keeps its cross-sectional structure and simply produces far too little of it.

Measured 2026-09-08 with that module, unmodified, on the live champion's actual
meta-training design matrix::

    v3.0-meta-2026-09-04-cc3271ea   share 0.702720   xsec_sd 0.010350
    v3.0-meta-2026-08-14-119e069b   share 0.919372   xsec_sd 0.048588
                                    (incumbent COEFFICIENTS, SAME panel)

The champion clears the 0.10 share floor by 7x. Its first served batch collapsed
to ``alpha_stdev`` 0.005819 — 0.41x the trailing 10-day median, below the 0.015
absolute serving floor, zero high-confidence names. The catchable number existed
at fit time: 0.010350 was ALREADY below the serving floor before that model ever
served a batch, and nothing gated on it.

Why the panel must be held constant
-----------------------------------
``xsec_sd`` has units. It moves with the target's scale, the horizon, and the
dispersion of the vintage the model was fitted on. Comparing a candidate's
number against the incumbent's STORED number therefore measures the two panels
as much as the two models. The only comparison that isolates the model is the
incumbent's frozen coefficients scored on the candidate's own rows, which is
what this module computes.

Relationship to ``training/incumbent_rescore.py``
-------------------------------------------------
Same bundle, same ETag verification (alpha-engine-config-I9028), same
feature-contract rules, same never-raises posture. Those seams are imported from
that module rather than restated: a second set of rules for loading a frozen
incumbent is a second thing to keep true, and the I5949 zero-fill trap is
exactly the kind of rule that gets restated wrong.

Posture
-------
NEVER raises. A failure yields a status and a null number; the promotion veto
reads a null as ``uncomputable`` and — per alpha-engine-config-I10185 — refuses
a below-floor candidate anyway, because a missing reference cannot exonerate
one. Absence is never a pass.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


# Seams, so a test can substitute them and so there is exactly one
# implementation of each rule in the tree.
def _resolve_served_version(s3, bucket: str):
    from training.incumbent_rescore import resolve_served_version

    return resolve_served_version(s3, bucket)


def _verify_bundle(s3, bucket: str, version_id: str) -> None:
    from model.registry import verify_bundle

    verify_bundle(s3, bucket, version_id)


def _load_meta_model(s3, bucket: str, version_id: str):
    from training.served_slice_dispersion import load_meta_model

    return load_meta_model(s3, bucket, version_id)


def incumbent_xsec_sd_on_candidate_panel(
    s3,
    bucket: str,
    *,
    meta_X,
    dates,
    train_meta_features,
    served_version: str | None = None,
) -> dict:
    """The incumbent's ``xsec_sd`` over the CANDIDATE's training panel.

    Returns a JSON-serializable block for the manifest::

        {"status": "ok" | "no_incumbent" | "feature_contract_mismatch"
                   | "bundle_unverifiable" | "coefficients_unavailable" | "error",
         "reason": str | None,
         "incumbent_version_id": str | None,
         "xsec_sd": float | None,
         "xsec_variance_share": float | None,
         "n_rows": int, "n_dates": int | None,
         "basis": "incumbent_coefficients_on_candidate_panel"}

    ``basis`` is recorded because the whole claim rests on it: this is the
    incumbent's frozen coefficients on THIS vintage's rows, never its own
    stored number and never a refit.
    """
    import numpy as np

    from training.xsec_variance_share import cross_sectional_variance_share

    X = np.asarray(meta_X, dtype=float)
    names = list(train_meta_features)
    out: dict = {
        "status": "error",
        "reason": None,
        "incumbent_version_id": None,
        "xsec_sd": None,
        "xsec_variance_share": None,
        "n_rows": int(X.shape[0]) if X.ndim == 2 else None,
        "n_dates": None,
        "basis": "incumbent_coefficients_on_candidate_panel",
    }
    try:
        vid = served_version or _resolve_served_version(s3, bucket)
        out["incumbent_version_id"] = vid
        if not vid:
            out["status"] = "no_incumbent"
            out["reason"] = (
                "the live manifest names no served_version — nothing is "
                "serving, so there is no incumbent to score on this panel "
                "(champion-challenger-policy §9.1 bootstrap)"
            )
            log.warning("incumbent xsec_sd: %s", out["reason"])
            return out

        try:
            _verify_bundle(s3, bucket, vid)
        except Exception as exc:  # noqa: BLE001 — labelled status, never a raise
            out["status"] = "bundle_unverifiable"
            out["reason"] = (
                f"registry bundle {vid} failed ETag verification ({exc}) — "
                f"refusing to score weights that are not provably the served "
                f"ones (alpha-engine-config-I9028)"
            )
            log.error("incumbent xsec_sd: %s", out["reason"])
            return out

        model = _load_meta_model(s3, bucket, vid)
        model_names = list(getattr(model, "_feature_names", None) or [])
        if not model_names:
            out["status"] = "feature_contract_mismatch"
            out["reason"] = (
                f"incumbent {vid}'s meta_model.pkl embeds no feature_names, so "
                f"its columns cannot be aligned to this vintage's "
                f"{len(names)} features"
            )
            log.warning("incumbent xsec_sd: %s", out["reason"])
            return out

        coefficients = dict(getattr(model, "_coefficients", None) or {})
        if not coefficients:
            out["status"] = "coefficients_unavailable"
            out["reason"] = (
                f"incumbent {vid} exposes no _coefficients. The candidate's own "
                f"xsec_sd is computed from its coefficient vector by "
                f"training/xsec_variance_share.py, so scoring the incumbent any "
                f"other way (predict(), a re-derivation) would not be the same "
                f"arithmetic and the two numbers would not be comparable"
            )
            log.warning("incumbent xsec_sd: %s", out["reason"])
            return out

        missing = sorted(set(model_names) - set(names))
        extra = sorted(set(names) - set(model_names))
        if missing or extra:
            out["status"] = "feature_contract_mismatch"
            out["reason"] = (
                f"incumbent {vid} was fit on a different feature set than this "
                f"vintage: it needs {missing or 'nothing'} that this run does "
                f"not produce, and this run adds {extra or 'nothing'}. Zero-"
                f"filling the difference is alpha-engine-config-I5949, and here "
                f"it would UNDERSTATE the incumbent's dispersion — which "
                f"flatters the candidate's ratio and turns a defect into a "
                f"pass. No number is produced; the veto reads that as "
                f"uncomputable and still refuses a below-floor candidate."
            )
            out["incumbent_feature_names"] = model_names
            out["vintage_feature_names"] = names
            log.warning("incumbent xsec_sd: %s", out["reason"])
            return out

        # Same feature SET: the candidate's columns are scored with the
        # incumbent's coefficients looked up BY NAME, so a column reorder is a
        # non-event rather than a silent mis-pairing.
        summary = cross_sectional_variance_share(
            X, dates, names,
            {n: float(coefficients.get(n, 0.0)) for n in names},
            intercept=float(coefficients.get("intercept", 0.0)),
        )
        out["n_dates"] = summary.get("n_dates")
        if summary.get("xsec_sd") is None:
            out["status"] = summary.get("status") or "error"
            out["reason"] = (
                f"the incumbent {vid} could not be measured on this panel: "
                f"{summary.get('reason')}"
            )
            log.warning("incumbent xsec_sd: %s", out["reason"])
            return out
        out.update({
            "status": "ok",
            "xsec_sd": float(summary["xsec_sd"]),
            "xsec_variance_share": summary.get("xsec_variance_share"),
        })
        log.info(
            "incumbent xsec_sd (alpha-engine-config-I10185): %s scores "
            "xsec_sd=%.6g / share=%.6g on THIS candidate's panel (%s rows, %s "
            "dates), frozen coefficients. This is the like-for-like reference a "
            "candidate's magnitude leg is measured against — never the "
            "incumbent's own stored number, which was measured on a different "
            "panel.",
            vid, out["xsec_sd"], out["xsec_variance_share"],
            out["n_rows"], out["n_dates"],
        )
        return out
    except Exception as exc:  # noqa: BLE001 — never fails a rotation
        out["status"] = "error"
        out["reason"] = f"incumbent xsec_sd on candidate panel failed: {exc}"
        log.warning(
            "incumbent xsec_sd failed — the magnitude leg's relative reference "
            "will be uncomputable and NAMED. A candidate below the absolute "
            "floor is still refused (alpha-engine-config-I10185); absence is "
            "never a pass.",
            exc_info=True,
        )
        return out
