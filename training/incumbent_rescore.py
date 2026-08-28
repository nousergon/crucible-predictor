"""training/incumbent_rescore.py — the incumbent, scored on TODAY's vintage.

alpha-engine-config-I9024 section 2.

The defect this closes
----------------------
``model_zoo._resolve_incumbent_from_bundle`` reads the incumbent's CPCV mean IC
from ``predictor/registry/{served_version}/manifest.json``. That number is
honest about WHERE it came from — I9018 fixed exactly that, killing the ``x >= x``
tautology — but it is a score the incumbent earned during its OWN training run,
weeks ago, on that vintage's data and that vintage's folds. Every candidate in
this rotation is scored on today's data and today's folds.

So the weekly comparison is CROSS-VINTAGE: a stale number ranked against fresh
ones. Both directions are unfair, and both promotion and demotion hang off it.
An incumbent that has decayed keeps a flattering old score; an incumbent that
trained through a hard regime carries a pessimistic one and gets displaced by a
challenger that never faced it.

What this does instead
----------------------
Evaluate the incumbent's FROZEN weights over the current rotation's CPCV folds —
**predict only, never a refit**. ``cpcv_meta_oos_ic`` already takes a
``fit_predict_fn``; the one built here ignores the training split entirely and
predicts the test fold from the model loaded out of
``predictor/registry/{served_version}/``. Same ``meta_X``, same ``meta_y``, same
folds, same embargo, same group/k parameters as every candidate in the run. That
identity is what makes the comparison apples-to-apples, and it is why this is
computed inside the training run rather than at select time: the folds only exist
here.

Loading a frozen bundle for evaluation is an exercised path
(``inference/stages/shadow_versions.py`` does it for shadow scoring), and
``model.registry.verify_bundle`` (alpha-engine-config-I9028) checks the bundle
against its own recorded ETags first, so the weights being scored are provably
the weights that are serving.

The feature contract, and why a mismatch must NOT raise
-------------------------------------------------------
The incumbent's ``feature_list.json`` may not match the current vintage's
``TRAIN_META_FEATURES`` — any feature addition makes it so. A frozen ridge cannot
be evaluated on a feature matrix it was not fit on, and zero-filling the
difference is the ``alpha-engine-config-I5949`` defect (a model silently scored
on zeros).

So a mismatch returns ``status: "feature_contract_mismatch"`` and the caller
falls back to the bundle's stored number. It must NOT raise: raising would break
every rotation after any feature change, which is a worse failure than an
unfair comparison. But the fallback is LABELLED — ``serving_ic_basis`` on the
leaderboard says ``rescored_current_vintage`` or ``stored_training_vintage``,
with the reason. An unlabelled fallback is the shape of the original I9018 bug:
the reader could not tell which comparison actually happened.

Both numbers are kept
---------------------
``serving_ic_stored`` (what it scored when trained) and ``serving_ic_rescored``
(what it scores on current data) are both carried. Their difference is a
MODEL-DECAY signal this loop has never had, and it costs nothing to surface.
"""
from __future__ import annotations

import json
import logging

import config as cfg

log = logging.getLogger(__name__)

_REGISTRY_PREFIX = "predictor/registry/"


def resolve_served_version(s3, bucket: str) -> str | None:
    """The live manifest's ``served_version`` POINTER — never a score.

    Returns None when there is no live manifest (bootstrap) or it names no
    served version; the caller reports that rather than guessing.
    """
    try:
        raw = s3.get_object(
            Bucket=bucket, Key=cfg.META_MANIFEST_KEY)["Body"].read()
        return (json.loads(raw) or {}).get("served_version") or None
    except Exception as exc:  # noqa: BLE001 — reported by the caller, never a pass
        log.warning(
            "incumbent re-score: no readable live manifest at %s (%s) — the "
            "incumbent cannot be identified, so no re-score is attempted.",
            getattr(cfg, "META_MANIFEST_KEY", "?"), exc,
        )
        return None


def _bundle_feature_list(s3, bucket: str, version_id: str) -> list:
    raw = s3.get_object(
        Bucket=bucket,
        Key=f"{_REGISTRY_PREFIX}{version_id}/feature_list.json",
    )["Body"].read()
    return list((json.loads(raw) or {}).get("l2_features") or [])


def _column_permutation(model_names: list, train_names: list):
    """Indices mapping ``train_names``-ordered columns into ``model_names`` order.

    A pure REORDER of the same feature set is not a contract mismatch — the
    ridge consumes columns positionally, so the matrix is simply reindexed.
    Returns None when the two are not the same set, which IS a mismatch.
    """
    if list(model_names) == list(train_names):
        return None  # already aligned; no reindex needed
    if set(model_names) != set(train_names):
        return False  # a genuine contract mismatch
    pos = {name: i for i, name in enumerate(train_names)}
    return [pos[name] for name in model_names]


def rescore_incumbent_on_current_vintage(
    s3, bucket: str, *, meta_X, meta_y, row_dates, train_meta_features,
    forward_days: int, embargo_days: int | None, n_groups: int, k_test: int,
    served_version: str | None = None,
) -> dict:
    """The incumbent's CPCV mean IC over THIS rotation's folds. Never raises.

    Returns a JSON-serializable block for the manifest::

        {"status": "ok" | "no_incumbent" | "feature_contract_mismatch"
                   | "bundle_unverifiable" | "error",
         "reason": str | None,
         "incumbent_version_id": str | None,
         "mean_ic": float | None,
         "n_combos": int | None, "n_groups": int | None, "k_test": int | None,
         "forward_days": int, "embargo_days": int | None,
         "n_rows": int, "n_features": int,
         "fit_mode": "frozen_predict_only"}

    ``fit_mode`` is recorded because the whole claim rests on it: the incumbent
    is EVALUATED on this vintage, never refitted to it. A refit would measure the
    architecture, not the model that is serving.
    """
    import numpy as np

    out: dict = {
        "status": "error", "reason": None, "incumbent_version_id": None,
        "mean_ic": None, "n_combos": None, "n_groups": None, "k_test": None,
        "forward_days": int(forward_days), "embargo_days": embargo_days,
        "n_rows": int(np.asarray(meta_y).shape[0]),
        "n_features": len(list(train_meta_features)),
        "fit_mode": "frozen_predict_only",
    }
    try:
        vid = served_version or resolve_served_version(s3, bucket)
        out["incumbent_version_id"] = vid
        if not vid:
            out["status"] = "no_incumbent"
            out["reason"] = (
                "the live manifest names no served_version — nothing is "
                "serving, so there is no incumbent to re-score "
                "(champion-challenger-policy §9.1 bootstrap)"
            )
            log.warning("incumbent re-score: %s", out["reason"])
            return out

        # alpha-engine-config-I9028 — the bundle checks itself against its own
        # recorded ETags BEFORE its weights are scored. A drifted bundle must
        # not become the number a promotion decision is made against.
        from model.registry import verify_bundle
        try:
            verify_bundle(s3, bucket, vid)
        except Exception as exc:  # noqa: BLE001 — labelled fallback, not a raise
            out["status"] = "bundle_unverifiable"
            out["reason"] = (
                f"registry bundle {vid} failed ETag verification ({exc}) — "
                f"refusing to score weights that are not provably the served "
                f"ones (alpha-engine-config-I9028)"
            )
            log.error("incumbent re-score: %s", out["reason"])
            return out

        from training.served_slice_dispersion import load_meta_model
        model = load_meta_model(s3, bucket, vid)
        model_names = list(getattr(model, "_feature_names", None) or [])
        train_names = list(train_meta_features)
        bundle_names = _bundle_feature_list(s3, bucket, vid)
        if not model_names:
            out["status"] = "feature_contract_mismatch"
            out["reason"] = (
                f"incumbent {vid}'s meta_model.pkl embeds no feature_names, so "
                f"its columns cannot be aligned to this vintage's "
                f"{len(train_names)} features"
            )
            log.warning("incumbent re-score: %s", out["reason"])
            return out
        if bundle_names and set(bundle_names) != set(model_names):
            log.warning(
                "incumbent re-score: %s's feature_list.json and its pickled "
                "feature_names disagree (%s vs %s names) — the PICKLE is "
                "authoritative because predict consumes columns positionally.",
                vid, len(bundle_names), len(model_names),
            )
        perm = _column_permutation(model_names, train_names)
        if perm is False:
            missing = sorted(set(model_names) - set(train_names))
            extra = sorted(set(train_names) - set(model_names))
            out["status"] = "feature_contract_mismatch"
            out["reason"] = (
                f"incumbent {vid} was fit on a different feature set than this "
                f"vintage: it needs {missing or 'nothing'} that this run does "
                f"not produce, and this run adds {extra or 'nothing'}. A frozen "
                f"ridge cannot be evaluated on a matrix it was not fit on, and "
                f"zero-filling the difference is alpha-engine-config-I5949. "
                f"Falling back to the bundle's STORED training-vintage CPCV — "
                f"labelled as such on the leaderboard, never substituted "
                f"silently."
            )
            out["incumbent_feature_names"] = model_names
            out["vintage_feature_names"] = train_names
            log.warning("incumbent re-score: %s", out["reason"])
            return out

        X = np.asarray(meta_X, dtype=float)
        if perm is not None:
            X = X[:, perm]
            log.info(
                "incumbent re-score: reindexed %s feature columns into %s's "
                "own order (same set, different order — not a mismatch).",
                len(perm), vid,
            )

        def _frozen_predict(_Xtr, _ytr, Xte):
            """Predict the test fold from FROZEN weights.

            ``_Xtr`` / ``_ytr`` are deliberately ignored: this measures the model
            that is serving, on this vintage's held-out rows. Refitting here
            would measure the incumbent's ARCHITECTURE and would be the same
            quantity champion-arch already reports.
            """
            return np.asarray(model.predict(np.asarray(Xte, dtype=float))).ravel()

        from training.leakfree_meta_ic import cpcv_meta_oos_ic
        cpcv = cpcv_meta_oos_ic(
            X, meta_y, row_dates,
            fit_predict_fn=_frozen_predict,
            forward_days=forward_days,
            embargo_days=embargo_days,
            n_groups=n_groups,
            k_test=k_test,
        )
        mean_ic = cpcv.get("mean_ic")
        mean_ic = None if mean_ic is None or mean_ic != mean_ic else float(mean_ic)
        if cpcv.get("status") != "ok" or mean_ic is None:
            out["status"] = "error"
            out["reason"] = (
                f"CPCV over this vintage's folds returned status="
                f"{cpcv.get('status')!r} for incumbent {vid} — no usable mean IC"
            )
            log.warning("incumbent re-score: %s", out["reason"])
            return out
        out.update({
            "status": "ok", "mean_ic": round(mean_ic, 6),
            "n_combos": cpcv.get("n_combos"), "n_groups": cpcv.get("n_groups"),
            "k_test": cpcv.get("k_test"),
            "std_ic": cpcv.get("std_ic"), "frac_positive": cpcv.get("frac_positive"),
            "n_backtest_paths": cpcv.get("n_backtest_paths"),
        })
        log.info(
            "incumbent re-score (alpha-engine-config-I9024 §2): %s scores "
            "mean CPCV IC %.6f on THIS vintage's folds over %s combos, frozen "
            "weights, predict-only. This is the number a candidate must beat — "
            "not the score it earned on its own training vintage.",
            vid, mean_ic, cpcv.get("n_combos"),
        )
        return out
    except Exception as exc:  # noqa: BLE001 — labelled fallback; never fails a rotation
        out["status"] = "error"
        out["reason"] = f"incumbent re-score failed: {exc}"
        log.warning(
            "incumbent re-score failed — the leaderboard will fall back to the "
            "bundle's STORED training-vintage CPCV and say so "
            "(serving_ic_basis=stored_training_vintage). NOT a pass.",
            exc_info=True,
        )
        return out
