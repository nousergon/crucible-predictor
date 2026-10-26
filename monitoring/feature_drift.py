"""Feature-drift monitoring: inference-vs-training KS test (config#859).

Produces the ``feature_drift_ks`` block in ``predictor/metrics/latest.json``
for the evaluator report-card ``feature_drift_ks`` component (Predictor tile,
diagnostic). Lower KS = inference feature distribution still matches training.

Design:
  - Training (weekly) saves a per-feature subsample reference to
    ``predictor/weights/meta/feature_drift_reference.json``.
  - Inference (daily) loads it and runs a 2-sample Kolmogorov-Smirnov test
    (``scipy.stats.ks_2samp``) per feature: today's cross-sectional
    distribution vs the saved training subsample. The headline is ``max_ks``
    (worst-feature drift).

Only CROSS-SECTIONAL features are tested. The macro_* / regime_intensity_z
META_FEATURES are MARKET-WIDE — a single value shared by every ticker on a
date — so an inference "distribution" is one repeated value, and KS vs the
multi-date training history would always read as huge drift (one date vs
many) with no real signal. Per-ticker features are where cross-sectional
drift is informative.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# S3 key for the training-time reference (sibling of manifest.json /
# feature_list.json under the meta-weights prefix). Distinct from the
# legacy ``predictor/metrics/training_feature_stats.json`` the z-score
# drift_detector expects, to avoid feature-set entanglement.
# alpha-engine-config-I11478 — training writes this FILENAME into its own
# per-run staging prefix, so the reference travels inside the registry bundle
# and ``model.registry.promote_to_champion`` lands it at the live key together
# with the model it describes.
FEATURE_DRIFT_REFERENCE_FILENAME = "feature_drift_reference.json"
FEATURE_DRIFT_REFERENCE_KEY = f"predictor/weights/meta/{FEATURE_DRIFT_REFERENCE_FILENAME}"

# Per-ticker META_FEATURES whose cross-sectional distribution is meaningful to
# KS-test. Excludes the market-wide macro_* / regime_intensity_z columns.
CROSS_SECTIONAL_DRIFT_FEATURES: tuple[str, ...] = (
    "research_calibrator_prob",
    "momentum_score",
    "expected_move",
    "research_composite_score",
    "research_conviction",
    "sector_macro_modifier",
)

# alpha-engine-config-I10063 — WHICH producer made a feature's values.
#
# ``research_calibrator_prob`` has two producers that emit different
# distributions by construction: the ResearchGBMScorer (canonical since
# 2026-05-09; near-continuous) and the bucket-lookup ResearchCalibrator (the
# fallback when the GBM is not fitted or not loaded; a handful of discrete hit
# rates). A training run whose purged split cannot support the GBM fits the
# meta-Ridge on bucket-lookup values, while serving keeps using whichever
# ``research_gbm.pkl`` the live prefix holds. A KS between those two is a
# statistic about two different producers, not about drift — measured
# 2026-10-05: KS 0.986, reference (2026-10-02 run, GBM `insufficient`) has 7
# distinct values, the served 2026-08-14 champion's GBM emitted 2.
#
# The KS numbers are left exactly as computed. What changes is that the block
# now SAYS which producer each side used, so a reader (and the evaluator's
# grader) can tell a measurement of drift from a comparison of producers.
RESEARCH_CALIBRATOR_PROB = "research_calibrator_prob"
PRODUCER_RESEARCH_GBM = "research_gbm"
PRODUCER_BUCKET_LOOKUP = "bucket_lookup"

# alpha-engine-config-I10063 — which reference the KS ran against.
#
# The live key is not the served model's reference: the weekly
# champion-architecture refresh rewrites it (the grandfathered second writer,
# alpha-engine-config-I9029) whether or not that run is promoted. Measured
# 2026-10-12: the served champion was ``v3.0-meta-2026-08-14-119e069b`` while
# the live key held the 2026-10-09 refresh's reference, and its
# research_calibrator_prob mean moved 0.482 / 0.492 / 0.569 / 0.443 across four
# weekly rewrites while the served feature stayed at 0.49-0.51. Since
# alpha-engine-config-I11478 every registry bundle carries the reference its
# own training run built, so the served champion's bundle is the reference
# that describes the model actually serving. The live key is the fallback for
# a champion whose bundle predates I11478, and the block says which was used.
SERVED_MANIFEST_KEY = "predictor/weights/meta/manifest.json"  # == config.META_MANIFEST_KEY
REGISTRY_PREFIX = "predictor/registry/"  # == model.registry.DEFAULT_REGISTRY_PREFIX
REFERENCE_SOURCE_SERVED_BUNDLE = "served_bundle"
REFERENCE_SOURCE_LIVE_KEY = "live_key"

EXCLUDED_PRODUCER_MISMATCH = "producer_mismatch"

# alpha-engine-config-I10063 option (a) — WHICH ROWS the reference describes.
#
# Inference KS-tests the ~30 names it scores: each date's predictor cut
# (``attractiveness_top_20``) plus the book (``load_universe``). The reference
# used to be a thinned sample of the WHOLE training panel, ~900 names per date.
# Those are different populations by construction: measured 2026-10-25,
# ``research_composite_score`` on the 10-23 served names had median 0.983 (26 of
# 29 at or above 0.90) against a near-uniform panel reference, KS 0.81 with no
# drift at all. A ranking input reads RED every day under that design, and the
# other features correlated with the ranking (``momentum_score``) inherit part
# of it. The reference is therefore built from the panel rows the per-date
# population rule would have served (``training/served_cut_mask.py``), so the
# KS compares like with like. The labels equal that module's ``BASIS_*``
# constants (asserted by tests), so one word names one population everywhere.
REFERENCE_POPULATION_SERVED_CUT = "served_cut"  # == served_cut_mask.BASIS_SERVED_CUT
REFERENCE_POPULATION_FIT_PANEL = "fit_panel"    # == served_cut_mask.BASIS_FIT_PANEL

PRODUCER_PARITY_MATCH = "match"            # every declared feature: same producer
PRODUCER_PARITY_MISMATCH = "mismatch"      # >=1 feature: producers differ
PRODUCER_PARITY_UNDECLARED = "undeclared"  # a side did not record its producer

_MAX_REFERENCE_SAMPLES = 2000  # per-feature subsample cap (keeps the JSON small)
_MIN_KS_SAMPLES = 30           # mirrors the consumer grader's n_floor


def _finite(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    return a[np.isfinite(a)]


def _subsample(values: np.ndarray, cap: int = _MAX_REFERENCE_SAMPLES) -> list[float]:
    """Deterministically thin ``values`` to at most ``cap`` points via
    evenly-spaced indices (no RNG — reproducible across runs)."""
    if values.size <= cap:
        return [round(float(v), 6) for v in values]
    idx = np.linspace(0, values.size - 1, cap).astype(int)
    return [round(float(v), 6) for v in values[idx]]


def _column_map(
    matrix: np.ndarray, feature_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Map the cross-sectional feature names to their (finite) columns of
    ``matrix`` (shape n_rows x len(feature_names))."""
    out: dict[str, np.ndarray] = {}
    for j, name in enumerate(feature_names):
        if name in CROSS_SECTIONAL_DRIFT_FEATURES:
            out[name] = _finite(matrix[:, j])
    return out


def research_calibrator_producer(gbm_used: bool) -> dict[str, str]:
    """The ``producers`` entry for ``research_calibrator_prob``: the GBM when
    it produced the values, else the bucket lookup (I10063)."""
    return {
        RESEARCH_CALIBRATOR_PROB: (
            PRODUCER_RESEARCH_GBM if gbm_used else PRODUCER_BUCKET_LOOKUP
        ),
    }


def build_training_reference(
    matrix: np.ndarray, feature_names: Sequence[str], *, trained_date: Optional[str] = None,
    producers: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Build the training feature reference from the training META_FEATURES
    matrix (n_samples x n_features) and its column order ``feature_names``.

    Stores a thinned per-feature subsample (for the 2-sample KS at inference)
    plus mean/std (cheap context). Cross-sectional features only.

    ``producers`` (I10063) records which producer made a feature's training
    values, for features that have more than one (``research_calibrator_prob``).
    A reference written before this key existed reads as ``undeclared``.
    """
    cols = _column_map(np.asarray(matrix, dtype=float), feature_names)
    samples = {f: _subsample(v) for f, v in cols.items() if v.size}
    reference: dict[str, Any] = {
        "schema_version": 1,
        "trained_date": trained_date,
        "features": list(samples.keys()),
        "n_samples": int(min((len(s) for s in samples.values()), default=0)),
        "samples": samples,
        "mean": {f: round(float(np.mean(cols[f])), 6) for f in samples},
        "std": {f: round(float(np.std(cols[f])), 6) for f in samples},
    }
    if producers:
        reference["producers"] = dict(producers)
    return reference


def build_served_population_reference(
    matrix: np.ndarray, feature_names: Sequence[str], served_mask: Any, *,
    trained_date: Optional[str] = None,
    producers: Optional[Mapping[str, str]] = None,
    mask_report: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """The training reference over the rows inference would have SCORED.

    alpha-engine-config-I10063 option (a). ``served_mask`` is the boolean row
    mask from ``training.served_cut_mask.resolve_served_cut_mask`` (positionally
    aligned with ``matrix``): per panel date, that date's predictor cut plus the
    held book — the population ``inference/stages/load_universe.py`` scores and
    ``_set_feature_drift_ks`` therefore tests. The reference is built from those
    rows only, and records ``population`` = ``served_cut``.

    When the mask is missing, misaligned or selects nothing, the reference falls
    back to the whole panel and SAYS so: ``population`` = ``fit_panel`` plus the
    mask's own reason, logged at ERROR. The fallback is labelled rather than
    dropped because a bundle with no reference sends inference to the live key
    (another run's reference), which is the identity defect I10063 already
    closed; a labelled panel reference keeps the block honest about what it
    compared. ``compute_feature_drift_ks`` carries the label into the block as
    ``reference_population``.
    """
    m = np.asarray(matrix, dtype=float)
    report = dict(mask_report or {})
    mask = None
    if served_mask is not None:
        mask = np.asarray(served_mask, dtype=bool)
        if mask.shape != (m.shape[0],):
            report["reason"] = (
                f"served-cut mask has shape {mask.shape}, the panel has "
                f"{m.shape[0]} rows — not aligned"
            )
            mask = None
    if mask is not None and bool(mask.any()):
        reference = build_training_reference(
            m[mask], feature_names, trained_date=trained_date, producers=producers,
        )
        reference["population"] = REFERENCE_POPULATION_SERVED_CUT
        reference["population_detail"] = {
            "n_rows": int(mask.sum()),
            "n_rows_panel": int(m.shape[0]),
            "n_dates": report.get("n_dates_resolved"),
            "cut_names": report.get("cut_names"),
        }
        return reference

    reason = report.get("reason") or (
        "served-cut mask selected no rows" if mask is not None
        else "no served-cut mask was resolved"
    )
    logger.error(
        "[feature_drift] reference built from the WHOLE training panel, not the "
        "served cut (%s). Inference tests only the served names, so this "
        "reference's KS mixes population selection with drift; the block says "
        "reference_population=%s (alpha-engine-config-I10063).",
        reason, REFERENCE_POPULATION_FIT_PANEL,
    )
    reference = build_training_reference(
        m, feature_names, trained_date=trained_date, producers=producers,
    )
    reference["population"] = REFERENCE_POPULATION_FIT_PANEL
    reference["population_detail"] = {
        "n_rows": int(m.shape[0]),
        "n_rows_panel": int(m.shape[0]),
        "reason": reason,
    }
    return reference


def producer_parity(
    reference_producers: Optional[Mapping[str, str]],
    serving_producers: Optional[Mapping[str, str]],
    features: Sequence[str],
) -> tuple[str, list[str]]:
    """Compare the producer each side declared for ``features``.

    Returns ``(parity, mismatched_features)``. ``mismatch`` wins over
    ``undeclared``: one feature known to differ already says the comparison
    is not like-for-like. ``undeclared`` when either side records nothing for
    a feature the other side declares — absence is never read as a match.
    """
    ref = dict(reference_producers or {})
    serv = dict(serving_producers or {})
    declared = [f for f in features if f in ref or f in serv]
    mismatched = sorted(
        f for f in declared if f in ref and f in serv and ref[f] != serv[f]
    )
    if mismatched:
        return PRODUCER_PARITY_MISMATCH, mismatched
    if not declared or any(f not in ref or f not in serv for f in declared):
        return PRODUCER_PARITY_UNDECLARED, []
    return PRODUCER_PARITY_MATCH, []


def compute_feature_drift_ks(
    matrix: np.ndarray, feature_names: Sequence[str], reference: dict[str, Any],
    *, serving_producers: Optional[Mapping[str, str]] = None,
) -> Optional[dict[str, Any]]:
    """2-sample KS per feature: today's inference distribution vs the training
    reference. Returns the ``feature_drift_ks`` block, or ``None`` when there's
    no usable reference / not enough inference samples (caller emits N/A).

    ``serving_producers`` (I10063) is which producer made today's values for
    features that have more than one. The block carries both sides'
    producers plus ``producer_parity`` / ``producer_mismatch``.

    A feature whose two sides were made by DIFFERENT producers has no valid
    reference: its KS compares two producers, not two periods (measured
    2026-10-12: 0.66-0.99 on every session since 2026-09-14 while the served
    values held 0.49-0.51). It is therefore moved out of ``per_feature`` and
    the ``max_ks`` / ``mean_ks`` headline into ``excluded`` — with its KS and
    the reason, so nothing is hidden — and ``n_features_excluded`` counts it.
    ``undeclared`` parity cannot be judged and stays in the headline.
    """
    from scipy.stats import ks_2samp

    ref_samples = (reference or {}).get("samples") or {}
    if not ref_samples:
        return None
    infer_cols = _column_map(np.asarray(matrix, dtype=float), feature_names)

    per_feature: dict[str, float] = {}
    n_used = 0
    for feat, ref_vals in ref_samples.items():
        inf = infer_cols.get(feat)
        ref = _finite(ref_vals)
        if inf is None or inf.size < _MIN_KS_SAMPLES or ref.size < _MIN_KS_SAMPLES:
            continue
        # Degenerate (all-constant) inference column → KS undefined-ish; skip.
        if np.ptp(inf) == 0 and np.ptp(ref) == 0:
            per_feature[feat] = 0.0
            n_used = max(n_used, int(inf.size))
            continue
        stat = float(ks_2samp(inf, ref).statistic)
        per_feature[feat] = round(stat, 4)
        n_used = max(n_used, int(inf.size))

    if not per_feature:
        return None
    ref_producers = dict(reference.get("producers") or {})
    serv_producers = dict(serving_producers or {})
    parity, mismatched = producer_parity(
        ref_producers, serv_producers, list(per_feature),
    )
    excluded = {
        f: {"ks": per_feature.pop(f), "reason": EXCLUDED_PRODUCER_MISMATCH}
        for f in mismatched
    }
    if parity == PRODUCER_PARITY_MISMATCH:
        logger.warning(
            "[feature_drift] producer mismatch on %s — reference (trained %s) "
            "used %s, serving used %s; those features' KS compares two "
            "producers, not two periods, and is excluded from the headline "
            "(alpha-engine-config-I10063).",
            mismatched, reference.get("trained_date"),
            {f: ref_producers.get(f) for f in mismatched},
            {f: serv_producers.get(f) for f in mismatched},
        )
    ks_values = list(per_feature.values())
    return {
        "max_ks": round(max(ks_values), 4) if ks_values else None,
        "mean_ks": round(float(np.mean(ks_values)), 4) if ks_values else None,
        "n_features": len(per_feature),
        "n_features_excluded": len(excluded),
        "n_samples": n_used,
        "per_feature": dict(sorted(per_feature.items(), key=lambda kv: -kv[1])),
        "excluded": excluded,
        "reference_trained_date": reference.get("trained_date"),
        # I10063 option (a): which rows the reference describes. ``None`` for
        # a reference written before the key existed (the whole panel).
        "reference_population": reference.get("population"),
        "reference_source": reference.get("reference_source"),
        "reference_version_id": reference.get("reference_version_id"),
        "served_version_id": reference.get("served_version_id"),
        "producers": {"reference": ref_producers, "serving": serv_producers},
        "producer_parity": parity,
        "producer_mismatch": mismatched,
    }


# ── S3 helpers ────────────────────────────────────────────────────────────


def save_training_reference(
    reference: dict[str, Any], *, bucket: str, region: str = "us-east-1",
    key: str = FEATURE_DRIFT_REFERENCE_KEY, s3_client: Any | None = None,
) -> str:
    import boto3

    client = s3_client if s3_client is not None else boto3.client("s3", region_name=region)
    client.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(reference, default=str).encode(),
        ContentType="application/json",
    )
    return key


def load_training_reference(
    *, bucket: str, region: str = "us-east-1",
    key: str = FEATURE_DRIFT_REFERENCE_KEY, s3_client: Any | None = None,
) -> Optional[dict[str, Any]]:
    """Load the reference, or None when absent (first inference after this
    ships, before a training run has written it) — caller degrades to N/A."""
    import boto3
    from botocore.exceptions import ClientError

    client = s3_client if s3_client is not None else boto3.client("s3", region_name=region)
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(resp["Body"].read())


def _read_served_version(client: Any, bucket: str, manifest_key: str) -> Optional[str]:
    """``served_version`` from the live manifest (restamped by every
    promotion, ``model.registry.promote_to_champion``), or None."""
    resp = client.get_object(Bucket=bucket, Key=manifest_key)
    served = (json.loads(resp["Body"].read()) or {}).get("served_version")
    return str(served) if served else None


def load_served_model_reference(
    *, bucket: str, region: str = "us-east-1", s3_client: Any | None = None,
    manifest_key: str = SERVED_MANIFEST_KEY, registry_prefix: str = REGISTRY_PREFIX,
) -> Optional[dict[str, Any]]:
    """The drift reference that describes the model actually serving (I10063).

    Prefers ``{registry_prefix}{served_version}/feature_drift_reference.json``
    — the reference the served champion's own training run built, carried in
    its registry bundle since alpha-engine-config-I11478. Falls back to the
    live key only when the served bundle has none (a champion promoted before
    I11478) or cannot be read; the live key may describe a refresh run that
    was never promoted, and the returned dict says so.

    The returned reference carries ``reference_source`` (``served_bundle`` |
    ``live_key``), ``reference_version_id`` (the bundle read, else None) and
    ``served_version_id``, which ``compute_feature_drift_ks`` copies into the
    block. None when no reference exists anywhere (caller degrades to N/A).
    """
    import boto3

    client = s3_client if s3_client is not None else boto3.client("s3", region_name=region)
    served: Optional[str] = None
    try:
        served = _read_served_version(client, bucket, manifest_key)
    except Exception as exc:  # noqa: BLE001 — fall back to the live key, recorded below
        logger.warning(
            "[feature_drift] could not read served_version from s3://%s/%s (%s); "
            "using the live reference key (alpha-engine-config-I10063).",
            bucket, manifest_key, exc,
        )
    if served:
        bundle_key = f"{registry_prefix}{served}/{FEATURE_DRIFT_REFERENCE_FILENAME}"
        try:
            ref = load_training_reference(bucket=bucket, key=bundle_key, s3_client=client)
        except Exception as exc:  # noqa: BLE001 — fall back to the live key, recorded below
            logger.warning(
                "[feature_drift] could not read the served champion's reference "
                "s3://%s/%s (%s); using the live reference key.", bucket, bundle_key, exc,
            )
            ref = None
        if ref is not None:
            ref.update(
                reference_source=REFERENCE_SOURCE_SERVED_BUNDLE,
                reference_version_id=served, served_version_id=served,
            )
            return ref
        logger.info(
            "[feature_drift] served champion %s carries no %s in its bundle "
            "(promoted before alpha-engine-config-I11478); using the live key, "
            "which may describe an unpromoted refresh run.",
            served, FEATURE_DRIFT_REFERENCE_FILENAME,
        )
    ref = load_training_reference(bucket=bucket, s3_client=client)
    if ref is None:
        return None
    ref.update(
        reference_source=REFERENCE_SOURCE_LIVE_KEY,
        reference_version_id=None, served_version_id=served,
    )
    return ref
