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
    producers plus ``producer_parity`` / ``producer_mismatch``; the KS values
    themselves are unchanged. A ``mismatch`` means that feature's KS compares
    two producers, not two periods.
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
    ks_values = list(per_feature.values())
    ref_producers = dict(reference.get("producers") or {})
    serv_producers = dict(serving_producers or {})
    parity, mismatched = producer_parity(
        ref_producers, serv_producers, list(per_feature),
    )
    if parity == PRODUCER_PARITY_MISMATCH:
        logger.warning(
            "[feature_drift] producer mismatch on %s — reference (trained %s) "
            "used %s, serving used %s; those features' KS compares two "
            "producers, not two periods (alpha-engine-config-I10063).",
            mismatched, reference.get("trained_date"),
            {f: ref_producers.get(f) for f in mismatched},
            {f: serv_producers.get(f) for f in mismatched},
        )
    return {
        "max_ks": round(max(ks_values), 4),
        "mean_ks": round(float(np.mean(ks_values)), 4),
        "n_features": len(per_feature),
        "n_samples": n_used,
        "per_feature": dict(sorted(per_feature.items(), key=lambda kv: -kv[1])),
        "reference_trained_date": reference.get("trained_date"),
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
