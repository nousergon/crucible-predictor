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
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# S3 key for the training-time reference (sibling of manifest.json /
# feature_list.json under the meta-weights prefix). Distinct from the
# legacy ``predictor/metrics/training_feature_stats.json`` the z-score
# drift_detector expects, to avoid feature-set entanglement.
FEATURE_DRIFT_REFERENCE_KEY = "predictor/weights/meta/feature_drift_reference.json"

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


def build_training_reference(
    matrix: np.ndarray, feature_names: Sequence[str], *, trained_date: Optional[str] = None,
) -> dict[str, Any]:
    """Build the training feature reference from the training META_FEATURES
    matrix (n_samples x n_features) and its column order ``feature_names``.

    Stores a thinned per-feature subsample (for the 2-sample KS at inference)
    plus mean/std (cheap context). Cross-sectional features only.
    """
    cols = _column_map(np.asarray(matrix, dtype=float), feature_names)
    samples = {f: _subsample(v) for f, v in cols.items() if v.size}
    return {
        "schema_version": 1,
        "trained_date": trained_date,
        "features": list(samples.keys()),
        "n_samples": int(min((len(s) for s in samples.values()), default=0)),
        "samples": samples,
        "mean": {f: round(float(np.mean(cols[f])), 6) for f in samples},
        "std": {f: round(float(np.std(cols[f])), 6) for f in samples},
    }


def compute_feature_drift_ks(
    matrix: np.ndarray, feature_names: Sequence[str], reference: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """2-sample KS per feature: today's inference distribution vs the training
    reference. Returns the ``feature_drift_ks`` block, or ``None`` when there's
    no usable reference / not enough inference samples (caller emits N/A).
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
    return {
        "max_ks": round(max(ks_values), 4),
        "mean_ks": round(float(np.mean(ks_values)), 4),
        "n_features": len(per_feature),
        "n_samples": n_used,
        "per_feature": dict(sorted(per_feature.items(), key=lambda kv: -kv[1])),
        "reference_trained_date": reference.get("trained_date"),
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
