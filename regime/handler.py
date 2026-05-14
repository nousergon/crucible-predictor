"""
regime/handler.py — Lambda entry point for the weekly RegimeSubstrate run.

Orchestrates the Saturday SF ``RegimeSubstrate`` state (inserted between
``RAGIngestion`` and ``Research`` per regime-v3-260514.md §5.4):

1. Fetch the historical macro feature DataFrame from data-side parquets
   (``regime/features.py``).
2. Fit a fresh HMM weekly on the 10y rolling history (always-refit
   design — no weights persistence in this PR; refitting is cheap
   relative to Lambda cold-start and keeps the artifact reproducible
   from the data side alone).
3. Assemble the substrate payload via ``regime/substrate.py``.
4. Publish the canonical-shape artifact to S3
   (``s3://alpha-engine-research/regime/{YYMMDDHHMM}.json`` +
   ``latest.json`` sidecar).

The macro economist agent (Stage C) reads ``latest.json`` and consumes
the substrate as a strong prior. The agent remains the final authority;
this Lambda is informational.

Why always-refit
----------------
HMM fit on ~520 weekly observations is seconds; Lambda cold-start +
boto3 init dominates. Persisting weights and partial-refitting adds
complexity without saving runtime, and obscures forensic replay (the
artifact is reproducible from the price-cache parquets alone). Weight
persistence may be revisited if HMM fit time grows materially or if
inter-week posterior continuity becomes a downstream requirement.

Invocation modes
----------------
- ``action == "produce"`` (default) — fit HMM, build substrate, write to S3.
- ``action == "dry_run"`` — fit HMM, build substrate, return payload
  without writing (for ad-hoc inspection + Stage A backfill validation).

Event payload (all optional with sensible defaults):

    {
      "action": "produce" | "dry_run",
      "bucket": "alpha-engine-research",
      "price_cache_prefix": "predictor/price_cache/",
      "regime_prefix": "regime",
      "lookback_weeks": 520,
      "hmm_feature_columns": ["spy_20d_return", "vix_level", "hy_oas_bps"]
    }

Lambda configuration (set at deploy time — separate alpha-engine-config PR):
- Runtime: container image (predictor's existing ECR image, separate CMD)
- Memory: 1024 MB (HMM fit is the bottleneck; ~520 × 3 features is tiny)
- Timeout: 300 seconds (fit ~10s + S3 IO ~5s + headroom)
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

# Ensure project root on sys.path for sibling imports (regime/, etc.).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from regime.features import (
    DEFAULT_PRICE_CACHE_PREFIX,
    current_features_from_history,
    fetch_macro_feature_history,
)
from regime.hmm import HMMRegimeClassifier
from regime.substrate import (
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_PREFIX,
    REGIME_FEATURES,
    build_regime_substrate,
    write_regime_substrate,
)


logger = logging.getLogger(__name__)


# HMM feature subset — must include PIN_FEATURE (``spy_20d_return``) and
# at minimum two more for K=3 to discover meaningful states. The default
# matches what's available in the FRED price-cache today; can be
# overridden via event payload as more features land.
DEFAULT_HMM_FEATURE_COLUMNS: tuple[str, ...] = (
    "spy_20d_return",
    "vix_level",
    "hy_oas_bps",
)


def produce_regime_substrate(
    *,
    s3_client: Any,
    bucket: str = DEFAULT_S3_BUCKET,
    price_cache_prefix: str = DEFAULT_PRICE_CACHE_PREFIX,
    regime_prefix: str = DEFAULT_S3_PREFIX,
    lookback_weeks: int = 520,
    hmm_feature_columns: tuple[str, ...] = DEFAULT_HMM_FEATURE_COLUMNS,
    write: bool = True,
) -> dict[str, Any]:
    """Run the substrate pipeline end-to-end.

    Returns the substrate payload dict. When ``write=True``, also
    publishes to S3 and includes ``artifact_key`` + ``latest_key`` in
    the return value alongside the payload.
    """
    history = fetch_macro_feature_history(
        s3_client=s3_client,
        bucket=bucket,
        prefix=price_cache_prefix,
        lookback_weeks=lookback_weeks,
    )
    if history.empty:
        raise RuntimeError(
            f"No macro feature history available at s3://{bucket}/{price_cache_prefix} — "
            f"upstream FRED ingestion likely failed."
        )

    # Drop any HMM-feature columns that came back all-NaN (e.g., breadth
    # before its historical series lands) before fitting.
    available_hmm_cols = tuple(
        c for c in hmm_feature_columns
        if c in history.columns and not history[c].dropna().empty
    )
    if "spy_20d_return" not in available_hmm_cols:
        raise RuntimeError(
            "spy_20d_return is required for HMM state pinning and was not "
            "available in fetched history."
        )

    hmm_input = history[list(available_hmm_cols)].dropna()
    hmm = HMMRegimeClassifier(n_states=3, random_state=42).fit(hmm_input, available_hmm_cols)

    current = current_features_from_history(history)

    payload = build_regime_substrate(
        feature_history=hmm_input,  # only the cols HMM saw at fit time
        current_features=current,
        hmm=hmm,
        fit_window_start=str(hmm_input.index.min().date()),
        fit_window_end=str(hmm_input.index.max().date()),
    )

    if not write:
        return {"payload": payload, "wrote": False}

    keys = write_regime_substrate(
        payload, s3_client=s3_client, bucket=bucket, prefix=regime_prefix,
    )
    return {"payload": payload, "wrote": True, **keys}


def lambda_handler(event: dict | None, context: Any) -> dict[str, Any]:
    """AWS Lambda entry point.

    Routes ``event["action"]`` to either ``produce`` (default, writes
    to S3) or ``dry_run`` (returns payload without writing).
    """
    event = event or {}
    action = event.get("action", "produce")

    import boto3
    s3_client = boto3.client("s3")

    kwargs = {
        "s3_client": s3_client,
        "bucket": event.get("bucket", DEFAULT_S3_BUCKET),
        "price_cache_prefix": event.get("price_cache_prefix", DEFAULT_PRICE_CACHE_PREFIX),
        "regime_prefix": event.get("regime_prefix", DEFAULT_S3_PREFIX),
        "lookback_weeks": int(event.get("lookback_weeks", 520)),
        "hmm_feature_columns": tuple(event.get("hmm_feature_columns", DEFAULT_HMM_FEATURE_COLUMNS)),
    }

    if action == "produce":
        result = produce_regime_substrate(**kwargs, write=True)
        return {
            "statusCode": 200,
            "action": action,
            "run_id": result["payload"]["run_id"],
            "artifact_key": result["artifact_key"],
            "latest_key": result["latest_key"],
            "hmm_argmax": result["payload"]["hmm"]["argmax"],
            "regime_change_signal": result["payload"]["bocpd"]["change_signal"],
        }
    elif action == "dry_run":
        result = produce_regime_substrate(**kwargs, write=False)
        return {
            "statusCode": 200,
            "action": action,
            "payload": result["payload"],
        }
    else:
        return {
            "statusCode": 400,
            "error": f"unknown action {action!r}; expected 'produce' or 'dry_run'",
        }
