"""Dated data-manifest writer (predictor-local).

Health enrichment writes live in ``nousergon_lib.health`` (config#1727).
This module keeps the dated ``data_manifest/`` PUT helper that is not part
of the shared health schema.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)


def write_data_manifest(
    bucket: str,
    module_name: str,
    run_date: str,
    manifest: dict,
) -> None:
    """Write a dated data manifest to S3 at data_manifest/{module}/{date}.json.

    Unlike health files (overwritten each run), manifests are dated and never
    overwritten — the collection of dated files IS the time series.
    """
    payload = {
        "module": module_name,
        "run_date": run_date,
        "written_at": datetime.now(timezone.utc).isoformat(),
        **manifest,
    }
    try:
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=f"data_manifest/{module_name}/{run_date}.json",
            Body=json.dumps(payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("Data manifest written: %s/%s", module_name, run_date)
    except Exception as e:
        logger.warning("Failed to write data manifest for %s: %s", module_name, e)
