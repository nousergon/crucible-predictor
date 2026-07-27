"""config#2891 — consumer-side staleness assertion for config/predictor_params.json.

Verifies the WARN-only staleness signal fires when the S3 pointer's
LastModified is older than the 2-weekly-cycle threshold, and stays silent
when fresh or absent — this is a defense-in-depth signal (config#1724), it
must never raise or block param loading.
"""
from __future__ import annotations

import json
import logging
from datetime import timezone, datetime, timedelta
from io import BytesIO
from unittest.mock import MagicMock

import inference.stages.write_output as wo


def _make_s3_response(body: dict, last_modified) -> dict:
    return {"Body": BytesIO(json.dumps(body).encode()), "LastModified": last_modified}


def test_check_predictor_params_staleness_warns_past_threshold(caplog):
    old = datetime.now(timezone.utc) - timedelta(
        hours=wo._PREDICTOR_PARAMS_STALE_HOURS + 1
    )
    with caplog.at_level(logging.ERROR):
        wo._check_predictor_params_staleness(old)
    assert any("STALE config/predictor_params.json" in r.message for r in caplog.records)


def test_check_predictor_params_staleness_silent_when_fresh(caplog):
    fresh = datetime.now(timezone.utc) - timedelta(hours=1)
    with caplog.at_level(logging.ERROR):
        wo._check_predictor_params_staleness(fresh)
    assert not any("STALE" in r.message for r in caplog.records)


def test_check_predictor_params_staleness_silent_when_absent(caplog):
    with caplog.at_level(logging.ERROR):
        wo._check_predictor_params_staleness(None)
    assert not any("STALE" in r.message for r in caplog.records)


def test_load_predictor_params_from_s3_warns_on_stale_write(monkeypatch, caplog, tmp_path):
    """End-to-end: a simulated stale write on the real S3 read path logs the WARN."""
    wo._predictor_params_loaded = False
    wo._predictor_params_cache = None
    # Redirect the fault-tolerance local cache write away from the shared
    # default /tmp path — this repo has no existing test exercising the
    # real S3-success branch, and leaving the default path would leak a
    # cached veto_confidence into unrelated tests (e.g. test_veto_threshold.py's
    # "no S3, no cache" case) via the on-disk fallback file.
    monkeypatch.setattr(wo, "_PREDICTOR_PARAMS_CACHE_PATH", tmp_path / "predictor_params_cache.json")
    stale_time = datetime.now(timezone.utc) - timedelta(days=30)
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = _make_s3_response(
        {"veto_confidence": 0.3}, stale_time
    )
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3
    monkeypatch.setitem(__import__("sys").modules, "boto3", mock_boto3)
    with caplog.at_level(logging.ERROR):
        wo._load_predictor_params_from_s3("alpha-engine-research")
    assert any("STALE config/predictor_params.json" in r.message for r in caplog.records)
    wo._predictor_params_loaded = False
    wo._predictor_params_cache = None
