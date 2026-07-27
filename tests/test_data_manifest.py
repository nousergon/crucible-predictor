"""Tests for data_manifest — dated S3 manifest writes."""

import json
from unittest.mock import MagicMock, patch

from data_manifest import write_data_manifest


def _captured_put_object():
    """Return (s3_mock, capture_dict) where capture_dict gets populated on put."""
    capture = {}
    s3 = MagicMock()

    def fake_put(**kwargs):
        capture.update(kwargs)
        return {}

    s3.put_object.side_effect = fake_put
    return s3, capture


def test_write_data_manifest_writes_dated_key():
    s3, capture = _captured_put_object()
    with patch("data_manifest.boto3.client", return_value=s3):
        write_data_manifest(
            bucket="bucket-x", module_name="data", run_date="2026-04-15",
            manifest={"tickers_processed": 920, "ok_ratio": 0.985},
        )

    assert capture["Key"] == "data_manifest/data/2026-04-15.json"
    payload = json.loads(capture["Body"].decode())
    assert payload["module"] == "data"
    assert payload["run_date"] == "2026-04-15"
    assert payload["tickers_processed"] == 920
    assert payload["ok_ratio"] == 0.985
    assert "written_at" in payload


def test_write_data_manifest_swallows_s3_failure():
    s3 = MagicMock()
    s3.put_object.side_effect = RuntimeError("S3 down")
    with patch("data_manifest.boto3.client", return_value=s3):
        write_data_manifest("bucket-x", "data", "2026-04-15", {})
