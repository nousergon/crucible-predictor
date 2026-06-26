"""Tests for inference.stages.fetch_alt_data — alt-data sources fail independently.

Options (O12) retired here: the predictor's options path was orphaned — the
inference feature matrix is built exclusively from ArcticDB
(``run_inference._load_precomputed_features_from_arcticdb``) and never read
``ctx.options_all``. The dead ``data/options_fetcher.py`` + the O12 fetch block
were removed; the stage now fetches earnings, revisions, and fundamentals only.
"""

import json
import sys
from unittest.mock import MagicMock

import pytest

from inference.pipeline import PipelineContext
from inference.stages.fetch_alt_data import run


def _ctx(tickers=None, date_str="2026-04-15", bucket="bucket-x"):
    ctx = PipelineContext()
    ctx.date_str = date_str
    ctx.bucket = bucket
    ctx.tickers = tickers or ["AAPL", "MSFT"]
    return ctx


@pytest.fixture
def fake_earnings(monkeypatch):
    earnings = MagicMock(name="earnings_fetcher")
    earnings.fetch_earnings_data = MagicMock(return_value={"AAPL": {"x": 1}})
    earnings.cache_earnings_to_s3 = MagicMock()
    earnings.fetch_revision_history = MagicMock(return_value={"AAPL": {"r": 1}})
    monkeypatch.setitem(sys.modules, "data.earnings_fetcher", earnings)
    return earnings


@pytest.fixture
def fake_boto(monkeypatch):
    s3 = MagicMock()
    body = MagicMock()
    body.read.return_value = json.dumps({"AAPL": {"f": 1}, "MSFT": {"f": 2}}).encode()
    s3.get_object.return_value = {"Body": body}
    fake_boto3 = MagicMock()
    fake_boto3.client.return_value = s3
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    return s3


# ── Happy paths ─────────────────────────────────────────────────────────────


def test_run_populates_three_sources(fake_earnings, fake_boto):
    ctx = _ctx()
    run(ctx)
    assert ctx.earnings_all == {"AAPL": {"x": 1}}
    assert ctx.revision_all == {"AAPL": {"r": 1}}
    assert ctx.fundamental_all == {"AAPL": {"f": 1}, "MSFT": {"f": 2}}
    fake_earnings.cache_earnings_to_s3.assert_called_once()


def test_run_does_not_import_retired_options_module(fake_earnings, fake_boto, monkeypatch):
    """Regression guard: the options path is retired. The stage must not import
    ``data.options_fetcher`` (and PipelineContext must not carry ``options_all``)."""
    monkeypatch.delitem(sys.modules, "data.options_fetcher", raising=False)
    ctx = _ctx()
    run(ctx)
    assert "data.options_fetcher" not in sys.modules
    assert not hasattr(ctx, "options_all")


# ── Per-source independent failure ──────────────────────────────────────────


def test_run_earnings_fetch_failure_logged_others_continue(fake_earnings, fake_boto, caplog):
    fake_earnings.fetch_earnings_data.side_effect = RuntimeError("earnings API down")
    ctx = _ctx()
    with caplog.at_level("WARNING"):
        run(ctx)
    assert ctx.earnings_all == {}  # failure → default
    assert ctx.fundamental_all == {"AAPL": {"f": 1}, "MSFT": {"f": 2}}  # others succeed
    assert any("Earnings data fetch failed" in r.message for r in caplog.records)


def test_run_revision_fetch_failure_logged(fake_earnings, fake_boto, caplog):
    fake_earnings.fetch_revision_history.side_effect = RuntimeError("revision API down")
    ctx = _ctx()
    with caplog.at_level("WARNING"):
        run(ctx)
    assert ctx.revision_all == {}
    assert any("Revision data fetch failed" in r.message for r in caplog.records)


def test_run_fundamentals_today_key_path(fake_earnings, fake_boto):
    """When archive/fundamentals/{date}.json exists, it wins immediately
    without scanning."""
    ctx = _ctx()
    run(ctx)
    fake_boto.list_objects_v2.assert_not_called()


def test_run_fundamentals_scan_fallback(fake_earnings, fake_boto):
    """If today's key is missing, scan list_objects_v2 for most recent."""
    # First get_object call (today) raises; second succeeds (most recent scan).
    def get_object_side_effect(Bucket, Key):
        if Key == "archive/fundamentals/2026-04-15.json":
            raise RuntimeError("not found")
        body = MagicMock()
        body.read.return_value = json.dumps({"AAPL": {"f": "from-scan"}}).encode()
        return {"Body": body}

    fake_boto.get_object.side_effect = get_object_side_effect
    fake_boto.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "archive/fundamentals/2026-04-10.json"},
            {"Key": "archive/fundamentals/2026-04-08.json"},
        ],
    }
    ctx = _ctx()
    run(ctx)
    # Should pick the most recent key (sorted reverse → 2026-04-10)
    assert ctx.fundamental_all == {"AAPL": {"f": "from-scan"}}


def test_run_fundamentals_all_failure_logged(fake_earnings, fake_boto, caplog):
    fake_boto.get_object.side_effect = RuntimeError("S3 down")
    fake_boto.list_objects_v2.side_effect = RuntimeError("S3 down")
    ctx = _ctx()
    with caplog.at_level("WARNING"):
        run(ctx)
    assert ctx.fundamental_all == {}
    assert any("Fundamentals: S3 load failed" in r.message for r in caplog.records)


def test_run_all_sources_fail_logs_error(fake_earnings, fake_boto, caplog):
    fake_earnings.fetch_earnings_data.side_effect = RuntimeError("down")
    fake_earnings.fetch_revision_history.side_effect = RuntimeError("down")
    fake_boto.get_object.side_effect = RuntimeError("S3 down")
    fake_boto.list_objects_v2.return_value = {"Contents": []}

    ctx = _ctx()
    with caplog.at_level("WARNING"):
        run(ctx)

    assert any(
        "ALL alternative data sources failed" in r.message for r in caplog.records
    )
    assert ctx.earnings_all == {}
    assert ctx.revision_all == {}
    assert ctx.fundamental_all == {}
