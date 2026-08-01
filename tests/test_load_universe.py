"""Tests for inference/stages/load_universe.py — local file loading."""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inference.stages.load_universe import (
    get_universe_tickers,
    load_watchlist,
)


class TestLoadWatchlistLocal:
    def test_signals_file(self):
        data = {
            "date": "2026-04-08",
            "universe": [
                {"ticker": "AAPL", "score": 82},
                {"ticker": "MSFT", "score": 75},
                {"ticker": "NVDA", "score": 90},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            f.flush()
            tickers, sources, raw = load_watchlist(f.name)
            assert len(tickers) == 3
            assert "AAPL" in tickers
            assert sources["AAPL"] == "tracked"

    def test_population_file(self):
        data = {
            "population": [
                {"ticker": "GOOG", "sector": "Technology"},
                {"ticker": "AMZN", "sector": "Consumer"},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            f.flush()
            tickers, sources, raw = load_watchlist(f.name)
            assert len(tickers) == 2
            assert sources["GOOG"] == "population"

    def test_empty_universe(self):
        data = {"date": "2026-04-08", "universe": []}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            f.flush()
            tickers, sources, raw = load_watchlist(f.name)
            assert tickers == []

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_watchlist("/nonexistent/signals.json")

    def test_tickers_uppercased(self):
        data = {"universe": [{"ticker": "aapl"}, {"ticker": "msft"}]}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            f.flush()
            tickers, _, _ = load_watchlist(f.name)
            assert all(t.isupper() for t in tickers)

    def test_tickers_sorted(self):
        data = {"universe": [{"ticker": "NVDA"}, {"ticker": "AAPL"}, {"ticker": "MSFT"}]}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            f.flush()
            tickers, _, _ = load_watchlist(f.name)
            assert tickers == sorted(tickers)

    def test_auto_without_bucket_raises(self):
        with pytest.raises(ValueError, match="s3_bucket"):
            load_watchlist("auto")


# ── shared S3 mock ───────────────────────────────────────────────────────────


def _mock_s3_with_keys(payloads_by_key: dict[str, dict]):
    """Build a MagicMock boto3 s3 client whose get_object returns the given
    JSON payloads keyed by S3 Key. Missing keys raise NoSuchKey ClientError.
    """
    from botocore.exceptions import ClientError

    def get_object(Bucket, Key):
        payload = payloads_by_key.get(Key)
        if payload is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )
        body = json.dumps(payload).encode("utf-8")
        return {"Body": io.BytesIO(body)}

    s3 = MagicMock()
    s3.get_object.side_effect = get_object
    return s3


# ── auto path: population-first resolution (alpha-engine-config#3284) ────────


class TestLoadWatchlistAutoResolvesPopulation:
    """The universe resolves from population/latest.json first, with
    signals.json providing buy_candidates union + macro field overlay.
    Falls back to the signals chain when population is unavailable.
    """

    def test_population_is_primary_source(self):
        s3 = _mock_s3_with_keys({
            "population/latest.json": {
                "population": [
                    {"ticker": "AAPL", "sector": "Technology"},
                    {"ticker": "MSFT", "sector": "Technology"},
                    {"ticker": "NVDA", "sector": "Technology"},
                ],
            },
        })
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-27",
            )
        assert set(tickers) == {"AAPL", "MSFT", "NVDA"}
        assert sources["AAPL"] == "population"
        assert sources["MSFT"] == "population"

    def test_buy_candidates_are_unioned(self):
        s3 = _mock_s3_with_keys({
            "population/latest.json": {
                "population": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
            },
            "signals/2026-07-27/signals.json": {
                "buy_candidates": [{"ticker": "ABT"}, {"ticker": "AAPL"}],
            },
        })
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-27",
            )
        assert set(tickers) == {"AAPL", "MSFT", "ABT"}
        assert sources["ABT"] == "buy_candidate"
        # AAPL is in both population and buy_candidates
        assert sources["AAPL"] == "both"

    def test_buy_candidates_as_bare_strings(self):
        s3 = _mock_s3_with_keys({
            "population/latest.json": {
                "population": [{"ticker": "AAPL"}],
            },
            "signals/2026-07-27/signals.json": {
                "buy_candidates": ["abt", "DVA"],
            },
        })
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-27",
            )
        assert {"ABT", "DVA"} <= set(tickers)
        assert sources["ABT"] == "buy_candidate"

    def test_macro_context_overlaid_from_signals(self):
        # signals.json is the canonical source for market_regime /
        # sector_modifiers — the population writer may carry drift.
        s3 = _mock_s3_with_keys({
            "population/latest.json": {
                "population": [{"ticker": "AAPL"}],
                "market_regime": "stale_regime",
            },
            "signals/2026-07-27/signals.json": {
                "market_regime": "bull",
                "sector_modifiers": {"Technology": 1.1},
            },
        })
        with patch("boto3.client", return_value=s3):
            _, _, data = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-27",
            )
        # signals value wins over population's stale value
        assert data["market_regime"] == "bull"
        assert data["sector_modifiers"] == {"Technology": 1.1}

    def test_missing_signals_leaves_population_intact(self):
        # No signals anywhere: the brief loses macro context, but the
        # universe is population-sourced and therefore unaffected.
        s3 = _mock_s3_with_keys({
            "population/latest.json": {
                "population": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
            },
        })
        with patch("boto3.client", return_value=s3):
            tickers, _, data = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-27",
            )
        assert set(tickers) == {"AAPL", "MSFT"}

    def test_falls_back_to_signals_when_population_missing(self):
        # population/latest.json doesn't exist → falls back to signals chain.
        s3 = _mock_s3_with_keys({
            "signals/2026-07-27/signals.json": {
                "universe": [
                    {"ticker": "AAPL", "score": 85},
                    {"ticker": "MSFT", "score": 78},
                ],
            },
        })
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-27",
            )
        assert set(tickers) == {"AAPL", "MSFT"}
        assert sources["AAPL"] == "tracked"

    def test_signals_fallback_walks_back_through_weekdays(self):
        # Today's signals key doesn't exist, but Friday's does.
        # 2026-07-28 is Tuesday → fallback walks: Tue, Mon, Fri.
        s3 = _mock_s3_with_keys({
            "signals/2026-07-24/signals.json": {  # Friday
                "universe": [{"ticker": "AAPL"}],
            },
        })
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-28",
            )
        assert tickers == ["AAPL"]


class TestLoadWatchlistFailsWhenNoSourceAvailable:
    """When neither population nor signals resolves, the run must fail
    loudly rather than scoring a silently-wrong universe."""

    def test_raises_when_both_population_and_signals_missing(self):
        s3 = _mock_s3_with_keys({})
        with patch("boto3.client", return_value=s3):
            with pytest.raises(RuntimeError, match="No signals found"):
                load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")


class TestSourceLabelIsPopulation:
    """alpha-engine-config#3284 — when population/latest.json resolves,
    the source label is ``"population"``, matching the artifact read."""

    def test_source_label_is_population(self):
        s3 = _mock_s3_with_keys({
            "population/latest.json": {
                "population": [
                    {"ticker": "AAPL"},
                    {"ticker": "MSFT"},
                ],
            },
        })
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-27",
            )
        assert set(tickers) == {"AAPL", "MSFT"}
        assert sources["AAPL"] == "population"
        assert sources["MSFT"] == "population"


class TestGetUniverseTickersFallback:
    """Regression: get_universe_tickers used to hardcode
    `signals/{date}/signals.json` and return `{}` signals_data on miss.
    Post-fix it walks the same fallback chain everywhere else in this
    module uses."""

    def test_falls_back_to_signals_latest_when_today_missing(self):
        """No dated key → uses signals/latest.json universe + signals_data."""
        signals_latest = {
            "signals": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
            "market_regime": "bull",
        }
        s3 = _mock_s3_with_keys({"signals/latest.json": signals_latest})
        with patch("boto3.client", return_value=s3):
            tickers, data = get_universe_tickers("b", date_str="2026-05-11")
        assert set(tickers) == {"AAPL", "MSFT"}
        # Critical regression: signals_data is NOT empty when only the
        # dated key is missing. Pre-fix this returned ({fallback_tickers}, {}).
        assert data["market_regime"] == "bull"

    def test_fallback_universe_when_no_signals_anywhere(self):
        """All signals keys missing → falls back to _FALLBACK_TICKERS,
        signals_data is `{}` (the only place where empty data is correct)."""
        from inference.stages.load_universe import _FALLBACK_TICKERS
        s3 = _mock_s3_with_keys({})
        with patch("boto3.client", return_value=s3):
            tickers, data = get_universe_tickers("b", date_str="2026-05-11")
        assert tickers == _FALLBACK_TICKERS
        assert data == {}
