"""Tests for inference/stages/load_universe.py — local file loading."""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inference.stages.load_universe import (
    MEMBERSHIP_MAX_AGE_DAYS,
    UniverseResolutionError,
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


# ── 2026-04-27: main pass auto-includes signals.buy_candidates ──────────────


def _mock_s3_with_keys(payloads_by_key: dict[str, dict], eod_csv: str | None = None):
    """Build a MagicMock boto3 s3 client whose get_object returns the given
    JSON payloads keyed by S3 Key. Missing keys raise NoSuchKey ClientError.

    ``eod_csv`` serves trades/eod_pnl.csv (raw CSV text) for the holdings union.
    """
    from botocore.exceptions import ClientError

    def get_object(Bucket, Key):
        if Key == "trades/eod_pnl.csv":
            if eod_csv is None:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            return {"Body": io.BytesIO(eod_csv.encode("utf-8"))}
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


def _membership(
    cut_tickers, run_date: str = "2026-07-24", generated_at: str | None = "2026-07-24T12:49:30+00:00"
) -> dict:
    """A universe_membership artifact whose predictor cut is ``cut_tickers``."""
    return {
        "schema_version": 1,
        "run_date": run_date,
        "generated_at": generated_at,
        "predictor_universe_cut": "scanner_candidates",
        "cuts": {
            "scanner_candidates": {
                "basis": "scanner_gate",
                "size": len(cut_tickers),
                "tickers": sorted(cut_tickers),
                "source": f"candidates/{run_date}/candidates.json::scanner_tickers",
            },
        },
        "ranks": {t: {"attractiveness_rank": i + 1, "attractiveness_score": 90.0 - i}
                  for i, t in enumerate(sorted(cut_tickers))},
    }


def _eod_csv(positions: dict) -> str:
    """A minimal trades/eod_pnl.csv whose last row carries ``positions``."""
    snapshot = json.dumps(positions).replace('"', '""')
    return (
        "date,portfolio_nav,positions_snapshot\n"
        f'2026-07-24,100000,"{snapshot}"\n'
    )


class TestMembershipChokepointGuard:
    """alpha-engine-config-I6495 — structural guard against the #422 shape.

    A groomer "signals helper consolidation" deleted membership resolution from
    ``load_universe.py`` while leaving tests that still asserted
    population-first. These string/chokepoint asserts make that unconstructible:
    deleting ``universe_membership`` / ``MEMBERSHIP_MAX_AGE_DAYS`` /
    ``predictor_universe_cut`` fails CI before a silent frozen-universe regress.
    """

    def test_source_file_keeps_membership_chokepoint_strings(self):
        src = Path("inference/stages/load_universe.py").read_text()
        for needle in (
            "universe_membership",
            "MEMBERSHIP_MAX_AGE_DAYS",
            "predictor_universe_cut",
            "resolve_universe_from_membership",
            "UniverseResolutionError",
        ):
            assert needle in src, (
                f"load_universe.py lost membership chokepoint {needle!r} — "
                "restore from crucible-predictor#429/#431 (config-I6495)"
            )

    def test_membership_max_age_is_positive_int(self):
        assert isinstance(MEMBERSHIP_MAX_AGE_DAYS, int)
        assert MEMBERSHIP_MAX_AGE_DAYS == 10


class TestLoadWatchlistAutoResolvesMembership:
    """The universe resolves from the versioned membership artifact
    (alpha-engine-config-I4818), NOT from population/latest.json.

    Regression being pinned: population/latest.json's producer was retired with
    the multi-agent research graph. The read kept SUCCEEDING against a file that
    had stopped changing, so the predictor scored a frozen 2026-07-10 list of 25
    names for three weekly cycles with every daily run green.
    """

    def test_universe_is_the_named_predictor_cut(self):
        s3 = _mock_s3_with_keys({
            "universe_membership/latest.json": _membership(["AAPL", "MSFT", "NVDA"]),
        })
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-27",
            )
        assert set(tickers) == {"AAPL", "MSFT", "NVDA"}
        # The source label is the CUT'S NAME, not a fixed literal — a
        # hardcoded label went stale the moment predictor_universe_cut
        # moved to attractiveness_top_20 (alpha-engine-config-I4983).
        assert sources["AAPL"] == "scanner_candidates"

    def test_population_latest_is_never_read(self):
        # The whole point of I4818: even when a (stale) population artifact is
        # sitting right there, it must not contribute a single ticker.
        s3 = _mock_s3_with_keys({
            "population/latest.json": {"population": [{"ticker": "STALE"}]},
            "universe_membership/latest.json": _membership(["AAPL"]),
        })
        with patch("boto3.client", return_value=s3):
            tickers, _, _ = load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")
        assert "STALE" not in tickers
        assert set(tickers) == {"AAPL"}
        for call in s3.get_object.call_args_list:
            assert call.kwargs.get("Key") != "population/latest.json"

    def test_holdings_are_unioned_so_held_names_are_never_scored_blind(self):
        s3 = _mock_s3_with_keys(
            {"universe_membership/latest.json": _membership(["AAPL"])},
            eod_csv=_eod_csv({"HELD": {"shares": 10}, "AAPL": {"shares": 5}}),
        )
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-27",
            )
        assert set(tickers) == {"AAPL", "HELD"}
        assert sources["HELD"] == "held"
        assert sources["AAPL"] == "both"      # in the cut AND held

    def test_unreadable_holdings_narrow_the_universe_but_do_not_fail(self):
        # Holdings only WIDEN the universe, so an unreadable book yields a
        # narrower-but-valid run. The load-bearing part (the cut) still came
        # from the membership artifact.
        s3 = _mock_s3_with_keys({
            "universe_membership/latest.json": _membership(["AAPL"]),
        })
        with patch("boto3.client", return_value=s3):
            tickers, _, _ = load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")
        assert tickers == ["AAPL"]

    def test_buy_candidates_are_unioned(self):
        s3 = _mock_s3_with_keys({
            "universe_membership/latest.json": _membership(["AAPL", "MSFT"]),
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
        assert sources["AAPL"] == "both"

    def test_buy_candidates_as_bare_strings(self):
        s3 = _mock_s3_with_keys({
            "universe_membership/latest.json": _membership(["AAPL"]),
            "signals/2026-07-27/signals.json": {"buy_candidates": ["abt", "DVA"]},
        })
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-27",
            )
        assert {"ABT", "DVA"} <= set(tickers)
        assert sources["ABT"] == "buy_candidate"

    def test_macro_context_comes_from_signals(self):
        # signals.json is now the SOLE macro producer the predictor reads —
        # the population overlay that reconciled two producers is gone.
        s3 = _mock_s3_with_keys({
            "universe_membership/latest.json": _membership(["AAPL"]),
            "signals/2026-07-27/signals.json": {
                "market_regime": "bull",
                "sector_modifiers": {"Technology": 1.1},
            },
        })
        with patch("boto3.client", return_value=s3):
            _, _, data = load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")
        assert data["market_regime"] == "bull"
        assert data["sector_modifiers"] == {"Technology": 1.1}
        # The resolved cycle is stamped on the payload so the morning brief and
        # any postmortem can see WHICH membership produced the day's universe.
        assert data["universe_membership_run_date"] == "2026-07-24"

    def test_missing_signals_leaves_universe_intact(self):
        # No signals anywhere: the brief loses macro context, but the universe
        # is membership-sourced and therefore unaffected.
        s3 = _mock_s3_with_keys({
            "universe_membership/latest.json": _membership(["AAPL", "MSFT"]),
        })
        with patch("boto3.client", return_value=s3):
            tickers, _, data = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-27",
            )
        assert set(tickers) == {"AAPL", "MSFT"}
        assert data["universe_membership_run_date"] == "2026-07-24"


class TestMembershipProvenanceIsStamped:
    """alpha-engine-config-I6786. ``run_date`` names the CYCLE; it does not name
    the INSTANCE, and both S3 keys in the resolution chain are pointers that the
    postclose-chained exercise Scanner rewrites for the same trading day.

    Measured 2026-08-07: the surviving ``universe_membership/2026-08-07/
    membership.json`` carried a ``generated_at`` 17 hours after the predictions
    it fed, and four names stamped ``attractiveness_top_20`` in those
    predictions were absent from it. Re-reading the key alone cannot reproduce
    what was scored; ``generated_at`` can."""

    def test_stamps_generated_at_and_the_resolved_key(self):
        s3 = _mock_s3_with_keys({
            "universe_membership/latest.json": _membership(["AAPL", "MSFT"]),
        })
        with patch("boto3.client", return_value=s3):
            _, _, data = load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")
        assert data["universe_membership_generated_at"] == "2026-07-24T12:49:30+00:00"
        assert data["universe_membership_key"] == "universe_membership/latest.json"

    def test_the_key_names_the_dated_artifact_when_the_pointer_is_absent(self):
        # Resolution falls back to the dated walk; provenance must follow it
        # rather than reporting the pointer that did not answer.
        s3 = _mock_s3_with_keys({
            "universe_membership/2026-07-24/membership.json": _membership(["AAPL"]),
        })
        with patch("boto3.client", return_value=s3):
            _, _, data = load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")
        assert data["universe_membership_key"] == "universe_membership/2026-07-24/membership.json"

    def test_absent_generated_at_is_recorded_as_none_not_invented(self):
        # A producer that predates the field must yield null, not a substituted
        # run_date — a fabricated instance stamp is worse than a missing one.
        s3 = _mock_s3_with_keys({
            "universe_membership/latest.json": _membership(["AAPL"], generated_at=None),
        })
        with patch("boto3.client", return_value=s3):
            _, _, data = load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")
        assert data["universe_membership_generated_at"] is None
        assert data["universe_membership_run_date"] == "2026-07-24"

    def test_future_stamped_membership_logs_error_but_still_scores(self, caplog):
        # Provenance check, not a trading gate: halting inference on a clock
        # artifact would cost a trading day to protect a metadata field.
        s3 = _mock_s3_with_keys({
            "universe_membership/latest.json": _membership(
                ["AAPL"], generated_at="2026-08-01T00:00:00+00:00"
            ),
        })
        with caplog.at_level("ERROR"):
            with patch("boto3.client", return_value=s3):
                tickers, _, data = load_watchlist(
                    "auto", s3_bucket="b", date_str="2026-07-27",
                )
        assert tickers == ["AAPL"]
        assert data["universe_membership_generated_at"] == "2026-08-01T00:00:00+00:00"
        assert any("AFTER this run's date" in r.message for r in caplog.records)

    def test_unparseable_generated_at_is_kept_verbatim(self):
        s3 = _mock_s3_with_keys({
            "universe_membership/latest.json": _membership(["AAPL"], generated_at="not-a-date"),
        })
        with patch("boto3.client", return_value=s3):
            _, _, data = load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")
        assert data["universe_membership_generated_at"] == "not-a-date"


class TestLoadWatchlistFailsOnStaleOrMissingMembership:
    """Staleness must be fatal. This is the actual I4818 defect: the artifact
    resolved fine, it just described a dead cycle — invisible to any check that
    only asks whether the read succeeded."""

    def test_stale_membership_raises(self):
        s3 = _mock_s3_with_keys({
            "universe_membership/latest.json": _membership(["AAPL"], run_date="2026-07-10"),
        })
        with patch("boto3.client", return_value=s3):
            with pytest.raises(UniverseResolutionError, match="no universe membership"):
                load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")

    def test_missing_membership_raises(self):
        s3 = _mock_s3_with_keys({})
        with patch("boto3.client", return_value=s3):
            with pytest.raises(UniverseResolutionError, match="no universe membership"):
                load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")

    def test_empty_named_cut_raises_rather_than_scoring_nothing(self):
        empty = _membership([])
        s3 = _mock_s3_with_keys({"universe_membership/latest.json": empty})
        with patch("boto3.client", return_value=s3):
            with pytest.raises(UniverseResolutionError, match="empty or absent"):
                load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")

    def test_cut_named_but_absent_raises(self):
        broken = _membership(["AAPL"])
        broken["predictor_universe_cut"] = "a_cut_that_does_not_exist"
        s3 = _mock_s3_with_keys({"universe_membership/latest.json": broken})
        with patch("boto3.client", return_value=s3):
            with pytest.raises(UniverseResolutionError, match="empty or absent"):
                load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")

    def test_dated_key_used_when_latest_pointer_is_absent(self):
        # Resolution must not depend on a single mutable pointer — a latest.json
        # that was never written (as during this artifact's own rollout) must
        # degrade to the most recent dated cycle, not to an outage.
        s3 = _mock_s3_with_keys({
            "universe_membership/2026-07-24/membership.json": _membership(["AAPL"]),
        })
        with patch("boto3.client", return_value=s3):
            tickers, _, _ = load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")
        assert tickers == ["AAPL"]

    def test_freshest_wins_when_pointer_lags_dated_key(self):
        s3 = _mock_s3_with_keys({
            "universe_membership/latest.json": _membership(["OLD"], run_date="2026-07-20"),
            "universe_membership/2026-07-24/membership.json": _membership(["NEW"]),
        })
        with patch("boto3.client", return_value=s3):
            tickers, _, _ = load_watchlist("auto", s3_bucket="b", date_str="2026-07-27")
        assert tickers == ["NEW"]


class TestGetUniverseTickersFallback:
    """Regression: get_universe_tickers used to hardcode
    `signals/{date}/signals.json` and return `{}` signals_data on miss.
    Post-fix it walks the same fallback chain everywhere else in this
    module uses."""

    def test_falls_back_to_signals_latest_when_today_missing(self):
        """No dated key → uses signals/latest.json universe + signals_data."""
        # dict keyed by ticker, matching the frozen contract shape
        # (nousergon_lib/contracts/signals.schema.json) — fixed 2026-08-18
        # (alpha-engine-config-I7627), previously a list-of-dicts fixture
        # that restated the consumer's own (wrong) assumption.
        signals_latest = {
            "signals": {"AAPL": {"ticker": "AAPL"}, "MSFT": {"ticker": "MSFT"}},
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


class TestSourceLabelTracksTheCut:
    """alpha-engine-config-I4983 — the annotation must name the cut that
    actually resolved.

    Before this, `resolve_universe_from_membership` hardcoded
    `"scanner_candidate"`. When the champion moved to `attractiveness_top_20`,
    every attractiveness-selected name was still annotated as a scanner
    candidate in `predictions/{date}.json` — an artifact stating something
    false about its own provenance. Fails against the pre-fix implementation.
    """

    def test_label_is_the_resolved_cut_name(self):
        membership = {
            "run_date": "2026-07-27",
            "predictor_universe_cut": "attractiveness_top_20",
            "cuts": {
                "attractiveness_top_20": {
                    "basis": "attractiveness_rank",
                    "size": 2,
                    "tickers": ["AAPL", "MSFT"],
                },
                "scanner_candidates": {
                    "basis": "scanner_gate",
                    "size": 1,
                    "tickers": ["ZZZZ"],
                },
            },
        }
        s3 = _mock_s3_with_keys({"universe_membership/latest.json": membership})
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-07-27",
            )

        assert set(tickers) == {"AAPL", "MSFT"}
        assert sources["AAPL"] == "attractiveness_top_20"
        assert sources["MSFT"] == "attractiveness_top_20"
        assert "scanner_candidate" not in set(sources.values()), (
            "the resolved cut was attractiveness_top_20, but a name is still "
            "annotated as a scanner candidate"
        )
