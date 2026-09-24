"""alpha-engine-config-I11483: the research-free producer scores the scanner's
evaluated universe plus the same run's membership cuts, not only the quant-pass
pool.

The pinned ``attractiveness_top_60`` shared 8 of 60 names with the quant-pass
pool on 2026-09-23 (the live scanner champion re-marks ``quant_filter_pass``),
so ``predictor_from_60`` failed its 90% join floor. These tests pin:

* the population rule, and that the pinned cut is fully covered;
* the membership doc is read at the CANDIDATES date, never ``latest.json``;
* adding cut-only names never changes any other name's prediction;
* a missing membership doc is recorded, never silent, and never fatal.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference.research_free_inference import (  # noqa: E402
    _cuts_by_ticker,
    load_membership_cuts,
    run_research_free_inference,
    scored_population,
)

CANDIDATES_DATE = "2026-07-10"
MEMBERSHIP_KEY = f"universe_membership/{CANDIDATES_DATE}/membership.json"
CANDIDATES_KEY = f"candidates/{CANDIDATES_DATE}/candidates.json"


class _FakeMetaModel:
    is_fitted = True
    _feature_names = [
        "momentum_score", "residual_momentum_score", "expected_move",
        "research_calibrator_prob", "research_composite_score",
        "research_conviction", "sector_macro_modifier",
    ]

    def predict_single(self, features: dict) -> float:
        return float(sum(features.values()))


class _RankSensitiveVolScorer:
    """Its output depends on the cross-sectional rank, so any change to the
    cross-section shows up in ``expected_move``."""
    feature_names = ["realized_vol_20d"]

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0]


def _prices(tickers, n=300):
    idx = pd.bdate_range("2025-01-01", periods=n)
    out = {}
    for i, t in enumerate(tickers):
        rng = np.random.default_rng(abs(hash(t)) % (2**31))
        close = 50 + i * 10 + np.cumsum(rng.normal(0, 1, n))
        out[t] = pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n),
            "momentum_5d": rng.normal(0, 0.02, n),
            "momentum_20d": rng.normal(0, 0.05, n),
            "price_vs_ma50": rng.normal(0, 0.03, n),
            "rsi_14": rng.uniform(30, 70, n),
            "realized_vol_20d": rng.uniform(0.1, 0.4, n),
        }, index=idx)
    return out


def _no_such_key(key):
    return ClientError({"Error": {"Code": "NoSuchKey", "Message": key}}, "GetObject")


def _s3(objects: dict):
    """An S3 double that serves exactly ``objects`` and 404s everything else,
    and records every key read and written."""
    s3 = MagicMock()
    s3.reads = []
    s3.written = {}

    def _get(Bucket, Key):  # noqa: N803 - boto3 kwarg names
        s3.reads.append(Key)
        if Key not in objects:
            raise _no_such_key(Key)
        body = MagicMock()
        body.read.return_value = json.dumps(objects[Key]).encode()
        return {"Body": body}

    def _put(**kwargs):
        s3.written[kwargs["Key"]] = json.loads(kwargs["Body"])

    s3.get_object.side_effect = _get
    s3.put_object.side_effect = _put
    return s3


EVAL_LOG = [
    {"ticker": "AAPL", "quant_filter_pass": 1},
    {"ticker": "MSFT", "quant_filter_pass": 0},
    {"ticker": "GOOG", "quant_filter_pass": 0},
]
# The pinned cut draws mostly from names the quant filter FAILED, the shape
# measured on 2026-09-23, plus NVDA, which this scanner run did not evaluate.
CUTS = {"attractiveness_top_60": {"tickers": ["MSFT", "GOOG", "NVDA"]},
        "scanner_top_20": {"tickers": ["AAPL"]}}


def _run(objects, *, vol_scorer=None, universe=("AAPL", "MSFT", "GOOG", "NVDA", "SPY")):
    s3 = _s3(objects)
    prices = _prices(list(universe))
    with patch("inference.research_free_inference.load_meta_model",
               return_value=_FakeMetaModel()), \
         patch("inference.research_free_inference.load_volatility_scorer",
               return_value=vol_scorer), \
         patch("nousergon_lib.arcticdb.load_universe_ohlcv",
               side_effect=lambda bucket, symbols, end: {t: prices[t] for t in symbols if t in prices}), \
         patch("nousergon_lib.arcticdb.load_macro_series",
               return_value={"SPY": prices["SPY"]}):
        result = run_research_free_inference(
            CANDIDATES_DATE, bucket="b", s3_client=s3, dry_run=False,
        )
    envelope = s3.written[f"predictor/predictions_research_free/{CANDIDATES_DATE}.json"]
    return result, envelope, s3


def _by_ticker(envelope):
    return {e["ticker"]: e for e in envelope["predictions"]}


class TestPopulationRule:
    def test_union_of_evaluated_universe_and_cut_members(self):
        artifact = {"scanner_eval_log": EVAL_LOG}
        cuts = {"a": ["MSFT", "NVDA"], "b": ["AAPL"]}
        assert scored_population(artifact, cuts) == ["AAPL", "GOOG", "MSFT", "NVDA"]

    def test_no_cuts_is_the_evaluated_universe(self):
        assert scored_population({"scanner_eval_log": EVAL_LOG}, {}) == ["AAPL", "GOOG", "MSFT"]

    def test_cuts_by_ticker_lists_every_containing_cut_sorted(self):
        got = _cuts_by_ticker({"z": ["X"], "a": ["X", "Y"]})
        assert got == {"X": ["a", "z"], "Y": ["a"]}


class TestPinnedCutIsCovered:
    def test_every_cut_member_is_scored_including_quant_filter_failures(self):
        _, env, _ = _run({
            CANDIDATES_KEY: {"scanner_eval_log": EVAL_LOG},
            MEMBERSHIP_KEY: {"run_date": CANDIDATES_DATE, "cuts": CUTS},
        })
        preds = _by_ticker(env)
        cut = CUTS["attractiveness_top_60"]["tickers"]
        # The pre-fix producer would have covered 0 of these 3.
        assert all(t in preds for t in cut)
        assert preds["AAPL"]["in_scanner_pool"] is True
        assert preds["MSFT"]["in_scanner_pool"] is False
        assert preds["NVDA"]["cuts"] == ["attractiveness_top_60"]
        assert preds["AAPL"]["cuts"] == ["scanner_top_20"]
        assert env["n_scanner_pool"] == 1
        assert env["n_scored_population"] == 4
        assert env["scored_population"] == "scanner_eval_log+membership_cuts"

    def test_quant_pass_pool_is_recoverable_from_the_artifact(self):
        _, env, _ = _run({
            CANDIDATES_KEY: {"scanner_eval_log": EVAL_LOG},
            MEMBERSHIP_KEY: {"run_date": CANDIDATES_DATE, "cuts": CUTS},
        })
        pool = sorted(e["ticker"] for e in env["predictions"] if e["in_scanner_pool"])
        assert pool == ["AAPL"]


class TestMembershipDateAlignment:
    def test_membership_is_read_at_the_candidates_date_not_latest(self):
        _, env, s3 = _run({
            CANDIDATES_KEY: {"scanner_eval_log": EVAL_LOG},
            MEMBERSHIP_KEY: {"run_date": CANDIDATES_DATE, "cuts": CUTS},
            "universe_membership/latest.json": {"run_date": "2099-01-01", "cuts": {}},
        })
        assert MEMBERSHIP_KEY in s3.reads
        assert "universe_membership/latest.json" not in s3.reads
        assert env["candidates_date"] == CANDIDATES_DATE
        assert env["membership"] == {
            "key": MEMBERSHIP_KEY, "run_date": CANDIDATES_DATE, "status": "ok",
        }

    def test_lookback_resolved_candidates_date_drives_the_membership_key(self):
        # Producer runs on 07-14, newest candidates are 07-10: membership must
        # come from 07-10 too, the same scanner run.
        s3 = _s3({
            CANDIDATES_KEY: {"scanner_eval_log": EVAL_LOG},
            MEMBERSHIP_KEY: {"run_date": CANDIDATES_DATE, "cuts": CUTS},
        })
        prices = _prices(["AAPL", "MSFT", "GOOG", "NVDA", "SPY"])
        with patch("inference.research_free_inference.load_meta_model",
                   return_value=_FakeMetaModel()), \
             patch("inference.research_free_inference.load_volatility_scorer",
                   return_value=None), \
             patch("nousergon_lib.arcticdb.load_universe_ohlcv", return_value=prices), \
             patch("nousergon_lib.arcticdb.load_macro_series",
                   return_value={"SPY": prices["SPY"]}):
            run_research_free_inference("2026-07-14", bucket="b", s3_client=s3)
        env = s3.written["predictor/predictions_research_free/2026-07-14.json"]
        assert env["candidates_date"] == CANDIDATES_DATE
        assert env["membership"]["key"] == MEMBERSHIP_KEY
        assert "universe_membership/2026-07-14/membership.json" not in s3.reads


class TestNoCrossSectionContamination:
    def test_cut_only_names_do_not_change_any_other_prediction(self):
        scorer = _RankSensitiveVolScorer()
        _, with_extra, _ = _run({
            CANDIDATES_KEY: {"scanner_eval_log": EVAL_LOG},
            MEMBERSHIP_KEY: {"run_date": CANDIDATES_DATE, "cuts": CUTS},
        }, vol_scorer=scorer)
        _, without_extra, _ = _run({
            CANDIDATES_KEY: {"scanner_eval_log": EVAL_LOG},
            MEMBERSHIP_KEY: {"run_date": CANDIDATES_DATE,
                             "cuts": {"attractiveness_top_60": {"tickers": ["MSFT", "GOOG"]}}},
        }, vol_scorer=scorer)
        a, b = _by_ticker(with_extra), _by_ticker(without_extra)
        assert "NVDA" in a and "NVDA" not in b
        for t in ("AAPL", "MSFT", "GOOG"):
            assert a[t]["predicted_alpha"] == b[t]["predicted_alpha"], t


class TestMembershipFailureIsRecordedNotFatal:
    @pytest.mark.parametrize("objects, status", [
        ({}, "absent"),
        ({MEMBERSHIP_KEY: {"run_date": CANDIDATES_DATE, "cuts": {}}}, "no_cuts"),
    ])
    def test_status_rides_the_envelope_and_the_universe_is_still_scored(self, objects, status):
        _, env, _ = _run({CANDIDATES_KEY: {"scanner_eval_log": EVAL_LOG}, **objects})
        assert env["membership"]["status"] == status
        assert sorted(_by_ticker(env)) == ["AAPL", "GOOG", "MSFT"]
        assert all(e["cuts"] == [] for e in env["predictions"])

    def test_access_denied_is_unreadable_not_absent(self):
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetObject")
        cuts, prov = load_membership_cuts("b", CANDIDATES_DATE, s3_client=s3)
        assert cuts == {}
        assert prov["status"] == "unreadable"
