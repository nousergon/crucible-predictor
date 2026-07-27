"""Producer-side contract test for the predictor →
predictions_research_free.json boundary (config#2365, champion-loop epic
child 1 / config#2364).

Mirrors ``tests/test_predictions_producer_contract.py`` (the sibling
``predictions/{date}.json`` contract): pins the field set THIS producer must
keep emitting via AST source-binding of the envelope/entry dict literals in
``inference/research_free_inference.py``, so a future edit that silently
drops a contract field fails loud at test time rather than at the executor
consumer (config#2364 child 2, once it lands) discovering a missing key via
a ``.get(...)`` default.

Also exercises the REAL assembly path (``run_research_free_inference`` with
the S3/ArcticDB seams mocked) to validate the schema-versioned envelope shape
end-to-end, matching ``tests/test_predictions_schema_conformance.py``'s
"capture from the real path, not hand-built" posture for the sibling
contract.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "inference", "research_free_inference.py")

# The envelope-level fields this producer commits to emitting every run.
_REQUIRED_ENVELOPE = {
    "schema_version", "date", "n_predictions", "predictions",
}
# The per-entry fields config#2365's "Closes when" pins verbatim:
# [{ticker, prediction_date, predicted_alpha, n_research_features_missing}].
_REQUIRED_PER_ITEM = {
    "ticker", "prediction_date", "predicted_alpha", "n_research_features_missing",
}


def _dict_literals_with_key(path: str, marker_key: str) -> list[set[str]]:
    """String-key sets of every dict literal in `path` containing a string
    key == marker_key. AST-based so it's robust to formatting (mirrors
    test_predictions_producer_contract.py's locator)."""
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    out: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            k.value for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        if marker_key in keys:
            out.append(keys)
    return out


def test_per_ticker_entry_dict_carries_every_required_field():
    candidates = [
        keys for keys in _dict_literals_with_key(_SRC, "ticker")
        if "predicted_alpha" in keys
    ]
    assert candidates, (
        "could not locate the per-ticker research-free entry dict literal in "
        "research_free_inference.py (expected a dict literal with both "
        "'ticker' and 'predicted_alpha'). If the builder moved, update this "
        "test's locator."
    )
    for keys in candidates:
        missing = _REQUIRED_PER_ITEM - keys
        assert not missing, (
            f"per-ticker research-free entry dict dropped contract field(s): "
            f"{sorted(missing)}. config#2365's 'Closes when' pins this exact "
            f"shape: [{{ticker, prediction_date, predicted_alpha, "
            f"n_research_features_missing}}]."
        )


def test_envelope_dict_carries_every_required_field():
    candidates = [
        keys for keys in _dict_literals_with_key(_SRC, "predictions")
        if "n_predictions" in keys
    ]
    assert candidates, (
        "could not locate the predictions_research_free.json envelope dict "
        "literal in research_free_inference.py (expected a dict literal "
        "with both 'predictions' and 'n_predictions')."
    )
    for keys in candidates:
        missing = _REQUIRED_ENVELOPE - keys
        assert not missing, (
            f"predictions_research_free.json envelope dropped contract "
            f"field(s): {sorted(missing)}."
        )


def test_schema_version_is_a_positive_integer_constant():
    from inference.research_free_inference import SCHEMA_VERSION

    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 1


# ── End-to-end envelope shape (real assembly path, mocked IO boundary) ─────

class _FakeMetaModel:
    is_fitted = True
    _feature_names = ["momentum_score", "research_calibrator_prob"]

    def predict_single(self, features: dict) -> float:
        return float(sum(features.values()))


def _synthetic_price_data(tickers, n=300):
    idx = pd.bdate_range("2025-01-01", periods=n)
    out = {}
    for i, t in enumerate(tickers):
        rng = np.random.default_rng(hash(t) % (2**31))
        close = 50 + i * 10 + np.cumsum(rng.normal(0, 1, n))
        out[t] = pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n),
            "momentum_5d": rng.normal(0, 0.02, n),
            "momentum_20d": rng.normal(0, 0.05, n),
            "price_vs_ma50": rng.normal(0, 0.03, n),
            "rsi_14": rng.uniform(30, 70, n),
        }, index=idx)
    return out


def _capture_written_envelope() -> dict:
    """Run run_research_free_inference through the real assembly + S3-write
    path (S3 client mocked, ArcticDB/universe seams mocked); return the
    dated-key envelope actually written."""
    universe = ["AAPL", "SPY"]
    price_data = _synthetic_price_data(universe)
    macro = {"SPY": price_data["SPY"]}

    candidates_payload = {
        "scanner_eval_log": [{"ticker": "AAPL", "quant_filter_pass": 1}]
    }
    s3 = MagicMock()
    body = MagicMock()
    body.read.return_value = json.dumps(candidates_payload).encode()
    s3.get_object.return_value = {"Body": body}

    written: dict[str, str] = {}

    def _capture_put(**kwargs):
        written[kwargs["Key"]] = kwargs["Body"]

    s3.put_object.side_effect = _capture_put

    with patch(
        "inference.research_free_inference.load_meta_model",
        return_value=_FakeMetaModel(),
    ), patch(
        "inference.research_free_inference.load_volatility_scorer",
        return_value=None,
    ), patch(
        "nousergon_lib.arcticdb.load_universe_ohlcv",
        return_value=price_data,
    ), patch(
        "nousergon_lib.arcticdb.load_macro_series",
        return_value=macro,
    ):
        from inference.research_free_inference import run_research_free_inference

        run_research_free_inference(
            "2026-07-10", bucket="test-bucket", s3_client=s3, dry_run=False,
        )

    dated_key = "predictor/predictions_research_free/2026-07-10.json"
    assert dated_key in written, f"producer did not write {dated_key}"
    return json.loads(written[dated_key])


class TestEndToEndEnvelopeConformsToContract:
    def test_envelope_has_all_required_fields(self):
        envelope = _capture_written_envelope()
        missing = _REQUIRED_ENVELOPE - set(envelope)
        assert not missing, f"assembled envelope missing: {sorted(missing)}"

    def test_entries_have_all_required_fields(self):
        envelope = _capture_written_envelope()
        assert envelope["predictions"], "expected at least one entry (AAPL passed the pool filter)"
        for entry in envelope["predictions"]:
            missing = _REQUIRED_PER_ITEM - set(entry)
            assert not missing, f"assembled entry missing: {sorted(missing)}: {entry}"

    def test_dated_and_latest_keys_both_written_identically(self):
        # Both keys are written from the SAME envelope in
        # write_research_free_predictions — verified via the dated-key
        # capture above; this test independently confirms latest.json
        # round-trips to an identical payload.
        universe = ["AAPL", "SPY"]
        price_data = _synthetic_price_data(universe)
        macro = {"SPY": price_data["SPY"]}
        candidates_payload = {
            "scanner_eval_log": [{"ticker": "AAPL", "quant_filter_pass": 1}]
        }
        s3 = MagicMock()
        body = MagicMock()
        body.read.return_value = json.dumps(candidates_payload).encode()
        s3.get_object.return_value = {"Body": body}
        written: dict[str, str] = {}
        s3.put_object.side_effect = lambda **kw: written.__setitem__(kw["Key"], kw["Body"])

        with patch(
            "inference.research_free_inference.load_meta_model",
            return_value=_FakeMetaModel(),
        ), patch(
            "inference.research_free_inference.load_volatility_scorer",
            return_value=None,
        ), patch(
            "nousergon_lib.arcticdb.load_universe_ohlcv",
            return_value=price_data,
        ), patch(
            "nousergon_lib.arcticdb.load_macro_series",
            return_value=macro,
        ):
            from inference.research_free_inference import run_research_free_inference

            run_research_free_inference(
                "2026-07-10", bucket="test-bucket", s3_client=s3, dry_run=False,
            )

        dated = json.loads(written["predictor/predictions_research_free/2026-07-10.json"])
        latest = json.loads(written["predictor/predictions_research_free/latest.json"])
        assert dated == latest
