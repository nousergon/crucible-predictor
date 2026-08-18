"""Consumer-side schema conformance for the ``signals`` contract (Slot R
boundary, M0 / alpha-engine-config#638), the predictor consumer half of
config#3026's wave 2 ("no PR in crucible-predictor implements the
signals->predictor consumer contract").

``inference/stages/load_universe.py`` is this repo's ONLY reader of
``signals/{date}/signals.json`` / ``signals/latest.json``
(``_read_buy_candidates_from_signals``, ``_load_signals_payload_with_fallback``,
``get_universe_tickers``) and, before this file, imported nothing from
``nousergon_lib.contracts`` for that boundary — ``tests/test_load_universe.py``
exercises the S3 fallback-chain logic against hand-built temp-file fixtures
that restate the consumer's own assumptions back to itself, never against the
frozen producer contract.

This file validates payloads shaped by the ACTUAL frozen
``nousergon_lib.contracts`` signals schema (v1, ``signals.schema.json``) and
checks what ``load_universe.py``'s functions do when handed one — not a
hand-rolled fixture, the schema's own required-field set plus its
``$defs.signal_entry`` shape.

Consumer call sites traced (both walk the SAME
``nousergon_lib.signals.fallback_research_date_keys`` chain):

  * ``_read_buy_candidates_from_signals`` reads ``buy_candidates`` (schema:
    array of ``signal_entry``, each REQUIRED to declare ``ticker``) —
    entries may be a ``signal_entry`` object OR a bare string; the consumer
    tolerates both, which is a WIDER read than the schema promises, not a
    dependency the schema could break.
  * ``get_universe_tickers`` reads ``signals`` (schema: ``type: object`` —
    "Legacy v2 per-ticker dict format (deprecated; retained for backward
    compatibility)") as though it were an ARRAY of per-ticker dicts each
    carrying ``ticker`` (``signals_data.get("signals", [])`` then
    ``[s["ticker"] for s in signals_list if "ticker" in s]``). That is a
    real drift: the schema does not promise ``signals`` is iterable-as-a-
    list-of-dicts at all — see ``TestSignalsFieldShapeDrift`` below, xfail,
    not silently patched here (config#3026's instruction: report, don't
    widen the schema and don't quietly "fix" the consumer's assumption
    without a ruling on which of the two is wrong).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

contracts = pytest.importorskip(
    "nousergon_lib.contracts",
    reason="needs nousergon-lib[contracts] (jsonschema) installed",
)

from inference.stages.load_universe import (
    _load_signals_payload_with_fallback,
    _read_buy_candidates_from_signals,
    get_universe_tickers,
)

_BUCKET = "test-bucket"
_DATE = "2026-06-11"


def _signal_entry(**overrides) -> dict:
    entry = {
        "ticker": "TEST",
        "signal": "ENTER",
        "score": 72.5,
        "rating": "BUY",
        "conviction": "rising",
        "sector": "Information Technology",
        "sector_rating": "overweight",
        "price_target_upside": 0.18,
    }
    entry.update(overrides)
    return entry


def _signals_payload(**overrides) -> dict:
    """A payload built from the FROZEN schema's own required fields —
    never a fixture that merely restates load_universe.py's assumptions."""
    payload = {
        "date": _DATE,
        "market_regime": "neutral",
        "sector_ratings": {"Technology": {"rating": "overweight", "modifier": 1.1}},
        "sector_modifiers": {"Technology": 1.1},
        "universe": [_signal_entry()],
        "buy_candidates": [_signal_entry(ticker="BUYME", signal="ENTER")],
    }
    payload.update(overrides)
    return payload


def _s3_returning(payload_by_key: dict[str, dict]) -> MagicMock:
    """A minimal boto3-shaped s3 client stub: get_object(Bucket, Key) returns
    a Body whose .read() yields the JSON bytes for a known key, or raises
    NoSuchKey (via botocore.exceptions.ClientError) for anything else —
    exactly the contract nousergon_lib.signals.try_read_s3_json expects."""
    import json as _json

    from botocore.exceptions import ClientError

    def _get_object(Bucket, Key):  # noqa: N803 - matches boto3's signature
        if Key not in payload_by_key:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "no"}}, "GetObject"
            )
        body = MagicMock()
        body.read.return_value = _json.dumps(payload_by_key[Key]).encode("utf-8")
        return {"Body": body}

    s3 = MagicMock()
    s3.get_object.side_effect = _get_object
    return s3


class TestSchemaConformantPayloadValidates:
    def test_minimal_payload_conforms(self):
        contracts.validate("signals", _signals_payload())


class TestBuyCandidatesConsumption:
    """_read_buy_candidates_from_signals against a schema-conformant
    payload, reached via the REAL fallback-chain S3 read path — not a
    direct-call unit test of the function in isolation."""

    def test_extracts_tickers_from_conforming_payload(self):
        payload = _signals_payload(
            buy_candidates=[_signal_entry(ticker="AAA"), _signal_entry(ticker="BBB")]
        )
        contracts.validate("signals", payload)
        s3 = _s3_returning({f"signals/{_DATE}/signals.json": payload})
        tickers = _read_buy_candidates_from_signals(s3, _BUCKET, _DATE)
        assert tickers == ["AAA", "BBB"]

    def test_bare_string_entries_are_tolerated_beyond_the_schema(self):
        # The schema's buy_candidates items are ALWAYS signal_entry objects
        # (required: ticker, signal, score, ...) — a bare string entry is
        # not schema-conformant. load_universe.py tolerates one anyway
        # (entry if not a dict). This is a WIDER read than the contract
        # promises, never a dependency the schema could break by narrowing,
        # so it is documented here rather than xfailed.
        payload = _signals_payload(buy_candidates=["RAWTICKER"])
        s3 = _s3_returning({f"signals/{_DATE}/signals.json": payload})
        tickers = _read_buy_candidates_from_signals(s3, _BUCKET, _DATE)
        assert tickers == ["RAWTICKER"]

    def test_no_candidates_key_returns_empty(self):
        payload = _signals_payload()
        del payload["buy_candidates"]
        # Not schema-conformant (buy_candidates is required) — the consumer
        # must still degrade honestly (empty list, not a crash) since a
        # producer regression dropping a required field is exactly the
        # failure mode this contract test exists to catch upstream of.
        s3 = _s3_returning({f"signals/{_DATE}/signals.json": payload})
        assert _read_buy_candidates_from_signals(s3, _BUCKET, _DATE) == []


class TestSignalsPayloadFallback:
    def test_walks_fallback_chain_to_latest(self):
        payload = _signals_payload()
        contracts.validate("signals", payload)
        s3 = _s3_returning({"signals/latest.json": payload})
        result = _load_signals_payload_with_fallback(s3, _BUCKET, _DATE)
        assert result["date"] == _DATE
        assert result["market_regime"] == "neutral"


class TestSignalsFieldShapeDrift:
    """KNOWN GAP (report, not silently patched — config#3026's instruction).

    ``get_universe_tickers`` (inference/stages/load_universe.py) reads:

        signals_list = signals_data.get("signals", []) if signals_data else []
        tickers = [s["ticker"] for s in signals_list if "ticker" in s]

    treating ``signals`` as an array of per-ticker dicts. The FROZEN v1
    schema declares ``signals`` as ``{"type": "object", "description":
    "Legacy v2 per-ticker dict format (deprecated; retained for backward
    compatibility)"}`` — an OBJECT, not an array. A schema-conformant
    ``signals`` value is a dict (e.g. ``{"AAPL": {...}, "MSFT": {...}}``),
    and iterating a dict with ``for s in signals_list`` yields its KEYS
    (ticker strings) — ``"ticker" in s`` then tests substring membership
    against a ticker string (almost never true), so the primary code path
    silently returns zero tickers and falls through to the hardcoded
    ``_FALLBACK_TICKERS`` list, rather than raising. The exception handler
    two lines up would also swallow a genuine ``TypeError`` from
    ``s["ticker"]`` on a dict-not-object shape, masking either failure mode
    identically as "signals load failed — using fallback universe".

    This is exactly the class of drift config#3026 asks to surface, not
    patch: the consumer's assumption (array of dicts) and the frozen
    contract's declared shape (object) do not agree, and it is not this
    test's job to decide which one is wrong. xfail, strict — this must
    start FAILING (proving the gap closed) the moment either side is fixed,
    so the xfail marker gets deleted then rather than rotting as a false
    green forever.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="get_universe_tickers (inference/stages/load_universe.py) "
        "reads payload['signals'] as an array of {'ticker': ...} dicts; "
        "the frozen nousergon_lib.contracts signals schema (v1) declares "
        "'signals' as type object (deprecated legacy per-ticker dict), "
        "not an array — config#3026.",
    )
    def test_get_universe_tickers_extracts_from_schema_conformant_signals_field(self):
        # A schema-conformant 'signals' object, per its own description:
        # a per-ticker dict, keyed by ticker.
        payload = _signals_payload(
            signals={"AAA": _signal_entry(ticker="AAA"), "BBB": _signal_entry(ticker="BBB")}
        )
        contracts.validate("signals", payload)
        s3 = _s3_returning({f"signals/{_DATE}/signals.json": payload})
        with patch("boto3.client", return_value=s3):
            tickers, _signals_data = get_universe_tickers(_BUCKET, _DATE)
        # What the consumer's code intends: universe tickers sourced from
        # the 'signals' field. Against a conformant payload this
        # currently returns the hardcoded _FALLBACK_TICKERS instead,
        # because dict-iteration silently yields no matches — proving the
        # drift rather than asserting the (wrong) actual behavior as
        # correct.
        assert sorted(tickers) == ["AAA", "BBB"]
