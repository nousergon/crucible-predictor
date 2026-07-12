"""Consumer contract for the ``signals`` slot boundary (predictor side).

config#2343 boundary 1 (sibling of config#692): today only the executor has a
``signals`` consumer contract test — the predictor reads ``signals.json`` in
``inference/stages/load_universe.py`` with fixture-only unit tests, so a
research-side rename/drop of a field the predictor depends on (e.g.
``sector_modifiers`` feeding the ``sector_macro_modifier`` META_FEATURE, or the
canonical ``market_regime`` overlay) would pass every predictor test and only
surface as silently-wrong inference in production.

This binds the predictor's ACTUAL signals reads to the lib-hosted versioned
schema (``nousergon_lib.contracts`` ``signals`` v1) — the same single source of
truth the executor/backtester/dashboard bind to. Two guards:

  1. Every key the predictor reads off a ``signals.json`` payload is a property
     the ``signals`` contract declares — so the contract cannot drop a field the
     predictor relies on without failing this test.
  2. A representative payload shaped like exactly the reads the predictor
     performs conforms to the contract (``contracts.validate``).

Per the established per-repo pattern (per-repo CI can't import the config repo /
the heavy inference deps), the predictor's declared read set is a documented
mirror, cross-checked against ``load_universe.py`` SOURCE via AST/text so a new
read or a changed macro-field tuple can't dodge the pin.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

contracts = pytest.importorskip(
    "nousergon_lib.contracts",
    reason="needs nousergon-lib[contracts]",
)

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "inference" / "stages" / "load_universe.py"

# Canonical macro fields the predictor overlays from signals.json onto the
# population payload (load_universe.py::_merge_canonical_macro_into): the
# post-critic regime + sector context feeding META_FEATURES. Read from SOURCE so
# a change to the tuple must update this mirror.
_MACRO_CONST = "_CANONICAL_SIGNALS_MACRO_FIELDS"

# Top-level signals.json keys the predictor reads. Macro subset is asserted to
# equal _CANONICAL_SIGNALS_MACRO_FIELDS below; the list keys carry per-ticker
# entries the predictor pulls tickers from.
DECLARED_TOP_LEVEL_READS = {
    "market_regime",     # canonical regime overlay (macro)
    "sector_modifiers",  # sector_macro_modifier META_FEATURE input (macro)
    "sector_ratings",    # sector context overlay (macro)
    "buy_candidates",    # research's buy list -> universe union (entry.ticker)
    "universe",          # full tracked population -> ticker extraction (entry.ticker)
    "signals",           # legacy list form: get_universe_tickers reads .get("signals", [])
}

# Per-entry keys the predictor reads off each universe/buy_candidates entry.
DECLARED_ENTRY_READS = {"ticker"}


def _macro_fields_from_source() -> set[str]:
    """AST-extract the string values of the _CANONICAL_SIGNALS_MACRO_FIELDS
    tuple from load_universe.py — no import (keeps the test free of the heavy
    inference deps), mirroring the params consumer-contract pattern."""
    tree = ast.parse(_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == _MACRO_CONST for t in node.targets
        ):
            elts = getattr(node.value, "elts", [])
            return {
                e.value for e in elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
    raise AssertionError(f"{_MACRO_CONST} not found in {_SRC}")


def test_signals_is_registered_contract():
    # Bind to the lib schema, not a predictor-authored fixture.
    assert "signals" in contracts.CONTRACT_SCHEMAS
    schema = contracts.load_schema("signals")
    assert "signals.v1.json" in schema["$id"]


def test_macro_overlay_matches_declared_reads():
    # The consumer code's macro tuple and this contract mirror must agree, so a
    # code-side change to what the predictor overlays trips the test.
    assert _macro_fields_from_source() == {
        "market_regime", "sector_modifiers", "sector_ratings",
    }


def test_all_declared_top_level_reads_are_in_contract():
    props = set(contracts.load_schema("signals").get("properties", {}))
    missing = DECLARED_TOP_LEVEL_READS - props
    assert not missing, (
        f"predictor reads signals.json key(s) the contract no longer declares: "
        f"{sorted(missing)} — a research-side drop/rename would silently break "
        f"predictor inference. Restore the field or migrate the read (+ bump the "
        f"signals schema version)."
    )


def test_entry_reads_are_required_in_contract():
    entry = contracts.load_schema("signals")["$defs"]["signal_entry"]
    entry_props = set(entry.get("properties", {}))
    required = set(entry.get("required", []))
    for key in DECLARED_ENTRY_READS:
        assert key in entry_props, f"signal entry contract missing {key!r}"
        assert key in required, f"predictor relies on entry.{key}; must be required"


def test_predictor_actually_reads_the_declared_list_keys():
    # Guard the declared list-key mirror against the source: each must appear as
    # a real read in load_universe.py (else it's a stale declaration).
    src = _SRC.read_text()
    for key in ("buy_candidates", "universe", "signals"):
        assert f'"{key}"' in src or f"'{key}'" in src, (
            f"declared read {key!r} no longer appears in load_universe.py"
        )


def _representative_signals(**overrides):
    """A signals.json payload shaped like exactly the reads the predictor
    performs; must conform to the lib contract."""
    entry = {
        "ticker": "AAPL",
        "signal": "ENTER",
        "score": 82.0,
        "rating": "BUY",
        "conviction": "rising",
        "sector": "Technology",
        "sector_rating": "overweight",
        "price_target_upside": 0.18,
    }
    payload = {
        "date": "2026-07-11",
        "market_regime": "bull",
        "sector_ratings": {"Technology": {"rating": "overweight", "modifier": 1.1}},
        "sector_modifiers": {"Technology": 1.1},
        "universe": [entry],
        "buy_candidates": [entry],
    }
    payload.update(overrides)
    return payload


def test_representative_predictor_read_shape_conforms():
    contracts.validate("signals", _representative_signals())


def test_representative_shape_with_schema_version_conforms():
    contracts.validate("signals", _representative_signals(schema_version=1))
