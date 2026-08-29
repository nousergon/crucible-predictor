"""Regression tests for the row/date input-completeness gate (I9290).

Every test here is written against the LIVE 2026-08-15 failure: ArcticDB
``macro/VIX3M`` held 16 rows against SPY's 2514, ``coverage_ratio`` read
``1.0``, and ``meta_trainer`` zero-filled six ``macro_*`` features plus
``regime_intensity_z`` for every OOS date across two full rotations.

Guard-fails-without-the-fix (champion-challenger-policy §7.4):
  * ``test_symbol_count_ratio_is_blind_to_a_16_row_series`` asserts the OLD
    measure reads 1.0 on the exact failing input, and the NEW one does not.
  * ``test_required_short_history_raises`` fails on any build where the gate
    does not exist or does not raise.
  * ``test_fold_loop_refuses_to_zero_fill`` reads the source of the branch
    that wrote the constants and asserts it raises. It fails against every
    commit before this one.
"""
from __future__ import annotations

import inspect

import pandas as pd
import pytest

from training.data_completeness import (
    INPUT_REGISTER,
    DataCompletenessError,
    assert_trainable,
    evaluate_completeness,
    measure_symbol_coverage,
    summarize_universe_rows,
)


def _series(n_rows: int, end: str = "2026-08-28") -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=n_rows)
    return pd.DataFrame({"Close": [1.0] * n_rows}, index=idx)


def _healthy_per_symbol(**overrides) -> dict:
    base = {
        spec.symbol: measure_symbol_coverage(_series(2514))
        for spec in INPUT_REGISTER
    }
    base.update(overrides)
    return base


class TestMeasureSymbolCoverage:
    def test_reports_rows_and_span(self):
        cov = measure_symbol_coverage(_series(2514))
        assert cov["rows"] == 2514
        assert cov["last_date"] == "2026-08-28"
        assert cov["first_date"] is not None

    def test_empty_frame_is_zero_rows_with_no_invented_dates(self):
        cov = measure_symbol_coverage(pd.DataFrame())
        assert cov == {"rows": 0, "first_date": None, "last_date": None}


class TestTheDefectThatShipped:
    def test_symbol_count_ratio_is_blind_to_a_16_row_series(self):
        """The 2026-08-15 input, graded by both measures.

        n_written/n_expected == 1.0 (every symbol present) while the row
        measure sees VIX3M at 16/2514. This is the whole finding.
        """
        per_symbol = _healthy_per_symbol(
            VIX3M=measure_symbol_coverage(_series(16))
        )
        n_expected = len(per_symbol)
        n_written = sum(1 for v in per_symbol.values() if v["rows"])
        assert n_written / n_expected == 1.0  # the OLD number, unchanged

        block = evaluate_completeness(per_symbol)
        assert block["status"] == "failed"
        assert block["input_completeness_ratio"] < 1.0
        assert block["inputs"]["VIX3M"]["status"] == "short_history"
        assert block["inputs"]["VIX3M"]["rows"] == 16
        assert block["inputs"]["VIX3M"]["rows_ratio"] < 0.01

    def test_required_short_history_raises(self):
        block = evaluate_completeness(
            _healthy_per_symbol(VIX3M=measure_symbol_coverage(_series(16)))
        )
        with pytest.raises(DataCompletenessError) as exc:
            assert_trainable(block)
        assert "VIX3M" in str(exc.value)
        assert "short_history" in str(exc.value)

    def test_a_frozen_producer_is_stale_not_complete(self):
        """HYOAS: 786 rows, last date 2026-05-07, rewritten weekly.

        Row count alone clears no bar it should — the series is long. The
        finding is that it stopped MOVING, which only last_date can see.
        """
        block = evaluate_completeness(
            _healthy_per_symbol(
                HYOAS=measure_symbol_coverage(_series(786, end="2026-05-07"))
            )
        )
        hy = block["inputs"]["HYOAS"]
        assert hy["status"] == "stale"
        assert hy["stale_days"] > 100
        # optional severity → a NAMED degradation, never a silent default
        assert block["status"] == "degraded"
        assert [d["input"] for d in block["degradations"]] == ["HYOAS"]
        assert "hy_oas_level" in block["degradations"][0]["features_affected"]
        assert_trainable(block)  # does not raise for an optional input

    def test_absent_optional_is_named_never_silent(self):
        per = _healthy_per_symbol()
        per.pop("BAA10Y")
        block = evaluate_completeness(per)
        assert block["inputs"]["BAA10Y"]["status"] == "absent"
        assert block["status"] == "degraded"
        assert "baa10y_level" in block["degradations"][0]["features_affected"]

    def test_a_long_series_that_starts_too_late_fails_span(self):
        """Rows alone cannot see this: the series is full length and fresh,
        but begins after the scored window opens."""
        per = _healthy_per_symbol(
            VIX3M=measure_symbol_coverage(_series(2514, end="2026-08-28"))
        )
        block = evaluate_completeness(
            per, oos_window=("2000-01-03", "2026-08-28")
        )
        assert block["inputs"]["VIX3M"]["status"] == "insufficient_span"
        assert block["status"] == "failed"


class TestHealthyRun:
    def test_all_complete_is_ok_and_does_not_raise(self):
        block = evaluate_completeness(_healthy_per_symbol())
        assert block["status"] == "ok"
        assert block["input_completeness_ratio"] == 1.0
        assert block["failures"] == [] and block["degradations"] == []
        assert_trainable(block)

    def test_every_registered_input_declares_severity_and_consumers(self):
        for spec in INPUT_REGISTER:
            assert spec.severity in {"required", "optional"}, spec.symbol
            assert spec.consumers, spec.symbol

    def test_register_covers_every_load_close_call_in_meta_trainer(self):
        """A new ``_load_close`` without a register row is the VIX3M failure
        mode repeating — the gate would be blind to the new input."""
        import re

        from training import meta_trainer

        src = inspect.getsource(meta_trainer.run_meta_training)
        loaded = set(re.findall(r'_load_close\(\s*"([A-Z0-9]+)\.parquet"', src))
        registered = {s.symbol for s in INPUT_REGISTER}
        assert loaded <= registered, (
            f"macro series loaded but NOT in INPUT_REGISTER: "
            f"{sorted(loaded - registered)} — add a row to "
            f"training/data_completeness.INPUT_REGISTER in the same PR."
        )


class TestUniverseRowSummary:
    def test_summary_excludes_registered_macros_and_names_the_worst(self):
        per = _healthy_per_symbol()
        per["AAPL"] = measure_symbol_coverage(_series(2514))
        per["NEWCO"] = measure_symbol_coverage(_series(26))
        s = summarize_universe_rows(per)
        assert s["n_symbols"] == 2
        assert s["min_rows"] == 26
        assert s["worst"][0]["symbol"] == "NEWCO"
        assert "SPY" not in [w["symbol"] for w in s["worst"]]


class TestNoSilentSubstitutionRemains:
    def test_fold_loop_refuses_to_zero_fill(self):
        """The branch at the heart of I9290 must RAISE, not assign 0.0.

        Located by AST rather than by string-slicing so a reformat cannot
        quietly turn this guard green. Fails against every commit before
        this one, where the ``else`` of the regime-row lookup read
        ``for meta_name in MACRO_FEATURE_META_MAP.values(): macro_row[...] = 0.0``.
        """
        import ast
        import textwrap

        from training import meta_trainer

        tree = ast.parse(textwrap.dedent(inspect.getsource(meta_trainer.run_meta_training)))
        lookups = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and isinstance(n.test.ops[0], ast.In)
            and "regime_features_df.index" in ast.unparse(n.test.comparators[0])
        ]
        assert lookups, "the regime-row lookup in the fold loop was not found"
        for node in lookups:
            assert node.orelse, "the regime-row lookup lost its else branch"
            body = "\n".join(ast.unparse(x) for x in node.orelse)
            assert "DataCompletenessError" in body, body
            assert "= 0.0" not in body, (
                "the macro zero-fill is back — a missing regime row must "
                "raise, never be substituted with a constant "
                "(alpha-engine-config-I9290)"
            )

        # And no per-row constant for the DERIVED regime columns either.
        whole = ast.unparse(tree)
        assert "macro_row[meta_name] = 0.0" not in whole

    def test_arctic_reader_returns_per_symbol_rows_and_dates(self):
        import store.arctic_reader as ar

        doc = ar.download_from_arctic.__doc__ or ""
        assert "per_symbol" in doc
        src = inspect.getsource(ar.download_from_arctic)
        assert "measure_symbol_coverage" in src
        # the empty-DataFrame skip must be RECORDED, not silent
        assert "empty_universe.append" in src
        assert "empty_macro.append" in src

    def test_regime_panel_coverage_gate_exists_and_raises(self):
        from training import meta_trainer

        src = inspect.getsource(meta_trainer.run_meta_training)
        assert "regime_panel_coverage" in src
        assert "REGIME_PANEL_COVERAGE_FLOOR" in src
        assert "assert_trainable(data_completeness)" in src
