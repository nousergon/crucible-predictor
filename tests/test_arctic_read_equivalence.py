"""The batched Arctic read returns the same model inputs as the per-symbol read.

Nothing here talks to S3 or the production library. Frames live in a local
``lmdb://`` ArcticDB, which implements the same ``read`` / ``read_batch`` /
``row_range`` / ``date_range`` contract the Lambda uses.

The feature path used to ``read(symbol).data.iloc[-1]``. ``row_range=(-1, None)``
is that row. A calendar window is not: a symbol whose last row is older than
the window would disappear and the prediction would change. Prices stay on the
existing 730-day window, which is what regime features and the macro z-score
are computed from today.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import arcticdb as adb
import pandas as pd
import pytest

from inference.pipeline import PipelineAbort, PipelineContext
from inference.stages import load_prices
from inference.stages.run_inference import _load_precomputed_features_from_arcticdb
from model.momentum_scorer import predict_dict
from store.arctic_batch import (
    DEFAULT_IO_THREADS,
    IO_THREADS_ENV,
    LAST_ROW_RANGE,
    configure_arcticdb_io_threads,
    iter_symbol_reads,
)

_REPO = Path(__file__).resolve().parent.parent


def _frame(index: pd.DatetimeIndex, close_0: float) -> pd.DataFrame:
    close = close_0 + pd.Series(range(len(index)), index=index, dtype="float64")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000.0,
            "momentum_5d": close * 0.001,
            "momentum_20d": close * -0.002,
            "price_vs_ma50": 1.0 + close_0 / 1000.0,
            "rsi_14": 40.0 + (close_0 % 17),
        },
        index=index,
    )


def _library(tmp_path: Path):
    arctic = adb.Arctic(f"lmdb://{tmp_path / 'store'}")
    universe = arctic.get_library("universe", create_if_missing=True)
    macro = arctic.get_library("macro", create_if_missing=True)
    return universe, macro


def _legacy_last_rows(lib, symbols: list[str]) -> dict[str, pd.Series]:
    """The pre-change feature read: full symbol, then iloc[-1]."""
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            df = lib.read(sym).data
        except Exception:
            continue
        if df is None or df.empty:
            continue
        out[sym] = df.iloc[-1]
    return out


def _legacy_window(lib, symbols: list[str], start, end, columns: list[str]) -> dict[str, pd.DataFrame]:
    """The pre-change price read: one lib.read per symbol, then sort_index."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = lib.read(sym, date_range=(start, end), columns=columns).data
        except Exception:
            continue
        if df is None or df.empty:
            continue
        out[sym] = df.sort_index()
    return out


def test_last_stored_row_matches_full_read_iloc_and_momentum_score(tmp_path, monkeypatch):
    """Feature rows and the momentum L1 score are identical to the old path.

    OLD ended in 2010. A 730-day window ending on the inference date does not
    contain that row; the old path still scored it. The new path must too.
    """
    universe, _macro = _library(tmp_path)
    recent = pd.bdate_range("2018-01-02", periods=600)
    stale = pd.bdate_range("2010-01-04", periods=40)
    universe.write("AAA", _frame(recent, 10))
    universe.write("BBB", _frame(recent, 20))
    universe.write("OLD", _frame(stale, 7))

    tickers = ["AAA", "BBB", "OLD"]
    legacy = _legacy_last_rows(universe, tickers)
    monkeypatch.setattr(
        "nousergon_lib.arcticdb.open_universe_lib", lambda bucket: universe,
    )
    ctx = PipelineContext(date_str="2020-06-15", bucket="local-test", tickers=tickers)
    fresh = _load_precomputed_features_from_arcticdb(ctx)

    assert set(fresh) == set(legacy) == set(tickers)
    for sym in tickers:
        pd.testing.assert_series_equal(fresh[sym], legacy[sym], check_names=True)
        assert predict_dict(fresh[sym].to_dict()) == predict_dict(legacy[sym].to_dict())

    # The price window would have dropped OLD. That is why features do not use it.
    end = pd.Timestamp("2020-06-15")
    start = end - pd.Timedelta(days=730)
    windowed = universe.read("OLD", date_range=(start, end)).data
    assert windowed.empty
    assert fresh["OLD"]["Close"] == legacy["OLD"]["Close"]


def test_price_window_matches_per_symbol_reads(tmp_path, monkeypatch):
    """OHLCV frames on the 730-day window match the old per-symbol read.

    VIX is only in the macro library (Close), which is the daily_append shape.
    SPY is in the universe, which is the backfill shape. Both paths agree.
    """
    universe, macro = _library(tmp_path)
    recent = pd.bdate_range("2018-01-02", periods=600)
    universe.write("AAA", _frame(recent, 10))
    universe.write("BBB", _frame(recent, 20))
    universe.write("SPY", _frame(recent, 100))
    vix = _frame(recent, 18)[["Close"]]
    macro.write("VIX", vix)

    date_str = "2020-06-15"
    end = pd.Timestamp(date_str).normalize()
    start = end - pd.Timedelta(days=730)
    tickers = ["AAA", "BBB"]
    monkeypatch.setattr(load_prices, "_connect_arctic", lambda bucket: (universe, macro))
    price_data, macro_out = load_prices.load_price_data_from_arctic(
        tickers, date_str, "local-test",
    )

    legacy_stocks = _legacy_window(
        universe, tickers, start, end, load_prices._ARCTIC_OHLCV_COLS,
    )
    for sym, frame in legacy_stocks.items():
        pd.testing.assert_frame_equal(price_data[sym], frame)

    spy = _legacy_window(universe, ["SPY"], start, end, load_prices._ARCTIC_OHLCV_COLS)
    pd.testing.assert_frame_equal(price_data["SPY"], spy["SPY"])
    vix_legacy = _legacy_window(macro, ["VIX"], start, end, ["Close"])
    pd.testing.assert_frame_equal(price_data["VIX"], vix_legacy["VIX"])
    assert macro_out["SPY"].equals(price_data["SPY"]["Close"].dropna())
    assert macro_out["VIX"].equals(price_data["VIX"]["Close"].dropna())


def test_one_missing_symbol_stays_under_the_error_gate(tmp_path, monkeypatch):
    universe, macro = _library(tmp_path)
    idx = pd.bdate_range("2019-01-02", periods=30)
    good = [f"T{i:02d}" for i in range(24)]
    for i, sym in enumerate(good):
        universe.write(sym, _frame(idx, float(i + 1)))
    tickers = good + ["MISSING"]
    monkeypatch.setattr(load_prices, "_connect_arctic", lambda bucket: (universe, macro))
    price_data, _macro_out = load_prices.load_price_data_from_arctic(
        tickers, "2019-02-15", "local-test",
    )
    assert set(good).issubset(price_data)
    assert "MISSING" not in price_data


def test_error_rate_above_five_percent_still_aborts(tmp_path, monkeypatch):
    universe, macro = _library(tmp_path)
    idx = pd.bdate_range("2019-01-02", periods=5)
    universe.write("AAA", _frame(idx, 1))
    monkeypatch.setattr(load_prices, "_connect_arctic", lambda bucket: (universe, macro))
    with pytest.raises(PipelineAbort, match="5%"):
        load_prices.load_price_data_from_arctic(
            ["AAA", "NOPE1", "NOPE2"], "2019-01-08", "local-test",
        )


def test_read_batch_is_chunked_and_still_returns_every_row(tmp_path):
    universe, _macro = _library(tmp_path)
    idx = pd.bdate_range("2019-01-02", periods=12)
    symbols = ["A", "B", "C", "D", "E"]
    for i, sym in enumerate(symbols):
        universe.write(sym, _frame(idx, float(i + 3)))

    class _Counting:
        def __init__(self, inner):
            self.inner = inner
            self.calls: list[list[str]] = []

        def read_batch(self, requests):
            self.calls.append([req.symbol for req in requests])
            return self.inner.read_batch(requests)

    wrapped = _Counting(universe)
    rows = {
        sym: df
        for sym, df, err in iter_symbol_reads(wrapped, symbols, chunk_size=2)
        if err is None and df is not None
    }
    assert [len(call) for call in wrapped.calls] == [2, 2, 1]
    for sym in symbols:
        pd.testing.assert_frame_equal(rows[sym], universe.read(sym).data)


def test_last_row_range_is_the_documented_iloc_tail():
    assert LAST_ROW_RANGE == (-1, None)


def test_configure_does_not_override_an_explicit_pool_size(monkeypatch):
    monkeypatch.setenv(IO_THREADS_ENV, "7")
    assert configure_arcticdb_io_threads(32) == "7"
    monkeypatch.delenv(IO_THREADS_ENV)
    assert configure_arcticdb_io_threads() == DEFAULT_IO_THREADS


def test_env_key_is_the_one_arcticdb_reads_at_import():
    """A misspelled key is ignored. Prove the key in a fresh interpreter."""
    code = (
        "import os; os.environ['ARCTICDB_VersionStore_NumIOThreads_int']='32'; "
        "import arcticdb; from arcticdb_ext import get_config_int; "
        "print(get_config_int('VersionStore.NumIOThreads'))"
    )
    env = os.environ.copy()
    env.pop(IO_THREADS_ENV, None)
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "32"


def _module_body(path: Path) -> list[ast.stmt]:
    return ast.parse(path.read_text(encoding="utf-8")).body


def _import_roots(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [(node.module or "").split(".")[0]]
    return []


def test_io_pool_is_sized_before_arcticdb_import_in_every_reader():
    """Setting the variable after ``import arcticdb`` leaves the CPU-sized pool."""
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    # store.arctic_batch setdefaults at import and does not import arcticdb.
    allowed_before = stdlib | {"store"}

    def _assert_before(path: Path) -> None:
        body = _module_body(path)
        saw_guard = False
        for node in body:
            roots = _import_roots(node)
            if roots == ["arcticdb"] or (roots and "arcticdb" in roots):
                assert saw_guard, f"{path} imports arcticdb before sizing the I/O pool"
                return
            if any(root == "arcticdb" for root in roots):
                raise AssertionError(path)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                name = func.attr if isinstance(func, ast.Attribute) else ""
                if name in {"setdefault", "configure_arcticdb_io_threads"}:
                    saw_guard = True
            for root in roots:
                if not saw_guard:
                    assert root in allowed_before, (
                        f"{path} imports {root!r} before the I/O pool is sized"
                    )
        raise AssertionError(f"{path} never imports arcticdb and never sizes the pool")

    for rel in (
        "inference/stages/load_prices.py",
        "inference/stages/run_inference.py",
        "scripts/dry_run_meta_training.py",
    ):
        _assert_before(_REPO / rel)

    batch = (_REPO / "store" / "arctic_batch.py").read_text(encoding="utf-8")
    assert f'os.environ.setdefault(IO_THREADS_ENV, DEFAULT_IO_THREADS)' in batch
    # The module-level body must not import arcticdb. The import lives in the function.
    batch_body = _module_body(_REPO / "store" / "arctic_batch.py")
    for node in batch_body:
        assert "arcticdb" not in _import_roots(node)


def test_entrypoint_sets_the_pool_before_any_third_party_import():
    stdlib = set(sys.stdlib_module_names) | {"__future__"}

    def _guard_index(path: Path) -> int:
        body = _module_body(path)
        for i, node in enumerate(body):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "setdefault"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == IO_THREADS_ENV
            ):
                return i
        raise AssertionError(f"{path} does not setdefault {IO_THREADS_ENV}")

    for rel in ("inference/handler.py", "training/train_handler.py"):
        body = _module_body(_REPO / rel)
        idx = _guard_index(_REPO / rel)
        for node in body[:idx]:
            for root in _import_roots(node):
                assert root in stdlib, f"{rel} imports {root!r} before the I/O pool is sized"


def test_local_bench_last_rows_match():
    """The timing script's comparison holds on a tiny local store.

    The default 40x800 run is for a human timing note. This only checks
    identity, which is what the production change is allowed to depend on.
    """
    import importlib.util

    path = _REPO / "scripts" / "bench_arctic_read_shape.py"
    spec = importlib.util.spec_from_file_location("bench_arctic_read_shape", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.run(n_symbols=3, n_rows=20)
    assert result["identical"] is True


def test_load_prices_still_does_not_dedup():
    import inspect

    src = inspect.getsource(load_prices.load_price_data_from_arctic)
    assert ".duplicated(" not in src


def test_readers_size_the_pool_before_they_open_the_library():
    import inspect

    from inference.research_free_inference import run_research_free_inference
    from store import arctic_reader

    feature = inspect.getsource(run_research_free_inference)
    assert feature.index("configure_arcticdb_io_threads()") < feature.index(
        "import load_universe_ohlcv"
    )
    training = inspect.getsource(arctic_reader.download_from_arctic)
    assert training.index("configure_arcticdb_io_threads()") < training.index(
        "import (\n        open_universe_lib"
    )
    assert "iter_symbol_reads" in training
    assert "gc.collect" in training
