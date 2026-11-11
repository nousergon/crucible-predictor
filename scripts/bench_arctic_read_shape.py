"""Local timing of the old per-symbol full read versus the batched last-row read.

This does not contact S3, AWS, or the production ArcticDB. It uses a local
``lmdb://`` store. The numbers are the cost of the read shape on this machine,
not the Lambda's S3 round trips.

The S3 estimate printed at the end is the scanner's measurement on the live
library (alpha-engine-config-I12271), not a number this script produced:

  150 symbols, 3 I/O threads -> 32 I/O threads: 45-47 s -> 5-6 s.
  A technical read went 53 s -> 168 s as weekday daily_append grew segments
  from 1 to 6. Saturday compaction still rewrites those segments and is
  unchanged by this work.

Usage:
    python scripts/bench_arctic_read_shape.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Size the pool before arcticdb loads. Harmless for lmdb; required for the
# import-order contract the production modules follow.
os.environ.setdefault("ARCTICDB_VersionStore_NumIOThreads_int", "32")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import arcticdb as adb  # noqa: E402
import pandas as pd  # noqa: E402

from store.arctic_batch import LAST_ROW_RANGE, iter_symbol_reads  # noqa: E402


def _frame(n_rows: int, close_0: float) -> pd.DataFrame:
    idx = pd.bdate_range("2016-01-04", periods=n_rows)
    close = close_0 + pd.Series(range(n_rows), index=idx, dtype="float64")
    data = {
        "Open": close,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": 1_000_000.0,
    }
    for i in range(12):
        data[f"f{i}"] = close * (0.001 * (i + 1))
    return pd.DataFrame(data, index=idx)


def run(n_symbols: int = 40, n_rows: int = 800) -> dict[str, float | bool]:
    """Time both shapes and check the last rows match. Returns the timings."""
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    with tempfile.TemporaryDirectory(prefix="arctic-bench-") as tmp:
        lib = adb.Arctic(f"lmdb://{tmp}/store").get_library("universe", create_if_missing=True)
        for i, sym in enumerate(symbols):
            lib.write(sym, _frame(n_rows, float(i + 1)))

        t0 = time.perf_counter()
        legacy: dict[str, pd.Series] = {}
        for sym in symbols:
            legacy[sym] = lib.read(sym).data.iloc[-1]
        legacy_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        fresh: dict[str, pd.Series] = {}
        for sym, df, err in iter_symbol_reads(lib, symbols, row_range=LAST_ROW_RANGE):
            if err is not None or df is None or df.empty:
                raise RuntimeError(f"{sym}: {err}")
            fresh[sym] = df.iloc[-1]
        fresh_s = time.perf_counter() - t1

    identical = True
    for sym in symbols:
        if not fresh[sym].equals(legacy[sym]):
            identical = False
            break
    return {
        "n_symbols": float(n_symbols),
        "n_rows": float(n_rows),
        "legacy_full_read_s": legacy_s,
        "batched_last_row_s": fresh_s,
        "identical": identical,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=40)
    parser.add_argument("--rows", type=int, default=800)
    args = parser.parse_args(argv)
    result = run(n_symbols=args.symbols, n_rows=args.rows)
    print(
        "LOCAL lmdb:// ArcticDB (not S3, not the production library): "
        f"{int(result['n_symbols'])} symbols x {int(result['n_rows'])} rows"
    )
    print(f"  per-symbol full read + iloc[-1]: {result['legacy_full_read_s']:.3f} s")
    print(f"  read_batch last row, chunks of 150: {result['batched_last_row_s']:.3f} s")
    print(f"  last rows identical: {result['identical']}")
    print(
        "ESTIMATED S3 gain (not measured here; scanner measurement on the "
        "live library, alpha-engine-config-I12271): 150 symbols, 3 -> 32 I/O "
        "threads, 45-47 s -> 5-6 s. The predictor previously did that style of "
        "read twice per run, and the feature pass read every row to keep "
        "iloc[-1]. Saturday compaction is unchanged."
    )
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
