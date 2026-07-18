"""
store/arctic_reader.py — Load universe data from ArcticDB for predictor training.

Reads per-ticker DataFrames (OHLCV + 53 pre-computed features) from the
ArcticDB universe library and writes them as parquets to a local cache
directory. This makes ArcticDB data compatible with the existing
build_regression_arrays() pipeline in dataset.py.

Also writes macro series (SPY, VIX, etc.) and sector_map.json so the
downstream pipeline finds everything it expects.

Usage:
    from store.arctic_reader import download_from_arctic

    coverage = download_from_arctic(bucket, local_dir)
    n_files = coverage["n_written"]

config#2882 — per-ticker/per-macro-series read failures used to be swallowed
at DEBUG (invisible at default INFO logging) with no record of HOW MANY
reads failed. A partial ArcticDB throttling/connectivity episode could
silently drop most of the universe while still returning a nonzero
``n_files``, so the only downstream gate (``n_files == 0``) never tripped —
a challenger could train on a crippled dataset, self-report a plausible
CPCV IC on the reduced data it saw, and still win weekly ModelZoo
promotion. Failures are now logged at WARN (ticker/series id + exception)
and the return value is a coverage dict — ``n_written`` (back-compat count),
``n_expected`` (symbols listed), ``n_failed``, and ``coverage_ratio`` — so
callers can gate on PARTIAL loss, not just total loss.
"""

from __future__ import annotations

import json
import logging
import os
import time

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"
ARCTIC_PREFIX = "arcticdb"


def download_from_arctic(
    bucket: str,
    local_dir: str | os.PathLike,
    universe_lib: str = "universe",
) -> dict:
    """
    Read all universe + macro symbols from ArcticDB and write as parquets
    to local_dir, matching the legacy per-ticker OHLCV parquet format
    (the now-removed S3 download_price_cache() fallback produced the same
    shape; ArcticDB is canonical since PR #6).

    The key difference: ArcticDB DataFrames include pre-computed feature
    columns alongside OHLCV. build_regression_arrays() in dataset.py
    detects these and skips inline compute_features().

    Parameters
    ----------
    universe_lib : str
        ArcticDB library to read the per-ticker stock universe from. Default
        ``"universe"`` (the canonical production library — live behaviour,
        unchanged: opened via the lib's ``open_universe_lib`` chokepoint). A
        total-return SHADOW run (PR7-7b) passes ``"universe_crsp"`` — the
        scratch library ne-data (#554) builds on a clean CRSP total-return
        basis (``Close`` = split-adjusted level + a new ``total_return_close``
        column, with the 53 features RECOMPUTED on the total-return series
        under the SAME column names). The macro library is always ``"macro"``
        regardless of basis.

    Returns
    -------
    dict with:
      ``n_written``       : total parquet files written (universe + macro) —
                             same count the pre-config#2882 int return gave.
      ``n_expected``      : total symbols listed (universe + macro) BEFORE
                             any read attempt — the coverage denominator.
      ``n_failed``        : symbols whose ArcticDB read raised (universe +
                             macro combined); empty-dataframe skips are NOT
                             counted as failures (that's valid "no data yet"
                             for a new listing, not a read error).
      ``coverage_ratio``  : ``n_written / n_expected`` (``1.0`` when
                             ``n_expected == 0`` — nothing was expected, so
                             nothing was missed).
      ``failed_universe`` : list of universe tickers whose read raised.
      ``failed_macro``    : list of macro series whose read raised.
    """
    t0 = time.time()
    local_dir = str(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    from nousergon_lib.arcticdb import (
        open_universe_lib, open_macro_lib, open_arctic,
    )
    if universe_lib == "universe":
        # Preserve the live chokepoint exactly (its RuntimeError wrapping).
        universe = open_universe_lib(bucket)
    else:
        # Shadow basis — open the named scratch library directly.
        arctic = open_arctic(bucket)
        try:
            universe = arctic.get_library(universe_lib)
        except Exception as exc:
            raise RuntimeError(
                f"ArcticDB {universe_lib!r} library open failed on bucket "
                f"{bucket!r}: {exc}"
            ) from exc
    macro_lib = open_macro_lib(bucket)

    n_written = 0
    failed_universe: list[str] = []
    failed_macro: list[str] = []

    # Write stock tickers from universe library
    symbols = universe.list_symbols()
    log.info(
        "[data_source=arcticdb] Reading %d symbols from library '%s'...",
        len(symbols), universe_lib,
    )

    for i, ticker in enumerate(symbols):
        try:
            df = universe.read(ticker).data
            if df.empty:
                continue
            out_path = os.path.join(local_dir, f"{ticker}.parquet")
            df.to_parquet(out_path, engine="pyarrow", compression="snappy")
            n_written += 1
        except Exception as exc:
            # config#2882 — WARN (not DEBUG): a partial read-failure episode
            # (throttling/connectivity/corrupted symbol) must be visible at
            # default INFO-level logging, not require a DEBUG-level re-run to
            # even notice data was silently dropped.
            failed_universe.append(ticker)
            log.warning("Failed to read ticker %s from ArcticDB: %s", ticker, exc)

        if (i + 1) % 200 == 0:
            log.info("  Written %d/%d symbols", i + 1, len(symbols))

    # Write macro series from macro library
    macro_symbols = macro_lib.list_symbols()
    for key in macro_symbols:
        try:
            df = macro_lib.read(key).data
            if df.empty:
                continue
            out_path = os.path.join(local_dir, f"{key}.parquet")
            df.to_parquet(out_path, engine="pyarrow", compression="snappy")
            n_written += 1
        except Exception as exc:
            # config#2882 — WARN, see above.
            failed_macro.append(key)
            log.warning("Failed to read macro series %s from ArcticDB: %s", key, exc)

    # Write sector_map.json from S3 (not stored in ArcticDB)
    try:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="data/sector_map.json")
        sector_map = json.loads(obj["Body"].read())
        map_path = os.path.join(local_dir, "sector_map.json")
        with open(map_path, "w") as f:
            json.dump(sector_map, f)
        log.info("[data_source=arcticdb] Wrote sector_map.json (%d mappings)", len(sector_map))
    except Exception as exc:
        log.warning("Failed to load sector_map.json from S3: %s", exc)

    n_expected = len(symbols) + len(macro_symbols)
    n_failed = len(failed_universe) + len(failed_macro)
    # n_expected == 0 → nothing was asked for, so nothing was missed (ratio
    # 1.0, not a divide-by-zero / NaN that a downstream threshold check would
    # have to special-case).
    coverage_ratio = (n_written / n_expected) if n_expected > 0 else 1.0

    elapsed = time.time() - t0
    log.info(
        "[data_source=arcticdb] Cache populated in %.1fs: %d/%d files written "
        "(coverage=%.4f, failed=%d) to %s",
        elapsed, n_written, n_expected, coverage_ratio, n_failed, local_dir,
    )
    if n_failed:
        log.warning(
            "[data_source=arcticdb] %d/%d symbol reads failed this run "
            "(coverage=%.4f) — universe_failed=%d macro_failed=%d. See WARN "
            "lines above for the individual ticker/series + exception.",
            n_failed, n_expected, coverage_ratio,
            len(failed_universe), len(failed_macro),
        )
    return {
        "n_written": n_written,
        "n_expected": n_expected,
        "n_failed": n_failed,
        "coverage_ratio": coverage_ratio,
        "failed_universe": failed_universe,
        "failed_macro": failed_macro,
    }
