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

alpha-engine-config-I9290 — ``coverage_ratio`` counts SYMBOLS, never ROWS
PER SYMBOL, and that blindness is what let ``macro/VIX3M`` run at 16 rows
against SPY's ~2514 for two full weekly rotations while every manifest
recorded ``coverage_ratio: 1.0``. This function now also returns
``per_symbol`` — ``{symbol: {rows, first_date, last_date}}`` — which
``training/data_completeness.py`` grades against the reference series and
the scored window. ``coverage_ratio`` is retained for back-compat and is
explicitly NOT the completeness number: read
``data_completeness.input_completeness_ratio`` for that.

Empty-DataFrame reads are no longer an unrecorded ``continue`` either: a
symbol that exists with zero rows is named in ``empty_universe`` /
``empty_macro`` and appears in ``per_symbol`` with ``rows: 0``, so "the
symbol is there but holds nothing" and "the symbol was written" stop
rendering identically.
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
      ``per_symbol``      : ``{symbol: {"rows", "first_date", "last_date"}}``
                             for every symbol READ, including empty ones
                             (alpha-engine-config-I9290). This is the input to
                             ``training.data_completeness.evaluate_completeness``,
                             which grades ROWS and DATE SPAN — the two things
                             ``coverage_ratio`` structurally cannot see.
      ``empty_universe``  : universe tickers that read OK holding ZERO rows.
      ``empty_macro``     : macro series that read OK holding ZERO rows.
                             Both were an unrecorded ``continue`` before I9290.
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
    # alpha-engine-config-I9290 — rows + date span per symbol. The map the
    # symbol-count coverage_ratio cannot express.
    per_symbol: dict[str, dict] = {}
    empty_universe: list[str] = []
    empty_macro: list[str] = []
    from training.data_completeness import measure_symbol_coverage

    # Write stock tickers from universe library
    symbols = universe.list_symbols()
    log.info(
        "[data_source=arcticdb] Reading %d symbols from library '%s'...",
        len(symbols), universe_lib,
    )

    for i, ticker in enumerate(symbols):
        try:
            df = universe.read(ticker).data
            per_symbol[ticker] = measure_symbol_coverage(df)
            if df.empty:
                # I9290 — NOT a silent skip. An empty symbol is a distinct
                # finding from a failed read and from a written file; it is
                # named here and carries rows=0 into per_symbol so the
                # completeness gate can see it.
                empty_universe.append(ticker)
                log.warning(
                    "ArcticDB universe symbol %s read OK but holds ZERO rows "
                    "— recorded as empty, not skipped silently.", ticker,
                )
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
            per_symbol[key] = measure_symbol_coverage(df)
            if df.empty:
                # I9290 — see the universe branch above.
                empty_macro.append(key)
                log.warning(
                    "ArcticDB macro series %s read OK but holds ZERO rows — "
                    "recorded as empty, not skipped silently.", key,
                )
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
    if empty_universe or empty_macro:
        log.warning(
            "[data_source=arcticdb] alpha-engine-config-I9290: %d symbol(s) "
            "read OK but held ZERO rows (universe=%d, macro=%d) — these are "
            "NOT counted as failures and NOT counted as written, so "
            "coverage_ratio cannot see them. universe=%s macro=%s",
            len(empty_universe) + len(empty_macro), len(empty_universe),
            len(empty_macro), empty_universe[:20], empty_macro[:20],
        )
    return {
        "n_written": n_written,
        "n_expected": n_expected,
        "n_failed": n_failed,
        # I9290 — SYMBOL-count ratio. Retained for back-compat; it is not
        # the completeness measure. See per_symbol + data_completeness.
        "coverage_ratio": coverage_ratio,
        "failed_universe": failed_universe,
        "failed_macro": failed_macro,
        "per_symbol": per_symbol,
        "empty_universe": empty_universe,
        "empty_macro": empty_macro,
    }
