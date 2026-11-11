"""Chunked ArcticDB ``read_batch`` and the S3 I/O pool size.

ArcticDB reads ``ARCTICDB_*`` configuration once, at import, and otherwise
sizes ``VersionStore.NumIOThreads`` from the CPU count (about 1.5x). On a
Lambda that is a handful of threads. A ``read_batch`` of hundreds of symbols
is bound by S3 round trips, not by bytes: crucible-research measured 150
symbols on this same library at 45-47 s with 3 threads and 5-6 s with 32
(alpha-engine-config-I12271). Weekday ``daily_append`` adds a segment per
session until Saturday compaction rewrites the symbol, and that segment
growth turned a 53 s technical read into 168 s. ``setdefault`` so an
explicit environment value still wins.

Importing this module sets the default and does not import arcticdb.
Call :func:`configure_arcticdb_io_threads` (or import this module) before
any ``import arcticdb``.

Chunk size 150 is the universe chunk nousergon-data settled on after the
2026-05-11 daily_append OOM (``UNIVERSE_CHUNK_SIZE``): one chunk of full
histories is about the resident set that fit, and it is the batch size the
32-thread measurement used. The predictor Lambda is 3072 MB. Last-row
inference chunks are far smaller than that; training still releases each
chunk before reading the next.

The live library was not contacted from this change. Those timings are the
scanner's measurement, reused as an estimate for this process. A local
``mem://`` timing lives in ``scripts/bench_arctic_read_shape.py`` and is
labeled as such.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

IO_THREADS_ENV = "ARCTICDB_VersionStore_NumIOThreads_int"
DEFAULT_IO_THREADS = "32"
READ_BATCH_CHUNK_SIZE = 150
# Same row ``DataFrame.iloc[-1]`` selects. ArcticDB's ``row_range`` matches
# ``iloc[start:end]``, including a negative start (documented on Library.read).
LAST_ROW_RANGE: tuple[int, None] = (-1, None)

# Module import is the earliest safe point: stdlib only, no arcticdb.
os.environ.setdefault(IO_THREADS_ENV, DEFAULT_IO_THREADS)


def configure_arcticdb_io_threads(n: int | str = DEFAULT_IO_THREADS) -> str:
    """Size the I/O pool if nothing has set it yet. Does not import arcticdb.

    Prefers ``nousergon_lib.arcticdb.configure_arcticdb_io_threads`` when the
    installed pin exports it. This repo's pin stays lockstep with
    crucible-backtester (``LibPinDriftGate`` / ``lib-pin-cross-repo-guard``),
    so until that bump the same ``setdefault`` above is what runs.
    """
    lib_configure = None
    try:
        from nousergon_lib.arcticdb import (
            configure_arcticdb_io_threads as lib_configure,
        )
    except ImportError:
        lib_configure = None
    if lib_configure is not None and lib_configure is not configure_arcticdb_io_threads:
        return str(lib_configure(n))
    os.environ.setdefault(IO_THREADS_ENV, str(n))
    return os.environ[IO_THREADS_ENV]


def iter_symbol_reads(
    lib: object,
    symbols: Iterable[str],
    *,
    date_range: tuple[object, object] | None = None,
    row_range: tuple[int | None, int | None] | None = None,
    columns: list[str] | None = None,
    chunk_size: int = READ_BATCH_CHUNK_SIZE,
    after_chunk: Callable[[], object] | None = None,
) -> Iterator[tuple[str, pd.DataFrame | None, str | None]]:
    """Yield ``(symbol, frame, error)`` for ``symbols``, in input order.

    One ``lib.read_batch`` per chunk. A per-symbol ``DataError`` (or a
    payload that is not a DataFrame) sets ``error`` and leaves ``frame``
    None. An empty DataFrame is a successful read: ``frame`` is empty and
    ``error`` is None, so the caller keeps its own empty-versus-failure
    rule. If the batch call itself raises, every symbol in that chunk is
    reported as an error and the next chunk is still attempted — the same
    per-symbol ``try`` the old ``lib.read`` loop used.

    ``date_range`` and ``row_range`` are passed through to ``ReadRequest``.
    Only one is set; ArcticDB rejects both. Omit both for a full-symbol read.
    """
    configure_arcticdb_io_threads()
    import arcticdb as adb
    import pandas as pd

    if date_range is not None and row_range is not None:
        raise ValueError("ArcticDB accepts date_range or row_range, not both")

    ordered = list(symbols)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    def _request(symbol: str) -> object:
        kwargs: dict[str, object] = {"symbol": symbol}
        if date_range is not None:
            kwargs["date_range"] = date_range
        if row_range is not None:
            kwargs["row_range"] = row_range
        if columns is not None:
            kwargs["columns"] = list(columns)
        return adb.ReadRequest(**kwargs)

    for start in range(0, len(ordered), chunk_size):
        chunk = ordered[start : start + chunk_size]
        try:
            results = lib.read_batch([_request(sym) for sym in chunk])  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — per-symbol loop used to catch this
            message = f"{type(exc).__name__}: {exc}"
            for sym in chunk:
                yield sym, None, message
            if after_chunk is not None:
                after_chunk()
            continue
        if len(results) != len(chunk):
            message = (
                f"read_batch returned {len(results)} results for {len(chunk)} symbols"
            )
            for sym in chunk:
                yield sym, None, message
            if after_chunk is not None:
                after_chunk()
            continue
        for sym, res in zip(chunk, results):
            yield sym, *_frame_or_error(res, pd)
        del results
        if after_chunk is not None:
            after_chunk()


def _frame_or_error(res: object, pd: object) -> tuple[pd.DataFrame | None, str | None]:
    """Split one ``read_batch`` element into a frame or an error string."""
    message = getattr(res, "exception_string", None)
    if isinstance(message, str) and message:
        return None, message
    data = getattr(res, "data", None)
    if not isinstance(data, pd.DataFrame):  # type: ignore[attr-defined]
        return None, "ArcticDB read returned no DataFrame"
    return data, None
