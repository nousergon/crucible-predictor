"""Stage: load_universe — Determine ticker universe and load sector map."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import config as cfg
from inference.pipeline import PipelineContext

log = logging.getLogger(__name__)

# Fallback universe if signals.json is unavailable
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B",
    "LLY", "JPM", "V", "UNH", "XOM", "COST", "TSLA", "HD",
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _signals_fallback_keys(date_str: str) -> list[str]:
    """Return S3 keys to try in priority order:
    today's signals → prior 5 weekdays' signals → signals/latest.json.

    Mirrors `compute_coverage_delta`'s lookup chain so the predictor's
    weekday paths and the executor's coverage gate agree on which
    research snapshot is current.
    """
    from datetime import date as _date, timedelta as _td

    keys: list[str] = []
    try:
        start = _date.fromisoformat(date_str)
        for days_back in range(6):
            candidate = start - _td(days=days_back)
            if candidate.weekday() >= 5:
                continue
            keys.append(f"signals/{candidate}/signals.json")
    except Exception:
        pass
    keys.append("signals/latest.json")
    return keys


def _read_s3_signals_payload(s3, bucket: str, key: str) -> dict | None:
    """Read a single S3 signals key. Return None on miss/permission/parse
    error so callers can walk to the next key in the fallback chain."""
    from botocore.exceptions import ClientError
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "AccessDenied", "403", "404"):
            return None
        raise
    except Exception:
        return None


def _load_signals_payload_with_fallback(
    s3, bucket: str, date_str: str
) -> dict:
    """Return the first non-empty signals.json payload in the fallback
    chain. Empty dict if nothing resolves — callers must tolerate.

    Brief regime defect (2026-05-11): population/latest.json carries a
    `market_regime` value that drifts from `signals/latest.json` (two
    research producers, no shared source of truth — population's writer
    today serves the pre-critic regime; signals.json carries the post-
    critic value). Without falling back to signals.json on weekday runs
    when today's daily snapshot doesn't exist yet, the predictor's
    morning brief displays the pre-critic regime, and downstream
    consumers (regime-conditional veto thresholds, sector_modifiers
    used for sector-by-sector boost) see drift. This helper exists so
    every load path in the file shares one fallback chain.
    """
    for key in _signals_fallback_keys(date_str):
        payload = _read_s3_signals_payload(s3, bucket, key)
        if payload:
            return payload
    return {}


def _read_buy_candidates_from_signals(
    s3, bucket: str, date_str: str
) -> list[str]:
    """Read research's `buy_candidates` ticker list from signals.

    Walks the same fallback chain `compute_coverage_delta` uses so weekday
    runs see Saturday's signals. Returns an empty list on any failure or
    empty payload — the caller treats that as "no candidates to union".
    """
    for key in _signals_fallback_keys(date_str):
        payload = _read_s3_signals_payload(s3, bucket, key)
        if payload is None:
            continue
        candidates = payload.get("buy_candidates") or []
        tickers: list[str] = []
        for entry in candidates:
            if isinstance(entry, dict):
                t = entry.get("ticker")
            else:
                t = entry
            if isinstance(t, str) and t:
                tickers.append(t.upper())
        if tickers:
            return tickers
    return []


# ── Universe membership resolution (alpha-engine-config-I4818) ───────────────
#
# The daily scoring universe resolves from the versioned membership artifact
# crucible-research's Scanner publishes each weekly cycle
# (``universe_membership/{date}/membership.json``), NOT from
# ``population/latest.json``.
#
# History that produced this contract: the population artifact's sole producer
# (the multi-agent research graph's ``archive_writer``) was retired with the
# research state, and nothing repointed this consumer. ``load_watchlist`` kept
# reading ``population/latest.json`` successfully — the read never failed, the
# file simply stopped changing — so the predictor scored a FROZEN 2026-07-10
# list of 25 names for three weekly cycles while every daily run looked green.
# The lesson encoded below: a universe read must fail on STALENESS, not just on
# absence. An artifact that resolves fine but describes a dead cycle is the
# failure mode, and it is invisible to any liveness check that only asks
# "did the read succeed?".

# Maximum age of the membership artifact before the predictor refuses to run.
# The producer's cadence is weekly (Saturday), so a healthy Friday read is at
# most 6 days old; 10 gives one weekend of slack for a delayed weekly SF while
# still catching a genuinely dead producer inside one cycle.
MEMBERSHIP_MAX_AGE_DAYS = 10

_MEMBERSHIP_LATEST_KEY = "universe_membership/latest.json"


def _membership_dated_key(date_str: str) -> str:
    return f"universe_membership/{date_str}/membership.json"


class UniverseResolutionError(RuntimeError):
    """Raised when no sufficiently fresh universe membership can be resolved.

    Deliberately fatal. Scoring an unknown or stale universe silently produces
    a normal-looking predictions file for the wrong names, which the executor
    then trades on — strictly worse than a red run (feedback_no_silent_fails).
    """


def _read_membership(s3, bucket: str, date_str: str) -> dict | None:
    """Resolve the freshest membership artifact, or None.

    Chain: ``latest.json`` first (the producer's own pointer), then dated keys
    walked backwards up to ``MEMBERSHIP_MAX_AGE_DAYS``. The dated walk is not
    redundant belt-and-braces — it makes resolution independent of a single
    mutable pointer, so a pointer that fails to update (or has not been written
    yet, as during the artifact's own rollout) degrades to "slightly older
    membership" rather than to a hard outage.

    Staleness is judged on the artifact's own ``run_date``, never on S3
    LastModified: a re-upload of an old cycle must not read as fresh.
    """
    from datetime import date as _date, timedelta as _td

    candidates: list[dict] = []
    latest = _read_s3_signals_payload(s3, bucket, _MEMBERSHIP_LATEST_KEY)
    if latest:
        candidates.append(latest)

    try:
        start = _date.fromisoformat(date_str)
    except Exception:
        start = None
    if start is not None:
        for days_back in range(MEMBERSHIP_MAX_AGE_DAYS + 1):
            probe = start - _td(days=days_back)
            payload = _read_s3_signals_payload(
                s3, bucket, _membership_dated_key(str(probe))
            )
            if payload:
                candidates.append(payload)
                break

    fresh: dict | None = None
    for payload in candidates:
        run_date = payload.get("run_date")
        if not run_date:
            continue
        try:
            age = (_date.fromisoformat(date_str) - _date.fromisoformat(run_date)).days
        except Exception:
            continue
        if age > MEMBERSHIP_MAX_AGE_DAYS:
            log.warning(
                "Universe membership %s is %d days old (limit %d) — rejecting",
                run_date, age, MEMBERSHIP_MAX_AGE_DAYS,
            )
            continue
        if fresh is None or run_date > fresh.get("run_date", ""):
            fresh = payload
    return fresh


def _read_holdings(s3, bucket: str) -> set[str]:
    """Currently-held tickers from the executor's published EOD surface
    (``trades/eod_pnl.csv``, last row's ``positions_snapshot``).

    Unioned into the scoring universe so a held name is never scored blind: the
    scanner cut is forward-looking (what to buy) and can legitimately drop a
    name the book still holds, but the executor needs a prediction to decide
    whether to keep holding it.

    Fail-soft by design — the holdings union WIDENS the universe, so an
    unreadable book yields a narrower-but-valid run rather than no run. The
    narrowing is logged, and the cut itself (the load-bearing part) still comes
    from the membership artifact, which is NOT fail-soft.
    """
    import csv
    import io as _io

    try:
        obj = s3.get_object(Bucket=bucket, Key="trades/eod_pnl.csv")
        rows = list(csv.DictReader(_io.StringIO(obj["Body"].read().decode("utf-8"))))
    except Exception as exc:  # noqa: BLE001 — see docstring; widening input only
        log.warning(
            "Holdings union: trades/eod_pnl.csv unreadable (%s) — universe will "
            "not include held names not already in the scanner cut", exc,
        )
        return set()
    if not rows:
        return set()
    snapshot_raw = rows[-1].get("positions_snapshot")
    if not snapshot_raw:
        return set()
    try:
        snapshot = json.loads(snapshot_raw)
    except Exception as exc:  # noqa: BLE001 — see docstring
        log.warning("Holdings union: positions_snapshot unparseable (%s)", exc)
        return set()
    if isinstance(snapshot, dict):
        return {str(t).upper() for t in snapshot}
    if isinstance(snapshot, list):
        return {
            str(p.get("ticker") or p.get("symbol") or "").upper()
            for p in snapshot
            if isinstance(p, dict)
        } - {""}
    return set()


def resolve_universe_from_membership(
    s3, bucket: str, date_str: str
) -> tuple[dict[str, str], dict]:
    """``({ticker: source}, membership_artifact)`` for ``date_str``.

    Universe = the membership artifact's designated predictor cut (today the
    scanner's ~60-name gate cut) ∪ currently-held names. ``buy_candidates`` are
    unioned by the caller from signals, preserving the existing coverage-gap
    contract with the Step Function.

    Raises ``UniverseResolutionError`` when no membership artifact within
    ``MEMBERSHIP_MAX_AGE_DAYS`` resolves, or when the artifact names a cut it
    does not contain.
    """
    membership = _read_membership(s3, bucket, date_str)
    if not membership:
        raise UniverseResolutionError(
            f"no universe membership artifact within {MEMBERSHIP_MAX_AGE_DAYS} "
            f"days of {date_str} (looked at {_MEMBERSHIP_LATEST_KEY} and dated "
            "keys). The weekly Scanner publishes it — check the weekly SF."
        )

    cut_name = membership.get("predictor_universe_cut")
    cut = (membership.get("cuts") or {}).get(cut_name) or {}
    tickers = [str(t).upper() for t in (cut.get("tickers") or [])]
    if not tickers:
        raise UniverseResolutionError(
            f"universe membership {membership.get('run_date')} names cut "
            f"{cut_name!r} but it is empty or absent — refusing to score an "
            "unknown universe"
        )

    sources = {t: "scanner_candidate" for t in tickers}
    for t in _read_holdings(s3, bucket):
        sources[t] = "both" if t in sources else "held"

    log.info(
        "Universe: %d tickers from membership %s (cut=%s: %d, held: %d)",
        len(sources), membership.get("run_date"), cut_name, len(tickers),
        sum(1 for v in sources.values() if v in ("held", "both")),
    )
    return sources, membership


# ``_merge_canonical_macro_into`` was REMOVED with I4818. It existed to reconcile
# two producers of the same macro fields — population/latest.json (pre-critic
# regime) and signals.json (post-critic) — after the 2026-05-11 "Market Regime:
# NEUTRAL" brief defect. With population/ no longer read, signals.json is the
# sole source of macro context and there is nothing left to overlay. Deleting it
# rather than leaving it unused is deliberate: a dormant merge helper invites a
# future caller to reintroduce exactly the multi-writer drift it was patching.


# ── Universe functions (migrated from daily_predict.py) ──────────────────────

def get_universe_tickers(
    s3_bucket: str, date_str: Optional[str] = None,
) -> tuple[list[str], dict]:
    """
    Read the active ticker universe from signals.json in S3, walking
    the fallback chain (today → prior 5 weekdays → signals/latest.json)
    so weekday runs see the most recent Saturday SF snapshot.

    Falls back to ``_FALLBACK_TICKERS`` only when no signals payload
    resolves anywhere in the chain.

    Parameters
    ----------
    s3_bucket : S3 bucket name.
    date_str :  Date string YYYY-MM-DD. Uses today if None.

    Returns
    -------
    (tickers, signals_data) — ticker list and full signals payload for email.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        import boto3
        s3 = boto3.client("s3")
        signals_data = _load_signals_payload_with_fallback(
            s3, s3_bucket, date_str
        )
        signals_list = signals_data.get("signals", []) if signals_data else []
        tickers = [s["ticker"] for s in signals_list if "ticker" in s]
        if tickers:
            log.info(
                "Universe: %d tickers from signals fallback chain", len(tickers)
            )
            return tickers, signals_data
        log.warning(
            "Universe: signals payload empty / no tickers extractable — "
            "using fallback universe"
        )
    except Exception as exc:
        log.info(
            "Universe: signals load failed (%s) — using fallback universe", exc
        )

    log.info("Using fallback universe: %d tickers", len(_FALLBACK_TICKERS))
    return _FALLBACK_TICKERS, {}


def load_watchlist(
    path: str,
    s3_bucket: Optional[str] = None,
    date_str: Optional[str] = None,
) -> tuple[list[str], dict[str, str]]:
    """
    Build a focused prediction universe from Research's population or signals.

    Resolution for ``path="auto"`` (alpha-engine-config-I4818):
      1. universe_membership/{latest|date}/membership.json — the versioned cut
         contract published by the weekly Scanner. Its designated predictor cut
         ∪ held positions ∪ signals.buy_candidates IS the universe. Fails LOUD
         (``UniverseResolutionError``) when absent or staler than
         ``MEMBERSHIP_MAX_AGE_DAYS`` — see the module-level contract note.
      2. signals/{date}/signals.json — read for macro context (regime, sector
         modifiers) and buy_candidates only; NOT as a universe fallback.

    ``population/latest.json`` is no longer read. Its producer was retired with
    the multi-agent research graph, and reading it kept the predictor pinned to
    a frozen 2026-07-10 universe for three weekly cycles.

    Parameters
    ----------
    path      : "auto" → resolve from S3 (membership + signals).
                Any other string → local file path to signals.json or
                population.json for offline / dry-run use.
    s3_bucket : S3 bucket name. Required when path="auto".
    date_str  : YYYY-MM-DD override. Defaults to today.

    Returns
    -------
    tickers : Deduplicated, sorted list of tickers.
    sources : Dict mapping ticker → "scanner_candidate" | "held" | "buy_candidate"
              | "both" (S3 path), or "population" | "tracked" (local-file path).
    data    : Raw JSON payload (signals) for the morning brief / macro context.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    data = None

    # ── Load raw payload ──────────────────────────────────────────────────────
    if path == "auto":
        if not s3_bucket:
            raise ValueError("s3_bucket is required when --watchlist auto is used")

        import boto3
        s3 = boto3.client("s3")

        # Universe = membership artifact's predictor cut ∪ held ∪ buy_candidates.
        # UniverseResolutionError propagates deliberately: no fallback universe
        # exists that is better than failing (see the contract note above).
        sources, membership = resolve_universe_from_membership(s3, s3_bucket, date_str)

        # Union research's signals.buy_candidates so the main pass scores them
        # inline. Without this, any buy_candidate outside the predictor's
        # universe required a second supplemental Lambda invocation via the
        # Step Function's coverage-gap Choice state — fragile for manual
        # recovery (the operator must remember the second call) and adds
        # latency to the weekday SF (~+30s per cold-start). The supplemental
        # path remains in place as a safety net for races (signals updated
        # between this read and check_coverage), but on the happy path it now
        # sees `has_gap=False` and never fires. See 2026-04-27 incident —
        # manual recovery was blocked by the executor's coverage-gap gate
        # after the SF aborted on the MorningEnrich SSM TimedOut, requiring a
        # second hand-crafted Lambda invoke for the 6 missing buy_candidates.
        buy_tickers = _read_buy_candidates_from_signals(s3, s3_bucket, date_str)
        if buy_tickers:
            new_tickers = []
            for t in buy_tickers:
                u = t.upper()
                if u in sources:
                    sources[u] = "both"
                else:
                    sources[u] = "buy_candidate"
                    new_tickers.append(u)
            log.info(
                "Watchlist: unioned %d buy_candidates from signals "
                "(%d new beyond the membership cut)",
                len(buy_tickers), len(new_tickers),
            )

        # signals.json supplies the macro context the morning brief and the
        # regime-conditional veto threshold read (market_regime,
        # sector_ratings, sector_modifiers). The membership artifact carries
        # membership only — deliberately, so the two producers cannot drift on
        # a shared field (see the 2026-05-11 multi-writer note below).
        data = _load_signals_payload_with_fallback(s3, s3_bucket, date_str) or {}
        if not data:
            log.warning(
                "Watchlist: no signals payload resolved for %s — universe is "
                "intact (membership-sourced) but the morning brief will omit "
                "macro context", date_str,
            )
        data["universe_membership_run_date"] = membership.get("run_date")

        tickers = sorted(sources.keys())
        log.info(
            "Watchlist: %d tickers from universe membership %s "
            "(+ holdings + buy_candidates union)",
            len(tickers), membership.get("run_date"),
        )
        return tickers, sources, data

    else:
        local_path = Path(path)
        if not local_path.exists():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                "Use --watchlist auto to pull from S3, or provide a local path."
            )
        data = json.loads(local_path.read_text())
        log.info("Watchlist: loaded from %s", path)

        # Check if this is a population file
        if "population" in data and isinstance(data["population"], list):
            pop_tickers = [p["ticker"] for p in data["population"] if "ticker" in p]
            if pop_tickers:
                sources = {t.upper(): "population" for t in pop_tickers}
                tickers = sorted(sources.keys())
                log.info("Watchlist: %d tickers from population file", len(tickers))
                return tickers, sources, data

    # ── Extract tickers from signals.json ───────────────────────────────────
    universe_tickers = {
        e["ticker"].upper() for e in data.get("universe", []) if "ticker" in e
    }

    sources: dict[str, str] = {}
    for t in universe_tickers:
        sources[t] = "tracked"

    tickers = sorted(sources.keys())
    log.info(
        "Watchlist: %d universe tickers",
        len(tickers),
    )
    return tickers, sources, data


# ── Stage entry point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Load ticker universe from watchlist or signals, plus sector map.

    Supplemental-scoring mode: if ``ctx.explicit_tickers`` is non-empty, the
    universe is forced to that list (regardless of signals.json content).
    signals_data is still loaded from S3 when available so the email renderer
    and ticker-source annotations have the research context; if signals.json
    is unavailable in this mode we proceed with an empty dict — the Step
    Function coverage-gap path has already validated the tickers exist in
    signals.json before invoking, so we don't re-enforce that here.
    """

    # ── Explicit-tickers (supplemental-scoring) short-circuit ────────────────
    if ctx.explicit_tickers:
        ctx.tickers = list(ctx.explicit_tickers)
        # The supplemental invocation is the terminal one in the Step Function
        # chain — its email carries the Research Brief. Research writes signals
        # on Saturdays, so `signals/{weekday}/signals.json` is absent; walk the
        # same fallback chain `compute_coverage_delta` uses so the brief
        # renders on weekday re-invokes.
        from inference.coverage_check import _read_s3_json
        # population/latest.json dropped from this chain with I4818 — its
        # producer is retired, so it can only ever contribute a stale regime.
        ctx.signals_data = (
            _read_s3_json(ctx.bucket, f"signals/{ctx.date_str}/signals.json")
            or _read_s3_json(ctx.bucket, "signals/latest.json")
            or {}
        )
        if not ctx.signals_data:
            log.warning(
                "Supplemental mode: no signals payload found (signals/%s, "
                "signals/latest both empty) — email will omit the "
                "Research Brief", ctx.date_str,
            )
        # Annotate sources from signals_data when possible
        buy_set = {
            e.get("ticker", "").upper()
            for e in (ctx.signals_data.get("buy_candidates") or [])
            if isinstance(e, dict)
        }
        universe_set = {
            e.get("ticker", "").upper()
            for e in (ctx.signals_data.get("universe") or [])
            if isinstance(e, dict)
        }
        ctx.ticker_sources = {
            t: (
                "buy_candidate" if t in buy_set
                else "tracked" if t in universe_set
                else "supplemental"
            )
            for t in ctx.tickers
        }
        log.info(
            "Supplemental-scoring mode: %d explicit tickers (%s)",
            len(ctx.tickers), ", ".join(ctx.tickers[:10]) + ("…" if len(ctx.tickers) > 10 else ""),
        )
    # ── Ticker universe ──────────────────────────────────────────────────────
    elif ctx.watchlist_path:
        ctx.tickers, ctx.ticker_sources, ctx.signals_data = load_watchlist(
            path=ctx.watchlist_path,
            s3_bucket=ctx.bucket,
            date_str=ctx.date_str,
        )
    else:
        ctx.tickers, ctx.signals_data = get_universe_tickers(ctx.bucket, ctx.date_str)

    # ── Sector map ───────────────────────────────────────────────────────────
    sector_map_path = Path("data/cache/sector_map.json")
    if sector_map_path.exists():
        try:
            ctx.sector_map = json.loads(sector_map_path.read_text())
            log.info("Sector map loaded: %d mappings", len(ctx.sector_map))
        except Exception as exc:
            log.warning("Could not load sector_map.json: %s — sector_vs_spy_5d will be 0", exc)
    else:
        log.warning(
            "data/cache/sector_map.json not found — sector_vs_spy_5d will be 0. "
            "The 10y price cache + sector reference data are produced by the "
            "alpha-engine-data weekly collector (collectors/prices.py, "
            "Saturday SF DataPhase1 — full-replace 10y refresh to "
            "s3://alpha-engine-research/predictor/price_cache/). "
            "Verify that collector ran and re-sync data/cache from S3."
        )
