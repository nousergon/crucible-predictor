"""inference/research_free_inference.py — daily research-free meta-ensemble
inference over the scanner-passing pool (config#2365, champion-loop epic
child 1 / config#2364).

Writes ``predictor/predictions_research_free/{date}.json``: one entry per
ticker where the CURRENT cycle's scanner run (``candidates/{date}/
candidates.json::scanner_eval_log``, produced by crucible-research's
Scanner Lambda) recorded ``quant_filter_pass=1``. Each entry runs the FULL
deployed meta-ensemble (the same ``model/meta_model.py::MetaModel`` the live
champion inference in ``inference/stages/run_inference.py`` uses) with the 4
research meta-features (``model/meta_model.py::RESEARCH_META_FEATURES`` —
``research_calibrator_prob`` / ``research_composite_score`` /
``research_conviction`` / ``sector_macro_modifier``) unconditionally zeroed —
this is the "research-free" counterfactual: what would the champion predict
using ONLY deterministic/macro signal, with no research-agent contribution?

This is the crucible-predictor-native sibling of crucible-backtester's
``analysis/scanner_predictor_research_free_backfill.py`` (PR#482/#486,
config#1405 arm 4). That module backfills HISTORICAL (ticker, eval_date)
pairs out of ``research.db``'s ``scanner_evaluations`` table into an S3
parquet, for the backtester's own offline lift analysis. This module instead
runs LIVE, once per trading day, and writes a schema-versioned JSON artifact
that other repos (the executor, per config#2364 child 2) can consume
directly — mirroring the existing ``predictor/predictions/{date}.json``
producer/consumer contract pattern (``tests/test_predictions_producer_contract.py``
+ ``tests/test_predictions_schema_conformance.py``), not the backtester's
sqlite-table-plus-parquet-artifact shape.

Design notes / gotchas (see crucible-backtester's module docstring for the
full discovery narrative — this repeats only what's load-bearing here):

- **Scanner pool source is candidates.json, not a shared sqlite table.**
  crucible-predictor has no ``research.db`` connection (that's a
  backtester/console-only artifact). The scanner's OWN daily S3 artifact,
  ``s3://{bucket}/candidates/{run_date}/candidates.json``, already carries
  ``scanner_eval_log`` — the per-ticker ``quant_filter_pass`` verdict
  (``crucible-research/data/scanner_orchestrator.py::build_candidates_artifact``,
  config#1458). This module reads THAT artifact directly; it never touches
  ``scanner_evaluations`` (sqlite) at all.
- **Residual-momentum features MUST be computed from close history via this
  repo's own ``data/residual_momentum_features.py``, not read as precomputed
  ArcticDB universe-library columns.** Confirmed by the backtester's build
  (PR#482): the ArcticDB row carries differently-named/differently-defined
  legacy columns (``residual_momentum_ratio`` / ``mom_12_1_pct`` /
  ``sector_mom_pct``) that do NOT match ``model/residual_momentum_scorer.py``'s
  expected input names (``resid_mom_vol_scaled`` / ``mom_12_1`` / ``mom_1m`` /
  ``mom_change`` / ``sector_mom``) — feeding them directly silently
  neutral-zeros every ticker's residual_momentum_score.
- **Momentum-score and volatility inputs, by contrast, ARE read as
  precomputed ArcticDB universe-library columns** — this matches
  ``inference/stages/run_inference.py``'s OWN live path
  (``_load_precomputed_features_from_arcticdb`` + ``cfg.VOLATILITY_FEATURES``
  / momentum's ``momentum_5d``/``momentum_20d``/``price_vs_ma50``/``rsi_14``),
  which is this repo's canonical feature source for those two components.
  The residual-momentum gotcha above is specific to THAT feature family, not
  a blanket "never trust ArcticDB columns" rule. Concretely this means
  reading via ``nousergon_lib.arcticdb.load_universe_ohlcv`` (full stored
  frame — precomputed columns AND Close history in one batched read), NOT
  ``inference/stages/load_prices.py::load_price_data_from_arctic`` — that
  reader hard-restricts to ``columns=[Open,High,Low,Close,Volume]`` and
  would silently drop every precomputed feature column, degrading
  momentum_score/expected_move to a neutral zero for every ticker exactly
  the way the residual-momentum gotcha above describes.
- **The rank-normalization / residual-momentum-benchmark universe is
  ``scanner_eval_log``'s own ticker population (pass AND fail), not
  ``inference/stages/load_universe.py::get_universe_tickers``.** That
  function reads ``signals.json`` — the RESEARCH module's tracked-ticker
  watchlist, a different and materially smaller population than what the
  Scanner Lambda actually evaluated. Cross-sectional rank-normalization for
  the volatility GBM input runs over the FULL scanned universe that day
  (mirrors ``run_inference.py``'s own rank-norm batch) — a scanner-subset-
  only (or wrong-population) rank would place every ticker near the same
  percentile and defeat the point of the normalization.
- **Weights are S3-only, never a local/checkout-relative path.** Reuses
  ``cfg.META_WEIGHTS_PREFIX`` (``predictor/weights/meta/``) — the SAME
  canonical prefix ``inference/stages/load_model.py`` reads for the live
  champion path. No sibling-checkout ``weights/`` directory is ever
  consulted (backtester-PR486 lesson).
- **Feature-schema-driven, not a hardcoded feature list.** Reads the
  LOADED ``MetaModel``'s own ``_feature_names`` (the same attribute
  ``predict_single`` falls back to) and builds exactly that feature set —
  robust to the deployed model swapping Layer-1 components (e.g. the
  2026-06-15 ``momentum_score`` → ``residual_momentum_score`` cutover)
  without a producer code change.
- **FAIL-HARD on a genuine precondition failure** (ArcticDB unreachable,
  MetaModel fails to load / isn't fitted, candidates.json missing/malformed)
  per config#1684 doctrine — this artifact becomes trade-load-bearing once
  consumed by the live executor path (config#2364 child 2). A per-ticker
  compute failure (one bad ArcticDB row) degrades that ticker only and is
  counted in the summary, matching the existing predictions.json producer's
  posture (``write_predictions`` writes partial output rather than aborting
  the whole run on one ticker).
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Canonical S3 key templates — single source of truth lives in config.py
# alongside the sibling PREDICTIONS_KEY / PREDICTIONS_LATEST_KEY.
PREDICTIONS_RESEARCH_FREE_KEY = cfg.PREDICTIONS_RESEARCH_FREE_KEY
PREDICTIONS_RESEARCH_FREE_LATEST_KEY = cfg.PREDICTIONS_RESEARCH_FREE_LATEST_KEY

_CANDIDATES_PREFIX = "candidates"



# ── Scanner-passing pool (candidates.json) ──────────────────────────────────

def load_candidates_artifact(
    bucket: str,
    date_str: str,
    *,
    s3_client=None,
) -> dict:
    """Return the raw ``candidates/{date_str}/candidates.json`` artifact dict.

    Raises RuntimeError if the artifact is missing or malformed — an absent
    scanner artifact for a trading day is a genuine upstream precondition
    failure (the Scanner Lambda didn't run / didn't publish), not something
    this producer should silently degrade to an empty output for (config#1684
    fail-hard doctrine; see module docstring).
    """
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    s3 = s3_client or boto3.client("s3")
    key = f"{_CANDIDATES_PREFIX}/{date_str}/candidates.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        artifact = json.loads(obj["Body"].read())
    except (ClientError, BotoCoreError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"failed to load candidates artifact from s3://{bucket}/{key}: {exc}"
        ) from exc
    if "scanner_eval_log" not in artifact:
        raise RuntimeError(
            f"s3://{bucket}/{key} has no 'scanner_eval_log' field — "
            f"cannot determine the scanner-passing pool"
        )
    return artifact


def load_scanner_pool(
    bucket: str,
    date_str: str,
    *,
    s3_client=None,
) -> list[str]:
    """Return the sorted, de-duplicated list of tickers where THIS cycle's
    ``candidates/{date_str}/candidates.json::scanner_eval_log`` recorded
    ``quant_filter_pass=1``. See :func:`load_candidates_artifact` for the
    fail-hard contract on a missing/malformed artifact."""
    artifact = load_candidates_artifact(bucket, date_str, s3_client=s3_client)
    eval_log = artifact["scanner_eval_log"] or []
    tickers = sorted({
        rec["ticker"] for rec in eval_log
        if rec.get("ticker") and rec.get("quant_filter_pass") == 1
    })
    log.info(
        "Scanner pool for %s: %d/%d eval_log entries passed quant_filter",
        date_str, len(tickers), len(eval_log),
    )
    return tickers


def _scanner_eval_log_universe(artifact: dict) -> list[str]:
    """Every ticker THIS cycle's scanner actually evaluated (pass or fail) —
    the true full cross-section the quant filter ran against, per
    ``scanner_eval_log``. This is the correct rank-normalization /
    residual-momentum-benchmark universe: it reflects exactly what the
    Scanner Lambda scored today, unlike ``inference/stages/load_universe.py
    ::get_universe_tickers`` (which reads signals.json — the RESEARCH
    module's tracked-ticker watchlist, a different and narrower population
    than what the scanner evaluated)."""
    eval_log = artifact.get("scanner_eval_log") or []
    return sorted({rec["ticker"] for rec in eval_log if rec.get("ticker")})


# ── Weights (S3-only — never a local checkout path) ─────────────────────────

def _download_weight(
    bucket: str,
    filename: str,
    *,
    prefix: str | None = None,
    s3_client=None,
    sidecar: bool = True,
) -> Path:
    """Download ``{prefix}{filename}`` from S3 to a temp dir and return the
    local path. Mirrors ``inference/stages/load_model.py``'s ``_dl`` helper
    (same canonical ``cfg.META_WEIGHTS_PREFIX`` the live champion path
    reads, and the same ``tempfile.gettempdir()`` + fixed-filename idiom —
    no per-call ``mkdtemp``, so a warm/reused Lambda container doesn't
    accumulate a fresh throwaway directory on every invocation) but raises
    on a failed PRIMARY download instead of that stage's per-artifact
    try/except — the deployed champion being unreachable must surface,
    never silently degrade this producer to an empty run.
    """
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    prefix = prefix if prefix is not None else cfg.META_WEIGHTS_PREFIX
    s3 = s3_client or boto3.client("s3")
    tmp_dir = Path(tempfile.gettempdir())
    local = tmp_dir / f"research_free_{filename}"
    key = f"{prefix}{filename}"
    try:
        s3.download_file(bucket, key, str(local))
    except (ClientError, BotoCoreError, OSError) as exc:
        raise RuntimeError(
            f"failed to download deployed weight artifact s3://{bucket}/{key}: {exc}"
        ) from exc
    if sidecar:
        try:
            s3.download_file(bucket, key + ".meta.json", str(local) + ".meta.json")
        except (ClientError, BotoCoreError, OSError):
            pass  # embedded schema (v3 pickles) wins when the sidecar is absent
    return local


def load_meta_model(bucket: str, *, prefix: str | None = None, s3_client=None):
    """Load the deployed MetaModel from canonical S3
    (``cfg.META_WEIGHTS_PREFIX`` + ``meta_model.pkl``) — never a local
    checkout path."""
    from model.meta_model import MetaModel

    path = _download_weight(bucket, "meta_model.pkl", prefix=prefix, s3_client=s3_client)
    mm = MetaModel.load(str(path))
    if not mm.is_fitted:
        raise RuntimeError(f"MetaModel loaded from s3 ({path}) is not fitted")
    return mm


def load_volatility_scorer(bucket: str, *, prefix: str | None = None, s3_client=None):
    """Best-effort load of the deployed volatility GBM. Returns None on any
    failure (missing artifact, lightgbm not installed) — callers treat that
    as "expected_move unavailable -> 0.0", matching
    ``inference/stages/run_inference.py``'s neutral default when
    ``vol_scorer is None``."""
    try:
        from model.gbm_scorer import GBMScorer

        path = _download_weight(
            bucket, "volatility_model.txt", prefix=prefix, s3_client=s3_client, sidecar=False,
        )
        return GBMScorer.load(str(path))
    except Exception as exc:  # noqa: BLE001 - graceful degrade, never fatal
        log.warning("volatility scorer load failed (expected_move -> 0.0): %s", exc)
        return None


# ── Per-component feature computers ─────────────────────────────────────────

def _compute_momentum_scores(rows: dict[str, pd.Series]) -> dict[str, float]:
    """model/momentum_scorer.py::predict_dict per ticker, fed by the
    precomputed ArcticDB row (momentum_5d / momentum_20d / price_vs_ma50 /
    rsi_14) — same source ``run_inference.py`` uses live for this component."""
    from model.momentum_scorer import predict_dict as _momentum_predict_dict

    return {t: float(_momentum_predict_dict(row.to_dict())) for t, row in rows.items()}


def _compute_residual_momentum_scores(
    tickers: list[str],
    close_history: dict[str, pd.Series],
    spy_close: pd.Series | None,
    as_of: pd.Timestamp,
) -> dict[str, float]:
    """model/residual_momentum_scorer.py::predict_dict per ticker, fed by
    ``data/residual_momentum_features.py::compute_residual_momentum_features``
    run on FULL close-price history — NOT the ArcticDB universe-library
    row's precomputed columns (see module docstring: those carry
    differently-named legacy columns that silently neutral-zero this
    scorer).

    Benchmark is SPY close for every ticker (not a per-ticker sector ETF) —
    the same documented simplification the backtester's reference
    implementation uses; ``compute_residual_momentum_features`` natively
    supports the ``benchmark_close=None -> SPY`` fallback.
    """
    from data.residual_momentum_features import compute_residual_momentum_features
    from model.residual_momentum_scorer import predict_dict as _resid_predict_dict

    out: dict[str, float] = {}
    for t in tickers:
        close = close_history.get(t)
        if close is None or close.empty:
            out[t] = 0.0
            continue
        close = close[close.index <= as_of]
        if close.empty:
            out[t] = 0.0
            continue
        try:
            feats_df = compute_residual_momentum_features(close, None, spy_close)
            latest = feats_df.iloc[-1].to_dict()
            out[t] = float(_resid_predict_dict(latest))
        except Exception as exc:  # noqa: BLE001 - one ticker must not abort the run
            log.warning("residual_momentum_score computation failed for %s: %s", t, exc)
            out[t] = 0.0
    return out


def _compute_expected_move(
    rows: dict[str, pd.Series],
    vol_scorer,
    as_of: str,
    feature_cols: tuple[str, ...] = (),
) -> dict[str, float]:
    """Cross-sectional rank-normalized volatility-GBM predict, ONE calendar
    date (``as_of``). ``rows`` must be the FULL per-ticker feature rows for
    tickers trading on this date (not just the scanner-passing subset) —
    mirrors ``run_inference.py``'s ``cross_sectional_rank_normalize`` +
    ``vol_scorer.predict`` batch. Returns 0.0 for every ticker if
    ``vol_scorer`` is None."""
    if vol_scorer is None or not rows:
        return {t: 0.0 for t in rows}
    from data.dataset import cross_sectional_rank_normalize as _rank_norm

    # GBMScorer.feature_names is the public accessor (booster-embedded
    # schema, config.py-independent); cfg.VOLATILITY_FEATURES is the
    # single-source-of-truth fallback when no scorer is available at all
    # (mirrors config.py's own _BASELINE_VOLATILITY_FEATURES default).
    cols = list(vol_scorer.feature_names or feature_cols or cfg.VOLATILITY_FEATURES)
    tickers = list(rows.keys())
    X_raw = np.stack([
        rows[t].reindex(cols).astype(np.float64).fillna(0.0).to_numpy()
        for t in tickers
    ]).astype(np.float32)
    same_date = [as_of] * len(tickers)  # single cross-section
    X_ranked = _rank_norm(X_raw, same_date).astype(np.float32)
    preds = vol_scorer.predict(X_ranked)
    return {t: float(p) for t, p in zip(tickers, preds)}


def _compute_macro_row(spy_s, vix_s, vix3m_s, tnx_s, irx_s, close_prices: dict) -> dict[str, float]:
    """The macro META_FEATURES + ``regime_intensity_z``, via
    ``model/regime_predictor.py::RegimePredictor.build_features`` — the same
    pure feature-engineering utility ``run_inference.py`` uses live. Returns
    an all-zero (neutral) dict on any failure so a single date's macro-build
    failure degrades one row rather than aborting the whole run."""
    from model.meta_model import MACRO_FEATURE_META_MAP, REGIME_DERIVED_FEATURE_META_MAP
    from model.regime_predictor import RegimePredictor

    macro_row = {name: 0.0 for name in MACRO_FEATURE_META_MAP.values()}
    for name in REGIME_DERIVED_FEATURE_META_MAP.values():
        macro_row[name] = 0.0
    if spy_s is None or len(spy_s) < 20:
        return macro_row
    try:
        regime_df = RegimePredictor().build_features(
            spy_s, vix_s, vix3m_s, tnx_s, irx_s, close_prices,
        )
        if regime_df.empty:
            return macro_row
        latest = regime_df.iloc[-1]
        for src_name, meta_name in MACRO_FEATURE_META_MAP.items():
            macro_row[meta_name] = float(latest.get(src_name, 0.0))
        for src_name, meta_name in REGIME_DERIVED_FEATURE_META_MAP.items():
            macro_row[meta_name] = float(latest.get(src_name, 0.0))
    except Exception as exc:  # noqa: BLE001 - one date's macro build must not abort the run
        log.warning("Macro feature build failed for one date (zero-fill fallback): %s", exc)
    return macro_row


# ── Pure per-ticker feature assembly (unit-testable, no IO) ────────────────

def _assemble_research_free_features(
    ticker: str,
    feat_names: list[str],
    *,
    momentum_scores: dict[str, float],
    resid_scores: dict[str, float],
    expected_moves: dict[str, float],
    macro_row: dict[str, float],
    research_meta_features: frozenset[str],
    log_context: str = "",
) -> dict[str, float]:
    """Build the research-free feature dict for one ticker, given the loaded
    model's own ``feat_names`` schema and this date's already-computed
    per-ticker/market-wide component scores.

    Pure (no IO) — the per-ticker heart of the research-free contract,
    directly unit-testable without ArcticDB/S3/a real MetaModel artifact.
    Any feature in ``research_meta_features`` is unconditionally zeroed
    regardless of whether a value happens to be available (research-free by
    construction, not by absence); any OTHER feature name the loaded model
    expects but this function doesn't know how to compute degrades to 0.0
    with a logged warning — the same graceful-degrade contract
    ``MetaModel.predict_single`` itself uses for a missing dict key.
    """
    feats: dict[str, float] = {}
    for f in feat_names:
        if f in research_meta_features:
            feats[f] = 0.0  # research-free by construction
        elif f == "momentum_score":
            feats[f] = momentum_scores.get(ticker, 0.0)
        elif f == "residual_momentum_score":
            feats[f] = resid_scores.get(ticker, 0.0)
        elif f == "expected_move":
            feats[f] = expected_moves.get(ticker, 0.0)
        elif f in macro_row:
            feats[f] = macro_row[f]
        else:
            log.warning(
                "%s%s: no feature computer registered for '%s' -> 0.0",
                (log_context + ": ") if log_context else "", ticker, f,
            )
            feats[f] = 0.0
    return feats


# ── Orchestration ────────────────────────────────────────────────────────────

def run_research_free_inference(
    date_str: str,
    *,
    bucket: str | None = None,
    s3_client=None,
    dry_run: bool = False,
) -> dict:
    """Compute + write research-free ``predicted_alpha`` for every ticker in
    the current cycle's scanner-passing pool.

    Returns a summary dict (``status``, ``n_written``, ``n_errors``,
    ``feature_names``, ``n_research_features_missing``, ``artifact_key``).

    Raises only on a genuine precondition failure (missing/malformed
    candidates.json, unloadable/unfitted MetaModel, unreachable ArcticDB) —
    a per-ticker compute failure degrades that ticker only (counted in
    ``n_errors``), matching the existing predictions.json producer's
    partial-output posture. See module docstring for the config#1684
    fail-hard rationale.
    """
    from model.meta_model import META_FEATURES, RESEARCH_META_FEATURES
    from nousergon_lib.arcticdb import load_universe_ohlcv, load_macro_series

    bucket = bucket or cfg.S3_BUCKET
    research_meta_features = frozenset(RESEARCH_META_FEATURES)

    candidates_artifact = load_candidates_artifact(bucket, date_str, s3_client=s3_client)
    pool_tickers = sorted({
        rec["ticker"] for rec in candidates_artifact["scanner_eval_log"]
        if rec.get("ticker") and rec.get("quant_filter_pass") == 1
    })
    if not pool_tickers:
        log.warning("Scanner pool for %s is empty — writing an empty artifact", date_str)

    mm = load_meta_model(bucket, s3_client=s3_client)
    vol_scorer = load_volatility_scorer(bucket, s3_client=s3_client)
    # Loaded model's own schema is the source of truth (mirrors
    # MetaModel.predict_single's OWN fallback — model/meta_model.py:434 —
    # so a pre-cutover pickle with an empty _feature_names doesn't silently
    # degrade to a zero-feature dict / constant predicted_alpha here).
    feat_names = list(mm._feature_names or META_FEATURES)
    research_feats_present = [f for f in feat_names if f in research_meta_features]
    n_research_missing = len(research_feats_present)
    if n_research_missing == 0:
        log.warning(
            "Loaded MetaModel's feature_names contain none of the known "
            "RESEARCH_META_FEATURES (%s) — n_research_features_missing will "
            "be recorded as 0. Either an unusual model variant or a "
            "feature-name drift this producer doesn't recognize yet.",
            sorted(research_meta_features),
        )

    # Full scanned universe for THIS date (cross-sectional rank-norm base +
    # residual-momentum benchmark) — every ticker THIS cycle's scanner
    # actually evaluated (pass or fail), per scanner_eval_log. NOT
    # inference/stages/load_universe.py::get_universe_tickers (that reads
    # signals.json, the research module's narrower tracked-ticker watchlist —
    # a materially different and smaller population than what the scanner
    # ran the quant filter against). See module docstring / _scanner_eval_log_
    # universe docstring for the full rationale.
    universe_tickers = _scanner_eval_log_universe(candidates_artifact)
    all_tickers = sorted(set(universe_tickers) | set(pool_tickers) | {"SPY"})

    # Full-column batch read (momentum/volatility precomputed feature columns
    # + Close history in ONE read) via the shared nousergon_lib reader — the
    # SAME source crucible-backtester's reference implementation
    # (analysis/scanner_predictor_research_free_backfill.py) uses, and
    # feature-complete unlike inference/stages/load_prices.py::
    # load_price_data_from_arctic (which hard-restricts to OHLCV-only
    # columns and would silently zero-fill momentum_score/expected_move).
    try:
        full_history = load_universe_ohlcv(bucket, symbols=all_tickers, end=date_str)
    except Exception as exc:
        raise RuntimeError(f"ArcticDB universe history load failed: {exc}") from exc
    if not full_history:
        raise RuntimeError("ArcticDB universe library returned zero symbols")

    macro_history: dict[str, pd.DataFrame] = {}
    if any(f.startswith("macro_") or f == "regime_intensity_z" for f in feat_names):
        try:
            # ArcticDB's macro library uses plain (non-Yahoo-prefixed) symbol
            # names: SPY/VIX/VIX3M/TNX/IRX — matches
            # crucible-backtester's confirmed-in-production symbol map.
            macro_history = load_macro_series(
                bucket, ["SPY", "VIX", "VIX3M", "TNX", "IRX"], end=date_str,
            )
        except Exception as exc:  # noqa: BLE001 - macro is best-effort (zero-fill fallback)
            log.warning("macro library load failed (macro features -> 0.0): %s", exc)

    rows_today: dict[str, pd.Series] = {}
    close_history: dict[str, pd.Series] = {}
    for t, df in full_history.items():
        if df is None or df.empty:
            continue
        close_history[t] = df["Close"] if "Close" in df.columns else pd.Series(dtype=float)
        rows_today[t] = df.iloc[-1]

    spy_close_full = close_history.get("SPY")
    d_ts = pd.Timestamp(date_str)

    target_rows = {t: rows_today[t] for t in pool_tickers if t in rows_today}
    missing = set(pool_tickers) - set(target_rows)
    n_errors = len(missing)
    if missing:
        log.warning(
            "%s: %d/%d scanner-passing tickers absent from ArcticDB as-of "
            "this date — skipped (n_errors)", date_str, len(missing), len(pool_tickers),
        )

    momentum_scores = (
        _compute_momentum_scores(target_rows) if "momentum_score" in feat_names else {}
    )
    resid_scores = (
        _compute_residual_momentum_scores(
            list(target_rows.keys()), close_history, spy_close_full, d_ts,
        )
        if "residual_momentum_score" in feat_names else {}
    )
    expected_moves = (
        _compute_expected_move(rows_today, vol_scorer, date_str)
        if "expected_move" in feat_names else {}
    )

    macro_row: dict[str, float] = {}
    if any(f.startswith("macro_") or f == "regime_intensity_z" for f in feat_names):
        def _asof_close(sym: str):
            df = macro_history.get(sym)
            if df is None or df.empty or "Close" not in df.columns:
                return None
            return df["Close"]

        spy_s = _asof_close("SPY")
        vix_s = _asof_close("VIX")
        vix3m_s = _asof_close("VIX3M")
        tnx_s = _asof_close("TNX")
        irx_s = _asof_close("IRX")
        close_prices = {t: s for t, s in close_history.items() if s is not None and not s.empty}
        macro_row = _compute_macro_row(spy_s, vix_s, vix3m_s, tnx_s, irx_s, close_prices)

    entries: list[dict] = []
    for t in pool_tickers:
        if t not in target_rows:
            continue
        try:
            feats = _assemble_research_free_features(
                t, feat_names,
                momentum_scores=momentum_scores,
                resid_scores=resid_scores,
                expected_moves=expected_moves,
                macro_row=macro_row,
                research_meta_features=research_meta_features,
                log_context=date_str,
            )
            alpha = float(mm.predict_single(feats))
        except Exception as exc:  # noqa: BLE001 - one ticker's failure must not abort the run
            log.warning("%s/%s: predict_single failed: %s", date_str, t, exc)
            n_errors += 1
            continue
        entries.append({
            "ticker": t,
            "prediction_date": date_str,
            "predicted_alpha": alpha,
            "n_research_features_missing": n_research_missing,
        })

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "date": date_str,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_predictions": len(entries),
        "n_scanner_pool": len(pool_tickers),
        "n_errors": n_errors,
        "feature_names": feat_names,
        "n_research_features_missing": n_research_missing,
        "predictions": entries,
    }

    artifact_key = None
    if dry_run:
        log.info("[DRY RUN] research-free predictions for %s: %s", date_str,
                  json.dumps(envelope, indent=2))
    else:
        artifact_key = write_research_free_predictions(envelope, bucket, s3_client=s3_client)

    return {
        "status": "ok",
        "n_written": len(entries),
        "n_errors": n_errors,
        "feature_names": feat_names,
        "n_research_features_missing": n_research_missing,
        "artifact_key": artifact_key,
    }


def write_research_free_predictions(
    envelope: dict,
    bucket: str,
    *,
    s3_client=None,
) -> str:
    """Write the schema-versioned envelope to the dated key + latest.json.
    Raises on failure — a computed-but-unpersisted run is indistinguishable
    from a never-run one to any consumer, exactly the silent failure mode
    this artifact exists to avoid (matches ``write_predictions``'s posture
    for the sibling ``predictions/{date}.json`` producer)."""
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    s3 = s3_client or boto3.client("s3")
    date_str = envelope["date"]
    dated_key = PREDICTIONS_RESEARCH_FREE_KEY.format(date=date_str)
    body = json.dumps(envelope, indent=2, default=str)
    try:
        s3.put_object(Bucket=bucket, Key=dated_key, Body=body,
                       ContentType="application/json")
        s3.put_object(Bucket=bucket, Key=PREDICTIONS_RESEARCH_FREE_LATEST_KEY, Body=body,
                       ContentType="application/json")
    except (ClientError, BotoCoreError, OSError) as exc:
        raise RuntimeError(
            f"failed to write research-free predictions to "
            f"s3://{bucket}/{dated_key}: {exc}"
        ) from exc
    log.info(
        "Wrote %d research-free predictions to s3://%s/%s",
        envelope["n_predictions"], bucket, dated_key,
    )
    return dated_key


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run daily research-free inference over the scanner-passing pool."
    )
    parser.add_argument("--date", default=None,
                         help="Override prediction date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Skip S3 writes; print the envelope to stdout.")
    parser.add_argument("--s3-bucket", default=None,
                         help=f"Override S3 bucket. Default: {cfg.S3_BUCKET}")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _date = args.date or datetime.now().strftime("%Y-%m-%d")
    result = run_research_free_inference(
        _date, bucket=args.s3_bucket, dry_run=args.dry_run,
    )
    log.info("research-free inference summary: %s", result)
