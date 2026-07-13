"""Consumer-side contract for the data→predictor feature-store boundary.

Refs nousergon/alpha-engine-config#692 (L4520 — cross-stage artifact contract).
This is #692's remaining "consumer side" tail: the SF pre-spend preflight gate
(#692 close-condition (b)) and the CI contract axis (a) already shipped; per
config#2343 the one piece left is the data→predictor feature-store consumer
contract, which this file provides.

WHAT IT PROTECTS
----------------
The predictor reads a per-ticker feature row from the ArcticDB ``universe``
library — ``inference/stages/run_inference.py::
_load_precomputed_features_from_arcticdb`` does ``universe.read(ticker).data``
then ``df.iloc[-1]`` — and feeds ``config.FEATURES`` (plus the direct
``momentum_20d`` / ``vix_level`` reads, all ⊆ ``config.FEATURES``) into the
GBM/MLP models. If the feature-store PRODUCER
(``nousergon-data/features/feature_engineer.py::FEATURES``, locked there by
``tests/test_schema_contract.py``: FEATURES ↔ CATALOG ↔ SCHEMA.md) ever drops
or renames a column this predictor requires, inference degrades SILENTLY: the
model receives an absent/NaN feature and emits a subtly-wrong prediction, and
nothing catches it. The freshness axis
(``feature_store_freshness_sentinel``) only asserts the library was *written*,
never that it still carries the *fields* this consumer reads. This test is the
production-coverage / field axis for the boundary — it fails at predictor CI
time the moment ``config.FEATURES`` requires a feature the producer (as last
mirrored below) does not emit.

WHY A STANDALONE PER-REPO TEST (not a PIPELINE_CONTRACT.yaml boundary)
---------------------------------------------------------------------
The feature store is ArcticDB (columnar), not an S3 JSON artifact. It is
monitored for freshness via the S3 ``feature_store_freshness_sentinel`` proxy
(config#1787, Brian's 2026-07-08 Option-B ruling) and is deliberately NOT
registered as a first-class, field-bearing artifact in ``ARTIFACT_REGISTRY``.
It therefore falls outside ``PIPELINE_CONTRACT.yaml``'s S3-JSON +
``artifact_id ∈ ARTIFACT_REGISTRY`` model, whose validator requires every
boundary to reference a registered artifact. The per-repo hard-coded-mirror
form used here is the same one the ``config/*`` boundaries and the config#939
feature mirror (``tests/test_config939_feature_mirror.py``) already use in this
repo — each side hard-codes its view; the mirror is the SoT snapshot. The
fully drift-proof lib-hosted feature schema (M0, like the signals/predictions
schemas in ``nousergon_lib.contracts``) is the wave-2 follow-on tracked under
config#2343 / epic config#2327.

MAINTENANCE
-----------
``PRODUCER_FEATURE_STORE_COLUMNS`` below is a snapshot of the producer's
``FEATURES`` list. When the producer adds/removes a column AND this predictor
starts requiring it, update this snapshot in the SAME PR that touches
``config.FEATURES``. Adding producer-only columns (producer set grows, consumer
does not read them) never breaks this test — only a consumer requiring an
un-produced field does.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

# Snapshot of nousergon-data `features/feature_engineer.py::FEATURES`
# (87 columns, locked there by `tests/test_schema_contract.py`). Sorted for
# stable diffs. SoT lives in the producer repo — see module docstring.
PRODUCER_FEATURE_STORE_COLUMNS = frozenset({
    "atr_14_pct", "atr_x_vix", "avg_volume_20d", "avg_volume_20d_raw",
    "beta_60d", "beta_60d_zscore", "bollinger_pct", "capex_growth_5y",
    "cmf_20_ratio", "current_ratio", "days_since_earnings", "debt_to_equity",
    "dist_from_20d_high", "dist_from_52w_high", "dist_from_52w_high_zscore", "dist_from_52w_low",
    "dist_from_5d_high", "dividend_yield", "earnings_surprise_pct", "ema_cross_8_21",
    "eps_growth_3y", "eps_revision_4w", "factor_momentum_ratio", "fcf_yield",
    "gold_mom_5d", "gross_margin", "hy_oas_credit_spread_pct", "idio_vol_60d",
    "idio_vol_60d_zscore", "intraday_return_5d", "iv_rank", "iv_vs_rv",
    "macd_above_zero", "macd_cross", "macd_line_last", "market_cap_raw",
    "max_drawdown_60d", "mom5d_x_vix", "mom_12_1_pct", "momentum_20d",
    "momentum_20d_zscore", "momentum_5d", "obv_slope_10d", "oil_mom_5d",
    "overnight_return_5d", "payout_ratio", "pb_ratio", "pe_ratio",
    "pe_ratio_zscore", "price_accel", "price_vs_ma200", "price_vs_ma50",
    "put_call_ratio", "realized_vol_20d", "realized_vol_63d", "realized_vol_63d_zscore",
    "rel_volume_ratio", "residual_momentum_ratio", "return_120d", "return_60d",
    "return_60d_zscore", "return_vs_spy_5d", "revenue_growth_3y", "revenue_growth_yoy",
    "revision_streak", "roe", "roe_zscore", "rsi_14",
    "rsi_slope_5d", "rsi_x_vix", "sector_mom_pct", "sector_vs_spy_10d",
    "sector_vs_spy_20d", "sector_vs_spy_5d", "sector_x_trend", "size_zscore",
    "vix_level", "vix_term_slope", "vol_of_vol_30d", "vol_ratio_10_60",
    "vol_trend_x_vix", "volume_price_div", "volume_trend", "vwap_divergence_pct",
    "xsect_dispersion", "yield_10y", "yield_curve_slope",
})

# Features read DIRECTLY by name in the inference read path, outside the
# config.FEATURES model-vector (momentum veto gate + research-free path). Each
# must also be produced. Kept explicit so a rename in one of these hard-coded
# call sites can't slip the contract.
DIRECT_INFERENCE_READS = frozenset({
    "momentum_20d",   # run_inference.py momentum veto gate
    "momentum_5d",    # research_free_inference.py
    "rsi_14",         # research_free_inference.py
    "vix_level",      # run_inference.py regime read
})


def test_all_model_features_are_produced():
    """Every feature the predictor feeds to its models must be emitted by the
    feature-store producer. A miss here = silent NaN-into-model at inference."""
    missing = set(cfg.FEATURES) - PRODUCER_FEATURE_STORE_COLUMNS
    assert not missing, (
        f"config.FEATURES requires feature-store column(s) the producer "
        f"(nousergon-data features/feature_engineer.py::FEATURES) does not "
        f"emit: {sorted(missing)}. Either the producer dropped/renamed them "
        f"(fix the producer or the predictor read), or a new required feature "
        f"was added here without confirming the producer emits it — update "
        f"PRODUCER_FEATURE_STORE_COLUMNS in the same PR."
    )


def test_fundamental_features_are_produced():
    """The v3.0 fundamental block is stored in the feature store even though it
    is currently GBM-excluded — it is still read from the row, so it must be
    produced."""
    missing = set(cfg.FUNDAMENTAL_FEATURES) - PRODUCER_FEATURE_STORE_COLUMNS
    assert not missing, (
        f"config.FUNDAMENTAL_FEATURES not emitted by the producer: {sorted(missing)}"
    )


def test_macro_features_are_produced():
    """Macro (per_ticker=False) features are read from the feature-store row
    like any other column; they must be produced."""
    missing = set(cfg.MACRO_FEATURES) - PRODUCER_FEATURE_STORE_COLUMNS
    assert not missing, (
        f"config.MACRO_FEATURES not emitted by the producer: {sorted(missing)}"
    )


def test_direct_inference_reads_are_produced():
    """Feature columns read by hard-coded name in the inference path (veto gate,
    research-free path) must be produced AND stay in config.FEATURES so the
    model-vector and the direct reads never diverge."""
    missing_producer = DIRECT_INFERENCE_READS - PRODUCER_FEATURE_STORE_COLUMNS
    assert not missing_producer, (
        f"Direct inference reads not emitted by the producer: {sorted(missing_producer)}"
    )
    missing_features = DIRECT_INFERENCE_READS - set(cfg.FEATURES)
    assert not missing_features, (
        f"Direct inference reads absent from config.FEATURES (model vector / "
        f"read-path drift): {sorted(missing_features)}"
    )


def test_producer_snapshot_is_a_strict_superset():
    """Guardrail on the mirror itself: the producer emits more columns than the
    predictor consumes (87 producer ⊇ 56 consumed). If this inverts, the
    snapshot was likely truncated by accident rather than the producer actually
    shrinking below the consumed set."""
    assert len(PRODUCER_FEATURE_STORE_COLUMNS) >= len(cfg.FEATURES), (
        f"PRODUCER_FEATURE_STORE_COLUMNS ({len(PRODUCER_FEATURE_STORE_COLUMNS)}) "
        f"is smaller than config.FEATURES ({len(cfg.FEATURES)}) — snapshot looks "
        f"truncated; re-mirror from the producer FEATURES list."
    )
