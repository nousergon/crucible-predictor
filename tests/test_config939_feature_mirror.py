"""config#939 — mirror of nousergon-data's 3 new feature-store columns
(vwap_divergence_pct, cmf_20_ratio, hy_oas_credit_spread_pct) into
config.py::FEATURES / MACRO_FEATURES / GBM_FEATURES.

Refs nousergon/alpha-engine-config#939.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg


_NEW_FEATURES = ("vwap_divergence_pct", "cmf_20_ratio", "hy_oas_credit_spread_pct")


def test_new_features_present_in_features_list():
    for name in _NEW_FEATURES:
        assert name in cfg.FEATURES, f"{name} missing from config.FEATURES"


def test_n_features_matches_list_length():
    assert cfg.N_FEATURES == len(cfg.FEATURES)


def test_hy_oas_credit_spread_pct_is_macro_excluded_from_gbm():
    # Macro (per_ticker=False, identical across every ticker on a date) —
    # must be excluded from GBM training the same way vix_level/yield_10y/
    # etc. are, per the MACRO_FEATURES contract.
    assert "hy_oas_credit_spread_pct" in cfg.MACRO_FEATURES
    assert "hy_oas_credit_spread_pct" not in cfg.GBM_FEATURES


def test_vwap_and_cmf_are_technical_and_included_in_gbm():
    # Per-ticker technical features — NOT macro, so they participate in
    # GBM training/inference like other technical columns.
    for name in ("vwap_divergence_pct", "cmf_20_ratio"):
        assert name not in cfg.MACRO_FEATURES
        assert name in cfg.GBM_FEATURES


def test_n_gbm_features_matches_list_length():
    assert cfg.N_GBM_FEATURES == len(cfg.GBM_FEATURES)


def test_no_collision_with_regime_predictor_hy_oas_level():
    # model/regime_predictor.py's MACRO_NORM_FEATURES already carries a
    # SEPARATE, already-shipped "hy_oas_level" / "hy_oas_change_21d" pair
    # (market-wide regime substrate, sourced independently via its own
    # HYOAS.parquet cache). This repo's per-ticker/date feature-store
    # mirror must use a distinct name so the two are never confused.
    assert "hy_oas_level" not in cfg.FEATURES
    assert "hy_oas_credit_spread_pct" in cfg.FEATURES
    # The regime-substrate pair still exists, untouched, in its own
    # namespace — confirms this PR didn't collide with or remove it.
    assert "hy_oas_level" in cfg.MACRO_NORM_FEATURES
    assert "hy_oas_change_21d" in cfg.MACRO_NORM_FEATURES
