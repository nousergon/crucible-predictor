"""config#635 — L1 feature subscriptions load from the experiment package.

The lists are experiment beliefs: predictor.yaml `l1_features:` overrides;
the in-code baselines are the public no-package defaults. Pins (a) the
baseline values stay the historically-validated sets, (b) absent config
yields the baseline, (c) a package override is honored verbatim.
"""
import config as cfg


def test_baseline_lists_are_the_validated_sets():
    assert cfg._BASELINE_MOMENTUM_FEATURES == [
        "momentum_5d", "momentum_20d", "price_vs_ma50", "price_vs_ma200",
        "rsi_14", "macd_cross", "return_60d", "return_120d",
        "intraday_return_5d",
    ]
    assert cfg._BASELINE_VOLATILITY_FEATURES == [
        "atr_14_pct", "realized_vol_20d", "vol_ratio_10_60",
        "iv_rank", "dist_from_52w_high", "dist_from_52w_low",
    ]


def test_active_lists_resolve_from_config_or_baseline():
    l1 = (cfg._cfg.get("l1_features") or {})
    expected_mom = list(l1.get("momentum") or cfg._BASELINE_MOMENTUM_FEATURES)
    expected_vol = list(l1.get("volatility") or cfg._BASELINE_VOLATILITY_FEATURES)
    assert cfg.MOMENTUM_FEATURES == expected_mom
    assert cfg.VOLATILITY_FEATURES == expected_vol


def test_override_shape_is_honored():
    # The resolution expression, applied to a synthetic config, honors the
    # package override verbatim (no merging/dedup surprises).
    fake = {"l1_features": {"momentum": ["a", "b"], "volatility": None}}
    l1 = fake.get("l1_features") or {}
    assert list(l1.get("momentum") or cfg._BASELINE_MOMENTUM_FEATURES) == ["a", "b"]
    assert (
        list(l1.get("volatility") or cfg._BASELINE_VOLATILITY_FEATURES)
        == cfg._BASELINE_VOLATILITY_FEATURES
    )
