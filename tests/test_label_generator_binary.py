"""Tests pinning the binary UP/DOWN label taxonomy on data/label_generator.

The 3-class UP/FLAT/DOWN taxonomy was retired 2026-05-12 per ROADMAP L1615.
These tests guard against accidental reintroduction of a FLAT band by
threshold-driven labeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.label_generator import compute_labels, label_distribution


def _series(values: list[float]) -> pd.DataFrame:
    """Helper: build a DataFrame with a Close column and DatetimeIndex."""
    idx = pd.date_range("2024-01-02", periods=len(values), freq="B")
    return pd.DataFrame({"Close": values}, index=idx)


def test_compute_labels_emits_only_up_and_down():
    """No row should be labeled FLAT under binary taxonomy, even when the
    forward return falls inside the legacy ±1% band."""
    # 10 days of monotonically increasing prices: every forward 5d return > 0.
    df = _series([100 + i * 0.1 for i in range(10)])
    out = compute_labels(df, forward_days=5)
    assert set(out["direction"].unique()).issubset({"UP", "DOWN"})
    assert "FLAT" not in out["direction"].unique()


def test_direction_int_is_binary():
    """direction_int must be 0 (DOWN) or 1 (UP) — not 2, not 1=FLAT."""
    df = _series([100, 101, 99, 102, 98, 103, 97, 100])
    out = compute_labels(df, forward_days=2)
    assert set(out["direction_int"].unique()).issubset({0, 1})


def test_sign_zero_is_up():
    """forward_return exactly 0 should map to UP (>=0 boundary is inclusive).
    Anchors the convention so a future refactor doesn't accidentally flip
    the boundary to strict >."""
    df = _series([100.0, 101.0, 100.0, 101.0])  # day 0 → day 2: 100 → 100
    out = compute_labels(df, forward_days=2)
    # Row 0: forward_return = (100 - 100) / 100 = 0.0 → UP under >= 0 rule
    assert out.iloc[0]["direction"] == "UP"
    assert out.iloc[0]["direction_int"] == 1


def test_threshold_args_ignored():
    """up_threshold / down_threshold / adaptive_thresholds must not influence
    the binary labels — passing extreme values must produce the same output
    as passing defaults."""
    df = _series([100.0 + i for i in range(8)])
    default = compute_labels(df, forward_days=3)
    # Pass absurd thresholds — they should have zero effect.
    extreme = compute_labels(
        df,
        forward_days=3,
        up_threshold=0.99,
        down_threshold=-0.99,
        adaptive_thresholds=True,
    )
    pd.testing.assert_series_equal(default["direction"], extreme["direction"])
    pd.testing.assert_series_equal(default["direction_int"], extreme["direction_int"])


def test_label_distribution_has_two_keys():
    """label_distribution should report only UP + DOWN, not FLAT."""
    df = _series([100, 102, 99, 101, 103, 98])
    out = compute_labels(df, forward_days=2)
    dist = label_distribution(out)
    assert set(dist.keys()) == {"UP", "DOWN"}
    assert "FLAT" not in dist
    assert dist["UP"] + dist["DOWN"] == 1.0


def test_label_distribution_empty():
    """Empty DataFrame should return zeros, no FLAT key."""
    dist = label_distribution(pd.DataFrame())
    assert dist == {"UP": 0.0, "DOWN": 0.0}


def test_benchmark_relative_label_is_still_binary():
    """Sector-neutral / market-relative path must also be binary."""
    df = _series([100 + i for i in range(8)])
    bench = pd.Series([100.0] * 8, index=df.index)  # flat benchmark
    out = compute_labels(df, forward_days=3, benchmark_returns=bench)
    assert set(out["direction"].unique()).issubset({"UP", "DOWN"})
