"""alpha-engine-config-I9255 — the 2026-08-21 macro-block outage.

Three regressions, each failing against the pre-fix code:

1. ``per_feature_standalone_ic`` reported ``{"xsec_ic": None, "n_dates": 0}``
   for every ``macro_*`` column in EVERY vintage ever produced, because those
   columns are constant WITHIN a date and ``cross_sectional_ic_series`` skips a
   date with no dispersion. A healthy market-wide feature and a dead all-zero
   column rendered BYTE-IDENTICALLY, so the outage was unobservable on the one
   surface built to observe it. It must now name which case it is, and measure
   the market-wide case the only way it can be measured (time series).

2. ``RegimePredictor.build_features`` ended with an unconditional
   ``df.dropna()``. The ArcticDB ``macro`` library's ``VIX3M`` symbol was
   truncated to 16 rows (2026-08-07 → 2026-08-28) against SPY's 2515, so
   ``vix_term_slope`` was NaN on every earlier date and the panel collapsed
   799 → 16 dates. A too-short series must instead take the neutral default
   the module already declares for an ABSENT series.

3. ``config.META_MACRO_MIN_ROW_COVERAGE`` exists as the hard stop, so a run
   whose macro block never arrives cannot produce a champion.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from model.regime_predictor import (
    MACRO_SERIES_MIN_COVERAGE,
    RegimePredictor,
)
from training.leakfree_meta_ic import (
    dead_standalone_features,
    per_feature_standalone_ic,
)


def _panel(n_dates: int = 40, n_names: int = 12, seed: int = 7):
    """A meta panel with one market-wide column and one cross-sectional one."""
    rng = np.random.default_rng(seed)
    dates, xsec, macro, y = [], [], [], []
    for d in range(n_dates):
        macro_value = float(rng.normal())
        for _ in range(n_names):
            x = float(rng.normal())
            dates.append(f"2026-01-{d + 1:02d}")
            xsec.append(x)
            macro.append(macro_value)          # identical for every name
            y.append(0.5 * x + 0.3 * macro_value + float(rng.normal(0, 0.2)))
    X = np.column_stack([np.array(xsec), np.array(macro)])
    return X, np.array(y), dates


def test_market_wide_feature_is_named_not_silently_blank():
    """A macro column must report WHY its xsec IC is absent, and carry a
    time-series IC instead of a bare n_dates=0."""
    X, y, dates = _panel()
    out = per_feature_standalone_ic(X, y, dates, ["xsec_feature", "macro_thing"])

    assert out["xsec_feature"]["status"] == "ok"
    assert out["xsec_feature"]["xsec_ic"] is not None

    macro = out["macro_thing"]
    # The pre-fix entry had NO "status" key at all — this is the assertion that
    # fails before the fix.
    assert macro["status"] == "constant_within_date"
    assert macro["xsec_ic"] is None, "xsec IC is undefined for a date-constant column"
    assert macro["ts_ic"] is not None, "the time-series read must actually happen"
    assert macro["ts_n_dates"] == 40
    # Constructed with a +0.3 loading on the market-wide column.
    assert macro["ts_ic"] > 0.0


def test_dead_column_is_distinguishable_from_a_market_wide_one():
    """The 2026-08-21 shape: the macro column arrives as a constant 0.0 for the
    whole panel. Pre-fix this produced the identical dict to a healthy
    market-wide feature."""
    X, y, dates = _panel()
    X_dead = X.copy()
    X_dead[:, 1] = 0.0

    healthy = per_feature_standalone_ic(X, y, dates, ["xsec_feature", "macro_thing"])
    dead = per_feature_standalone_ic(X_dead, y, dates, ["xsec_feature", "macro_thing"])

    assert dead["macro_thing"]["status"] == "constant_all_rows"
    assert dead["macro_thing"] != healthy["macro_thing"]
    assert dead_standalone_features(dead) == ["macro_thing"]
    assert dead_standalone_features(healthy) == []


def test_all_nan_column_is_named():
    X, y, dates = _panel()
    X_nan = X.copy()
    X_nan[:, 1] = np.nan
    out = per_feature_standalone_ic(X_nan, y, dates, ["xsec_feature", "macro_thing"])
    assert out["macro_thing"]["status"] == "no_finite_rows"
    assert dead_standalone_features(out) == ["macro_thing"]


def _series(n, start="2018-01-01", level=100.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(
        np.abs(rng.normal(level, level * 0.05, n)),
        index=pd.date_range(start, periods=n, freq="B"),
    )


def _closes(index, k=15, seed=3):
    rng = np.random.default_rng(seed)
    return {
        f"T{i}": pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index)))), index=index
        )
        for i in range(k)
    }


def test_truncated_vix3m_does_not_annihilate_the_panel():
    """The live failure. VIX3M covering 16 of 600 dates must NOT delete the
    other 584 — every one of those dates otherwise reaches the L2 with a
    constant-0.0 macro block."""
    n = 600
    spy = _series(n, level=400.0, seed=1)
    vix = _series(n, level=18.0, seed=2)
    # Present, but only the last 16 dates — the shape found in the ArcticDB
    # 'macro' library on 2026-08-29 (VIX3M: 16 rows vs VIX's 2515).
    vix3m = _series(n, level=20.0, seed=3).iloc[-16:]
    tnx = _series(n, level=42.0, seed=4)
    irx = _series(n, level=52.0, seed=5)

    df = RegimePredictor().build_features(
        spy, vix, vix3m, tnx, irx, _closes(spy.index),
    )

    assert len(df) > 400, (
        f"panel collapsed to {len(df)} dates — a 16-date VIX3M must be treated "
        "as ABSENT (neutral default), not allowed to dropna() the panel"
    )
    # And the OTHER five core macros must still be real, not constants.
    for col in ("spy_20d_return", "spy_20d_vol", "vix_level", "yield_curve_slope",
                "market_breadth"):
        assert float(df[col].std()) > 0.0, f"{col} went constant"
    # The demoted column takes its declared neutral, and says so.
    assert float(df["vix_term_slope"].std()) == 0.0
    assert df.attrs["macro_series_coverage"]["VIX3M"] < MACRO_SERIES_MIN_COVERAGE
    assert df.attrs["macro_series_coverage"]["VIX"] == 1.0
    assert df.attrs["macro_panel_retention"] > 0.5


def test_a_legitimately_short_series_is_still_used():
    """HYOAS is license-gated to 2023+ on FRED and covers ~0.31 of the SPY
    panel. The truncation guard must not demote it."""
    n = 600
    spy = _series(n, level=400.0, seed=1)
    hyoas = _series(n, level=3.5, seed=9).iloc[-300:]   # 0.5 coverage

    df = RegimePredictor().build_features(
        spy, _series(n, level=18.0, seed=2), _series(n, level=20.0, seed=3),
        _series(n, level=42.0, seed=4), _series(n, level=52.0, seed=5),
        _closes(spy.index), hyoas_series=hyoas,
    )
    assert df.attrs["macro_series_coverage"]["HYOAS"] >= MACRO_SERIES_MIN_COVERAGE
    assert float(df["hy_oas_level"].std()) > 0.0


def test_macro_row_coverage_floor_is_declared():
    """The hard stop must exist and must be a real fraction — the gate in
    meta_trainer reads it."""
    assert 0.0 < cfg.META_MACRO_MIN_ROW_COVERAGE <= 1.0
