"""alpha-engine-config-I9255 — the 2026-08-21 macro-block outage.

Two regressions, each failing against the pre-fix code:

1. ``per_feature_standalone_ic`` reported ``{"xsec_ic": None, "n_dates": 0}``
   for every ``macro_*`` column in EVERY vintage ever produced, because those
   columns are constant WITHIN a date and ``cross_sectional_ic_series`` skips a
   date with no dispersion. A healthy market-wide feature and a dead all-zero
   column rendered BYTE-IDENTICALLY, so the outage was unobservable on the one
   surface built to observe it. It must now name which case it is, and measure
   the market-wide case the only way it can be measured (time series).

2. ``config.META_MACRO_MIN_ROW_COVERAGE`` exists as the hard stop, so a run
   whose macro block never arrives cannot produce a champion. (The upstream
   half — ``RegimePredictor.build_features``' blanket ``dropna()`` letting a
   16-row VIX3M delete 2498 dates of regime history — is fixed by
   crucible-predictor-PR579 / alpha-engine-config-I9258, which MERGES FIRST.
   This gate is the backstop that stops a champion being produced anyway.)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
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


def test_macro_row_coverage_floor_is_declared():
    """The hard stop must exist and must be a real fraction — the gate in
    meta_trainer reads it."""
    assert 0.0 < cfg.META_MACRO_MIN_ROW_COVERAGE <= 1.0
