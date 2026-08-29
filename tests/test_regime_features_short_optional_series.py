"""Regression: one short OPTIONAL macro series must not delete regime history.

alpha-engine-config-I9258 (upstream data cause: alpha-engine-config-I9256).

Live failure this pins:

  ArcticDB's ``macro`` library held 16 rows for ``VIX3M`` (2026-08-07 ->
  2026-08-28) against ~2514 for SPY/VIX/TNX/IRX. ``build_features`` builds
  ``vix_term_slope`` from a ffill-reindexed VIX3M, so every date before
  2026-08-07 was NaN, and the terminal ``df.dropna()`` then truncated the
  WHOLE regime feature frame from 2514 dates to 16. Training logs went
  ``Regime data: 799 dates`` (2026-08-15) -> ``Regime data: 16 dates``
  (2026-08-22, 2026-08-29). Downstream, every OOS row's date fell outside
  ``regime_features_df.index``, took the 0.0 macro fill, and was labelled
  "neutral" — bull n=0 / bear n=0 in the 08-21 and 08-28manifests.

The function already declares a neutral fallback for each optional series
when it is ABSENT. Present-but-short must take the same fallback and be
recorded, not silently truncate history.
"""

import numpy as np
import pandas as pd
import pytest

from model.regime_predictor import RegimePredictor


def _spy(n: int = 2514) -> pd.Series:
    idx = pd.bdate_range("2016-08-29", periods=n)
    rng = np.random.default_rng(42)
    # Random walk with drift — gives a real spread of 20d returns so the
    # frame is not degenerate for reasons unrelated to this test.
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, n))), index=idx)


class TestShortOptionalMacroSeries:
    def test_16_row_vix3m_does_not_truncate_the_frame(self):
        spy = _spy()
        vix = pd.Series(18.0, index=spy.index)
        # The live shape: VIX3M covers only the last 16 dates.
        vix3m = pd.Series(19.0, index=spy.index[-16:])

        df = RegimePredictor().build_features(spy, vix, vix3m)

        # Before the fix this was 16.
        assert len(df) >= 2400, (
            f"regime frame truncated to {len(df)} dates by a 16-row VIX3M — "
            "one short optional series must not delete history"
        )
        # And the uncovered dates carry the SAME declared neutral the
        # absent-VIX3M branch uses, not NaN.
        assert df["vix_term_slope"].notna().all()
        assert df["vix_vix3m_ratio"].notna().all()
        assert df["vix_term_slope"].iloc[0] == pytest.approx(0.0)
        assert df["vix_vix3m_ratio"].iloc[0] == pytest.approx(1.0)

    def test_shortfall_is_recorded_not_swallowed(self):
        spy = _spy()
        vix = pd.Series(18.0, index=spy.index)
        vix3m = pd.Series(19.0, index=spy.index[-16:])

        df = RegimePredictor().build_features(spy, vix, vix3m)

        coverage = df.attrs["macro_coverage"]
        assert "VIX3M" in coverage
        assert coverage["VIX3M"]["coverage_ratio"] < 0.05
        assert coverage["VIX3M"]["n_covered"] == 16
        # A fully-covered sibling is recorded at 1.0, so a reader can tell
        # WHICH series is short.
        assert coverage["VIX"]["coverage_ratio"] == pytest.approx(1.0)

    def test_fully_covered_series_keeps_real_values(self):
        # The fallback must not fire when coverage is complete — otherwise the
        # fix would flatten a healthy signal.
        spy = _spy()
        vix = pd.Series(18.0, index=spy.index)
        vix3m = pd.Series(19.0, index=spy.index)

        df = RegimePredictor().build_features(spy, vix, vix3m)

        assert df.attrs["macro_coverage"]["VIX3M"]["coverage_ratio"] == pytest.approx(1.0)
        # (18 - 19) / 20 = -0.05 everywhere, not the 0.0 neutral.
        assert df["vix_term_slope"].iloc[-1] == pytest.approx(-0.05)

    def test_core_warmup_rows_are_still_dropped(self):
        # The dropna narrowing must not start admitting rows whose SPY-derived
        # core columns are NaN (the first 20 trading days).
        spy = _spy(300)
        df = RegimePredictor().build_features(spy, None, None)
        assert df["spy_20d_return"].notna().all()
        assert df["spy_20d_vol"].notna().all()
        assert len(df) < 300
