"""Tests for analysis/horizon_battery.py — standalone offline harness.

Per the 2026-05-07 predictor audit Track B (PR 4/N): the analysis
module reads OOS rows persisted by training and recomputes the full
overlap/non-overlap/per-regime/bootstrap-CI battery without requiring
a training rerun. Tests target the pure logic (compute_horizon_battery
+ format_report) using synthetic in-memory rows.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.horizon_battery import (
    _fmt,
    _fmt_ci,
    _round_or_none,
    compute_horizon_battery,
    compute_horizon_blend,
    format_blend_report,
    format_report,
)
from training.meta_trainer import _DIAGNOSTIC_HORIZONS, _ENSEMBLE_HORIZONS


def _synthetic_oos_rows(n_dates: int = 60, tickers_per_date: int = 20, seed: int = 0):
    """Build a DataFrame matching meta_trainer's OOS row schema.

    Includes ALL META_FEATURES + actual_fwd + actual_fwd_{h}d for each
    diagnostic horizon. Synthetic strong signal: actual is a noisy linear
    function of the META_FEATURES so the fitted Ridge will produce a
    meaningful IC.
    """
    from model.meta_model import META_FEATURES

    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2025-01-02")
    bdays = pd.bdate_range(start=base, periods=n_dates)
    for d in bdays:
        for t in range(tickers_per_date):
            row = {
                f: float(rng.normal(0, 1)) for f in META_FEATURES
            }
            # Construct an actual signal correlated with the features.
            actual = sum(row[f] for f in META_FEATURES) * 0.05 + rng.normal(0, 1.5)
            row["actual_fwd"] = float(actual)
            # Multi-horizon labels — noisier at long horizons (more drift).
            # Iterate the live diagnostic ladder so the fixture always carries a
            # column for every horizon the module measures (incl. the config#937
            # ratified ensemble ladder 42/63/126d).
            for h in _DIAGNOSTIC_HORIZONS:
                row[f"actual_fwd_{h}d"] = float(actual + rng.normal(0, 0.3 * h / 5))
            row["date"] = d.strftime("%Y-%m-%d")
            # Override macro_spy_20d_return for regime classification:
            # cycle through bull/neutral/bear so all three are populated.
            phase = (rng.integers(0, 3))
            row["macro_spy_20d_return"] = float(
                {0: 0.06, 1: 0.0, 2: -0.06}[int(phase)]
            )
            rows.append(row)
    return pd.DataFrame(rows)


# ── _round_or_none / formatting helpers ──────────────────────────────────

class TestFormattingHelpers:

    def test_round_or_none_finite(self):
        assert _round_or_none(0.123456789, ndigits=4) == 0.1235

    def test_round_or_none_nan(self):
        assert _round_or_none(float("nan")) is None

    def test_round_or_none_none(self):
        assert _round_or_none(None) is None

    def test_fmt_finite(self):
        assert _fmt(0.0123) == "+0.0123"

    def test_fmt_negative(self):
        assert _fmt(-0.0456) == "-0.0456"

    def test_fmt_none(self):
        assert _fmt(None) == "—"

    def test_fmt_ci_finite(self):
        assert _fmt_ci(0.01, 0.05) == "[+0.010,+0.050]"

    def test_fmt_ci_none_lo(self):
        assert _fmt_ci(None, 0.05) == "[—]"


# ── compute_horizon_battery ──────────────────────────────────────────────

class TestComputeHorizonBattery:

    def test_basic_shape(self):
        df = _synthetic_oos_rows(n_dates=40, tickers_per_date=15)
        report = compute_horizon_battery(df, bootstrap_iter=50)
        assert "horizons" in report
        assert "regime_distribution" in report
        assert "curve" in report
        assert "n_rows" in report
        assert report["n_rows"] == 40 * 15
        # All diagnostic horizons present in curve (if columns exist).
        for h in ["5d", "10d", "15d", "21d", "40d", "60d", "90d"]:
            assert h in report["curve"]

    def test_curve_per_horizon_fields(self):
        df = _synthetic_oos_rows(n_dates=60, tickers_per_date=20)
        report = compute_horizon_battery(df, bootstrap_iter=100)
        h5 = report["curve"]["5d"]
        # All required fields present.
        for field in (
            "spearman", "n", "spearman_ci_lo", "spearman_ci_hi",
            "spearman_nonoverlap", "n_nonoverlap",
            "spearman_nonoverlap_ci_lo", "spearman_nonoverlap_ci_hi",
            "by_regime",
        ):
            assert field in h5, f"missing field: {field}"
        # by_regime contains all three regimes.
        for regime in ("bull", "neutral", "bear"):
            assert regime in h5["by_regime"]
            assert "spearman" in h5["by_regime"][regime]
            assert "n" in h5["by_regime"][regime]

    def test_regime_distribution_sums_to_n_rows(self):
        df = _synthetic_oos_rows(n_dates=50, tickers_per_date=12)
        report = compute_horizon_battery(df, bootstrap_iter=50)
        dist = report["regime_distribution"]
        assert sum(dist.values()) == report["n_rows"]

    def test_horizons_argument_filters_curve(self):
        df = _synthetic_oos_rows(n_dates=30, tickers_per_date=10)
        report = compute_horizon_battery(
            df, horizons=[5, 21], bootstrap_iter=50,
        )
        assert set(report["curve"].keys()) == {"5d", "21d"}

    def test_bootstrap_iter_propagates_to_report(self):
        df = _synthetic_oos_rows(n_dates=30, tickers_per_date=10)
        report = compute_horizon_battery(df, bootstrap_iter=250)
        assert report["bootstrap_iter"] == 250

    def test_strong_signal_yields_finite_ic(self):
        # Synthetic signal should produce a finite IC (sign depends on
        # Ridge fit, but should not be NaN with enough rows).
        df = _synthetic_oos_rows(n_dates=80, tickers_per_date=20, seed=7)
        report = compute_horizon_battery(df, bootstrap_iter=100)
        # Overall 5d IC should be finite.
        assert report["curve"]["5d"]["spearman"] is not None

    def test_missing_horizon_column_skipped_gracefully(self):
        # Drop one of the horizon columns; the helper should log a
        # warning and skip that horizon, not crash.
        df = _synthetic_oos_rows(n_dates=20, tickers_per_date=8)
        df = df.drop(columns=["actual_fwd_60d"])
        report = compute_horizon_battery(df, bootstrap_iter=50)
        assert "60d" not in report["curve"]
        assert "5d" in report["curve"]  # other horizons still computed


# ── format_report ────────────────────────────────────────────────────────

class TestFormatReport:

    def test_includes_header(self):
        df = _synthetic_oos_rows(n_dates=20, tickers_per_date=8)
        report = compute_horizon_battery(df, bootstrap_iter=50)
        rendered = format_report(report)
        assert "Horizon battery" in rendered
        assert "Regime distribution" in rendered
        # Column headers present.
        assert "Horizon" in rendered
        assert "IC bull" in rendered

    def test_per_horizon_row_present(self):
        df = _synthetic_oos_rows(n_dates=25, tickers_per_date=10)
        report = compute_horizon_battery(df, horizons=[5, 21], bootstrap_iter=50)
        rendered = format_report(report)
        assert "5d" in rendered
        assert "21d" in rendered

    def test_handles_nan_ic_gracefully(self):
        # Force a horizon with too few finite labels by zeroing them.
        df = _synthetic_oos_rows(n_dates=20, tickers_per_date=5)
        df["actual_fwd_90d"] = float("nan")
        report = compute_horizon_battery(df, bootstrap_iter=50)
        rendered = format_report(report)
        # Should render without crashing; "—" or similar for NaN cells.
        assert "90d" in rendered


# ── ratified ensemble ladder (config#937) ────────────────────────────────

class TestEnsembleLadder:

    def test_ratified_ladder_is_subset_of_diagnostic(self):
        # The operator-ratified 10/21/42/63/126d must all be measurable, i.e.
        # present in the diagnostic ladder that drives OOS-row persistence.
        assert _ENSEMBLE_HORIZONS == [10, 21, 42, 63, 126]
        assert set(_ENSEMBLE_HORIZONS).issubset(set(_DIAGNOSTIC_HORIZONS))

    def test_legacy_horizons_retained(self):
        # Additive: pre-config#937 diagnostic horizons must survive so no
        # existing consumer loses a forward-return column.
        for h in (5, 10, 15, 21, 40, 60, 90):
            assert h in _DIAGNOSTIC_HORIZONS


# ── compute_horizon_blend (config#937 Phase 1) ───────────────────────────

class TestComputeHorizonBlend:

    def test_basic_shape(self):
        df = _synthetic_oos_rows(n_dates=80, tickers_per_date=20, seed=3)
        report = compute_horizon_blend(df, bootstrap_iter=100, n_cscv_blocks=6)
        assert report["target_horizon"] == "21d"
        assert report["ladder"] == [f"{h}d" for h in _ENSEMBLE_HORIZONS]
        assert "baseline" in report and "blends" in report
        assert set(report["blends"]) == {"ic_weighted", "ridge_blend"}
        assert "cscv_pbo" in report
        # Every ladder member has an IC + a normalized blend weight.
        assert set(report["members"]) == {f"{h}d" for h in _ENSEMBLE_HORIZONS}
        wsum = sum(m["blend_weight"] for m in report["members"].values())
        assert abs(wsum - 1.0) < 1e-6

    def test_uplift_is_ic_minus_baseline(self):
        df = _synthetic_oos_rows(n_dates=70, tickers_per_date=18, seed=5)
        report = compute_horizon_blend(df, bootstrap_iter=80, n_cscv_blocks=6)
        base = report["baseline"]["ic"]
        for name in ("ic_weighted", "ridge_blend"):
            bl = report["blends"][name]
            if bl["ic"] is not None and base is not None:
                # uplift rounds the raw diff; ic/base are each pre-rounded to 6dp,
                # so allow one ULP of rounding slack (max ~1.5e-6).
                assert abs(bl["uplift_vs_baseline"] - round(bl["ic"] - base, 6)) < 2e-6

    def test_singleton_ladder_blend_equals_baseline(self):
        # A ladder of only the target horizon must make both blends collapse to
        # the baseline signal — uplift ≈ 0 (sanity: blending nothing adds nothing).
        df = _synthetic_oos_rows(n_dates=70, tickers_per_date=18, seed=9)
        report = compute_horizon_blend(
            df, horizons=[21], target_horizon=21,
            bootstrap_iter=50, n_cscv_blocks=6,
        )
        assert report["ladder"] == ["21d"]
        for name in ("ic_weighted", "ridge_blend"):
            up = report["blends"][name]["uplift_vs_baseline"]
            assert up is None or abs(up) < 1e-6

    def test_target_absent_raises(self):
        df = _synthetic_oos_rows(n_dates=40, tickers_per_date=10)
        df = df.drop(columns=["actual_fwd_21d"])
        with pytest.raises(ValueError):
            compute_horizon_blend(df, target_horizon=21, bootstrap_iter=20)

    def test_pbo_in_unit_interval_or_none(self):
        df = _synthetic_oos_rows(n_dates=90, tickers_per_date=20, seed=11)
        report = compute_horizon_blend(df, bootstrap_iter=60, n_cscv_blocks=8)
        pbo = report["cscv_pbo"].get("pbo")
        assert pbo is None or (0.0 <= pbo <= 1.0)

    def test_format_blend_report_renders(self):
        df = _synthetic_oos_rows(n_dates=70, tickers_per_date=16, seed=13)
        report = compute_horizon_blend(df, bootstrap_iter=50, n_cscv_blocks=6)
        rendered = format_blend_report(report)
        assert "Temporal-ensemble blend" in rendered
        assert "Baseline" in rendered
        assert "CSCV/PBO" in rendered
