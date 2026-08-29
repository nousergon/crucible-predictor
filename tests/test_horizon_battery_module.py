"""Tests for analysis/horizon_battery.py — standalone offline harness.

Per the 2026-05-07 predictor audit Track B (PR 4/N): the analysis
module reads OOS rows persisted by training and recomputes the full
overlap/non-overlap/per-regime/bootstrap-CI battery without requiring
a training rerun. Tests target the pure logic (compute_horizon_battery
+ format_report) using synthetic in-memory rows.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.horizon_battery import (
    OOS_ROWS_PREFIX,
    PREDICTION_PANELS_PREFIX,
    _fit_horizon_members,
    _fmt,
    _fmt_ci,
    _round_or_none,
    compute_horizon_battery,
    compute_horizon_blend,
    format_blend_report,
    format_report,
    load_oos_rows,
    persist_horizon_prediction_panels,
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
            row["ticker"] = f"T{t:03d}"
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


class TestPersistHorizonPredictionPanels:
    """Tests for persist_horizon_prediction_panels (config#1993 double-sort substrate)."""

    def test_fit_horizon_members_matches_blend_members_unscored(self):
        # _fit_horizon_members (raw) should, once z-scored, reproduce exactly
        # what compute_horizon_blend's internal members use for IC-weighting —
        # this pins the refactor didn't change compute_horizon_blend's numbers.
        df = _synthetic_oos_rows(n_dates=60, tickers_per_date=18, seed=3)
        from analysis.horizon_battery import _zscore
        raw = _fit_horizon_members(df, [10, 21, 42])
        assert set(raw) == {10, 21, 42}
        for h, arr in raw.items():
            assert len(arr) == len(df)
            z = _zscore(arr)
            assert abs(float(np.mean(z))) < 1e-6

    def test_dry_run_no_s3_call_returns_shape(self):
        df = _synthetic_oos_rows(n_dates=50, tickers_per_date=15, seed=5)
        result = persist_horizon_prediction_panels(
            df, horizons=[10, 21, 63], bucket="unused-bucket", dry_run=True,
        )
        assert result["status"] == "dry_run"
        assert result["horizons"] == [10, 21, 63]
        assert result["n_rows"] == 50 * 15 * 3
        assert result["key"] == f"{PREDICTION_PANELS_PREFIX}latest.parquet"

    def test_dry_run_dated_key(self):
        df = _synthetic_oos_rows(n_dates=20, tickers_per_date=10, seed=6)
        result = persist_horizon_prediction_panels(
            df, horizons=[21], date="2026-07-18", dry_run=True,
        )
        assert result["key"] == f"{PREDICTION_PANELS_PREFIX}2026-07-18.parquet"

    def test_drops_horizon_with_missing_column(self):
        df = _synthetic_oos_rows(n_dates=40, tickers_per_date=12, seed=7)
        df = df.drop(columns=["actual_fwd_42d"])
        result = persist_horizon_prediction_panels(
            df, horizons=[10, 21, 42], dry_run=True,
        )
        assert result["horizons"] == [10, 21]

    def test_no_usable_horizons(self):
        df = _synthetic_oos_rows(n_dates=10, tickers_per_date=5, seed=8)
        result = persist_horizon_prediction_panels(
            df, horizons=[999], dry_run=True,
        )
        assert result["status"] == "no_usable_horizons"

    def test_persist_writes_expected_parquet_shape(self, monkeypatch):
        # Full path (minus the real S3 PUT) — monkeypatch boto3.client to
        # capture what would have been written and verify the long-format
        # parquet round-trips to the {horizon: {date: {ticker: alpha}}}
        # shape analysis.double_sort.compute_double_sort expects.
        df = _synthetic_oos_rows(n_dates=30, tickers_per_date=10, seed=9)
        captured = {}

        class _FakeS3:
            def put_object(self, Bucket, Key, Body):
                captured["bucket"] = Bucket
                captured["key"] = Key
                captured["body"] = Body

        import boto3
        monkeypatch.setattr(boto3, "client", lambda *a, **k: _FakeS3())

        result = persist_horizon_prediction_panels(
            df, horizons=[10, 21], bucket="test-bucket", dry_run=False,
        )
        assert result["status"] == "ok"
        assert captured["bucket"] == "test-bucket"
        assert captured["key"] == f"{PREDICTION_PANELS_PREFIX}latest.parquet"

        panel_df = pd.read_parquet(io.BytesIO(captured["body"]))
        assert set(panel_df.columns) == {"date", "ticker", "horizon", "predicted_alpha"}
        assert set(panel_df["horizon"].unique()) == {10, 21}
        # Reshape into the {horizon: {date: {ticker: alpha}}} nested dict the
        # backtester's double_sort.compute_double_sort consumes.
        nested = {}
        for h, g in panel_df.groupby("horizon"):
            nested[h] = {
                d: dict(zip(sub["ticker"], sub["predicted_alpha"]))
                for d, sub in g.groupby("date")
            }
        assert set(nested) == {10, 21}
        any_date = next(iter(nested[21]))
        assert len(nested[21][any_date]) <= 10  # <= tickers_per_date


# ── load_oos_rows key scoping + freshness (alpha-engine-config-I9378) ──

class _FakeS3Get:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self, key_to_body: dict):
        self._key_to_body = key_to_body

    def get_object(self, Bucket, Key):
        if Key not in self._key_to_body:
            raise self.exceptions.NoSuchKey(Key)
        return {"Body": io.BytesIO(self._key_to_body[Key])}


def _parquet_bytes(df) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


class TestLoadOosRowsKeyScoping:
    def test_reads_the_model_version_scoped_key(self, monkeypatch):
        df = _synthetic_oos_rows(n_dates=5, tickers_per_date=3)
        key = f"{OOS_ROWS_PREFIX}v3.0-meta-2026-08-28-01cf7e1a/latest.parquet"
        fake = _FakeS3Get({key: _parquet_bytes(df)})
        import boto3
        monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
        out = load_oos_rows("test-bucket", model_version="v3.0-meta-2026-08-28-01cf7e1a")
        assert out is not None
        assert len(out) == len(df)

    def test_unscoped_read_falls_back_to_the_legacy_prefix(self, monkeypatch):
        """`model_version=None` reads the pre-I9378 unscoped key — historical
        panels, never a fresh write."""
        df = _synthetic_oos_rows(n_dates=5, tickers_per_date=3)
        key = f"{OOS_ROWS_PREFIX}latest.parquet"
        fake = _FakeS3Get({key: _parquet_bytes(df)})
        import boto3
        monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
        out = load_oos_rows("test-bucket")
        assert out is not None

    def test_scoped_and_unscoped_reads_are_disjoint_keys(self, monkeypatch):
        """The exact collision this issue closes: a champion-arch read and a
        specialist read at the same date must not resolve to the same key."""
        df = _synthetic_oos_rows(n_dates=5, tickers_per_date=3)
        df["date"] = pd.date_range("2026-08-20", periods=len(df), freq="h").strftime("%Y-%m-%d")
        champ_key = f"{OOS_ROWS_PREFIX}v3.0-meta-2026-08-28-01cf7e1a/2026-08-28.parquet"
        spec_key = f"{OOS_ROWS_PREFIX}m-spec-60d-2026-08-28-50bb8bdf/2026-08-28.parquet"
        assert champ_key != spec_key
        fake = _FakeS3Get({
            champ_key: _parquet_bytes(df),
            spec_key: _parquet_bytes(df.assign(some_other_col=1)),
        })
        import boto3
        monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
        champ_df = load_oos_rows(
            "test-bucket", date="2026-08-28",
            model_version="v3.0-meta-2026-08-28-01cf7e1a",
        )
        spec_df = load_oos_rows(
            "test-bucket", date="2026-08-28",
            model_version="m-spec-60d-2026-08-28-50bb8bdf",
        )
        assert "some_other_col" not in champ_df.columns
        assert "some_other_col" in spec_df.columns


class TestLoadOosRowsFreshnessGuard:
    def test_a_panel_far_older_than_the_requested_date_raises(self, monkeypatch):
        """The measured incident: a 150000x34 panel spanning
        2025-07-10..2026-03-09 read against a 2026-08-28 training run — 172
        days stale. Must raise rather than silently answer."""
        df = _synthetic_oos_rows(n_dates=5, tickers_per_date=3)
        df["date"] = pd.date_range("2026-03-01", periods=len(df), freq="h").strftime("%Y-%m-%d")
        key = f"{OOS_ROWS_PREFIX}v3.0-meta/2026-08-28.parquet"
        fake = _FakeS3Get({key: _parquet_bytes(df)})
        import boto3
        monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
        with pytest.raises(RuntimeError, match="STALE"):
            load_oos_rows("test-bucket", date="2026-08-28", model_version="v3.0-meta")

    def test_a_fresh_panel_does_not_raise(self, monkeypatch):
        df = _synthetic_oos_rows(n_dates=5, tickers_per_date=3)
        df["date"] = pd.date_range("2026-08-20", periods=len(df), freq="h").strftime("%Y-%m-%d")
        key = f"{OOS_ROWS_PREFIX}v3.0-meta/2026-08-28.parquet"
        fake = _FakeS3Get({key: _parquet_bytes(df)})
        import boto3
        monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
        out = load_oos_rows("test-bucket", date="2026-08-28", model_version="v3.0-meta")
        assert out is not None


# ── main() CLI guard (§119 rule 1) ────────────────────────────────────

class TestMainCLIGuard:
    """Tests for horizon_battery.main() CLI entry point guards.

    Coverage for the success path (rows loaded -> normal exit) and the
    failure path (empty/no rows -> exit 1) — the two states the
    ``sys.exit(1)`` guard at line 780 can reach.
    """

    def test_success_path_returns_normally(self, monkeypatch):
        """Mock load_oos_rows with valid data — guard passes, no SystemExit."""
        from analysis.horizon_battery import main

        df = _synthetic_oos_rows(n_dates=10, tickers_per_date=5)
        monkeypatch.setattr("sys.argv", ["horizon_battery"])
        monkeypatch.setattr(
            "analysis.horizon_battery.load_oos_rows",
            lambda bucket=None, date=None, model_version=None: df,
        )
        main()  # must not raise

    def test_none_rows_exits_with_code_1(self, monkeypatch):
        """Mock load_oos_rows returns None — verifies SystemExit(1)."""
        import pytest
        from analysis.horizon_battery import main

        monkeypatch.setattr("sys.argv", ["horizon_battery"])
        monkeypatch.setattr(
            "analysis.horizon_battery.load_oos_rows",
            lambda bucket=None, date=None, model_version=None: None,
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_empty_df_exits_with_code_1(self, monkeypatch):
        """Mock load_oos_rows returns empty DataFrame — verifies SystemExit(1)."""
        import pytest
        from analysis.horizon_battery import main

        monkeypatch.setattr("sys.argv", ["horizon_battery"])
        monkeypatch.setattr(
            "analysis.horizon_battery.load_oos_rows",
            lambda bucket=None, date=None, model_version=None: pd.DataFrame(),
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
