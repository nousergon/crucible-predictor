"""Parity + contract tests for the research-calibrator label cutover onto the
long-format ``score_performance_outcomes`` store (config#1527, EPIC config#1483).

The acceptance bar: the migrated read must produce a label frame IDENTICAL to
the pre-migration wide-column read
(``SELECT score, beat_spy_21d, score_date FROM score_performance WHERE …``).
The fixture builds BOTH representations of the same ground truth exactly as
the producers do (wide: alpha-engine-data ``signal_returns`` Step 2; long:
Step 2c — decimal returns, primary-horizon rows carry the canonical label) and
asserts row-set equality plus identical fitted-calibrator behavior. The
full-history replay against a copy of the live research.db (the method proven
on config#1483) is run at PR time; this fixture pins the invariants in CI.
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd
import pytest
from nousergon_lib.quant.horizons import DEFAULT_POLICY

from training.meta_trainer import _load_research_calibrator_frame

_PRIMARY = DEFAULT_POLICY.primary_horizon

# (symbol, score_date, score, ret21, spy21) decimals; TSLA unresolved,
# PLTR resolved-but-scoreless (excluded by both reads).
_TRUTH = [
    ("AAPL", "2026-05-01", 78.0, 0.0432, 0.0201),
    ("MSFT", "2026-05-01", 55.0, -0.0311, 0.0201),
    ("NVDA", "2026-05-08", 91.0, 0.1023, 0.0250),
    ("KO", "2026-05-08", 62.0, 0.0260, 0.0250),
    ("JPM", "2026-05-15", 70.0, 0.0140, 0.0080),
    ("PLTR", "2026-05-15", None, 0.0500, 0.0080),
    ("TSLA", "2026-06-20", 83.0, None, None),
]


@pytest.fixture()
def research_db(tmp_path):
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE score_performance (
            symbol TEXT, score_date TEXT, score REAL,
            return_21d REAL, spy_21d_return REAL, beat_spy_21d INTEGER,
            log_alpha_21d REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE score_performance_outcomes (
            id INTEGER PRIMARY KEY, signal_id TEXT NOT NULL,
            symbol TEXT NOT NULL, score_date TEXT NOT NULL,
            horizon_days INTEGER NOT NULL, beat_spy INTEGER,
            stock_return REAL, spy_return REAL, log_alpha REAL,
            is_primary INTEGER NOT NULL, resolved_at TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1,
            UNIQUE(signal_id, horizon_days)
        )"""
    )
    for sym, d, score, r21, s21 in _TRUTH:
        resolved = r21 is not None
        la = round(float(np.log1p(r21) - np.log1p(s21)), 6) if resolved else None
        conn.execute(
            "INSERT INTO score_performance VALUES (?,?,?,?,?,?,?)",
            (
                sym, d, score,
                round(r21 * 100, 2) if resolved else None,
                round(s21 * 100, 2) if resolved else None,
                (1 if r21 > s21 else 0) if resolved else None,
                la,
            ),
        )
        if resolved:
            conn.execute(
                "INSERT INTO score_performance_outcomes "
                "(signal_id, symbol, score_date, horizon_days, beat_spy, "
                " stock_return, spy_return, log_alpha, is_primary, resolved_at) "
                "VALUES (?,?,?,?,?,?,?,?,1,'2026-06-27T00:00:00+00:00')",
                (f"{sym}:{d}", sym, d, _PRIMARY,
                 1 if r21 > s21 else 0, r21, s21, la),
            )
    conn.commit()
    conn.close()
    return str(db)


def _legacy_wide_frame(db_path: str) -> pd.DataFrame:
    """The pre-migration read, byte-for-byte (crucible-predictor pre-#1527)."""
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(
            "SELECT score, beat_spy_21d, score_date FROM score_performance "
            "WHERE beat_spy_21d IS NOT NULL AND score IS NOT NULL "
            "AND score_date IS NOT NULL",
            conn,
        )
    finally:
        conn.close()


def test_label_frame_parity_with_legacy_wide_read(research_db):
    """Row-set parity: same (score, label, score_date) triples as the legacy
    wide-column read (order-insensitive — every downstream use is
    order-independent: bucket-lookup calibrator fit + per-fold date masks)."""
    legacy = _legacy_wide_frame(research_db)
    migrated = _load_research_calibrator_frame(research_db)

    key = ["score_date", "score"]
    legacy_sorted = (
        legacy.rename(columns={"beat_spy_21d": "beat_spy"})
        .sort_values(key).reset_index(drop=True)
    )
    migrated_sorted = migrated.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        migrated_sorted[["score", "beat_spy", "score_date"]].astype(
            {"beat_spy": "int64"}
        ),
        legacy_sorted[["score", "beat_spy", "score_date"]].astype(
            {"beat_spy": "int64"}
        ),
    )
    # unresolved (TSLA) and scoreless (PLTR) rows excluded by both reads
    assert len(migrated) == len(legacy) == 5


def test_fitted_calibrator_identical_on_both_sources(research_db):
    from model.research_calibrator import ResearchCalibrator

    legacy = _legacy_wide_frame(research_db)
    migrated = _load_research_calibrator_frame(research_db)

    cal_legacy = ResearchCalibrator().fit(
        legacy["score"].to_numpy(), legacy["beat_spy_21d"].to_numpy().astype(int)
    )
    cal_migrated = ResearchCalibrator().fit(
        migrated["score"].to_numpy(), migrated["beat_spy"].to_numpy().astype(int)
    )
    grid = np.linspace(0, 100, 201)
    np.testing.assert_array_equal(
        cal_legacy.predict_batch(grid), cal_migrated.predict_batch(grid)
    )
    assert cal_legacy.metrics() == cal_migrated.metrics()


def test_missing_store_is_graceful_and_loud(tmp_path, caplog):
    """No long-store table → empty frame + a cause-naming WARN (the caller's
    existing priors fallback engages); never an exception, never silence."""
    db = tmp_path / "stale.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE score_performance (symbol TEXT, score REAL)")
    conn.commit()
    conn.close()
    with caplog.at_level(logging.WARNING):
        frame = _load_research_calibrator_frame(str(db))
    assert frame.empty
    assert list(frame.columns) == ["score", "beat_spy", "score_date"]
    assert any(
        "score_performance_outcomes ABSENT" in r.message for r in caplog.records
    )


def test_read_filters_to_the_policy_primary_horizon(research_db):
    """A diagnostic-horizon row must NOT leak into the label frame."""
    conn = sqlite3.connect(research_db)
    conn.execute(
        "INSERT INTO score_performance_outcomes "
        "(signal_id, symbol, score_date, horizon_days, beat_spy, "
        " stock_return, spy_return, log_alpha, is_primary, resolved_at) "
        "VALUES ('AAPL:2026-05-01', 'AAPL', '2026-05-01', ?, 0, "
        "        0.01, 0.02, NULL, 0, '2026-06-27T00:00:00+00:00')",
        (DEFAULT_POLICY.diagnostic_horizons[0],),
    )
    conn.commit()
    conn.close()
    frame = _load_research_calibrator_frame(research_db)
    # still exactly one AAPL@2026-05-01 row, carrying the PRIMARY label (=1)
    aapl = frame[(frame["score_date"] == "2026-05-01") & (frame["score"] == 78.0)]
    assert len(aapl) == 1
    assert int(aapl["beat_spy"].iloc[0]) == 1
