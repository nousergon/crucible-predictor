"""config#2889 (Brian's 2026-07-18 Decision Queue Option-B ruling) —
independent second-party IC recomputation from realized outcomes.

Pins: (1) compute_second_opinion_ic independently re-derives ground truth
from a FRESH score_performance_outcomes read (never trusts the caller's
"true" array — the whole point is it must NOT be poisoned by the same bug
that might poison meta_y), (2) low match-rate / too-few-rows degrade to a
non-blocking "unavailable-ish" status rather than crashing or asserting a
bogus IC, (3) evaluate_second_opinion_gate flags a real divergence
(sign flip, large gap, low match rate) and clears a corroborating one, and
never blocks by itself (the caller's enforce flag decides).
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from training.realized_ic_second_opinion import (
    DEFAULT_MAX_IC_GAP,
    MIN_MATCH_RATE,
    compute_second_opinion_ic,
    evaluate_second_opinion_gate,
)


def _make_db(tmp_path, rows, horizon_days=21):
    """rows: list of (symbol, score_date, log_alpha)."""
    db_path = tmp_path / "research.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE score_performance_outcomes "
        "(symbol TEXT, score_date TEXT, horizon_days INTEGER, log_alpha REAL)"
    )
    conn.executemany(
        "INSERT INTO score_performance_outcomes VALUES (?, ?, ?, ?)",
        [(sym, d, horizon_days, la) for sym, d, la in rows],
    )
    conn.commit()
    conn.close()
    return db_path


# ── compute_second_opinion_ic ───────────────────────────────────────────


def test_no_oos_predictions_returns_status_without_crashing(tmp_path):
    db_path = _make_db(tmp_path, [])
    result = compute_second_opinion_ic(
        [], [], [], db_path=db_path, horizon_days=21,
    )
    assert result["status"] == "no_oos_predictions"
    assert result["second_opinion_ic"] is None


def test_missing_table_degrades_to_unavailable_style_status(tmp_path):
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    preds = [0.1] * 40
    ids = [f"T{i % 5}" for i in range(40)]
    dates = ["2026-01-01"] * 40
    result = compute_second_opinion_ic(
        preds, ids, dates, db_path=db_path, horizon_days=21,
    )
    # no table -> zero rows fetched -> zero matched -> insufficient
    assert result["status"] == "insufficient_matched_rows"
    assert result["second_opinion_ic"] is None
    assert result["n_matched"] == 0


def test_independently_recovers_a_real_signal_bypassing_caller_supplied_truth():
    """The key property: ground truth comes ONLY from the DB, never from
    whatever the caller might claim is 'true'. Build predictions that
    correlate with the DB's log_alpha and confirm the recovered IC is high
    — using only (ticker, date) identity to rejoin, nothing else."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(3)
        n = 200
        tickers = [f"T{i}" for i in range(n)]
        dates = [f"2026-02-{(i % 25) + 1:02d}" for i in range(n)]
        true_alpha = rng.normal(size=n)
        preds = true_alpha + rng.normal(scale=0.05, size=n)  # near-perfect corroboration
        db_path = _make_db(
            Path(td), list(zip(tickers, dates, true_alpha.tolist())),
        )
        result = compute_second_opinion_ic(
            preds, tickers, dates, db_path=db_path, horizon_days=21,
        )
        assert result["status"] == "ok"
        assert result["n_matched"] == n
        assert result["second_opinion_ic"] > 0.9


def test_low_match_rate_flagged_via_status(tmp_path):
    """Most predicted rows have no corresponding realized outcome (a
    label/join integrity issue) — status must say so, not just report a
    thin-sample IC as if the data were healthy."""
    rng = np.random.default_rng(1)
    n = 100
    tickers = [f"T{i}" for i in range(n)]
    dates = [f"2026-03-{(i % 25) + 1:02d}" for i in range(n)]
    preds = rng.normal(size=n)
    # Only the first 35 rows (>= MIN_MATCHED_FOR_VERDICT) have a realized
    # outcome in the store — 35% match rate, well under MIN_MATCH_RATE.
    matched_n = 35
    db_path = _make_db(
        tmp_path,
        list(zip(tickers[:matched_n], dates[:matched_n], rng.normal(size=matched_n).tolist())),
    )
    result = compute_second_opinion_ic(preds, tickers, dates, db_path=db_path, horizon_days=21)
    assert result["status"] == "low_match_rate"
    assert result["match_rate"] < MIN_MATCH_RATE
    assert result["second_opinion_ic"] is not None  # still computed, just flagged


def test_bad_db_path_degrades_to_unavailable_never_raises(tmp_path):
    result = compute_second_opinion_ic(
        [0.1] * 40, [f"T{i}" for i in range(40)], ["2026-01-01"] * 40,
        db_path=tmp_path / "does" / "not" / "exist.db", horizon_days=21,
    )
    assert result["status"] == "unavailable"
    assert result["second_opinion_ic"] is None


# ── evaluate_second_opinion_gate ────────────────────────────────────────


def test_unavailable_second_opinion_never_flags_divergence():
    verdict = evaluate_second_opinion_gate(
        0.10, {"status": "insufficient_matched_rows", "second_opinion_ic": None},
    )
    assert verdict["divergence_detected"] is False


def test_corroborating_second_opinion_does_not_diverge():
    verdict = evaluate_second_opinion_gate(
        0.10, {"status": "ok", "second_opinion_ic": 0.09, "match_rate": 0.9},
    )
    assert verdict["divergence_detected"] is False


def test_sign_flip_flags_divergence():
    verdict = evaluate_second_opinion_gate(
        0.10, {"status": "ok", "second_opinion_ic": -0.08, "match_rate": 0.9},
    )
    assert verdict["divergence_detected"] is True
    assert "sign flip" in verdict["reason"]


def test_large_gap_flags_divergence():
    verdict = evaluate_second_opinion_gate(
        0.30, {"status": "ok", "second_opinion_ic": 0.01, "match_rate": 0.9},
        max_ic_gap=DEFAULT_MAX_IC_GAP,
    )
    assert verdict["divergence_detected"] is True
    assert "exceeds" in verdict["reason"]


def test_low_match_rate_flags_divergence_even_with_close_ics():
    verdict = evaluate_second_opinion_gate(
        0.10, {"status": "low_match_rate", "second_opinion_ic": 0.10, "match_rate": 0.2},
    )
    assert verdict["divergence_detected"] is True
    assert "rejoined" in verdict["reason"]


def test_no_cpcv_mean_ic_to_compare_is_not_a_divergence():
    verdict = evaluate_second_opinion_gate(
        None, {"status": "ok", "second_opinion_ic": 0.10, "match_rate": 0.9},
    )
    assert verdict["divergence_detected"] is False
