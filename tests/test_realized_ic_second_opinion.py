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
    """rows: list of (symbol, score_date, log_alpha).

    alpha-engine-config-I9038 — the ground truth is now ``universe_returns``,
    which stores the two log legs rather than the difference. `log_alpha` is
    split into (log_return, log_spy_return) so the module derives exactly the
    value the caller asked for, mirroring production where
    ``log_alpha == log_return_21d - log_spy_return_21d`` held on 534/534
    overlapping rows.
    """
    db_path = tmp_path / "research.db"
    conn = sqlite3.connect(str(db_path))
    h = int(horizon_days)
    conn.execute(
        f"CREATE TABLE universe_returns "
        f"(ticker TEXT, eval_date TEXT, log_return_{h}d REAL, log_spy_return_{h}d REAL)"
    )
    conn.executemany(
        f"INSERT INTO universe_returns VALUES (?, ?, ?, ?)",
        # spy leg fixed at 0.0 so the derived alpha equals the requested value.
        [(sym, d, la, 0.0) for sym, d, la in rows],
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


# ── alpha-engine-config-I9038 — coverage is not join integrity ───────────────
# Rows predating the ground truth's earliest resolved date cannot match however
# sound the join is. Folding them into match_rate turned a 0.5 join-integrity
# threshold into an archive-depth check no rotation could pass, and since
# low_match_rate blocks promotion unconditionally (I9030) that silently froze
# the model slot from 2026-07-24 onward.


def _rows(n, start_day=1, month="01"):
    return [(f"T{i}", f"2026-{month}-{start_day + i:02d}", 0.01 * (i - n / 2))
            for i in range(n)]


def test_out_of_coverage_rows_do_not_count_against_match_rate(tmp_path):
    """The production shape: a sound join over a short archive must read `ok`."""
    covered = _rows(40, start_day=1, month="03")
    db_path = _make_db(tmp_path, covered)
    # 40 in-coverage rows plus 120 that predate the archive entirely.
    ids = [t for t, _, _ in covered] + [f"OLD{i}" for i in range(120)]
    dates = [d for _, d, _ in covered] + [f"2025-07-{(i % 28) + 1:02d}" for i in range(120)]
    preds = [la for _, _, la in covered] + [0.0] * 120

    result = compute_second_opinion_ic(
        preds, ids, dates, db_path=str(db_path), horizon_days=21,
    )
    assert result["n_oos_rows"] == 160
    assert result["n_in_coverage"] == 40
    assert result["n_matched"] == 40
    assert result["match_rate"] == 1.0, "match_rate must use the coverable denominator"
    assert result["status"] == "ok"
    # The shortfall is still reported — it is a real finding, just a different one.
    assert result["coverage_fraction"] == 0.25
    assert result["coverage_window"] == ["2026-03-01", "2026-03-40"[:10]] or \
        result["coverage_window"][0] == "2026-03-01"
    assert result["ground_truth_source"] == "universe_returns"


def test_a_genuine_join_break_inside_coverage_still_reads_low_match_rate(tmp_path):
    """The guard must still bite: same window, mismatched keys."""
    covered = _rows(40, start_day=1, month="03")
    db_path = _make_db(tmp_path, covered)
    # Same dates (so fully in coverage) but tickers that do not exist.
    ids = [f"MISSING{i}" for i in range(40)] + [t for t, _, _ in covered[:5]]
    dates = [d for _, d, _ in covered] + [d for _, d, _ in covered[:5]]
    preds = [0.01 * i for i in range(45)]

    result = compute_second_opinion_ic(
        preds, ids, dates, db_path=str(db_path), horizon_days=21, min_matched=1,
    )
    assert result["n_in_coverage"] == 45
    assert result["n_matched"] == 5
    assert result["match_rate"] < 0.5
    assert result["status"] == "low_match_rate"


def test_absent_universe_returns_is_unavailable_not_a_join_failure(tmp_path):
    """A missing table means the check could not RUN — non-blocking per
    champion-challenger-policy 5.1, never a join-integrity verdict."""
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    result = compute_second_opinion_ic(
        [0.1] * 50, [f"T{i}" for i in range(50)],
        ["2026-03-01"] * 50, db_path=str(db_path), horizon_days=21,
    )
    assert result["status"] == "insufficient_matched_rows"
    assert result["n_matched"] == 0


def test_a_horizon_the_returns_table_does_not_carry_is_not_a_join_failure(tmp_path):
    """universe_returns carries a fixed ladder; asking for 999d is honest
    absence, not evidence against the candidate."""
    db_path = _make_db(tmp_path, _rows(40, start_day=1, month="03"), horizon_days=21)
    result = compute_second_opinion_ic(
        [0.1] * 40, [f"T{i}" for i in range(40)],
        ["2026-03-01"] * 40, db_path=str(db_path), horizon_days=999,
    )
    assert result["n_matched"] == 0
    assert result["status"] == "insufficient_matched_rows"


def test_the_old_denominator_is_what_blocked_every_rotation(tmp_path):
    """champion-challenger-policy 7.4 — demonstrate the pre-fix failure in place.

    Same fixture as the coverage test above. Under the OLD semantics the
    denominator was every OOS row, so a sound join over a short archive scored
    40/160 = 0.25 and tripped MIN_MATCH_RATE, which `alpha-engine-config-I9030`
    makes an unconditional promotion block. That is the mechanism that froze
    the model slot from 2026-07-24 onward, reproduced here so the regression
    cannot return silently.
    """
    covered = _rows(40, start_day=1, month="03")
    db_path = _make_db(tmp_path, covered)
    ids = [t for t, _, _ in covered] + [f"OLD{i}" for i in range(120)]
    dates = [d for _, d, _ in covered] + [f"2025-07-{(i % 28) + 1:02d}" for i in range(120)]
    preds = [la for _, _, la in covered] + [0.0] * 120

    result = compute_second_opinion_ic(
        preds, ids, dates, db_path=str(db_path), horizon_days=21,
    )
    old_style_rate = result["n_matched"] / result["n_oos_rows"]
    assert old_style_rate == 0.25
    assert old_style_rate < MIN_MATCH_RATE, "the old denominator failed the bar"
    assert result["match_rate"] >= MIN_MATCH_RATE, "the new denominator passes it"
    assert result["status"] == "ok"
