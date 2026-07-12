"""Tests for the non-overlapping date-mask helper used by Track B horizon battery.

Per the 2026-05-07 predictor audit: the existing horizon IC diagnostic in
``training/meta_trainer.py`` samples (date, ticker) rows with a forward
window that overlaps heavily across consecutive dates. The
``_nonoverlapping_date_mask`` helper subsamples one date per non-overlapping
h-day window so IC computed on the subsample reflects independent
observations.

Spacing is **trading-day-exact** on the NYSE calendar (config#937 Step 0):
the forward-return label is a row-shift on a trading-day-only-indexed series,
so a calendar-day proxy under-spaces samples at longer horizons and leaks
autocorrelation back into the diagnostic. These tests pin the trading-day
semantics — including the regression case a calendar-day proxy would fail.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from krepis.trading_calendar import add_trading_days, count_trading_days

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.meta_trainer import _nonoverlapping_date_mask


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _ladder(start: str, n: int) -> list[pd.Timestamp]:
    """``n`` consecutive NYSE trading sessions starting at trading day ``start``.

    On such a ladder ``count_trading_days(ladder[a], ladder[b]) == b - a``,
    so kept indices are exactly the multiples of the horizon — the greedy
    walk's behaviour is deterministic and hand-checkable.
    """
    d0 = pd.Timestamp(start).date()
    out = [d0]
    for _ in range(n - 1):
        out.append(add_trading_days(out[-1], 1))
    return [pd.Timestamp(d) for d in out]


class TestNonoverlappingDateMask:

    def test_empty_input_returns_empty(self):
        assert _nonoverlapping_date_mask([], 21) == []

    def test_single_date_kept(self):
        assert _nonoverlapping_date_mask([_ts("2026-01-05")], 21) == [True]

    def test_consecutive_sessions_5d_horizon_thins_to_every_fifth(self):
        # 11 consecutive trading sessions; horizon=5 keeps sessions 0, 5, 10.
        dates = _ladder("2026-01-05", 11)  # Mon 2026-01-05 is a trading day
        mask = _nonoverlapping_date_mask(dates, 5)
        assert [i for i, m in enumerate(mask) if m] == [0, 5, 10]

    def test_exact_horizon_boundary_kept(self):
        # Second date exactly h=21 trading sessions past the first → kept.
        d0 = pd.Timestamp("2026-01-05")
        d1 = pd.Timestamp(add_trading_days(d0.date(), 21))
        mask = _nonoverlapping_date_mask([d0, d1], 21)
        assert mask == [True, True]

    def test_one_session_inside_boundary_dropped(self):
        # 20 trading sessions apart, horizon=21 → second dropped.
        d0 = pd.Timestamp("2026-01-05")
        d1 = pd.Timestamp(add_trading_days(d0.date(), 20))
        mask = _nonoverlapping_date_mask([d0, d1], 21)
        assert mask == [True, False]

    def test_calendar_day_proxy_would_be_wrong(self):
        # Regression guard for the config#937 Step-0 bug: a calendar-day proxy
        # keeps any date >= h calendar days out; the trading-day-exact rule must
        # DROP a date that is h *calendar* days out but < h *trading* sessions
        # out. Fri 2026-01-09 + 5 calendar days = Wed 2026-01-14, which is only
        # 3 trading sessions later (Mon/Tue/Wed) — must be dropped at horizon=5.
        d0 = pd.Timestamp("2026-01-09")  # Friday, trading day
        d1 = d0 + pd.Timedelta(days=5)   # Wed 2026-01-14
        assert count_trading_days(d0.date(), d1.date()) == 3  # < 5 sessions
        mask = _nonoverlapping_date_mask([d0, d1], 5)
        assert mask == [True, False]

    def test_non_sorted_input_yields_sorted_greedy_selection(self):
        # Input out of order — helper sorts internally, returns mask aligned to
        # original positions. Build three dates each exactly 21 trading
        # sessions apart, then present them shuffled → all kept.
        a = pd.Timestamp("2026-01-05")
        b = pd.Timestamp(add_trading_days(a.date(), 21))
        c = pd.Timestamp(add_trading_days(b.date(), 21))
        dates = [c, a, b]  # shuffled
        mask = _nonoverlapping_date_mask(dates, 21)
        assert mask == [True, True, True]

    def test_horizon_exceeds_span_keeps_only_first(self):
        # 3 dates within ~40 trading sessions; horizon=90 → only the first.
        dates = [_ts("2026-01-05"), _ts("2026-02-02"), _ts("2026-03-02")]
        mask = _nonoverlapping_date_mask(dates, 90)
        kept = [i for i, m in enumerate(mask) if m]
        assert kept == [0]

    def test_long_horizon_consecutive_sessions_exact(self):
        # 500 consecutive trading sessions, horizon=90 → kept exactly at
        # multiples of 90 (0, 90, 180, 270, 360, 450).
        dates = _ladder("2024-01-02", 500)
        mask = _nonoverlapping_date_mask(dates, 90)
        assert [i for i, m in enumerate(mask) if m] == [0, 90, 180, 270, 360, 450]

    def test_short_horizon_consecutive_sessions_exact(self):
        # 500 consecutive trading sessions, horizon=5 → 100 kept (every 5th).
        dates = _ladder("2024-01-02", 500)
        mask = _nonoverlapping_date_mask(dates, 5)
        assert sum(mask) == 100
        assert [i for i, m in enumerate(mask) if m] == list(range(0, 500, 5))

    def test_accepts_string_dates(self):
        # Helper coerces via pd.Timestamp; string input should work. Three
        # dates ~21 trading sessions apart → all kept.
        d0 = "2026-01-05"
        d1 = str(add_trading_days(pd.Timestamp(d0).date(), 21))
        d2 = str(add_trading_days(pd.Timestamp(d1).date(), 21))
        mask = _nonoverlapping_date_mask([d0, d1, d2], 21)
        assert mask == [True, True, True]

    def test_duplicate_dates_keep_one(self):
        # Two rows with the same date — 0 sessions apart (< horizon) → drop one.
        d2 = pd.Timestamp(add_trading_days(pd.Timestamp("2026-01-05").date(), 21))
        dates = [_ts("2026-01-05"), _ts("2026-01-05"), d2]
        mask = _nonoverlapping_date_mask(dates, 21)
        kept = [i for i, m in enumerate(mask) if m]
        assert len(kept) == 2
        assert 0 in kept and 2 in kept
