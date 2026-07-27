"""Tests for inference.trading_day_gate + its handler dispatch.

The gate is pure NYSE-calendar math (config#1430). Tests use KNOWN fixed
dates — never "today" — so they are deterministic. The handler-dispatch test
asserts the gate returns BEFORE PredictorPreflight is ever constructed, which
is the whole point of the fix: the gate cannot be broken by infra/env issues.
"""
import sys
from unittest.mock import MagicMock

import pytest

from inference.trading_day_gate import check_trading_day


# ── Direct gate tests (fixed dates) ─────────────────────────────────────────


def test_known_weekday_trading_day():
    """2026-06-30 is a Tuesday and an NYSE trading day."""
    r = check_trading_day("2026-06-30")
    assert r["is_trading_day"] is True
    assert r["check_date"] == "2026-06-30"
    assert r["day_name"] == "Tuesday"
    assert r["marker"] == "TRADING DAY"
    # A trading day carries no reason / next_trading_day fields.
    assert "reason" not in r
    assert "next_trading_day" not in r


def test_saturday_is_weekend():
    """2026-06-27 is a Saturday — closed, reason weekend, has next_trading_day."""
    r = check_trading_day("2026-06-27")
    assert r["is_trading_day"] is False
    assert r["day_name"] == "Saturday"
    assert r["marker"] == "MARKET_CLOSED"
    assert r["reason"] == "weekend"
    # Next trading day is a real future date, not a weekend.
    assert r["next_trading_day"] > "2026-06-27"


def test_new_years_day_is_holiday():
    """2026-01-01 is a Thursday New Year's Day — closed as an NYSE holiday."""
    r = check_trading_day("2026-01-01")
    assert r["is_trading_day"] is False
    assert r["day_name"] == "Thursday"
    assert r["marker"] == "MARKET_CLOSED"
    assert r["reason"] == "NYSE holiday"
    assert "next_trading_day" in r


def test_none_returns_expected_keys():
    """date=None resolves to today (NYSE) and returns the standard shape."""
    r = check_trading_day(None)
    assert set(["is_trading_day", "check_date", "day_name", "marker"]).issubset(r)
    assert isinstance(r["is_trading_day"], bool)
    assert r["marker"] in ("TRADING DAY", "MARKET_CLOSED")


# ── Handler dispatch test (gate returns BEFORE preflight) ────────────────────


def test_handler_dispatches_before_preflight(monkeypatch):
    """handler(action=check_trading_day) returns the gate result WITHOUT ever
    constructing PredictorPreflight — proving zero preflight/infra dependency."""

    class _ExplodingPreflight:
        def __init__(self, *a, **k):
            raise AssertionError("PredictorPreflight must NOT be constructed for check_trading_day")

    fake_module = MagicMock(PredictorPreflight=_ExplodingPreflight)
    monkeypatch.setitem(sys.modules, "inference.preflight", fake_module)

    import inference.handler as h
    result = h.handler({"action": "check_trading_day", "date": "2026-06-30"}, None)

    assert result["is_trading_day"] is True
    assert result["check_date"] == "2026-06-30"
    assert result["marker"] == "TRADING DAY"


# ── Weekly run-day gate (config#1824) ────────────────────────────────────────
#
# Policy (Brian, 2026-07-06): the weekly SF runs exactly one calendar day
# after the LAST trading session of its Mon–Fri week. Fixed known dates,
# including the REAL 2026 July-4th week (Fri 7/3 observed holiday: last
# session Thu 7/2 → run day Fri 7/3, and Sat 7/4 must NOT be a run day).

from inference.trading_day_gate import check_weekly_run_day


def test_normal_week_saturday_is_run_day():
    """2026-07-11 is a Saturday after a full trading week (Fri 7/10 session)."""
    r = check_weekly_run_day("2026-07-11")
    assert r["is_weekly_run_day"] is True
    assert r["marker"] == "WEEKLY_RUN_DAY"
    assert r["last_week_session"] == "2026-07-10"
    assert "reason" not in r


def test_normal_week_friday_is_not_run_day():
    """2026-07-10 is a regular Friday — Thursday was a session but Friday
    itself is a later session in the same week."""
    r = check_weekly_run_day("2026-07-10")
    assert r["is_weekly_run_day"] is False
    assert r["marker"] == "NOT_WEEKLY_RUN_DAY"
    assert "2026-07-10" in r["reason"]


def test_holiday_friday_is_run_day_real_july4_week():
    """2026-07-03 (July 4th observed, NYSE closed): last session of the week
    was Thu 7/2 → Friday IS the run day. This is the real case that was
    hand-shifted on 2026-07-03."""
    r = check_weekly_run_day("2026-07-03")
    assert r["is_weekly_run_day"] is True
    assert r["last_week_session"] == "2026-07-02"


def test_holiday_week_saturday_is_not_run_day():
    """2026-07-04 Saturday: previous day (Fri 7/3) was a holiday, not a
    session — the run day already happened on Friday."""
    r = check_weekly_run_day("2026-07-04")
    assert r["is_weekly_run_day"] is False
    assert "2026-07-03" in r["reason"]


def test_sunday_never_run_day():
    r = check_weekly_run_day("2026-07-12")
    assert r["is_weekly_run_day"] is False


def test_midweek_never_run_day():
    """2026-07-08 Wednesday: Tuesday was a session but Wed/Thu/Fri sessions
    follow — not the week's last."""
    r = check_weekly_run_day("2026-07-08")
    assert r["is_weekly_run_day"] is False


def test_double_holiday_thursday_is_run_day(monkeypatch):
    """Synthetic Thu+Fri double-holiday week (no real NYSE precedent):
    Wed is the last session → Thursday is the run day."""
    import inference.trading_day_gate as g

    def fake_is_trading_day(d):
        # Week of Mon 2026-08-10: Wed 8/12 last session, Thu 8/13 + Fri 8/14
        # synthetic holidays.
        return d.isoformat() in ("2026-08-10", "2026-08-11", "2026-08-12")

    import nousergon_lib.trading_calendar as tc
    monkeypatch.setattr(tc, "is_trading_day", fake_is_trading_day)
    r = check_weekly_run_day("2026-08-13")
    assert r["is_weekly_run_day"] is True
    assert r["last_week_session"] == "2026-08-12"
    # And Friday of that synthetic week is NOT a run day (Thu already was).
    r2 = check_weekly_run_day("2026-08-14")
    assert r2["is_weekly_run_day"] is False


def test_handler_dispatches_weekly_gate_before_preflight(monkeypatch):
    """Mirror of the check_trading_day dispatch test — zero preflight
    dependency for the weekly gate too."""

    class _ExplodingPreflight:
        def __init__(self, *a, **k):
            raise AssertionError("PredictorPreflight must NOT be constructed for check_weekly_run_day")

    fake_module = MagicMock(PredictorPreflight=_ExplodingPreflight)
    monkeypatch.setitem(sys.modules, "inference.preflight", fake_module)

    import inference.handler as h
    result = h.handler({"action": "check_weekly_run_day", "date": "2026-07-11"}, None)

    assert result["is_weekly_run_day"] is True
    assert result["marker"] == "WEEKLY_RUN_DAY"
