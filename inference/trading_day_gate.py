"""Trading-day gate for the weekday Step Function.

Exposed as ``action=check_trading_day`` on the predictor Lambda handler so the
weekday pipeline can decide whether to start the executor BEFORE booting the
trading box — replacing the prior on-box SSM ``trading_calendar`` check, whose
stdout was unreliably captured on a just-cold-booted instance (2026-06-30 false
TradingDayCheckFailed alert; config#1430).

Pure NYSE-calendar math via ``nousergon_lib.trading_calendar`` — no S3, no
ArcticDB, no models, no GitHub. The gate must never depend on fragile infra, so
that the gate itself can't be the thing that breaks.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

_NYSE = ZoneInfo("America/New_York")


def check_weekly_run_day(date_str: str | None = None) -> dict:
    """Return whether ``date_str`` (or today in NYSE time) is the weekly-SF
    run day — exactly one calendar day after the LAST trading session of
    its Mon–Fri week (Brian-ratified policy, config#1824, 2026-07-06).

    Normal week → Saturday. Friday NYSE holiday → Friday (e.g. 2026-07-03
    for the July-4th week: last session Thu 7/2). Thursday AND Friday
    holidays → Thursday. Exposed as ``action=check_weekly_run_day`` so the
    weekly pipeline's THU–SAT cron can self-select the single correct
    firing. Same zero-infra posture as ``check_trading_day``: pure
    calendar math, evaluated BEFORE preflight.
    """
    from nousergon_lib.trading_calendar import is_trading_day

    check_date = date.fromisoformat(date_str) if date_str else datetime.now(_NYSE).date()
    prev = check_date - timedelta(days=1)
    result = {
        "check_date": check_date.isoformat(),
        "day_name": check_date.strftime("%A"),
    }

    if not is_trading_day(prev):
        run = False
        reason = f"previous day {prev.isoformat()} is not a trading session"
    else:
        # prev is a session (so prev.weekday() in 0..4). Run day iff no
        # session exists AFTER prev within prev's Mon-Fri week.
        later = [
            prev + timedelta(days=offset)
            for offset in range(1, 5 - prev.weekday())
            if is_trading_day(prev + timedelta(days=offset))
        ]
        if later:
            run = False
            reason = (
                f"{prev.isoformat()} is not the week's last session "
                f"(later session {later[0].isoformat()})"
            )
        else:
            run = True
            reason = None

    result["is_weekly_run_day"] = run
    result["marker"] = "WEEKLY_RUN_DAY" if run else "NOT_WEEKLY_RUN_DAY"
    if run:
        result["last_week_session"] = prev.isoformat()
    else:
        result["reason"] = reason
    return result


def check_trading_day(date_str: str | None = None) -> dict:
    """Return whether ``date_str`` (or today in NYSE time) is an NYSE trading day."""
    from nousergon_lib.trading_calendar import is_trading_day, next_trading_day

    check_date = date.fromisoformat(date_str) if date_str else datetime.now(_NYSE).date()
    trading = bool(is_trading_day(check_date))
    result = {
        "is_trading_day": trading,
        "check_date": check_date.isoformat(),
        "day_name": check_date.strftime("%A"),
        "marker": "TRADING DAY" if trading else "MARKET_CLOSED",
    }
    if not trading:
        result["reason"] = "weekend" if check_date.weekday() > 4 else "NYSE holiday"
        result["next_trading_day"] = next_trading_day(check_date).isoformat()
    return result
