"""Trading-day gate for the weekday Step Function.

Exposed as ``action=check_trading_day`` on the predictor Lambda handler so the
weekday pipeline can decide whether to start the executor BEFORE booting the
trading box — replacing the prior on-box SSM ``trading_calendar`` check, whose
stdout was unreliably captured on a just-cold-booted instance (2026-06-30 false
TradingDayCheckFailed alert; config#1430).

Also exposes ``action=check_market_hours`` (alpha-engine-config-I7111) — the
time-of-day half of the same question, gating the FIRST state of both trading
pipelines.

Pure NYSE-calendar math via ``nousergon_lib.trading_calendar`` — no S3, no
ArcticDB, no models, no GitHub. The gate must never depend on fragile infra, so
that the gate itself can't be the thing that breaks.

One deliberate exception, alpha-engine-config-I11384: the REMEDIATION probe
(``_todays_book_is_unfilled``) reads two S3 keys. It is admissible only
because it can move the verdict in exactly one direction — it can GRANT a
start the calendar would otherwise refuse, never refuse one the calendar
would allow — and because every one of its own failure modes returns
``False``. An S3 outage therefore makes this gate STRICTER, which is the
direction a boundary is allowed to fail in.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_NYSE = ZoneInfo("America/New_York")

# NYSE regular session, half-open ``[09:30, 16:00)`` ET. The close is
# EXCLUSIVE and that is load-bearing, not a detail: the daemon starts
# ``ne-postclose-trading-pipeline`` the moment its own ``is_market_hours()``
# goes False, so every observed ``eod-*`` execution begins at 16:00:0x ET.
# A close-inclusive boundary would have this gate refuse the settlement run
# that carries NAV continuity (sf-pipeline-policy §1.3).
_SESSION_OPEN_ET = time(9, 30)
_SESSION_CLOSE_ET = time(16, 0)


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


# An override is a deliberate, time-boxed authorisation to start a trading
# pipeline INSIDE the session. Bounded so a stale runbook line stops working
# rather than quietly staying valid: the operator has to re-state the intent
# for the run they are actually making.
_OVERRIDE_MAX_HORIZON_HOURS = 24
_OVERRIDE_KEY = "market_hours_override"
_OVERRIDE_REQUIRED_FIELDS = ("reason", "authorized_by", "expires_at")


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 instant, tolerating the trailing ``Z`` that Step
    Functions' ``$$.Execution.StartTime`` carries. A naive value is read as
    NYSE-local, matching every other date input on this Lambda."""
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return parsed.replace(tzinfo=_NYSE) if parsed.tzinfo is None else parsed


def _evaluate_override(execution_input: dict | None, now: datetime) -> dict:
    """Validate the market-hours override carried in the raw execution input.

    Returns a record that is always populated — ``present`` false when no
    override was offered — so the caller's audit trail is uniform.

    Fails CLOSED on anything malformed, and does so whether or not the market
    is actually open. A typo'd override is a mis-authorisation; surfacing it on
    the quiet run is the only way it is not discovered on the run that matters.
    """
    record: dict = {
        "present": False,
        "valid": False,
        "reason": None,
        "authorized_by": None,
        "expires_at": None,
        "rejection": None,
    }
    raw = (execution_input or {}).get(_OVERRIDE_KEY)
    if raw is None:
        return record

    record["present"] = True
    if not isinstance(raw, dict):
        record["rejection"] = (
            f"{_OVERRIDE_KEY} must be an object with "
            f"{', '.join(_OVERRIDE_REQUIRED_FIELDS)}; got {type(raw).__name__}"
        )
        return record

    missing = [
        f
        for f in _OVERRIDE_REQUIRED_FIELDS
        if not isinstance(raw.get(f), str) or not raw.get(f, "").strip()
    ]
    if missing:
        record["rejection"] = (
            f"{_OVERRIDE_KEY} is missing or blank: {', '.join(missing)}"
        )
        return record

    record["reason"] = raw["reason"].strip()
    record["authorized_by"] = raw["authorized_by"].strip()
    record["expires_at"] = raw["expires_at"].strip()

    try:
        expires = _parse_iso(record["expires_at"])
    except ValueError as exc:
        record["rejection"] = f"expires_at is not an ISO-8601 instant: {exc}"
        return record

    if expires <= now:
        record["rejection"] = (
            f"expires_at {expires.isoformat()} is not after the execution start "
            f"{now.isoformat()} — an override may not be back-dated or reused"
        )
        return record

    horizon = (expires - now).total_seconds() / 3600.0
    if horizon > _OVERRIDE_MAX_HORIZON_HOURS:
        record["rejection"] = (
            f"expires_at is {horizon:.1f}h after the execution start, beyond the "
            f"{_OVERRIDE_MAX_HORIZON_HOURS}h ceiling — an override is scoped to "
            f"the run being made, not left standing"
        )
        return record

    record["valid"] = True
    return record


def _todays_book_is_unfilled(check_date: date) -> dict:
    """Did today's preopen produce a tradeable book? Evidence, not assertion.

    alpha-engine-config-I11384. Brian's standing rule (2026-09-22): *"when the
    portfolio optimizer doesn't load correctly during trading hours, we need
    to FIX IT ASAP AND GET IT RUNNING SAME DAY ... We never wait for a close
    before implementing fixes that affect intraday trading."* Generalises his
    2026-08-13 ruling that a failed preopen is ALWAYS relaunched while the
    session is open.

    The gate cannot take that claim from the CALLER — a self-declared
    "this is a remediation" flag is an override with extra steps, and the
    whole point of I7111 is that the boundary binds every principal
    including the operator. So it is derived from what the morning actually
    left in S3:

      * the day's order book is absent, OR
      * it carries zero approved entries AND zero urgent exits, OR
      * the day's optimizer shadow log is absent or not ``shadow_status: ok``

    Any of those means the session is currently running on a book that did
    not trade, which is the only state this exemption exists for.

    **Every failure mode returns False.** No credentials, no bucket, S3
    error, malformed JSON, unexpected shape — all of it means "no evidence of
    a remediation", so the caller falls through to the normal BLOCKED path.
    That is what makes an S3 read admissible inside a gate whose whole design
    premise is that it cannot be taken out by infrastructure: this probe can
    only ever make the boundary stricter.
    """
    out = {"probed": True, "unfilled": False, "evidence": None, "error": None}
    day = check_date.isoformat()
    try:
        import boto3

        bucket = os.environ.get("S3_BUCKET", "alpha-engine-research")
        s3 = boto3.client("s3")

        try:
            body = s3.get_object(
                Bucket=bucket, Key=f"trades/order_book/{day}.json"
            )["Body"].read()
        except Exception:
            out["unfilled"] = True
            out["evidence"] = f"trades/order_book/{day}.json absent or unreadable"
            return out

        import json as _json

        book = _json.loads(body)
        n_entries = len(book.get("approved_entries") or [])
        n_exits = len(book.get("urgent_exits") or [])
        if n_entries == 0 and n_exits == 0:
            out["unfilled"] = True
            out["evidence"] = (
                f"trades/order_book/{day}.json carries 0 approved_entries and "
                f"0 urgent_exits"
            )
            return out

        try:
            shadow = _json.loads(
                s3.get_object(
                    Bucket=bucket,
                    Key=f"predictor/optimizer_shadow/{day}.json",
                )["Body"].read()
            )
        except Exception:
            out["unfilled"] = True
            out["evidence"] = (
                f"predictor/optimizer_shadow/{day}.json absent or unreadable "
                f"while the book carries {n_entries} entries / {n_exits} exits"
            )
            return out

        if shadow.get("shadow_status") != "ok":
            out["unfilled"] = True
            out["evidence"] = (
                f"predictor/optimizer_shadow/{day}.json shadow_status="
                f"{shadow.get('shadow_status')!r}"
            )
            return out

        out["evidence"] = (
            f"today's book carries {n_entries} approved entries and {n_exits} "
            f"urgent exits, and the optimizer solved ok"
        )
        return out
    except Exception as exc:  # noqa: BLE001 — see docstring: fails to False
        logger.warning(
            "market-hours remediation probe failed (%s) — NOT granting a "
            "remediation start; falling through to the normal boundary", exc,
        )
        return {
            "probed": True,
            "unfilled": False,
            "evidence": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def check_market_hours(
    now: str | None = None,
    execution_input: dict | None = None,
    pipeline: str | None = None,
) -> dict:
    """Return whether a trading pipeline may start at ``now``.

    The boundary alpha-engine-config#2932 ruled and #7111 re-sited: no trading
    pipeline starts inside the NYSE regular session. It lives here, at the
    pipeline's own first state, rather than on one caller's IAM role — IAM has
    no time-of-day condition key, which is why the ruled-then-abandoned
    mechanism needed a Lambda to simulate one and why the simulation could
    silently not exist for three weeks. A gate state binds every principal:
    the scheduler, the Overseer's ``overseer-sf-watch`` lane, and the operator
    reruns at 09:32 PT on 2026-08-07 and 08:32 PT on 2026-07-27 that no IAM
    variant would ever have covered.

    Args:
        now: the moment to judge, ISO-8601. Both pipelines pass
            ``$$.Execution.StartTime``, so the verdict is a property of the
            execution rather than of the Lambda's own clock — deterministic,
            replayable, and immune to a slow cold start pushing a refused run
            past the close. Defaults to the current NYSE-local moment.
        execution_input: the raw ``$$.Execution.Input``, read for
            ``market_hours_override``. Passed whole rather than as a single
            field so an ABSENT override is representable — an ASL
            ``"override.$": "$.market_hours_override"`` parameter throws
            States.Runtime on every normal run.
        pipeline: ``"preopen"`` or ``"postclose"``, a LITERAL in each state
            machine's own ``MarketHoursGate`` payload — never read from
            ``execution_input``, so no caller can claim to be the preopen
            chain. Only ``"preopen"`` is eligible for the remediation
            verdict: the postclose chain stops the box and reconciles the
            book, and what "the book did not trade" should mean there is a
            different question that has not been answered
            (alpha-engine-config-I11384).

    Returns a ``verdict`` the pipeline's Choice keys on:

      ``PROCEED``             market closed (or override offered but not needed)
      ``BLOCKED``             market open, no override, book already traded
      ``PROCEED_OVERRIDE``    market open, valid override — recorded, not silent
      ``PROCEED_REMEDIATION`` market open, no override, and today's book is
                              demonstrably unfilled — the same-day repair
                              Brian's 2026-09-22 ruling says must always
                              proceed. Recorded like an override; granted on
                              S3 EVIDENCE, never on a caller's claim.
      ``OVERRIDE_MALFORMED``  an override was offered and is not usable

    Precedence is deliberate. A malformed override still fails closed even on
    a genuinely unfilled day: a half-typed authorisation is a
    mis-authorisation, and discovering it on the run where it did not matter
    is the whole reason that check exists. A VALID override still wins over
    remediation, so an explicit human authorisation is never silently
    re-labelled as an automatic one in the record.
    """
    from nousergon_lib.trading_calendar import is_trading_day

    moment = _parse_iso(now) if now else datetime.now(_NYSE)
    moment_et = moment.astimezone(_NYSE)
    check_date = moment_et.date()
    clock = moment_et.time()

    trading = bool(is_trading_day(check_date))
    within_window = _SESSION_OPEN_ET <= clock < _SESSION_CLOSE_ET
    open_now = trading and within_window

    override = _evaluate_override(execution_input, moment_et)

    # Probed ONLY when it can change the answer: the market is open, no valid
    # override is carrying the start, and this is the preopen chain. An S3
    # read on the scheduled 05:15 PT path would be a dependency the gate does
    # not need and must not have.
    remediation = {"probed": False, "unfilled": False, "evidence": None,
                   "error": None}
    if (
        open_now
        and pipeline == "preopen"
        and not (override["present"] and not override["valid"])
        and not override["valid"]
    ):
        remediation = _todays_book_is_unfilled(check_date)

    if override["present"] and not override["valid"]:
        verdict = "OVERRIDE_MALFORMED"
    elif not open_now:
        verdict = "PROCEED"
    elif override["valid"]:
        verdict = "PROCEED_OVERRIDE"
    elif remediation["unfilled"]:
        verdict = "PROCEED_REMEDIATION"
    else:
        verdict = "BLOCKED"

    if not trading:
        why = "weekend" if check_date.weekday() > 4 else "NYSE holiday"
    elif not within_window:
        why = (
            "pre-open"
            if clock < _SESSION_OPEN_ET
            else "post-close"
        )
    else:
        why = "NYSE regular session is live"

    return {
        "is_market_hours": open_now,
        "verdict": verdict,
        "now_et": moment_et.isoformat(),
        "check_date": check_date.isoformat(),
        "day_name": check_date.strftime("%A"),
        "session_window_et": (
            f"{_SESSION_OPEN_ET.strftime('%H:%M')}-"
            f"{_SESSION_CLOSE_ET.strftime('%H:%M')}"
        ),
        "reason": why,
        "override": override,
        "remediation": remediation,
        "pipeline": pipeline,
        "marker": "MARKET_OPEN" if open_now else "MARKET_CLOSED",
    }
