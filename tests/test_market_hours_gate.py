"""alpha-engine-config-I7111 — ``action=check_market_hours``.

The boundary alpha-engine-config#2932 ruled: no trading pipeline starts inside
the NYSE regular session. #2932 put it on ``alpha-engine-sf-watch-executor-role``,
enforced by a Lambda flipping the role's inline policy on a 5-minute schedule.
That Lambda never existed in the account, so the boundary was never once in
force — and it could not have covered the operator reruns at 09:32 PT on
2026-08-07 or 08:32 PT on 2026-07-27, which were in-session starts by a
different principal.

Brian ruled option (c) on I7111: the boundary belongs to the pipeline. This
action is its evaluator. Both ``ne-preopen-trading-pipeline`` and
``ne-postclose-trading-pipeline`` gate their first state on the ``verdict``
these tests pin.

Two things are worth stating about what is tested where:

  * The session boundary is half-open ``[09:30, 16:00)`` ET. The exclusive
    close is not a style choice — the postclose pipeline is *triggered at*
    16:00:0x ET by the daemon's own shutdown, and a close-inclusive boundary
    would refuse the settlement run that carries NAV continuity
    (sf-pipeline-policy §1.3). ``TestPostcloseTriggerIsNotRefused`` uses the
    real observed execution start times.

  * The override is the ruling's second requirement: refusable deliberately,
    never bypassable accidentally. ``TestOverride`` is the proof — it must be
    complete, authored, unexpired, and bounded, and every rejection is a
    verdict the pipeline fails on rather than a field it ignores.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from inference.trading_day_gate import check_market_hours

# 2026-08-12 is a Wednesday and not an NYSE holiday. EDT == UTC-4, so
# 16:00Z == 12:00 ET (mid-session) and 20:00Z == 16:00 ET (the close).
_MIDSESSION = "2026-08-12T16:00:00Z"
_PREOPEN_CRON = "2026-08-12T12:15:41Z"   # 05:15 PT, the live preopen schedule
_CLOSE_INSTANT = "2026-08-12T20:00:00Z"


def _ok_override(now_iso: str = _MIDSESSION, hours: float = 2.0) -> dict:
    expires = datetime.fromisoformat(now_iso.replace("Z", "+00:00")) + timedelta(
        hours=hours
    )
    return {
        "market_hours_override": {
            "reason": "preopen failed at 05:15; positions unmanaged, re-running planner",
            "authorized_by": "brian",
            "expires_at": expires.isoformat(),
        }
    }


class TestSessionVerdict:
    def test_midsession_start_is_blocked(self):
        r = check_market_hours(_MIDSESSION)
        assert r["is_market_hours"] is True
        assert r["verdict"] == "BLOCKED"
        assert r["reason"] == "NYSE regular session is live"

    def test_scheduled_preopen_start_proceeds(self):
        # The live 05:15 PT cron. If this ever returns BLOCKED the gate has
        # broken the daily trading chain, not protected it.
        r = check_market_hours(_PREOPEN_CRON)
        assert r["is_market_hours"] is False
        assert r["verdict"] == "PROCEED"
        assert r["reason"] == "pre-open"

    def test_open_bell_is_inside_the_session(self):
        # 13:30Z == 09:30 ET in August.
        assert check_market_hours("2026-08-12T13:30:00Z")["verdict"] == "BLOCKED"
        assert check_market_hours("2026-08-12T13:29:59Z")["verdict"] == "PROCEED"

    def test_close_is_exclusive(self):
        assert check_market_hours("2026-08-12T19:59:59Z")["verdict"] == "BLOCKED"
        assert check_market_hours(_CLOSE_INSTANT)["verdict"] == "PROCEED"
        assert check_market_hours(_CLOSE_INSTANT)["reason"] == "post-close"

    def test_weekend_midsession_clock_proceeds(self):
        r = check_market_hours("2026-08-15T16:00:00Z")  # Saturday noon ET
        assert r["verdict"] == "PROCEED"
        assert r["reason"] == "weekend"

    def test_holiday_midsession_clock_proceeds(self):
        r = check_market_hours("2026-04-03T16:00:00Z")  # Good Friday
        assert r["verdict"] == "PROCEED"
        assert r["reason"] == "NYSE holiday"

    def test_dst_is_resolved_by_the_zone_not_a_fixed_offset(self):
        # January, EST (UTC-5): the bell is 14:30Z, not 13:30Z. A hardcoded
        # -4 offset would call 13:30Z in-session and refuse an hour of
        # legitimate pre-open window every winter.
        assert check_market_hours("2027-01-05T13:30:00Z")["verdict"] == "PROCEED"
        assert check_market_hours("2027-01-05T14:30:00Z")["verdict"] == "BLOCKED"
        assert check_market_hours("2027-01-05T21:00:00Z")["verdict"] == "PROCEED"

    def test_naive_now_is_read_as_eastern(self):
        assert check_market_hours("2026-08-12T12:00:00")["verdict"] == "BLOCKED"
        assert check_market_hours("2026-08-12T08:15:00")["verdict"] == "PROCEED"

    def test_verdict_is_always_one_of_the_four(self):
        for moment in (_MIDSESSION, _PREOPEN_CRON, _CLOSE_INSTANT):
            for payload in (None, {}, _ok_override(moment), {"market_hours_override": 7}):
                v = check_market_hours(moment, payload)["verdict"]
                assert v in {
                    "PROCEED",
                    "BLOCKED",
                    "PROCEED_OVERRIDE",
                    "OVERRIDE_MALFORMED",
                }


class TestPostcloseTriggerIsNotRefused:
    """The daemon starts ne-postclose-trading-pipeline the instant its own
    ``is_market_hours()`` goes False. Measured starts, 2026-07-08..2026-08-12,
    all land in the first minute after the close. Every one must PROCEED."""

    @pytest.mark.parametrize("second", [4, 15, 23, 36, 39, 47, 49, 52, 56])
    def test_every_observed_eod_start_proceeds(self, second):
        r = check_market_hours(f"2026-08-12T20:00:{second:02d}Z")
        assert r["verdict"] == "PROCEED", r
        assert r["reason"] == "post-close"

    def test_a_second_before_the_close_would_be_refused(self):
        # Stated so the margin is explicit rather than assumed: the daemon's
        # guard is what keeps the trigger on the safe side of this edge.
        assert check_market_hours("2026-08-12T19:59:59Z")["verdict"] == "BLOCKED"


class TestOverride:
    def test_valid_override_admits_an_in_session_start(self):
        r = check_market_hours(_MIDSESSION, _ok_override())
        assert r["verdict"] == "PROCEED_OVERRIDE"
        assert r["is_market_hours"] is True
        assert r["override"]["valid"] is True
        assert r["override"]["authorized_by"] == "brian"
        assert "positions unmanaged" in r["override"]["reason"]

    def test_override_is_recorded_even_when_it_was_not_needed(self):
        r = check_market_hours(_PREOPEN_CRON, _ok_override(_PREOPEN_CRON))
        # Market closed — the boundary was never engaged, so this is a plain
        # PROCEED. The override still appears in the record so the execution
        # history shows what was offered.
        assert r["verdict"] == "PROCEED"
        assert r["override"]["present"] is True
        assert r["override"]["valid"] is True

    def test_absent_override_yields_a_uniform_empty_record(self):
        for payload in (None, {}, {"pipeline_role": "daily"}):
            rec = check_market_hours(_MIDSESSION, payload)["override"]
            assert rec["present"] is False
            assert rec["valid"] is False
            assert rec["rejection"] is None

    @pytest.mark.parametrize("field", ["reason", "authorized_by", "expires_at"])
    def test_every_field_is_required(self, field):
        payload = _ok_override()
        del payload["market_hours_override"][field]
        r = check_market_hours(_MIDSESSION, payload)
        assert r["verdict"] == "OVERRIDE_MALFORMED"
        assert field in r["override"]["rejection"]

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_a_blank_field_is_not_an_authorisation(self, blank):
        payload = _ok_override()
        payload["market_hours_override"]["reason"] = blank
        r = check_market_hours(_MIDSESSION, payload)
        assert r["verdict"] == "OVERRIDE_MALFORMED"

    @pytest.mark.parametrize("junk", [7, "yes", ["brian"], True])
    def test_a_non_object_override_is_rejected(self, junk):
        r = check_market_hours(_MIDSESSION, {"market_hours_override": junk})
        assert r["verdict"] == "OVERRIDE_MALFORMED"
        assert "must be an object" in r["override"]["rejection"]

    def test_an_expired_override_is_rejected(self):
        payload = _ok_override(hours=-1)
        r = check_market_hours(_MIDSESSION, payload)
        assert r["verdict"] == "OVERRIDE_MALFORMED"
        assert "not after the execution start" in r["override"]["rejection"]

    def test_an_override_cannot_be_left_standing(self):
        # A year-long expiry is the failure mode this ceiling exists for: an
        # override pasted into a runbook once and then silently valid forever.
        payload = _ok_override(hours=24 * 365)
        r = check_market_hours(_MIDSESSION, payload)
        assert r["verdict"] == "OVERRIDE_MALFORMED"
        assert "ceiling" in r["override"]["rejection"]

    def test_the_horizon_ceiling_is_24h(self):
        assert check_market_hours(_MIDSESSION, _ok_override(hours=23.9))[
            "verdict"
        ] == "PROCEED_OVERRIDE"
        assert check_market_hours(_MIDSESSION, _ok_override(hours=24.1))[
            "verdict"
        ] == "OVERRIDE_MALFORMED"

    def test_an_unparseable_expiry_is_rejected(self):
        payload = _ok_override()
        payload["market_hours_override"]["expires_at"] = "tomorrow"
        r = check_market_hours(_MIDSESSION, payload)
        assert r["verdict"] == "OVERRIDE_MALFORMED"
        assert "ISO-8601" in r["override"]["rejection"]

    def test_a_malformed_override_fails_closed_even_out_of_session(self):
        # Deliberate: a typo'd override is a mis-authorisation. Discovering it
        # on the quiet run is the only way it is not discovered on the run
        # that matters.
        payload = _ok_override(_PREOPEN_CRON, hours=-1)
        assert check_market_hours(_PREOPEN_CRON, payload)["verdict"] == (
            "OVERRIDE_MALFORMED"
        )

    def test_expiry_is_judged_against_the_execution_start_not_the_clock(self):
        # The override's window is a property of the execution, so a replay
        # of a past execution reaches the same verdict.
        payload = {
            "market_hours_override": {
                "reason": "recovery",
                "authorized_by": "brian",
                "expires_at": "2026-08-12T18:00:00+00:00",
            }
        }
        assert check_market_hours("2026-08-12T16:00:00Z", payload)["verdict"] == (
            "PROCEED_OVERRIDE"
        )
        assert check_market_hours("2026-08-12T19:00:00Z", payload)["verdict"] == (
            "OVERRIDE_MALFORMED"
        )


class TestHandlerDispatch:
    def test_gate_returns_before_preflight_is_constructed(self, monkeypatch):
        """The gate decides whether a trading pipeline may start at all, so it
        must not be capable of failing for any reason but its own calendar
        math. A preflight dependency would let an ArcticDB or S3 problem take
        out the boundary and the pipeline together."""

        class _ExplodingPreflight:
            def __init__(self, *a, **k):
                raise AssertionError(
                    "PredictorPreflight must NOT be constructed for check_market_hours"
                )

        monkeypatch.setitem(
            sys.modules,
            "inference.preflight",
            MagicMock(PredictorPreflight=_ExplodingPreflight),
        )

        import inference.handler as h

        result = h.handler(
            {"action": "check_market_hours", "now": _MIDSESSION}, None
        )
        assert result["verdict"] == "BLOCKED"
        assert result["is_market_hours"] is True

    def test_handler_threads_the_execution_input_through(self, monkeypatch):
        class _ExplodingPreflight:
            def __init__(self, *a, **k):
                raise AssertionError("preflight must not run")

        monkeypatch.setitem(
            sys.modules,
            "inference.preflight",
            MagicMock(PredictorPreflight=_ExplodingPreflight),
        )

        import inference.handler as h

        result = h.handler(
            {
                "action": "check_market_hours",
                "now": _MIDSESSION,
                "execution_input": _ok_override(),
            },
            None,
        )
        assert result["verdict"] == "PROCEED_OVERRIDE"

    def test_handler_without_now_uses_the_current_clock(self, monkeypatch):
        class _ExplodingPreflight:
            def __init__(self, *a, **k):
                raise AssertionError("preflight must not run")

        monkeypatch.setitem(
            sys.modules,
            "inference.preflight",
            MagicMock(PredictorPreflight=_ExplodingPreflight),
        )

        import inference.handler as h

        result = h.handler({"action": "check_market_hours"}, None)
        assert isinstance(result["is_market_hours"], bool)
        assert result["verdict"] in {"PROCEED", "BLOCKED"}


class TestResultContract:
    """The SF Choice keys on ``verdict``; the SNS notify states format the
    rest. A renamed key here silently changes what the pipeline decides, so
    the shape is pinned rather than assumed."""

    def test_result_carries_every_key_the_pipeline_reads(self):
        r = check_market_hours(_MIDSESSION, _ok_override())
        assert {
            "is_market_hours",
            "verdict",
            "now_et",
            "check_date",
            "day_name",
            "session_window_et",
            "reason",
            "override",
            "marker",
        }.issubset(r)
        assert r["session_window_et"] == "09:30-16:00"
        assert r["marker"] in ("MARKET_OPEN", "MARKET_CLOSED")
        assert set(r["override"]) == {
            "present",
            "valid",
            "reason",
            "authorized_by",
            "expires_at",
            "rejection",
        }

    def test_now_et_is_reported_in_eastern_not_the_input_zone(self):
        r = check_market_hours(_MIDSESSION)
        assert r["now_et"].startswith("2026-08-12T12:00:00")
        assert datetime.fromisoformat(r["now_et"]).utcoffset() == timedelta(hours=-4)

    def test_is_market_hours_and_marker_never_disagree(self):
        for moment in (_MIDSESSION, _PREOPEN_CRON, _CLOSE_INSTANT):
            r = check_market_hours(moment)
            assert r["marker"] == ("MARKET_OPEN" if r["is_market_hours"] else "MARKET_CLOSED")

    def test_utc_and_offset_forms_of_the_same_instant_agree(self):
        a = check_market_hours("2026-08-12T16:00:00Z")
        b = check_market_hours("2026-08-12T12:00:00-04:00")
        c = check_market_hours(
            datetime(2026, 8, 12, 16, 0, tzinfo=timezone.utc).isoformat()
        )
        assert a["verdict"] == b["verdict"] == c["verdict"] == "BLOCKED"


# ── alpha-engine-config-I11384 — PROCEED_REMEDIATION ────────────────────────
#
# The gate refused hardest in the one case Brian's standing rule says must
# always proceed: a same-day rerun after the morning produced no tradeable
# book. On 2026-09-22 the optimizer raised, the executor held the whole book,
# the fix was merged and verified by 11:51 ET, and the rerun was refused at
# 11:45 ET by a boundary whose stated purpose is to stop a run acting on a
# STALE plan — while the thing being stopped was installing a CORRECTED one.

import json as _json
from datetime import date
from unittest.mock import MagicMock

import pytest

from inference.trading_day_gate import _todays_book_is_unfilled, check_market_hours

_IN_SESSION = "2026-09-22T15:45:00Z"  # 11:45 ET, a Tuesday
_PREOPEN_CRON = "2026-09-22T12:15:00Z"  # 05:15 PT, the scheduled start


def _s3(monkeypatch, *, book=None, shadow=None, book_err=None, shadow_err=None):
    client = MagicMock()

    def _get(Bucket, Key):  # noqa: N803 — boto3 kwarg names
        if "order_book" in Key:
            if book_err:
                raise book_err
            return {"Body": MagicMock(read=lambda: _json.dumps(book).encode())}
        if book_err is None and shadow_err:
            raise shadow_err
        return {"Body": MagicMock(read=lambda: _json.dumps(shadow).encode())}

    client.get_object.side_effect = _get
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: client)
    return client


_TRADED = {"approved_entries": [{"ticker": "AAPL"}], "urgent_exits": []}
_OK_SHADOW = {"shadow_status": "ok"}


class TestRemediationProbe:
    """Evidence, not a caller's claim — and every failure returns False."""

    def test_an_absent_order_book_is_an_unfilled_day(self, monkeypatch):
        _s3(monkeypatch, book_err=RuntimeError("404"))
        out = _todays_book_is_unfilled(date(2026, 9, 22))
        assert out["unfilled"] is True
        assert "absent" in out["evidence"]

    def test_a_book_with_no_entries_and_no_exits_is_unfilled(self, monkeypatch):
        _s3(monkeypatch, book={"approved_entries": [], "urgent_exits": []})
        out = _todays_book_is_unfilled(date(2026, 9, 22))
        assert out["unfilled"] is True
        assert "0 approved_entries" in out["evidence"]

    def test_a_failed_shadow_log_is_unfilled_even_behind_a_populated_book(
        self, monkeypatch
    ):
        # The 2026-09-22 shape after a partial rerun: orders exist but the
        # optimizer that should have sized them did not solve.
        _s3(monkeypatch, book=_TRADED, shadow={"shadow_status": "failed"})
        out = _todays_book_is_unfilled(date(2026, 9, 22))
        assert out["unfilled"] is True
        assert "shadow_status='failed'" in out["evidence"]

    def test_a_traded_day_is_not_unfilled(self, monkeypatch):
        _s3(monkeypatch, book=_TRADED, shadow=_OK_SHADOW)
        out = _todays_book_is_unfilled(date(2026, 9, 22))
        assert out["unfilled"] is False
        assert out["error"] is None

    @pytest.mark.parametrize(
        "boom",
        [ImportError("no boto3"), RuntimeError("no credentials")],
    )
    def test_every_probe_failure_returns_false_not_true(self, monkeypatch, boom):
        """The direction a boundary is allowed to fail in.

        The probe can only GRANT a start the calendar would refuse, so a
        failure that returned True would loosen the boundary via an S3
        outage — exactly what the gate's zero-infra design exists to prevent.
        """
        import boto3

        monkeypatch.setattr(
            boto3, "client", MagicMock(side_effect=boom)
        )
        out = _todays_book_is_unfilled(date(2026, 9, 22))
        assert out["unfilled"] is False
        assert out["error"] is not None, "and it says why, rather than going quiet"

    def test_malformed_json_returns_false(self, monkeypatch):
        client = MagicMock()
        client.get_object.return_value = {"Body": MagicMock(read=lambda: b"{{{")}
        import boto3

        monkeypatch.setattr(boto3, "client", lambda *a, **k: client)
        assert _todays_book_is_unfilled(date(2026, 9, 22))["unfilled"] is False


class TestRemediationVerdict:
    def test_an_unfilled_day_proceeds_in_session_with_no_override(
        self, monkeypatch
    ):
        _s3(monkeypatch, book={"approved_entries": [], "urgent_exits": []})
        out = check_market_hours(_IN_SESSION, {}, pipeline="preopen")
        assert out["is_market_hours"] is True
        assert out["verdict"] == "PROCEED_REMEDIATION"
        assert out["remediation"]["probed"] is True
        assert out["remediation"]["evidence"]

    def test_a_traded_day_is_still_BLOCKED(self, monkeypatch):
        _s3(monkeypatch, book=_TRADED, shadow=_OK_SHADOW)
        out = check_market_hours(_IN_SESSION, {}, pipeline="preopen")
        assert out["verdict"] == "BLOCKED", (
            "the boundary still refuses an opportunistic in-session start — "
            "that is the case I7111 exists for and it is unchanged"
        )

    def test_the_postclose_chain_is_never_eligible(self, monkeypatch):
        _s3(monkeypatch, book={"approved_entries": [], "urgent_exits": []})
        out = check_market_hours(_IN_SESSION, {}, pipeline="postclose")
        assert out["verdict"] == "BLOCKED"
        assert out["remediation"]["probed"] is False

    def test_an_unknown_pipeline_is_never_eligible(self, monkeypatch):
        _s3(monkeypatch, book={"approved_entries": [], "urgent_exits": []})
        assert check_market_hours(_IN_SESSION, {})["verdict"] == "BLOCKED"

    def test_the_scheduled_start_never_probes_s3(self, monkeypatch):
        """The 05:15 PT path must keep its zero-infra posture.

        A gate that reached S3 on every scheduled run would have taken on the
        dependency its whole design exists to avoid.
        """
        client = _s3(monkeypatch, book=_TRADED, shadow=_OK_SHADOW)
        out = check_market_hours(_PREOPEN_CRON, {}, pipeline="preopen")
        assert out["verdict"] == "PROCEED"
        assert out["remediation"]["probed"] is False
        client.get_object.assert_not_called()

    def test_a_valid_override_still_wins_over_remediation(self, monkeypatch):
        _s3(monkeypatch, book={"approved_entries": [], "urgent_exits": []})
        out = check_market_hours(
            _IN_SESSION,
            {"market_hours_override": {
                "reason": "manual",
                "authorized_by": "Brian McMahon",
                "expires_at": "2026-09-22T20:00:00Z",
            }},
            pipeline="preopen",
        )
        assert out["verdict"] == "PROCEED_OVERRIDE", (
            "an explicit human authorisation must never be silently "
            "re-labelled as an automatic one in the record"
        )
        assert out["remediation"]["probed"] is False

    def test_a_malformed_override_still_fails_closed_on_an_unfilled_day(
        self, monkeypatch
    ):
        _s3(monkeypatch, book={"approved_entries": [], "urgent_exits": []})
        out = check_market_hours(
            _IN_SESSION,
            {"market_hours_override": {"reason": "", "authorized_by": ""}},
            pipeline="preopen",
        )
        assert out["verdict"] == "OVERRIDE_MALFORMED", (
            "a half-typed authorisation is a mis-authorisation, and finding "
            "it on the run where it did not matter is the point"
        )
        assert out["remediation"]["probed"] is False
