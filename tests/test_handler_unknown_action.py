"""An unrecognised `action` must fail loud, never fall through to predict.

alpha-engine-config-I7111. The 2026-08-13 preopen run invoked
`action=check_market_hours` — the first state of both trading pipelines —
against a `live` alias frozen on a version predating that action. The handler
dispatched none of its `action ==` branches and ran the default prediction
path, answering the gate with `{"statusCode": 200, "body": "Predictions
written for today"}`. The Choice state keyed on `$.market_hours_gate.Payload
.verdict` then died with States.Runtime, 48s in, no orders placed.

The fall-through was wrong twice over: it answered a gate question with an
unrelated side-effecting job, and it made a version skew — the one condition a
gate most needs to detect — present as a successful invoke.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HANDLER_PY = REPO_ROOT / "inference" / "handler.py"


@pytest.fixture()
def handler_mod():
    from inference import handler as mod

    return mod


class TestUnknownActionRaises:
    @pytest.mark.parametrize(
        "action",
        [
            "check_market_hours_v2",
            "check_markethours",
            "CHECK_MARKET_HOURS",
            "check_trading_days",
            "",
            "nonsense",
        ],
    )
    def test_unknown_action_raises_before_any_work(self, handler_mod, action):
        with pytest.raises(handler_mod.UnknownAction) as exc:
            handler_mod.handler({"action": action}, None)
        # The message must name the skew, because that is the condition it
        # exists to surface — an operator reading it should not have to guess.
        assert "live" in str(exc.value)
        assert "not implemented by this build" in str(exc.value)

    def test_the_message_lists_what_this_build_can_do(self, handler_mod):
        with pytest.raises(handler_mod.UnknownAction) as exc:
            handler_mod.handler({"action": "nope"}, None)
        assert "check_market_hours" in str(exc.value)

    def test_raises_rather_than_returning_an_error_dict(self, handler_mod):
        # A returned dict is indistinguishable from a successful invoke to an
        # SF Choice state; only a FunctionError reaches the Task's Catch.
        assert issubclass(handler_mod.UnknownAction, Exception)

    def test_it_rejects_before_preflight_or_any_s3_write(self, handler_mod, monkeypatch):
        # Nothing downstream may run. Poison the first thing the dispatch would
        # otherwise reach and assert we never get there.
        import inference.preflight as preflight_mod

        def _boom(*a, **k):  # pragma: no cover - must never be called
            raise AssertionError(
                "preflight ran for an unknown action — the guard is too late"
            )

        monkeypatch.setattr(preflight_mod, "PredictorPreflight", _boom)
        with pytest.raises(handler_mod.UnknownAction):
            handler_mod.handler({"action": "nonsense"}, None)


class TestKnownActionsStayInSyncWithTheDispatch:
    def test_every_dispatched_action_is_declared_known(self, handler_mod):
        """A branch added without a KNOWN_ACTIONS entry would be rejected.

        That failure mode is the mirror of the one this guard fixes: the gate
        would fail closed instead of falling through, which is safe, but it
        would also be an outage — so bind the two together here.
        """
        src = HANDLER_PY.read_text()
        dispatched = set(re.findall(r'action == "([a-z_]+)"', src))
        assert dispatched, "no `action == \"...\"` branches found — did the dispatch move?"
        undeclared = dispatched - handler_mod.KNOWN_ACTIONS
        assert not undeclared, (
            f"inference/handler.py dispatches {sorted(undeclared)} but "
            f"KNOWN_ACTIONS does not list them — the guard would reject an "
            f"action this build actually implements"
        )

    def test_no_declared_action_is_undispatchable(self, handler_mod):
        src = HANDLER_PY.read_text()
        dispatched = set(re.findall(r'action == "([a-z_]+)"', src))
        for group in re.findall(r"_action in \(([^)]*)\)", src):
            dispatched |= set(re.findall(r'"([a-z_]+)"', group))
        declared = {a for a in handler_mod.KNOWN_ACTIONS if a and a != "predict"}
        stale = declared - dispatched
        assert not stale, (
            f"KNOWN_ACTIONS declares {sorted(stale)} but nothing dispatches "
            f"them — a caller would get a prediction run instead of an error"
        )


class TestTheDefaultIsPreserved:
    def test_omitted_action_is_still_predict(self, handler_mod):
        # The historical default. It must NOT be caught by the guard — the
        # deploy canary and every ad-hoc invoke rely on it.
        assert None in handler_mod.KNOWN_ACTIONS
        assert "predict" in handler_mod.KNOWN_ACTIONS
