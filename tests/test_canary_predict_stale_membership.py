"""The deploy canary's predict(dry_run) must not be gated on upstream freshness.

2026-11-03: the v1 weekly trigger was disabled 2026-10-28, so
``universe_membership/latest.json`` stayed at run_date 2026-10-23. The 05:02Z
deploy of a dependency-only change (krepis 0.59.90) published :655, and the
canary's ``predict(dry_run)`` died at load_universe with
``UniverseResolutionError`` (11 days old, limit 10) — ``FunctionError:
Unhandled`` — so deploy.sh refused to promote a healthy image. Every later
deploy would have been refused the same way, including one fixing the
check_market_hours / check_trading_day gates the trading pipelines start with.

The handler answers that case with the deploy.sh contract's reserved
``statusCode: "skipped"`` — and ONLY that case: a canary-stamped event and a
membership artifact that was READ and rejected for age. Everything else raises
exactly as before.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_load_universe import _membership, _mock_s3_with_keys  # noqa: E402

from inference.stages.load_universe import (  # noqa: E402
    UniverseMembershipStaleError,
    UniverseResolutionError,
    load_watchlist,
)

CANARY_EVENT = {"dry_run": True, "invocation_kind": "canary"}
STALE_KEYS = {
    "universe_membership/latest.json": _membership(["AAPL"], run_date="2026-10-23"),
}


class _NoPreflight:
    def __init__(self, *a, **k):
        pass

    def run(self, *a, **k):
        return None

    def run_for_drift_gate(self, *a, **k):
        return None


@pytest.fixture()
def handler_mod(monkeypatch):
    import inference.preflight as preflight_mod
    from inference import handler as mod

    monkeypatch.setattr(preflight_mod, "PredictorPreflight", _NoPreflight)
    return mod


def _main_that_resolves_the_universe(s3_keys, date_str="2026-11-03"):
    """Stand-in for daily_predict.main that runs the REAL universe resolution
    against a mocked bucket — the exact call the 11-03 canary died in."""

    def _main(**kwargs):
        with patch("boto3.client", return_value=_mock_s3_with_keys(s3_keys)):
            load_watchlist("auto", s3_bucket="alpha-engine-research", date_str=date_str)

    return _main


class TestCanaryPredictOnStaleMembership:
    def test_reproduces_the_11_03_failure_without_the_canary_stamp(self, handler_mod):
        # A hand-invoked dry run (no invocation_kind) still fails loud.
        with patch("inference.daily_predict.main",
                   _main_that_resolves_the_universe(STALE_KEYS)):
            with pytest.raises(UniverseMembershipStaleError, match="11 days old"):
                handler_mod.handler({"dry_run": True}, None)

    def test_canary_returns_the_reserved_noop_status(self, handler_mod):
        with patch("inference.daily_predict.main",
                   _main_that_resolves_the_universe(STALE_KEYS)):
            out = handler_mod.handler(dict(CANARY_EVENT), None)
        assert out["statusCode"] == "skipped"
        assert out["reason"] == "upstream_universe_membership_stale"
        assert "2026-10-23" in out["body"] and "11 days old" in out["body"]

    def test_production_predict_still_raises(self, handler_mod):
        # The SF's PredictorInference Task Catch depends on a raise; a returned
        # dict would let the executor proceed on stale predictions.
        with patch("inference.daily_predict.main",
                   _main_that_resolves_the_universe(STALE_KEYS)):
            with pytest.raises(UniverseMembershipStaleError):
                handler_mod.handler({"action": "predict"}, None)

    def test_canary_with_absent_membership_still_raises(self, handler_mod):
        # Nothing read can be this image's own defect — never skipped.
        with patch("inference.daily_predict.main",
                   _main_that_resolves_the_universe({})):
            with pytest.raises(UniverseResolutionError) as ei:
                handler_mod.handler(dict(CANARY_EVENT), None)
        assert not isinstance(ei.value, UniverseMembershipStaleError)

    def test_canary_with_any_other_failure_still_raises(self, handler_mod):
        def _boom(**kwargs):
            raise RuntimeError("scoring stage broke")

        with patch("inference.daily_predict.main", _boom):
            with pytest.raises(RuntimeError, match="scoring stage broke"):
                handler_mod.handler(dict(CANARY_EVENT), None)

    def test_skipped_is_in_the_deploy_allowlist(self):
        # The status the handler emits must be one deploy.sh accepts, or this
        # path is a FunctionError-free red canary instead of a no-op.
        deploy_sh = (Path(__file__).resolve().parents[1]
                     / "infrastructure" / "deploy.sh").read_text()
        assert 'CANARY_NOOP_STATUSES=("200" "204" "skipped")' in deploy_sh
