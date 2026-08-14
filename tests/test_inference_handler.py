"""Tests for inference.handler — Lambda action dispatch."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def stubbed_preflight(monkeypatch):
    """Replace PredictorPreflight with a no-op so tests don't hit S3."""
    pf = MagicMock()
    pf.return_value.run = MagicMock()
    pf.return_value.run_for_drift_gate = MagicMock()
    fake_module = MagicMock(PredictorPreflight=pf)
    monkeypatch.setitem(sys.modules, "inference.preflight", fake_module)
    return pf


def _fake_context(function_name="local", arn=""):
    ctx = MagicMock()
    ctx.function_name = function_name
    ctx.invoked_function_arn = arn
    return ctx


# ── action="train" (deprecated) ─────────────────────────────────────────────


def test_handler_train_returns_400_deprecated(stubbed_preflight):
    import inference.handler as h
    result = h.handler({"action": "train"}, _fake_context())
    assert result["statusCode"] == 400
    assert "Training has moved" in result["body"]


# ── action="check_coverage" ─────────────────────────────────────────────────


def test_handler_check_coverage_returns_delta(stubbed_preflight, monkeypatch):
    fake_delta = {
        "n_buy_candidates": 25,
        "n_predictions": 20,
        "missing_count": 5,
        "missing_tickers": ["A", "B", "C", "D", "E"],
    }
    fake_cov = MagicMock()
    fake_cov.compute_coverage_delta = MagicMock(return_value=fake_delta)
    monkeypatch.setitem(sys.modules, "inference.coverage_check", fake_cov)

    import inference.handler as h
    result = h.handler({"action": "check_coverage", "date": "2026-04-15"}, _fake_context())

    assert result == fake_delta
    fake_cov.compute_coverage_delta.assert_called_once()


# ── action="check_drift" ────────────────────────────────────────────────────


def test_handler_check_drift_writes_artifact(stubbed_preflight, monkeypatch):
    """check_drift dispatches to monitoring.drift_detector.check_drift, returns
    its structured result, and uses the light env+S3 preflight only — NOT the
    full predict preflight and NOT the deploy-drift gate (config#1282)."""
    fake_result = {
        "date": "2026-06-26",
        "status": "ok",
        "severity": None,
        "alerts": [],
        "alert_details": [],
        "skipped_checks": [],
        "n_alerts": 0,
    }
    fake_dd = MagicMock()
    fake_dd.check_drift = MagicMock(return_value=fake_result)
    monkeypatch.setitem(sys.modules, "monitoring.drift_detector", fake_dd)

    import inference.handler as h
    result = h.handler({"action": "check_drift", "date": "2026-06-26"}, _fake_context())

    assert result == fake_result
    fake_dd.check_drift.assert_called_once()
    call_kwargs = fake_dd.check_drift.call_args.kwargs
    assert call_kwargs["date_str"] == "2026-06-26"
    assert call_kwargs["dry_run"] is False
    # Light preflight: env + S3 only, no full predict bootstrap, no drift gate.
    stubbed_preflight.return_value.check_env_vars.assert_called_once_with("AWS_REGION")
    stubbed_preflight.return_value.check_s3_bucket.assert_called_once()


def test_handler_check_drift_dry_run_threaded_to_check_drift(
    stubbed_preflight, monkeypatch,
):
    """The deploy-time canary invokes check_drift with dry_run=true so it
    doesn't overwrite the real EOD-SF-produced drift_{date}.json on every
    deploy (config#3025 dim8) — the handler must thread that flag through."""
    fake_dd = MagicMock()
    fake_dd.check_drift = MagicMock(return_value={"date": "2026-06-26", "status": "ok"})
    monkeypatch.setitem(sys.modules, "monitoring.drift_detector", fake_dd)

    import inference.handler as h
    h.handler({"action": "check_drift", "dry_run": True}, _fake_context())

    call_kwargs = fake_dd.check_drift.call_args.kwargs
    assert call_kwargs["dry_run"] is True
    stubbed_preflight.return_value.run.assert_not_called()
    stubbed_preflight.return_value.run_for_drift_gate.assert_not_called()


# ── action="check_deploy_drift" ─────────────────────────────────────────────


def test_handler_check_deploy_drift_uses_run_for_drift_gate(
    stubbed_preflight, monkeypatch,
):
    fake_drift = MagicMock()
    fake_drift.check_deploy_drift = MagicMock(return_value={
        "upstream_sha": "abc1234567",
        "sf_sha": "abc1234567",
        "sf_drift": False,
        "stack_sha": "abc1234567",
        "cf_drift": False,
    })
    monkeypatch.setitem(sys.modules, "inference.deploy_drift", fake_drift)

    import inference.handler as h
    ctx = _fake_context(arn="arn:aws:lambda:us-east-1:123456789012:function:foo")
    result = h.handler({"action": "check_deploy_drift"}, ctx)

    assert result["sf_drift"] is False
    fake_drift.check_deploy_drift.assert_called_once_with(
        region="us-east-1", account_id="123456789012",
    )
    # Drift gate should use the lighter preflight path
    stubbed_preflight.return_value.run_for_drift_gate.assert_called_once()
    stubbed_preflight.return_value.run.assert_not_called()


def test_handler_check_deploy_drift_falls_back_to_env_account(
    stubbed_preflight, monkeypatch,
):
    monkeypatch.setenv("AWS_ACCOUNT_ID", "999999999999")
    fake_drift = MagicMock()
    fake_drift.check_deploy_drift = MagicMock(return_value={
        "upstream_sha": "abc",
        "sf_sha": "abc",
        "sf_drift": False,
        "stack_sha": "abc",
        "cf_drift": False,
    })
    monkeypatch.setitem(sys.modules, "inference.deploy_drift", fake_drift)

    import inference.handler as h
    result = h.handler({"action": "check_deploy_drift"}, None)
    fake_drift.check_deploy_drift.assert_called_once_with(
        region="us-east-1", account_id="999999999999",
    )


def test_handler_check_deploy_drift_never_receives_skip_even_with_dry_run(
    stubbed_preflight, monkeypatch,
):
    """Hard invariant (config#2731): the dedicated check_deploy_drift ACTION
    (the SF's first-state gate) must keep asserting drift unconditionally.
    Even if a caller mistakenly sets dry_run=true alongside this action, the
    skip must NOT be threaded through — run_for_drift_gate() is called with
    no kwargs (its default is skip_deploy_drift=False)."""
    fake_drift = MagicMock()
    fake_drift.check_deploy_drift = MagicMock(return_value={
        "upstream_sha": "abc1234567",
        "sf_sha": "abc1234567",
        "sf_drift": False,
        "stack_sha": "abc1234567",
        "cf_drift": False,
    })
    monkeypatch.setitem(sys.modules, "inference.deploy_drift", fake_drift)

    import inference.handler as h
    ctx = _fake_context(arn="arn:aws:lambda:us-east-1:123456789012:function:foo")
    h.handler({"action": "check_deploy_drift", "dry_run": True}, ctx)

    stubbed_preflight.return_value.run_for_drift_gate.assert_called_once_with()


# ── action="check_lib_pin_drift" / "check_pipeline_contract" ────────────────
# (config#2731 — deploy-time canary dry_run must skip ONLY the deploy-drift
# assertion inside run_for_drift_gate, not the contract/lib-pin check itself,
# and non-dry_run SF pre-spend invocations must retain drift checking.)


def test_handler_check_lib_pin_drift_sf_gate_defaults_drift_on(
    stubbed_preflight, monkeypatch,
):
    """The SF's own pre-spend invocation (no dry_run) retains its existing
    drift belt-and-suspenders — skip_deploy_drift defaults to False."""
    fake_lpd = MagicMock()
    fake_lpd.check_lib_pin_drift = MagicMock(return_value={
        "has_drift": False, "reason": "in_sync", "pins": {}, "offenders": [],
    })
    monkeypatch.setitem(sys.modules, "inference.lib_pin_drift", fake_lpd)

    import inference.handler as h
    result = h.handler({"action": "check_lib_pin_drift"}, _fake_context())

    assert result["has_drift"] is False
    stubbed_preflight.return_value.run_for_drift_gate.assert_called_once_with(
        skip_deploy_drift=False,
    )


def test_handler_check_lib_pin_drift_canary_dry_run_skips_drift(
    stubbed_preflight, monkeypatch,
):
    """deploy.sh's deploy-time canary invocation (dry_run=true) skips ONLY
    the deploy-drift assertion — the lib-pin check itself still runs and its
    result is still returned/validated."""
    fake_lpd = MagicMock()
    fake_lpd.check_lib_pin_drift = MagicMock(return_value={
        "has_drift": False, "reason": "in_sync", "pins": {}, "offenders": [],
    })
    monkeypatch.setitem(sys.modules, "inference.lib_pin_drift", fake_lpd)

    import inference.handler as h
    result = h.handler(
        {"action": "check_lib_pin_drift", "dry_run": True}, _fake_context(),
    )

    assert result["has_drift"] is False
    fake_lpd.check_lib_pin_drift.assert_called_once()
    stubbed_preflight.return_value.run_for_drift_gate.assert_called_once_with(
        skip_deploy_drift=True,
    )


def test_handler_check_pipeline_contract_sf_gate_defaults_drift_on(
    stubbed_preflight, monkeypatch,
):
    """The SF's own pre-spend invocation (no dry_run) retains its existing
    drift belt-and-suspenders — skip_deploy_drift defaults to False."""
    fake_pcc = MagicMock()
    fake_pcc.check_pipeline_contract = MagicMock(return_value={
        "has_violation": False, "violations": [], "boundary_count": 1,
        "reason": "in_sync",
    })
    monkeypatch.setitem(sys.modules, "inference.pipeline_contract_check", fake_pcc)

    import inference.handler as h
    result = h.handler({"action": "check_pipeline_contract"}, _fake_context())

    assert result["has_violation"] is False
    stubbed_preflight.return_value.run_for_drift_gate.assert_called_once_with(
        skip_deploy_drift=False,
    )


def test_handler_check_pipeline_contract_canary_dry_run_skips_drift_not_contract(
    stubbed_preflight, monkeypatch,
):
    """The core failure class from config#2731: a deploy-time canary
    (dry_run=true) invocation of check_pipeline_contract must PASS the
    drift gate (skip_deploy_drift=True threaded through) while STILL
    running and honoring the real contract-validation result — a genuine
    contract violation must still surface, dry_run or not."""
    fake_pcc = MagicMock()
    fake_pcc.check_pipeline_contract = MagicMock(return_value={
        "has_violation": True,
        "violations": ["boundaries[0] (id=x) artifact_id='ghost' dangling"],
        "boundary_count": 1,
        "reason": "violation_detected",
    })
    monkeypatch.setitem(sys.modules, "inference.pipeline_contract_check", fake_pcc)

    import inference.handler as h
    result = h.handler(
        {"action": "check_pipeline_contract", "dry_run": True}, _fake_context(),
    )

    # Contract check itself is not skipped/short-circuited by dry_run.
    assert result["has_violation"] is True
    fake_pcc.check_pipeline_contract.assert_called_once()
    # Only the deploy-drift assertion inside the preflight is skipped.
    stubbed_preflight.return_value.run_for_drift_gate.assert_called_once_with(
        skip_deploy_drift=True,
    )
    stubbed_preflight.return_value.run.assert_not_called()


# ── action="predict" (default) ─────────────────────────────────────────────


def test_handler_predict_default_path_invokes_main(
    stubbed_preflight, monkeypatch,
):
    fake_daily = MagicMock()
    fake_daily.main = MagicMock()
    monkeypatch.setitem(sys.modules, "inference.daily_predict", fake_daily)

    import inference.handler as h
    result = h.handler({"date": "2026-04-15"}, _fake_context())

    assert result["statusCode"] == 200
    assert "Predictions written for 2026-04-15" in result["body"]
    fake_daily.main.assert_called_once()
    call_kwargs = fake_daily.main.call_args.kwargs
    assert call_kwargs["date_str"] == "2026-04-15"
    assert call_kwargs["dry_run"] is False
    assert call_kwargs["local"] is False
    assert call_kwargs["explicit_tickers"] == []
    # Default predict path uses full preflight, drift-check enabled
    stubbed_preflight.return_value.run.assert_called_once()
    assert (
        stubbed_preflight.return_value.run.call_args.kwargs["skip_deploy_drift"]
        is False
    )


def test_handler_predict_dry_run_skips_heartbeat_write(
    stubbed_preflight, monkeypatch,
):
    """dry_run's contract is "no S3 writes" — the deploy canary invokes
    predict(dry_run=true) and must not leave any S3 side effect, including
    the flow-doctor heartbeat (config#3025 dim8)."""
    fake_daily = MagicMock()
    fake_daily.main = MagicMock()
    monkeypatch.setitem(sys.modules, "inference.daily_predict", fake_daily)

    fake_fd = MagicMock()
    fake_krepis_logging = MagicMock(get_flow_doctor=MagicMock(return_value=fake_fd))
    monkeypatch.setitem(sys.modules, "krepis.logging", fake_krepis_logging)

    import inference.handler as h
    result = h.handler({"date": "2026-04-15", "dry_run": True}, _fake_context())

    assert result["statusCode"] == 200
    fake_fd.emit_heartbeat.assert_not_called()


def test_handler_predict_non_dry_run_writes_heartbeat(
    stubbed_preflight, monkeypatch,
):
    fake_daily = MagicMock()
    fake_daily.main = MagicMock()
    monkeypatch.setitem(sys.modules, "inference.daily_predict", fake_daily)

    fake_fd = MagicMock()
    fake_krepis_logging = MagicMock(get_flow_doctor=MagicMock(return_value=fake_fd))
    monkeypatch.setitem(sys.modules, "krepis.logging", fake_krepis_logging)

    import inference.handler as h
    result = h.handler({"date": "2026-04-15"}, _fake_context())

    assert result["statusCode"] == 200
    fake_fd.emit_heartbeat.assert_called_once()


def test_handler_predict_supplemental_tickers_split_csv(
    stubbed_preflight, monkeypatch,
):
    fake_daily = MagicMock()
    fake_daily.main = MagicMock()
    monkeypatch.setitem(sys.modules, "inference.daily_predict", fake_daily)

    import inference.handler as h
    result = h.handler(
        {"action": "predict", "tickers": "aapl, msft , googl"},
        _fake_context(),
    )

    assert result["statusCode"] == 200
    assert "Supplemental predictions written" in result["body"]
    assert "3 tickers" in result["body"]
    call_kwargs = fake_daily.main.call_args.kwargs
    assert call_kwargs["explicit_tickers"] == ["AAPL", "MSFT", "GOOGL"]


def test_handler_predict_supplemental_tickers_list(
    stubbed_preflight, monkeypatch,
):
    fake_daily = MagicMock()
    fake_daily.main = MagicMock()
    monkeypatch.setitem(sys.modules, "inference.daily_predict", fake_daily)

    import inference.handler as h
    result = h.handler({"tickers": ["aapl", "msft"], "dry_run": True}, _fake_context())

    call_kwargs = fake_daily.main.call_args.kwargs
    assert call_kwargs["explicit_tickers"] == ["AAPL", "MSFT"]
    assert call_kwargs["dry_run"] is True
    # Canary (dry_run=true) must skip the deploy-drift check (config#1073)
    assert (
        stubbed_preflight.return_value.run.call_args.kwargs["skip_deploy_drift"]
        is True
    )




# ── alpha-engine-config-I7316 ───────────────────────────────────────────────
# The handler must survive the UNMEASURED payload of every gate probe that
# omits keys on its fail-open path.
#
# What happened: alpha-engine-config-I7048 taught these probes to OMIT the
# verdict key rather than report a false one, and guarded the handler's log
# line for THAT key with `.get()`. -I7277 and -I7171 then extended the same
# reasoning to the EVIDENCE lists (`violations`, `offenders`) — an empty list
# is as much a claim as a false verdict — but the handler's bare subscripts on
# those siblings were never updated. From that merge on, every unmeasured
# invocation raised KeyError inside the very log line written to report it.
#
# Measured on the 2026-08-14T03:10 deploy canary against version 479:
#   check_pipeline_contract: FunctionError: Unhandled, payload: 'violations'
#   check_lib_pin_drift:     FunctionError: Unhandled, payload: 'offenders'
# The canary refused the promote and the :live alias froze at v474.
#
# These tests drive the probes' REAL unmeasured payloads — built by the
# producer's own constructor, not hand-written here — through the handler. A
# third probe adopting the omit-the-key shape, or a fourth key joining the
# omitted set, fails here instead of at a deploy canary.


def _real_unknown_pipeline_contract_payload() -> dict:
    from inference.pipeline_contract_check import _unknown_result
    return _unknown_result("source_missing")


def _real_unmeasured_lib_pin_payload() -> dict:
    """The lib-pin probe's own fail-open payload, driven through the real
    function with every upstream read unresolvable."""
    from unittest.mock import patch as _patch

    from inference import lib_pin_drift as lpd

    with _patch.object(
        lpd, "_fetch_repo_pin",
        side_effect=lambda *a, **k: lpd.PinRead(None, lpd.UNREACHABLE, "patched"),
    ):
        return lpd.check_lib_pin_drift()


def test_unmeasured_pipeline_contract_payload_survives_the_handler(
    stubbed_preflight, monkeypatch,
):
    payload = _real_unknown_pipeline_contract_payload()
    # Precondition: this is the shape the bug needed. If the producer starts
    # emitting these keys again, this test is no longer covering anything and
    # should be re-pointed rather than silently passing.
    assert "violations" not in payload
    assert "has_violation" not in payload

    fake_pcc = MagicMock()
    fake_pcc.check_pipeline_contract = MagicMock(return_value=payload)
    monkeypatch.setitem(sys.modules, "inference.pipeline_contract_check", fake_pcc)

    import inference.handler as h
    result = h.handler({"action": "check_pipeline_contract"}, _fake_context())

    assert result == payload, "the handler must return the probe's payload verbatim"


def test_unmeasured_lib_pin_payload_survives_the_handler(
    stubbed_preflight, monkeypatch,
):
    payload = _real_unmeasured_lib_pin_payload()
    assert "has_drift" not in payload

    fake_lpd = MagicMock()
    fake_lpd.check_lib_pin_drift = MagicMock(return_value=payload)
    monkeypatch.setitem(sys.modules, "inference.lib_pin_drift", fake_lpd)

    import inference.handler as h
    result = h.handler({"action": "check_lib_pin_drift"}, _fake_context())

    assert result == payload


@pytest.mark.parametrize(("action", "module_path", "func_name"), [
    ("check_pipeline_contract", "inference.pipeline_contract_check",
     "check_pipeline_contract"),
    ("check_lib_pin_drift", "inference.lib_pin_drift", "check_lib_pin_drift"),
])
def test_gate_actions_survive_a_payload_carrying_only_a_reason(
    stubbed_preflight, monkeypatch, action, module_path, func_name,
):
    """The floor: a probe payload that carries nothing but a reason.

    Generalizes past the two keys that actually broke. Any future key the
    handler's log line reaches for must be optional, because the whole point
    of the unmeasured shape is that it asserts as little as possible.
    """
    fake = MagicMock()
    setattr(fake, func_name, MagicMock(return_value={"reason": "source_missing"}))
    monkeypatch.setitem(sys.modules, module_path, fake)

    import inference.handler as h
    result = h.handler({"action": action}, _fake_context())

    assert result == {"reason": "source_missing"}


@pytest.mark.parametrize(("action", "module_path", "func_name"), [
    ("check_pipeline_contract", "inference.pipeline_contract_check",
     "check_pipeline_contract"),
    ("check_lib_pin_drift", "inference.lib_pin_drift", "check_lib_pin_drift"),
])
def test_gate_actions_survive_an_entirely_empty_payload(
    stubbed_preflight, monkeypatch, action, module_path, func_name,
):
    """Even `reason` is not guaranteed by anything structural. The handler is
    a reporting surface; it must never be the thing that turns a degraded
    probe into an Unhandled Lambda error, because the SF then attributes the
    failure to the invocation rather than to the gate
    (alpha-engine-config-I7302)."""
    fake = MagicMock()
    setattr(fake, func_name, MagicMock(return_value={}))
    monkeypatch.setitem(sys.modules, module_path, fake)

    import inference.handler as h
    assert h.handler({"action": action}, _fake_context()) == {}


# ── alpha-engine-config-I7319: the other direction ──────────────────────────
#
# Every test above asserts that nothing RAISES on a payload missing its
# evidence list. None asserts that the evidence still arrives when it exists.
#
# That asymmetry is a real risk here rather than a theoretical one, because
# everything in this area has moved one way for two days: -I7048 omitted the
# verdict key, -I7277 and -I7171 omitted the evidence lists, -I7316 guarded
# every read. Each step was right. An edit that took the next step — dropping
# the `offenders=` clause entirely, say, while "simplifying" the log line —
# would pass the whole suite, because the suite only knows how to check that
# the handler survives.
#
# `offenders` is what a human acts on when the gate halts the weekly SF: it
# names the exact cross-repo pin mismatch to fix. Losing it silently costs
# the diagnostic at precisely the moment it is needed, which is the same
# reports-less-than-it-measured shape the whole -I7301/-I7316 arc is about.


def _captured_gate_log(monkeypatch, h) -> list:
    """Capture the handler's own log calls, rendered.

    Deliberately NOT caplog: `handler()` reconfigures root logging on its
    first invocation (``logging.getLogger().setLevel`` at handler.py:181,
    plus krepis' own setup), so caplog captures nothing in whichever test
    triggers that first — the assertion would pass or fail on collection
    order. Patching the module logger tests the same thing (what the gate
    line is asked to render) and cannot be perturbed by logging config.
    """
    rendered: list = []

    def _info(fmt, *args):
        rendered.append(fmt % args)

    monkeypatch.setattr(h.log, "info", _info)
    return rendered


def test_lib_pin_offenders_reach_the_log_when_the_gate_measured_them(
    stubbed_preflight, monkeypatch,
):
    fake_lpd = MagicMock()
    fake_lpd.check_lib_pin_drift = MagicMock(return_value={
        "status": "MEASURED",
        "has_drift": True,
        "parity_ok": False,
        "floor_ok": True,
        "min_lib_version": "0.39.0",
        "pins": {
            "nousergon/crucible-backtester": "v0.124.5",
            "nousergon/crucible-predictor": "v0.124.57",
        },
        "offenders": [
            "co-install parity: nousergon/crucible-backtester=v0.124.5 != "
            "nousergon/crucible-predictor=v0.124.57",
        ],
        "reason": "drift_detected",
    })
    monkeypatch.setitem(sys.modules, "inference.lib_pin_drift", fake_lpd)

    import inference.handler as h
    rendered = _captured_gate_log(monkeypatch, h)
    h.handler({"action": "check_lib_pin_drift"}, _fake_context())

    line = next(m for m in rendered if m.startswith("Lib-pin drift check:"))
    assert "offenders=co-install parity" in line
    # the offending versions, not just the label — this line is the only
    # place a human sees them without opening the SF execution
    assert "v0.124.5" in line and "v0.124.57" in line


def test_pipeline_contract_violations_reach_the_log_when_measured(
    stubbed_preflight, monkeypatch,
):
    fake_pcc = MagicMock()
    fake_pcc.check_pipeline_contract = MagicMock(return_value={
        "status": "MEASURED",
        "has_violation": True,
        "violations": ["producer/consumer mismatch: predictions/{date}.json"],
        "boundary_count": 12,
        "reason": "violation_detected",
    })
    monkeypatch.setitem(sys.modules, "inference.pipeline_contract_check", fake_pcc)

    import inference.handler as h
    rendered = _captured_gate_log(monkeypatch, h)
    h.handler({"action": "check_pipeline_contract"}, _fake_context())

    line = next(m for m in rendered if m.startswith("Pipeline-contract preflight:"))
    assert "violations=producer/consumer mismatch" in line
    assert "boundaries=12" in line
