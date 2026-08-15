"""Stage-coverage assertion wiring in this repo's three Lambda handlers
(config-I7214, sf-pipeline-policy.md §2.1 / §2.3a).

Two properties, per call site:

1. When `krepis.stage_coverage` IS importable, its verdict lands in
   the handler's returned payload under `stage_coverage`.
2. When it is NOT importable (the pinned nousergon-lib SHA predates the
   module — the live condition today), the handler's own outcome is
   UNCHANGED: no exception escapes, no key silently mutates a field the
   caller already reads, and the miss is logged loudly (not swallowed).

Handlers covered: `inference.handler.handler` (WeeklyRunDayGate,
LibPinDriftCheck, PipelineContractCheck), `regime.handler.lambda_handler`
(RegimeSubstrate), `regime.retrospective_eval_handler.lambda_handler`
(RegimeRetrospectiveEval).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


# ── Shared fake krepis.stage_coverage module ────────────────────────


def _install_fake_stage_coverage(monkeypatch, verdict: dict) -> MagicMock:
    """Install a fake `krepis.stage_coverage` submodule whose
    `assert_stage_coverage` returns `verdict` unconditionally, and return
    the mock so call args can be inspected.

    Only the submodule entry is patched into `sys.modules` — the real,
    already-installed `nousergon_lib` top-level package (pinned in
    requirements.txt) is left untouched, because other code on this path
    (`inference.trading_day_gate`) imports real siblings of it
    (`nousergon_lib.trading_calendar`) in the same call.
    """
    fake_fn = MagicMock(return_value=verdict)
    fake_module = types.ModuleType("krepis.stage_coverage")
    fake_module.assert_stage_coverage = fake_fn
    monkeypatch.setitem(sys.modules, "krepis.stage_coverage", fake_module)
    return fake_fn


def _uninstall_stage_coverage(monkeypatch) -> None:
    """Force `from krepis.stage_coverage import assert_stage_coverage`
    to raise ImportError, mirroring the live pinned-SHA condition.
    """
    monkeypatch.delitem(sys.modules, "krepis.stage_coverage", raising=False)
    # Also block a real installed copy from being found on sys.path, if any.
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "krepis.stage_coverage" or name.startswith("krepis.stage_coverage."):
            raise ImportError("no module named krepis.stage_coverage (test-forced)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)


# ── inference.handler — WeeklyRunDayGate ────────────────────────────────────


def test_weekly_run_day_gate_merges_verdict_when_lib_available(monkeypatch):
    fake_fn = _install_fake_stage_coverage(
        monkeypatch, {"verdict": "COVERED_NO_OUTPUT", "stage": "WeeklyRunDayGate"}
    )

    class _ExplodingPreflight:
        def __init__(self, *a, **k):
            raise AssertionError("PredictorPreflight must NOT be constructed for check_weekly_run_day")

    fake_preflight_module = MagicMock(PredictorPreflight=_ExplodingPreflight)
    monkeypatch.setitem(sys.modules, "inference.preflight", fake_preflight_module)

    import inference.handler as h

    result = h.handler({"action": "check_weekly_run_day", "date": "2026-07-11"}, None)

    assert result["is_weekly_run_day"] is True
    assert result["stage_coverage"] == {"verdict": "COVERED_NO_OUTPUT", "stage": "WeeklyRunDayGate"}
    fake_fn.assert_called_once()
    _, kwargs = fake_fn.call_args
    assert fake_fn.call_args[0][0] == "WeeklyRunDayGate"
    assert kwargs["run_date"] == "2026-07-11"
    assert kwargs["window_start"] is not None


def test_weekly_run_day_gate_outcome_unchanged_when_lib_absent(monkeypatch):
    _uninstall_stage_coverage(monkeypatch)

    class _ExplodingPreflight:
        def __init__(self, *a, **k):
            raise AssertionError("PredictorPreflight must NOT be constructed for check_weekly_run_day")

    fake_preflight_module = MagicMock(PredictorPreflight=_ExplodingPreflight)
    monkeypatch.setitem(sys.modules, "inference.preflight", fake_preflight_module)

    import inference.handler as h

    result = h.handler({"action": "check_weekly_run_day", "date": "2026-07-11"}, None)

    assert result["is_weekly_run_day"] is True
    assert result["marker"] == "WEEKLY_RUN_DAY"
    assert "stage_coverage" not in result


# ── inference.handler — LibPinDriftCheck / PipelineContractCheck ───────────


def _stub_preflight(monkeypatch):
    """check_lib_pin_drift / check_pipeline_contract still construct
    PredictorPreflight (the lightweight GitHub-reads-only path) — stub it so
    the test never touches AWS/GitHub. `PredictorPreflight` is imported
    LOCALLY inside `handler()` (`from inference.preflight import
    PredictorPreflight`), so the class must be patched on its defining
    module, not on `inference.handler`.
    """
    fake_pf_instance = MagicMock()
    fake_pf_instance.run_for_drift_gate.return_value = None
    fake_pf_cls = MagicMock(return_value=fake_pf_instance)
    monkeypatch.setattr("inference.preflight.PredictorPreflight", fake_pf_cls)


def test_lib_pin_drift_check_merges_verdict_when_lib_available(monkeypatch):
    fake_fn = _install_fake_stage_coverage(
        monkeypatch, {"verdict": "COVERED_NO_OUTPUT", "stage": "LibPinDriftCheck"}
    )
    fake_result = {"has_drift": False, "reason": "ok", "pins": {}, "offenders": []}
    monkeypatch.setattr(
        "inference.lib_pin_drift.check_lib_pin_drift", lambda: fake_result
    )
    _stub_preflight(monkeypatch)

    import inference.handler as h

    result = h.handler({"action": "check_lib_pin_drift"}, None)

    assert result["has_drift"] is False
    assert result["stage_coverage"] == {"verdict": "COVERED_NO_OUTPUT", "stage": "LibPinDriftCheck"}
    fake_fn.assert_called_once()
    assert fake_fn.call_args[0][0] == "LibPinDriftCheck"


def test_lib_pin_drift_check_outcome_unchanged_when_lib_absent(monkeypatch):
    _uninstall_stage_coverage(monkeypatch)
    fake_result = {"has_drift": False, "reason": "ok", "pins": {}, "offenders": []}
    monkeypatch.setattr(
        "inference.lib_pin_drift.check_lib_pin_drift", lambda: fake_result
    )
    _stub_preflight(monkeypatch)

    import inference.handler as h

    result = h.handler({"action": "check_lib_pin_drift"}, None)

    assert result["has_drift"] is False
    assert "stage_coverage" not in result


def test_pipeline_contract_check_merges_verdict_when_lib_available(monkeypatch):
    fake_fn = _install_fake_stage_coverage(
        monkeypatch, {"verdict": "COVERED_NO_OUTPUT", "stage": "PipelineContractCheck"}
    )
    fake_result = {
        "has_violation": False,
        "reason": "ok",
        "boundary_count": 3,
        "violations": [],
    }
    monkeypatch.setattr(
        "inference.pipeline_contract_check.check_pipeline_contract", lambda: fake_result
    )
    _stub_preflight(monkeypatch)

    import inference.handler as h

    result = h.handler({"action": "check_pipeline_contract"}, None)

    assert result["has_violation"] is False
    assert result["stage_coverage"] == {"verdict": "COVERED_NO_OUTPUT", "stage": "PipelineContractCheck"}
    assert fake_fn.call_args[0][0] == "PipelineContractCheck"


def test_pipeline_contract_check_outcome_unchanged_when_lib_absent(monkeypatch):
    _uninstall_stage_coverage(monkeypatch)
    fake_result = {
        "has_violation": False,
        "reason": "ok",
        "boundary_count": 3,
        "violations": [],
    }
    monkeypatch.setattr(
        "inference.pipeline_contract_check.check_pipeline_contract", lambda: fake_result
    )
    _stub_preflight(monkeypatch)

    import inference.handler as h

    result = h.handler({"action": "check_pipeline_contract"}, None)

    assert result["has_violation"] is False
    assert "stage_coverage" not in result


# ── regime.handler — RegimeSubstrate ────────────────────────────────────────


def test_regime_substrate_produce_merges_verdict_when_lib_available(monkeypatch):
    fake_fn = _install_fake_stage_coverage(
        monkeypatch, {"verdict": "COVERED", "stage": "RegimeSubstrate"}
    )

    import regime.handler as rh

    fake_payload = {
        "run_id": "2608130900",
        "calendar_date": "2026-08-13",
        "hmm": {"argmax": 1},
        "bocpd": {"change_signal": False},
    }
    fake_result = {
        "payload": fake_payload,
        "wrote": True,
        "artifact_key": "regime/2608130900.json",
        "latest_key": "regime/latest.json",
    }
    monkeypatch.setattr(rh, "produce_regime_substrate", lambda **kw: fake_result)
    monkeypatch.setitem(sys.modules, "boto3", MagicMock())

    result = rh.lambda_handler({"action": "produce"}, None)

    assert result["statusCode"] == 200
    assert result["stage_coverage"] == {"verdict": "COVERED", "stage": "RegimeSubstrate"}
    assert fake_fn.call_args[0][0] == "RegimeSubstrate"
    assert fake_fn.call_args[1]["run_date"] == "2026-08-13"


def test_regime_substrate_produce_outcome_unchanged_when_lib_absent(monkeypatch):
    _uninstall_stage_coverage(monkeypatch)

    import regime.handler as rh

    fake_payload = {
        "run_id": "2608130900",
        "calendar_date": "2026-08-13",
        "hmm": {"argmax": 1},
        "bocpd": {"change_signal": False},
    }
    fake_result = {
        "payload": fake_payload,
        "wrote": True,
        "artifact_key": "regime/2608130900.json",
        "latest_key": "regime/latest.json",
    }
    monkeypatch.setattr(rh, "produce_regime_substrate", lambda **kw: fake_result)
    monkeypatch.setitem(sys.modules, "boto3", MagicMock())

    result = rh.lambda_handler({"action": "produce"}, None)

    assert result["statusCode"] == 200
    assert result["run_id"] == "2608130900"
    assert "stage_coverage" not in result


def test_regime_substrate_dry_run_never_asserts(monkeypatch):
    """dry_run declares by design that it writes nothing — asserting there
    would be a guaranteed-false MISSING, so the dry_run path must not call
    the assertion at all.
    """
    fake_fn = _install_fake_stage_coverage(
        monkeypatch, {"verdict": "COVERED", "stage": "RegimeSubstrate"}
    )

    import regime.handler as rh

    fake_payload = {"run_id": "2608130900", "calendar_date": "2026-08-13"}
    fake_result = {"payload": fake_payload, "wrote": False}
    monkeypatch.setattr(rh, "produce_regime_substrate", lambda **kw: fake_result)
    monkeypatch.setitem(sys.modules, "boto3", MagicMock())

    result = rh.lambda_handler({"action": "dry_run"}, None)

    assert result["statusCode"] == 200
    assert "stage_coverage" not in result
    fake_fn.assert_not_called()


# ── regime.retrospective_eval_handler — RegimeRetrospectiveEval ────────────


def test_retrospective_eval_produce_merges_verdict_when_lib_available(monkeypatch):
    fake_fn = _install_fake_stage_coverage(
        monkeypatch, {"verdict": "COVERED", "stage": "RegimeRetrospectiveEval"}
    )

    import regime.retrospective_eval_handler as reh

    fake_payload = {
        "run_id": "2608130900",
        "calendar_date": "2026-08-13",
        "score": {
            "n_pairings": 10,
            "asymmetric_weighted_agreement_rate": 0.8,
            "rolling_window_score": 0.75,
        },
    }
    fake_result = {
        "payload": fake_payload,
        "wrote": True,
        "artifact_key": "regime/retrospective/2608130900.json",
        "latest_key": "regime/retrospective/latest.json",
    }
    monkeypatch.setattr(reh, "produce_t1_eval", lambda **kw: fake_result)
    monkeypatch.setitem(sys.modules, "boto3", MagicMock())

    result = reh.lambda_handler({"action": "produce"}, None)

    assert result["statusCode"] == 200
    assert result["stage_coverage"] == {"verdict": "COVERED", "stage": "RegimeRetrospectiveEval"}
    assert fake_fn.call_args[0][0] == "RegimeRetrospectiveEval"
    assert fake_fn.call_args[1]["run_date"] == "2026-08-13"


def test_retrospective_eval_produce_outcome_unchanged_when_lib_absent(monkeypatch):
    _uninstall_stage_coverage(monkeypatch)

    import regime.retrospective_eval_handler as reh

    fake_payload = {
        "run_id": "2608130900",
        "calendar_date": "2026-08-13",
        "score": {
            "n_pairings": 10,
            "asymmetric_weighted_agreement_rate": 0.8,
            "rolling_window_score": 0.75,
        },
    }
    fake_result = {
        "payload": fake_payload,
        "wrote": True,
        "artifact_key": "regime/retrospective/2608130900.json",
        "latest_key": "regime/retrospective/latest.json",
    }
    monkeypatch.setattr(reh, "produce_t1_eval", lambda **kw: fake_result)
    monkeypatch.setitem(sys.modules, "boto3", MagicMock())

    result = reh.lambda_handler({"action": "produce"}, None)

    assert result["statusCode"] == 200
    assert result["run_id"] == "2608130900"
    assert "stage_coverage" not in result


def test_retrospective_eval_dry_run_never_asserts(monkeypatch):
    fake_fn = _install_fake_stage_coverage(
        monkeypatch, {"verdict": "COVERED", "stage": "RegimeRetrospectiveEval"}
    )

    import regime.retrospective_eval_handler as reh

    fake_payload = {"run_id": "2608130900", "calendar_date": "2026-08-13"}
    fake_result = {"payload": fake_payload, "wrote": False}
    monkeypatch.setattr(reh, "produce_t1_eval", lambda **kw: fake_result)
    monkeypatch.setitem(sys.modules, "boto3", MagicMock())

    result = reh.lambda_handler({"action": "dry_run"}, None)

    assert result["statusCode"] == 200
    assert "stage_coverage" not in result
    fake_fn.assert_not_called()
