"""A probe must write no verdict, and every verdict must say who wrote it.

`alpha-engine-config/scripts/check_prespend_gate_arm.py` (merged 2026-08-28,
`alpha-engine-config-I9168`) invokes each pre-spend gate's Lambda with the
Task's own Payload and `$.run_date` filled from `next_run_date()`
(`_payload_for` at `:455`, `probe_gate` at `:496`). Off the Step Function, with
a real-looking run_date and no execution behind it — the exact shape
`is_synthetic_invocation`'s docstring named as the reason the synthetic check
runs BEFORE the run_date check.

Measured 2026-08-29 on `s3://alpha-engine-research/_stage_coverage/2026-08-28/`
(13 objects): `LibPinDriftCheck`, `PipelineContractCheck` and
`EvaluatorDeployDriftCheck` rewritten at 03:18:34–39Z, `WeeklyRunDayGate` and
`EvaluatorDirectorDeployDriftCheck` at 02:14Z, by no execution. That prefix is
the denominator of `alpha-engine-config-I8155`'s own gate predicate, so a probe
write lets the gate clear green on verdicts no run produced.

Both halves are pinned here:

* `"probe"` is synthetic, so the invocation writes nothing at all;
* every verdict record that IS written carries `execution_arn` and
  `invocation_kind`. Before this, every object in that partition read
  `execution_arn: None` — including `SignalsEnvelope`, which no probe touched —
  so a probe write and a real write were indistinguishable in the artifact and
  no consumer could have excluded one from a count.
"""

from __future__ import annotations

import json
import logging
import sys
import types

import pytest

from stage_coverage_safety import (
    EXECUTION_ARN_KEY,
    INVOCATION_KIND_KEY,
    SYNTHETIC_INVOCATION_KINDS,
    VERDICT_KEY_PREFIX,
    StageCoverageAttributionError,
    _AttributingS3Client,
    invocation_attribution,
    is_synthetic_invocation,
    resolve_execution_arn,
    safe_assert_stage_coverage,
    unmeasured_stage_coverage,
)

_EXEC_ARN = (
    "arn:aws:states:us-east-1:123456789012:execution:"
    "ne-weekly-freshness-pipeline:2026-08-28-test"
)


class _RecordingS3:
    """Captures every put_object the code under test causes."""

    def __init__(self) -> None:
        self.puts: list[dict] = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        return {}


# ── The closed set gained exactly one member ─────────────────────────────────


def test_probe_is_synthetic_and_the_set_is_still_closed() -> None:
    assert SYNTHETIC_INVOCATION_KINDS == frozenset({"canary", "probe"})
    assert is_synthetic_invocation({INVOCATION_KIND_KEY: "probe"}) is True
    assert is_synthetic_invocation({INVOCATION_KIND_KEY: "PROBE"}) is True
    assert is_synthetic_invocation({INVOCATION_KIND_KEY: " probe "}) is True
    # Still closed: a kind nobody declared is a real run.
    assert is_synthetic_invocation({INVOCATION_KIND_KEY: "prober"}) is False
    assert is_synthetic_invocation({"run_date": "2026-08-28"}) is False


def test_a_probe_invocation_writes_no_verdict_object(monkeypatch) -> None:
    """Mirrors the canary case: synthetic is read BEFORE run_date.

    `check_prespend_gate_arm.py` supplies a real-looking `run_date`, so the
    exemption must not depend on the run_date being absent.
    """
    reached = []

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        reached.append((args, kwargs))
        raise AssertionError("assert_stage_coverage reached from a probe")

    monkeypatch.setattr(
        "krepis.stage_coverage.assert_stage_coverage", _boom, raising=False
    )
    log = logging.getLogger("test.attribution.probe")
    verdict = safe_assert_stage_coverage(
        "LibPinDriftCheck",
        event={
            "action": "check_lib_pin_drift",
            "run_date": "2026-08-29",
            INVOCATION_KIND_KEY: "probe",
        },
        window_start=None,
        log=log,
    )

    assert reached == []
    assert verdict is not None
    assert verdict["status"] == "UNMEASURED"
    assert verdict[INVOCATION_KIND_KEY] == "probe"
    assert verdict[EXECUTION_ARN_KEY] is None


def test_a_probe_records_unmeasured_without_a_single_error_record(caplog) -> None:
    """A refusal, said out loud at INFO — not a swallow, and not a page.

    `inference/handler.py` attaches flow-doctor's handler at ERROR; the
    prespend prober runs on a schedule, so an ERROR here would page on every
    arm check.
    """
    log = logging.getLogger("test.attribution.probe.level")
    with caplog.at_level(logging.INFO, logger=log.name):
        safe_assert_stage_coverage(
            "WeeklyRunDayGate",
            event={"run_date": "2026-08-29", INVOCATION_KIND_KEY: "probe"},
            window_start=None,
            log=log,
        )
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "No verdict written" in message


# ── Attribution resolution ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "event, expected",
    [
        ({"execution_arn": _EXEC_ARN}, _EXEC_ARN),
        ({"execution_id": _EXEC_ARN}, _EXEC_ARN),
        ({"execution_arn": f"  {_EXEC_ARN} "}, _EXEC_ARN),
        # Never fabricated, never derived.
        ({"run_date": "2026-08-28"}, None),
        ({"execution_arn": ""}, None),
        ({"execution_arn": "   "}, None),
        ({"execution_arn": None}, None),
        ({"execution_arn": 17}, None),
    ],
)
def test_resolve_execution_arn(event: dict, expected) -> None:
    assert resolve_execution_arn(event) == expected


def test_absent_attribution_is_none_not_empty_string() -> None:
    """`None` so a consumer's IS NOT NULL filter reads the same everywhere."""
    assert invocation_attribution({"run_date": "2026-08-28"}) == {
        EXECUTION_ARN_KEY: None,
        INVOCATION_KIND_KEY: None,
    }
    assert invocation_attribution(
        {"execution_arn": _EXEC_ARN, INVOCATION_KIND_KEY: "probe"}
    ) == {EXECUTION_ARN_KEY: _EXEC_ARN, INVOCATION_KIND_KEY: "probe"}


def test_the_degrade_shape_carries_both_fields() -> None:
    degraded = unmeasured_stage_coverage("LibPinDriftCheck", "because")
    assert degraded[EXECUTION_ARN_KEY] is None
    assert degraded[INVOCATION_KIND_KEY] is None


# ── The WRITTEN record — the half a gate predicate actually reads ────────────


def test_a_written_verdict_record_carries_both_attribution_fields() -> None:
    inner = _RecordingS3()
    client = _AttributingS3Client(
        inner, {EXECUTION_ARN_KEY: _EXEC_ARN, INVOCATION_KIND_KEY: None}
    )
    client.put_object(
        Bucket="alpha-engine-research",
        Key=f"{VERDICT_KEY_PREFIX}2026-08-28/LibPinDriftCheck.json",
        Body=json.dumps({"stage": "LibPinDriftCheck", "status": "COVERED_NO_OUTPUT"}).encode(),
        ContentType="application/json",
    )

    assert len(inner.puts) == 1
    record = json.loads(inner.puts[0]["Body"])
    assert EXECUTION_ARN_KEY in record and INVOCATION_KIND_KEY in record
    assert record[EXECUTION_ARN_KEY] == _EXEC_ARN
    assert record[INVOCATION_KIND_KEY] is None
    # Nothing the verdict already said is lost.
    assert record["stage"] == "LibPinDriftCheck"
    assert record["status"] == "COVERED_NO_OUTPUT"


def test_only_verdict_keys_are_stamped() -> None:
    """The registry read and the head_object probes proxy through untouched."""
    inner = _RecordingS3()
    client = _AttributingS3Client(inner, {EXECUTION_ARN_KEY: _EXEC_ARN})
    body = b"not json at all"
    client.put_object(Bucket="b", Key="some/other/prefix/thing.bin", Body=body)
    assert inner.puts[0]["Body"] is body


def test_a_non_json_verdict_body_raises_rather_than_writing_unattributed() -> None:
    """Loud: `record_verdict` wraps put_object in a fail-soft ERROR log."""
    client = _AttributingS3Client(_RecordingS3(), {EXECUTION_ARN_KEY: None})
    with pytest.raises(StageCoverageAttributionError):
        client.put_object(
            Bucket="b", Key=f"{VERDICT_KEY_PREFIX}2026-08-28/X.json", Body=b"nope"
        )


def test_safe_assert_writes_an_attributed_record_end_to_end(monkeypatch) -> None:
    """The client this module hands krepis is the one that reaches S3.

    Stands in for `krepis.stage_coverage.record_verdict`, which serialises
    `StageVerdict.to_dict()` and `put_object`s it under `VERDICT_KEY_PREFIX`.
    """
    inner = _RecordingS3()

    def _fake_assert(stage, *, run_date, window_start=None, s3_client=None, **_):
        assert s3_client is not None, "no attributing client was passed to krepis"
        verdict = {"stage": stage, "status": "COVERED_NO_OUTPUT", "run_date": run_date}
        s3_client.put_object(
            Bucket="alpha-engine-research",
            Key=f"{VERDICT_KEY_PREFIX}{run_date}/{stage}.json",
            Body=json.dumps(verdict).encode(),
            ContentType="application/json",
        )
        return verdict

    fake_module = types.ModuleType("krepis.stage_coverage")
    fake_module.assert_stage_coverage = _fake_assert
    fake_module.StageCoverageContractError = RuntimeError
    monkeypatch.setitem(sys.modules, "krepis.stage_coverage", fake_module)
    monkeypatch.setattr(
        "stage_coverage_safety._attributing_s3_client",
        lambda attribution, log: _AttributingS3Client(inner, attribution),
    )

    returned = safe_assert_stage_coverage(
        "LibPinDriftCheck",
        event={"run_date": "2026-08-28", "execution_arn": _EXEC_ARN},
        window_start=None,
        log=logging.getLogger("test.attribution.e2e"),
    )

    assert len(inner.puts) == 1
    written = json.loads(inner.puts[0]["Body"])
    assert written[EXECUTION_ARN_KEY] == _EXEC_ARN
    assert written[INVOCATION_KIND_KEY] is None
    # And the handler's own response says the same thing.
    assert returned[EXECUTION_ARN_KEY] == _EXEC_ARN
    assert returned[INVOCATION_KIND_KEY] is None


# ── The sanctioned mirrored constant (policy-shared-code §4) ─────────────────


def test_the_mirrored_verdict_prefix_matches_krepis() -> None:
    """A mirrored constant is legitimate only while a test proves it matches."""
    krepis_stage_coverage = pytest.importorskip("krepis.stage_coverage")
    assert VERDICT_KEY_PREFIX == f"{krepis_stage_coverage.VERDICT_PREFIX}/"
