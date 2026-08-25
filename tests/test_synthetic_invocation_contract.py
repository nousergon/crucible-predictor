"""A deploy canary is a probe, not a run — and must not page (I8155).

`infrastructure/deploy.sh` invokes the freshly published Lambda VERSION
directly, off the Step Function, to prove its wiring. Such an invocation has
no `$.run_date` and must never acquire one. Measured on
`/aws/lambda/alpha-engine-predictor-inference` 2026-08-25: every deploy
emitted three `log.error` lines from `stage_coverage_safety`'s
`run_date is None` branch — one per canaried gate action, versions 528, 529,
530 and 531, one burst per deploy. `inference/handler.py` attaches
flow-doctor's handler at ERROR and `flow-doctor.yaml` caps the day at 10
alerts, so each deploy spent 30% of the alert budget reporting invocations
that were healthy by construction.

The fix is a declaration rather than a silence, and this suite pins both
halves of it:

* the canary DECLARES itself, at the one chokepoint every canary call site
  passes through (`run_canary_action`), so no payload literal can drift out
  of it — the way `dry_run` already drifted off `check_lib_pin_drift`'s
  canary while the handler comment still claimed I7954's suppression applied
  to it;
* a REAL Step Functions invocation that arrives without its run_date still
  reaches the ERROR branch untouched, because that is the shape the whole
  detector exists to catch.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

from stage_coverage_safety import (
    INVOCATION_KIND_KEY,
    STATUS_UNMEASURED,
    is_synthetic_invocation,
    safe_assert_stage_coverage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SH = REPO_ROOT / "infrastructure" / "deploy.sh"
HANDLER_PY = REPO_ROOT / "inference" / "handler.py"


# ── The predicate ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "event, expected",
    [
        ({"action": "check_weekly_run_day", INVOCATION_KIND_KEY: "canary"}, True),
        ({"action": "check_weekly_run_day", INVOCATION_KIND_KEY: "CANARY"}, True),
        ({"action": "check_weekly_run_day", INVOCATION_KIND_KEY: " canary "}, True),
        # A real SF Task Payload: run_date, never invocation_kind.
        ({"action": "check_weekly_run_day", "run_date": "2026-08-29"}, False),
        ({"action": "check_weekly_run_day"}, False),
        # The set is CLOSED — an undeclared or unknown kind is a real run.
        ({INVOCATION_KIND_KEY: ""}, False),
        ({INVOCATION_KIND_KEY: "smoke"}, False),
        ({INVOCATION_KIND_KEY: None}, False),
        ({INVOCATION_KIND_KEY: True}, False),
    ],
)
def test_only_a_declared_canary_is_synthetic(event: dict, expected: bool) -> None:
    assert is_synthetic_invocation(event) is expected


# ── The log level, which is what actually pages ──────────────────────────────


def test_a_canary_records_unmeasured_without_a_single_error_record(caplog) -> None:
    """INFO, not ERROR: handler.py attaches flow-doctor's handler at ERROR."""
    log = logging.getLogger("test.synthetic.canary")
    with caplog.at_level(logging.INFO, logger=log.name):
        verdict = safe_assert_stage_coverage(
            "WeeklyRunDayGate",
            event={"action": "check_weekly_run_day", INVOCATION_KIND_KEY: "canary"},
            window_start=None,
            log=log,
        )

    assert verdict is not None
    assert verdict["status"] == STATUS_UNMEASURED
    assert verdict["run_date"] == ""
    assert "synthetic" in verdict["reason"]
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert any(r.levelno == logging.INFO for r in caplog.records)


def test_a_real_invocation_missing_its_run_date_still_errors(caplog) -> None:
    """The half that had to survive — this is the shape the detector is for."""
    log = logging.getLogger("test.synthetic.real")
    with caplog.at_level(logging.INFO, logger=log.name):
        verdict = safe_assert_stage_coverage(
            "WeeklyRunDayGate",
            event={"action": "check_weekly_run_day"},
            window_start=None,
            log=log,
        )

    assert verdict is not None
    assert verdict["status"] == STATUS_UNMEASURED
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "carries no run_date" in errors[0].getMessage()


def test_a_canary_carrying_a_run_date_still_writes_no_verdict(monkeypatch) -> None:
    """Synthetic is checked BEFORE run_date, deliberately.

    A canary that somehow carried one would otherwise land a verdict in a real
    execution's prefix and count toward its denominator.
    """
    called = []

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        called.append((args, kwargs))
        raise AssertionError("assert_stage_coverage reached from a canary")

    monkeypatch.setattr(
        "krepis.stage_coverage.assert_stage_coverage", _boom, raising=False
    )
    verdict = safe_assert_stage_coverage(
        "WeeklyRunDayGate",
        event={
            "action": "check_weekly_run_day",
            "run_date": "2026-08-29",
            INVOCATION_KIND_KEY: "canary",
        },
        window_start=None,
        log=logging.getLogger("test.synthetic.both"),
    )
    assert called == []
    assert verdict is not None and verdict["status"] == STATUS_UNMEASURED


# ── The stamp, exercised as deploy.sh actually runs it ───────────────────────


def _extract_stamp_snippet() -> str:
    """Return the `python3 -c '...'` body `run_canary_action` stamps with.

    Runs the REAL snippet rather than a transcription of it, so an edit to
    deploy.sh that breaks the stamp fails here instead of on the next deploy.
    """
    text = DEPLOY_SH.read_text()
    match = re.search(r"payload=\$\(python3 -c '\n(.*?)\n' \"\$payload\"\)", text, re.S)
    assert match, "run_canary_action no longer stamps the payload via python3 -c"
    return match.group(1)


def _run_stamp(payload: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _extract_stamp_snippet(), payload],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "payload",
    [
        '{"action": "check_weekly_run_day"}',
        '{"action": "check_lib_pin_drift"}',
        '{"action": "check_pipeline_contract", "dry_run": true}',
        '{"dry_run": true}',
    ],
)
def test_the_stamp_declares_every_canary_payload(payload: str) -> None:
    result = _run_stamp(payload)
    assert result.returncode == 0, result.stderr
    stamped = json.loads(result.stdout)
    assert stamped[INVOCATION_KIND_KEY] == "canary"
    # Everything the call site declared survives untouched.
    for key, value in json.loads(payload).items():
        assert stamped[key] == value


@pytest.mark.parametrize(
    "payload", ["not json", "[]", '"scalar"', '{"invocation_kind": "smoke"}']
)
def test_the_stamp_fails_loud_rather_than_passing_a_payload_through(payload) -> None:
    """No silent pass-through: an unstamped canary is the defect being fixed."""
    result = _run_stamp(payload)
    assert result.returncode != 0, result.stdout


def test_every_canary_call_site_goes_through_the_stamping_helper() -> None:
    """No canary may bypass `run_canary_action` with a bare invoke-canary."""
    body = DEPLOY_SH.read_text()
    invocations = [
        line
        for line in body.splitlines()
        if "krepis.aws invoke-canary" in line and not line.lstrip().startswith("#")
    ]
    assert len(invocations) == 1, (
        "every canary must go through run_canary_action, which stamps "
        f"invocation_kind; found {len(invocations)} invoke-canary call(s): "
        f"{invocations!r}"
    )


# ── I7954's suppression, which drifted off its own call site ─────────────────


def test_probe_suppression_is_keyed_on_the_central_marker_not_only_dry_run() -> None:
    """I7954 claimed `probe=dry_run` covered check_lib_pin_drift's canary.

    It did not: that canary's payload carries no `dry_run`. Both probe call
    sites now read the marker `run_canary_action` stamps, which no payload
    literal can drift out of.
    """
    handler = HANDLER_PY.read_text()
    for call in ("check_lib_pin_drift(", "check_pipeline_contract("):
        index = handler.index(call)
        window = handler[index : index + 200]
        assert "dry_run or is_synthetic_invocation(event)" in window, call
