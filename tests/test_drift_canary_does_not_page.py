"""alpha-engine-config-I10281 — the deploy canary must not raise an OPERATOR alert.

Measured 2026-09-08: six deploys of `alpha-engine-predictor-inference` each
invoked `action=check_drift` with `dry_run=True` (the canary in
`infrastructure/deploy.sh`) against a freshly-published, not-yet-live version.
Every one logged a byte-identical

    DRIFT ALERT [CRITICAL] Confidence collapse: mean prediction confidence
    0.059 is 68% below this book's own 28-day median of 0.184 ...

for the SAME 2026-09-08 condition, and four of them reached Brian as separate
pages (flow-doctor report ids 01a082e8…, 01a08320…, 01a083ca…, 01a08406…).

`DRIFT ALERT` is the operator-facing token a log-pattern subscriber pages on.
A synthetic probe of a not-yet-live artifact observes no production run, so it
must not emit it — which is what every other canary stage already declares in
its own log line (`invocation_kind='canary'` is synthetic, -I8155).

The structured result must be IDENTICAL in both modes: the canary's whole job
is to exercise this action's read path, and `infrastructure/deploy.sh` asserts
on the shape of what comes back.
"""
import logging

import pytest

from monitoring import drift_detector as dd


@pytest.fixture()
def _one_critical_finding(monkeypatch):
    """One CRITICAL finding, so the only variable under test is dry_run."""
    finding = dd._alert(
        code="confidence_collapse",
        severity=dd.CRITICAL,
        headline="Confidence collapse",
        detail="mean prediction confidence 0.059 is 68% below the 28-day median",
        cause="the served champion changed",
        action="check the champion",
        date="2026-09-08",
    )
    monkeypatch.setattr(dd, "check_prediction_drift",
                        lambda *a, **k: [dict(finding)])

    class _S3:
        def __init__(self):
            self.puts = []

        def put_object(self, **kw):
            self.puts.append(kw)

    s3 = _S3()

    class _Boto:
        @staticmethod
        def client(_name):
            return s3

    monkeypatch.setitem(dd.__dict__, "_TEST_S3", s3)
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)
    return s3


def _run(caplog, *, dry_run):
    caplog.set_level(logging.INFO, logger=dd.logger.name)
    with caplog.at_level(logging.INFO, logger=dd.logger.name):
        result = dd.check_drift(bucket="b", date_str="2026-09-08", dry_run=dry_run)
    return result, "\n".join(r.getMessage() for r in caplog.records)


def test_the_canary_does_not_emit_the_operator_alert_token(caplog, _one_critical_finding):
    """dry_run=True is the deploy canary — no `DRIFT ALERT`, at any level.

    This is the assertion that fails on the pre-fix code: it emitted the token
    unconditionally, once per deploy.
    """
    result, logged = _run(caplog, dry_run=True)
    assert "DRIFT ALERT" not in logged, (
        "the deploy canary paged an operator from a synthetic probe of a "
        "not-yet-live version — this is the 2026-09-08 four-pages defect"
    )
    assert "canary" in logged.lower()
    # The finding itself is still visible for anyone reading the deploy log.
    assert "Confidence collapse" in logged
    assert result["severity"] == dd.CRITICAL


def test_the_real_eod_run_still_pages(caplog, _one_critical_finding):
    """The genuine EOD Step Functions invocation never passes dry_run, and its
    operator-facing behaviour must be untouched — suppressing the canary must
    not suppress the alert that matters."""
    _, logged = _run(caplog, dry_run=False)
    assert "DRIFT ALERT" in logged
    assert "Confidence collapse" in logged


def test_the_structured_result_is_identical_in_both_modes(caplog, _one_critical_finding):
    """Only the operator-facing TOKEN is withheld. deploy.sh asserts on this
    shape, so a canary that returned less would break the deploy gate."""
    canary, _ = _run(caplog, dry_run=True)
    caplog.clear()
    live, _ = _run(caplog, dry_run=False)
    assert canary["status"] == live["status"] == "alert"
    assert canary["severity"] == live["severity"]
    assert canary["alerts"] == live["alerts"]
    assert canary["n_alerts"] == live["n_alerts"] == 1
