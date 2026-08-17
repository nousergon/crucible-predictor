"""Tests for monitoring.drift_detector — prediction drift checks.

Prediction-drift only (config#1853): feature-distribution drift is a separate
layer owned by ``monitoring/feature_drift.py`` (``feature_drift_ks``,
config#859), which already runs at daily inference time — this module
deliberately does not duplicate it.

check_prediction_drift returns a list of STRUCTURED alert dicts
(``{code, severity, headline, detail, cause, action, line, ...}``) so each
alert is self-describing (severity + distance-from-threshold + trend + cause +
action). ``check_drift`` keeps ``alerts`` as a backward-compatible list[str]
(the rendered ``line`` of each) and adds ``severity``, ``alert_details`` and
``skipped_checks`` (checks that could not RUN — currently only the confidence
relative-drop check before its trailing baseline exists; never a finding, and
never raises ``status`` to ``alert``).
"""

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from monitoring.drift_detector import (
    ALPHA_MIN_STDEV,
    CONFIDENCE_BASELINE_MIN_DAYS,
    CONFIDENCE_DEGENERATE_MEAN,
    CONFIDENCE_RELATIVE_DROP,
    CONSECUTIVE_DAYS_THRESHOLD,
    CRITICAL,
    DIRECTION_CLUSTER_THRESHOLD,
    INFO,
    WARN,
    _load_json,
    _max_severity,
    check_drift,
    check_prediction_drift,
    format_alert_report,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _lines(alerts):
    """Rendered ``line`` of each structured alert (what flows to alerts/SNS)."""
    return [a["line"] for a in alerts]


def _codes(alerts):
    return [a["code"] for a in alerts]


# ── S3 helpers ──────────────────────────────────────────────────────────────


def _s3_with_json(payload):
    """An S3 client that returns json bytes for any key."""
    s3 = MagicMock()
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode()
    s3.get_object.return_value = {"Body": body}
    return s3


def _s3_with_routes(routes):
    """routes: dict[key, "json"|"missing", payload]."""
    s3 = MagicMock()

    def get_object(*, Bucket, Key):
        if Key not in routes:
            raise RuntimeError(f"NoSuchKey: {Key}")
        kind, payload = routes[Key]
        body = MagicMock()
        if kind == "json":
            body.read.return_value = json.dumps(payload).encode()
        else:
            raise RuntimeError(f"Unknown kind {kind}")
        return {"Body": body}

    s3.get_object.side_effect = get_object
    return s3


def test_load_json_success():
    s3 = _s3_with_json({"hello": "world"})
    assert _load_json(s3, "bucket", "key") == {"hello": "world"}


def test_load_json_failure_returns_none():
    s3 = MagicMock()
    s3.get_object.side_effect = RuntimeError("NoSuchKey")
    assert _load_json(s3, "bucket", "key") is None


# ── severity helper ─────────────────────────────────────────────────────────


def test_max_severity_orders_correctly():
    assert _max_severity([INFO, WARN, CRITICAL]) == CRITICAL
    assert _max_severity([INFO, WARN]) == WARN
    assert _max_severity([INFO]) == INFO
    assert _max_severity([]) is None


# ── check_prediction_drift ─────────────────────────────────────────────────


def _make_preds(directions=None, confidences=None, alphas=None, n=20):
    """Build a predictions JSON payload.

    Confidence values are on the LIVE axis — ``|p_up - 0.5| * 2`` ∈ [0, 1], zero
    at coin-flip (PR #143). A healthy daily book sits around 0.10–0.20; fixtures
    in the old ``max(p_up, p_down)`` ∈ [0.5, 1.0] range describe a predictor that
    has not existed since 2026-05-12.
    """
    if directions is None:
        directions = ["UP"] * n
    if confidences is None:
        confidences = [0.15] * n
    if alphas is None:
        alphas = [0.05] * n
    preds = []
    for i, (d, c, a) in enumerate(zip(directions, confidences, alphas)):
        preds.append({
            "ticker": f"T{i}",
            "predicted_direction": d,
            "prediction_confidence": c,
            "predicted_alpha": a,
        })
    return {"predictions": preds}


def test_check_prediction_drift_no_recent_alerts():
    s3 = MagicMock()
    s3.get_object.side_effect = RuntimeError("missing")
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert _codes(alerts) == ["no_recent_predictions"]
    assert alerts[0]["severity"] == CRITICAL


def test_check_prediction_drift_empty_today_alerts():
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", {"predictions": []}),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert _codes(alerts) == ["today_predictions_empty"]
    assert alerts[0]["severity"] == CRITICAL
    assert any("empty" in ln.lower() for ln in _lines(alerts))


def test_check_prediction_drift_flags_single_day_clustering():
    """90% UP on today only → single-day cluster alert (WARN)."""
    preds = _make_preds(directions=["UP"] * 18 + ["DOWN"] * 2)
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    cluster = [a for a in alerts if a["code"] == "direction_clustering"]
    assert cluster and cluster[0]["dominant_direction"] == "UP"
    assert cluster[0]["severity"] == WARN
    # Single day → no persistent cluster
    assert "persistent_direction_clustering" not in _codes(alerts)


def test_check_prediction_drift_persistent_clustering_alert():
    """N consecutive trading days all clustered → PERSISTENT alert (CRITICAL)."""
    clustered_preds = _make_preds(directions=["UP"] * 19 + ["DOWN"] * 1)

    routes = {}
    days = ["2026-04-15", "2026-04-14", "2026-04-13", "2026-04-12", "2026-04-11"]
    for d in days:
        routes[f"predictor/predictions/{d}.json"] = ("json", clustered_preds)

    s3 = _s3_with_routes(routes)
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    persistent = [a for a in alerts if a["code"] == "persistent_direction_clustering"]
    assert persistent and persistent[0]["severity"] == CRITICAL
    assert str(CONSECUTIVE_DAYS_THRESHOLD) in persistent[0]["line"]


_MIXED_DIRECTIONS = ["UP", "DOWN"] * 11  # diversified so clustering never fires


def _conf_routes(target: str, conf_by_offset, n_days: int = 25):
    """Routes for ``n_days`` consecutive calendar days back from ``target``.

    ``conf_by_offset(offset) -> float`` gives that day's per-ticker confidence.
    Consecutive calendar days (weekends included) keep the fixture readable —
    the detector only cares that a key exists.
    """
    routes = {}
    t = date.fromisoformat(target)
    for offset in range(n_days):
        d = (t - timedelta(days=offset)).isoformat()
        routes[f"predictor/predictions/{d}.json"] = ("json", _make_preds(
            directions=_MIXED_DIRECTIONS,
            confidences=[conf_by_offset(offset)] * 21,
            n=21,
        ))
    return routes


def test_low_but_stable_confidence_is_not_an_alert():
    """The regression pin for alpha-engine-config#6952.

    A live book sits near 0.11 mean confidence — that is ``|p_up - 0.5| * 2``
    with p_up ≈ 0.555, an ordinary calibrated daily-direction edge, not a
    collapse. The retired absolute floor (0.45, authored against the pre-2026-
    04-15 3-class max-class-probability convention) rated exactly this batch
    CRITICAL every trading day from 2026-05-12 onward. A detector must not fire
    on the system's normal operating point.
    """
    s3 = _s3_with_routes(_conf_routes("2026-04-15", lambda _o: 0.11))
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert [a for a in alerts if a["code"].startswith("confidence")] == []


def test_confidence_collapse_chronic_is_warn():
    """Well under the book's own baseline for days → CHRONIC, WARN."""
    # Baseline days at 0.20; the recent 5 at 0.05 — far below the 50%-of-median bar.
    s3 = _s3_with_routes(_conf_routes(
        "2026-04-15", lambda o: 0.05 if o < 5 else 0.20))
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    cc = [a for a in alerts if a["code"] == "confidence_collapse"]
    assert cc, "expected a confidence_collapse alert"
    assert cc[0]["trend"] == "chronic"
    assert cc[0]["severity"] == WARN
    assert "CHRONIC" in cc[0]["line"]
    assert cc[0]["baseline"] == pytest.approx(0.20, abs=0.005)
    assert cc[0]["threshold"] == pytest.approx(0.20 * CONFIDENCE_RELATIVE_DROP, abs=0.005)
    assert cc[0]["pct_below_threshold"] == pytest.approx((0.20 - 0.05) / 0.20, abs=0.01)


def test_confidence_collapse_acute_is_critical():
    """Healthy prior days, only today drops → ACUTE, CRITICAL."""
    s3 = _s3_with_routes(_conf_routes(
        "2026-04-15", lambda o: 0.05 if o == 0 else 0.20))
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    cc = [a for a in alerts if a["code"] == "confidence_collapse"]
    assert cc and cc[0]["trend"] == "acute"
    assert cc[0]["severity"] == CRITICAL
    assert "ACUTE" in cc[0]["line"]


def test_confidence_collapse_baseline_excludes_today():
    """Today must not lower the bar it is judged against.

    With a 25-day window a single collapsed day would barely move a median that
    included it, but the exclusion is the property worth pinning: the baseline
    is reported over prior days only.
    """
    s3 = _s3_with_routes(_conf_routes(
        "2026-04-15", lambda o: 0.05 if o == 0 else 0.20))
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    cc = [a for a in alerts if a["code"] == "confidence_collapse"]
    assert cc and cc[0]["baseline"] == pytest.approx(0.20, abs=0.005)
    assert cc[0]["baseline_days"] == 24


def test_confidence_degenerate_is_critical():
    """Mean at coin-flip → p_up ≡ 0.5 → no directional signal at all."""
    s3 = _s3_with_routes(_conf_routes(
        "2026-04-15", lambda o: 0.0 if o == 0 else 0.20))
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    deg = [a for a in alerts if a["code"] == "confidence_degenerate"]
    assert deg and deg[0]["severity"] == CRITICAL
    assert deg[0]["threshold"] == CONFIDENCE_DEGENERATE_MEAN
    # Degenerate supersedes the relative check — one finding, not two.
    assert "confidence_collapse" not in _codes(alerts)


def test_confidence_relative_check_is_recorded_as_skipped_without_a_baseline():
    """Too little history to judge → recorded as unrun, NOT as clean."""
    s3 = _s3_with_routes(_conf_routes("2026-04-15", lambda _o: 0.11, n_days=3))
    skipped: list[dict] = []
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15", skipped)
    assert [a for a in alerts if a["code"].startswith("confidence")] == []
    assert [s["check"] for s in skipped] == ["confidence_relative_drop"]
    assert str(CONFIDENCE_BASELINE_MIN_DAYS) in skipped[0]["reason"]


def test_check_prediction_drift_flags_alpha_degeneration():
    preds = _make_preds(
        directions=_MIXED_DIRECTIONS,
        confidences=[0.15] * 21,
        alphas=[0.05] * 21,  # zero stdev
    )
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    deg = [a for a in alerts if a["code"] == "alpha_degeneration"]
    assert deg and deg[0]["severity"] == CRITICAL


def test_check_prediction_drift_clean_no_alerts():
    rng = np.random.default_rng(7)
    preds = _make_preds(
        directions=list(rng.choice(["UP", "DOWN"], 30)),
        confidences=list(rng.uniform(0.05, 0.25, 30)),
        alphas=list(rng.normal(0.0, 0.02, 30)),
        n=30,
    )
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert alerts == []


def test_check_prediction_drift_ignores_none_directions():
    """A prediction with predicted_direction=None should NOT contribute to clustering counts."""
    payload = {"predictions": [
        {"ticker": f"T{i}", "predicted_direction": None,
         "prediction_confidence": 0.15, "predicted_alpha": 0.05}
        for i in range(20)
    ]}
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", payload),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert "direction_clustering" not in _codes(alerts)


# ── check_drift (top-level orchestrator) ───────────────────────────────────


def test_check_drift_ok_when_no_alerts():
    rng = np.random.default_rng(11)
    preds = _make_preds(
        directions=list(rng.choice(["UP", "DOWN"], 30)),
        confidences=list(rng.uniform(0.05, 0.25, 30)),
        alphas=list(rng.normal(0.0, 0.02, 30)),
        n=30,
    )
    fake_s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    fake_s3.put_object = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15")

    assert result["status"] == "ok"
    assert result["alerts"] == []
    assert result["n_alerts"] == 0
    assert result["severity"] is None
    assert result["date"] == "2026-04-15"
    # One day of predictions is not a baseline, so the relative-drop check has
    # no verdict — recorded as unrun, and NOT counted as a finding: status stays
    # "ok" and severity stays None.
    assert [s["check"] for s in result["skipped_checks"]] == ["confidence_relative_drop"]
    # Result persisted to S3
    fake_s3.put_object.assert_called_once()
    args = fake_s3.put_object.call_args.kwargs
    assert args["Key"] == "predictor/metrics/drift_2026-04-15.json"


def test_check_drift_dry_run_skips_s3_write():
    """dry_run=True must not overwrite the real EOD-SF-produced
    drift_{date}.json — added so the deploy-time canary can exercise this
    action's read path without a production side effect (config#3025 dim8)."""
    rng = np.random.default_rng(11)
    preds = _make_preds(
        directions=list(rng.choice(["UP", "DOWN"], 30)),
        confidences=list(rng.uniform(0.05, 0.25, 30)),
        alphas=list(rng.normal(0.0, 0.02, 30)),
        n=30,
    )
    fake_s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    fake_s3.put_object = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15", dry_run=True)

    assert result["status"] == "ok"
    fake_s3.put_object.assert_not_called()


def test_check_drift_alert_status_and_severity():
    preds = _make_preds(directions=["UP"] * 20)  # 100% clustering → WARN single-day
    fake_s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    fake_s3.put_object = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15")

    assert result["status"] == "alert"
    assert result["n_alerts"] >= 1
    assert result["severity"] in (WARN, CRITICAL)
    assert any("Direction clustering" in a for a in result["alerts"])
    # alerts stays a list[str]; structured detail is additive
    assert all(isinstance(a, str) for a in result["alerts"])
    assert all("severity" in d for d in result["alert_details"])


def test_check_drift_severity_is_max_across_alerts():
    """A CRITICAL alpha-degeneration outranks a WARN clustering → overall CRITICAL."""
    preds = _make_preds(directions=["UP"] * 20, alphas=[0.05] * 20)  # cluster WARN + alpha CRITICAL
    fake_s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    fake_s3.put_object = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15")
    assert result["severity"] == CRITICAL


def test_check_drift_swallows_put_object_failure():
    fake_s3 = _s3_with_routes({})
    fake_s3.put_object = MagicMock(side_effect=RuntimeError("S3 down"))
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15")
    assert result["status"] == "alert"  # missing preds → "No recent predictions"


def test_check_drift_missing_date_defaults_to_last_closed_trading_day():
    """alpha-engine-config-I2722 (2026-07-16): check_drift is being re-homed
    onto a direct EventBridge trigger that can no longer thread
    $.trading_day_gate.Payload.check_date through the Payload — a missing
    date_str must resolve via last_closed_trading_day() (NYSE-aware), NOT
    date.today() (a bare calendar date is not a trade-decision-key axis; see
    this repo's Date Conventions rule). Uses a Sunday reference so today() and
    last_closed_trading_day() provably diverge — Sunday's last closed trading
    day is the preceding Friday."""
    fake_s3 = _s3_with_routes({
        "predictor/predictions/2026-04-17.json": ("json", _make_preds(n=5)),
    })
    fake_s3.put_object = MagicMock()
    with patch("boto3.client", return_value=fake_s3), \
         patch("krepis.trading_calendar.last_closed_trading_day",
               return_value=date(2026, 4, 17)) as mock_lctd:
        result = check_drift(bucket="bucket", date_str=None)

    mock_lctd.assert_called_once_with()
    assert result["date"] == "2026-04-17"
    assert "2026-04-19" not in result["date"], (
        "must not fall back to date.today() — that is the trade-decision-key "
        "bug class this fix closes"
    )


def test_format_alert_report_is_severity_led():
    preds = _make_preds(directions=["UP"] * 20, alphas=[0.05] * 20)
    fake_s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    fake_s3.put_object = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15")
    report = format_alert_report(result)
    assert report.startswith("SEVERITY: ")
    assert "CRITICAL" in report
    # each alert renders its labeled block
    assert "Likely cause:" in report
    assert "Action:" in report
