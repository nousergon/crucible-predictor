"""
Automated drift detection for prediction patterns.

Compares recent inference output against itself (direction clustering,
confidence collapse, alpha degeneration) to detect model degradation before
it affects portfolio performance. Feature-distribution drift (inference-vs-
training) is a SEPARATE layer owned by ``monitoring/feature_drift.py``
(``feature_drift_ks``, config#859), which already runs at daily inference
time — this module deliberately does not duplicate it (config#1853).

Every finding is emitted as a SEVERITY-CLASSIFIED, self-describing alert so an
operator can judge urgency from the alert alone — no S3 forensics required. Each
alert carries: a severity (INFO / WARN / CRITICAL), the metric value vs its
threshold and the distance between them, a TREND (chronic vs acute, derived from
the recent-days window), a plain-language likely-cause, and a recommended
action.

Three of those actions prescribe champion replacement, which runs through
``PredictorTraining`` on ``ne-weekly-freshness-pipeline``. This module therefore
reads ONE external fact beyond the predictions themselves: the most recent
SUCCEEDED cycle of that pipeline, from
``s3://<bucket>/_sf_completion/ne-weekly-freshness-pipeline/{cycle_key}.json``
(per-key GET, no listing, no Step Functions API — see
``_last_successful_weekly_cycle``). When it is stale the alert names the stale
pipeline instead of prescribing a lever that is disconnected
(alpha-engine-config#7536).

Usage:
    python -m monitoring.drift_detector                    # check today
    python -m monitoring.drift_detector --date 2026-04-03  # check specific date
    python -m monitoring.drift_detector --alert            # send SNS on drift
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"

# Thresholds
DIRECTION_CLUSTER_THRESHOLD = 0.80  # >80% same direction = degenerate
ALPHA_MIN_STDEV = 0.001            # Alpha stdev below this = degenerate
CONSECUTIVE_DAYS_THRESHOLD = 3     # Direction clustering must persist N days

# ── Confidence checks ─────────────────────────────────────────────────────────
# ``prediction_confidence`` semantics since 2026-05-12 (PR #143, ROADMAP L1615):
#
#     prediction_confidence = |p_up - 0.5| * 2   ∈ [0, 1],  0 == coin-flip
#
# It is a DISTANCE FROM COIN-FLIP, not a winner-class probability. The previous
# convention was ``max(p_up, p_down)`` ∈ [0.5, 1.0]; the two map linearly via
# ``new = (old - 0.5) * 2``. The absolute floor this module used to carry
# (``CONFIDENCE_MIN_MEAN = 0.45``) was authored against the pre-2026-04-15
# 3-class max-class-probability convention and was never rescaled, so from
# 2026-05-12 it sat above every value the predictor can realistically emit and
# fired CRITICAL every single day (alpha-engine-config#6952, #6850, #5986).
#
# An absolute floor is the wrong instrument regardless of its value: the mean
# distance-from-coin-flip of a calibrated daily-direction book is a property of
# the calibration convention and the market, not of model health — a genuinely
# healthy book sits near 0.1–0.2. What IS a health signal is (a) the batch
# collapsing onto p_up ≡ 0.5 exactly, and (b) today's conviction falling far
# below the book's OWN recent norm. Both are scale-free, so neither silently
# inverts the next time the confidence convention changes.
CONFIDENCE_DEGENERATE_MEAN = 0.01     # mean at/below this ⇒ p_up ≡ 0.5 ⇒ no signal
CONFIDENCE_RELATIVE_DROP = 0.50       # today below this fraction of its own baseline
CONFIDENCE_BASELINE_MIN_DAYS = 10     # trading days of history the baseline needs
CONFIDENCE_BASELINE_MAX_DAYS = 30     # trading days the baseline is computed over
CONFIDENCE_BASELINE_LOOKBACK_DAYS = 45  # calendar days scanned to find them

# ── Champion-replacement availability ─────────────────────────────────────────
# Three of this module's alerts prescribe promoting a challenger or reverting to
# a known-good champion. All three run through PredictorTraining on the weekly
# freshness pipeline, so when that pipeline is down the prescription points at a
# disconnected lever (alpha-engine-config#7536).
WEEKLY_SF_NAME = "ne-weekly-freshness-pipeline"
TRAINING_STALE_CYCLES = 2      # cycles without a SUCCEEDED run ⇒ remedy unavailable
TRAINING_LOOKBACK_CYCLES = 6   # weekly cycles probed before giving up

# ── Severity model ────────────────────────────────────────────────────────────
# A small, explicit ladder so the most urgent finding can set the SNS subject and
# the operator can triage by reading one word. INFO = visible-but-no-action
# (e.g. a check that was skipped); WARN = degraded / standing model-quality
# condition (advisory, no trading halt); CRITICAL = acute regression or a
# degenerate/empty model output that needs a look now.
INFO, WARN, CRITICAL = "INFO", "WARN", "CRITICAL"
_SEVERITY_ORDER = {INFO: 0, WARN: 1, CRITICAL: 2}


def _max_severity(severities: list[str]) -> str | None:
    """Return the highest severity in the list, or None if empty."""
    sevs = [s for s in severities if s in _SEVERITY_ORDER]
    return max(sevs, key=lambda s: _SEVERITY_ORDER[s]) if sevs else None


def _alert(
    *,
    code: str,
    severity: str,
    headline: str,
    detail: str,
    cause: str | None = None,
    action: str | None = None,
    **context,
) -> dict:
    """Build one structured, self-describing alert.

    ``line`` is a single human-readable string that folds severity + detail +
    (optionally) cause + action together — it is what flows into the
    backward-compatible ``alerts`` list[str] and the SNS body, so the alert
    explains its own severity without a second lookup. ``context`` carries the
    raw numbers (value/threshold/trend/…) for programmatic consumers.
    """
    parts = [f"[{severity}] {headline}: {detail}"]
    if cause:
        parts.append(f"Likely cause: {cause}")
    if action:
        parts.append(f"Action: {action}")
    line = " ".join(parts)
    return {
        "code": code,
        "severity": severity,
        "headline": headline,
        "detail": detail,
        "cause": cause,
        "action": action,
        "line": line,
        **context,
    }


def _load_json(s3, bucket: str, key: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def _load_json_maybe_wrapped(s3, bucket: str, key: str) -> dict | None:
    """``_load_json`` plus one unwrap of a JSON-string-inside-JSON body.

    The ``_sf_completion/`` records are written double-encoded — the object body
    is a JSON *string* whose content is the JSON object. Decoding once yields a
    ``str``, and reading ``.get("status")`` off that silently yields nothing
    rather than raising, which is exactly how a freshness check turns into a
    permanent quiet pass. Unwrap explicitly, once, and return None on anything
    that is still not a mapping."""
    data = _load_json(s3, bucket, key)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (ValueError, TypeError):
            return None
    return data if isinstance(data, dict) else None


def _last_successful_weekly_cycle(
    s3, bucket: str, target: date,
) -> tuple[str, int] | None:
    """(cycle_key, cycles_ago) of the most recent SUCCEEDED weekly-freshness run,
    or None if none of the recent cycles succeeded.

    Champion replacement runs through ``PredictorTraining`` on
    ``ne-weekly-freshness-pipeline``, so "can the prescribed remedy run today"
    reduces to "has that pipeline completed recently". The pipeline stamps a
    per-cycle completion record keyed by its Saturday ``cycle_key``; this reads
    those by constructed key — the same GET-and-tolerate-404 shape the
    predictions window above uses — rather than listing the prefix or querying
    Step Functions directly. That choice is deliberate: the executions API would
    couple a monitoring Lambda to Step Functions IAM for a fact already sitting
    in the bucket it reads, and a per-key GET needs no new permission at all.

    ``cycles_ago`` counts weekly cycles, not days, because staleness here is a
    property of the pipeline's own cadence: one missed Saturday is a miss, two
    is a standing outage.
    """
    # Most recent Saturday at or before the target (weekday(): Mon=0 … Sat=5).
    last_saturday = target - timedelta(days=(target.weekday() - 5) % 7)
    for cycles_ago in range(TRAINING_LOOKBACK_CYCLES):
        cycle = last_saturday - timedelta(weeks=cycles_ago)
        rec = _load_json_maybe_wrapped(
            s3, bucket,
            f"_sf_completion/{WEEKLY_SF_NAME}/{cycle.isoformat()}.json",
        )
        if rec and rec.get("status") == "SUCCEEDED":
            return cycle.isoformat(), cycles_ago
    return None


def _champion_remedy(
    training: tuple[str, int] | None, checked_cycles: int,
) -> tuple[str, bool]:
    """(action text, remedy_available) for an alert prescribing champion replacement.

    When the pipeline that performs champion replacement has not completed
    recently, the alert names THAT instead of prescribing a lever that is
    disconnected. An advisory whose remedy cannot run is worse than one with no
    remedy: it converts operator attention into wasted motion and reads as
    though someone checked. Live during the 2026-08-08..08-12 outage
    (alpha-engine-config#6949), when the weekly SF failed 11 consecutive times
    while the drift alert kept pointing at the challenger pipeline.
    """
    if training is not None and training[1] < TRAINING_STALE_CYCLES:
        return (
            "resolve via champion replacement (the challenger pipeline), not an "
            "inference hotfix; advisory — does not halt trading"
        ), True
    if training is None:
        seen = (f"no SUCCEEDED cycle in the last {checked_cycles} weekly cycles")
    else:
        seen = f"last SUCCEEDED cycle was {training[0]}, {training[1]} cycles ago"
    return (
        f"do NOT reach for champion replacement yet — {WEEKLY_SF_NAME} is stale "
        f"({seen}), and champion promotion runs through its PredictorTraining "
        f"stage. Fix the pipeline first; this alert cannot be resolved by the "
        f"remedy it would otherwise prescribe."
    ), False


def _daily_confidence_means(recent_preds: list[dict]) -> list[tuple[str, float]]:
    """Per-day mean ``prediction_confidence`` over the recent-days window
    (most-recent-first), skipping days with no confidence values."""
    out: list[tuple[str, float]] = []
    for day in recent_preds:
        cs = [p.get("prediction_confidence") for p in day["predictions"]
              if p.get("prediction_confidence") is not None]
        if cs:
            out.append((day["date"], float(np.mean(cs))))
    return out


def check_prediction_drift(
    s3, bucket: str, date_str: str, skipped: list[dict] | None = None,
) -> list[dict]:
    """Check prediction distribution patterns for degenerate model behavior.

    Returns a list of structured alert dicts (see ``_alert``). When ``skipped``
    is supplied, checks that could not RUN (as distinct from checks that ran and
    found nothing) append a record to it — a check with no verdict is reported
    as absent rather than silently counted as healthy."""
    alerts: list[dict] = []

    # Load recent predictions. The clustering + degeneracy checks read the most
    # recent 5 trading days; the confidence baseline reads the whole window
    # (``CONFIDENCE_BASELINE_MAX_DAYS`` trading days) so today's conviction is
    # judged against the book's own recent norm rather than a fixed constant.
    target = date.fromisoformat(date_str)
    window_preds: list[dict] = []
    for offset in range(CONFIDENCE_BASELINE_LOOKBACK_DAYS):
        d = (target - timedelta(days=offset)).isoformat()
        data = _load_json(s3, bucket, f"predictor/predictions/{d}.json")
        if data and "predictions" in data:
            preds = data["predictions"]
            window_preds.append({"date": d, "predictions": preds})
        if len(window_preds) >= CONFIDENCE_BASELINE_MAX_DAYS:
            break
    recent_preds = window_preds[:5]

    # Resolved once: is the pipeline that performs champion replacement up?
    # Every alert below that prescribes promoting a challenger consults this,
    # so a single outage cannot leave one alert honest and two prescribing into
    # it (alpha-engine-config#7536).
    _training = _last_successful_weekly_cycle(s3, bucket, target)
    champion_action, champion_remedy_available = _champion_remedy(
        _training, TRAINING_LOOKBACK_CYCLES,
    )
    training_ctx = {
        "champion_remedy_available": champion_remedy_available,
        "last_successful_training_cycle": _training[0] if _training else None,
    }

    if not recent_preds:
        alerts.append(_alert(
            code="no_recent_predictions",
            severity=CRITICAL,
            headline="No recent predictions found",
            detail=f"no predictions JSON in the 7 days ending {date_str} — the predictor produced nothing",
            cause="inference Lambda did not write predictions (deploy/coverage/staleness failure)",
            action="check the daily PredictorInference run; the executor is running blind without predictions",
            date=date_str,
        ))
        return alerts

    # Check today's predictions
    today = recent_preds[0]["predictions"]
    if not today:
        alerts.append(_alert(
            code="today_predictions_empty",
            severity=CRITICAL,
            headline="Today's predictions are empty",
            detail=f"the predictions file for {recent_preds[0]['date']} contains zero predictions",
            cause="inference produced an empty set (universe/coverage collapse)",
            action="check the latest PredictorInference run; downstream sizing has no signal",
            date=date_str,
        ))
        return alerts

    # Direction clustering (single day). Filter None — missing
    # predicted_direction shouldn't count as its own cluster.
    directions = [d for d in (p.get("predicted_direction") for p in today) if d]
    if directions:
        from collections import Counter
        counts = Counter(directions)
        dominant_pct = counts.most_common(1)[0][1] / len(directions)
        if dominant_pct > DIRECTION_CLUSTER_THRESHOLD:
            dominant = counts.most_common(1)[0][0]
            alerts.append(_alert(
                code="direction_clustering",
                severity=WARN,
                headline="Direction clustering",
                detail=(f"{dominant_pct:.0%} of names predict {dominant} today "
                        f"(>{DIRECTION_CLUSTER_THRESHOLD:.0%} same-direction floor)"),
                cause="the model has little cross-sectional discrimination today — it is "
                      "taking a near-uniform directional view instead of ranking names",
                action="tolerable as a one-off (a strong market day); watch for the persistent "
                       "variant below, which indicates a genuinely degenerate model",
                value=round(dominant_pct, 3),
                threshold=DIRECTION_CLUSTER_THRESHOLD,
                dominant_direction=dominant,
                date=date_str,
            ))

    # Check if clustering persists across multiple days
    if len(recent_preds) >= CONSECUTIVE_DAYS_THRESHOLD:
        consecutive_clustered = 0
        for day_data in recent_preds[:CONSECUTIVE_DAYS_THRESHOLD]:
            day_dirs = [d for d in (p.get("predicted_direction") for p in day_data["predictions"]) if d]
            if day_dirs:
                day_counts: dict[str, int] = {}
                for d in day_dirs:
                    day_counts[d] = day_counts.get(d, 0) + 1
                day_dominant_pct = max(day_counts.values()) / len(day_dirs)
                if day_dominant_pct > DIRECTION_CLUSTER_THRESHOLD:
                    consecutive_clustered += 1
        if consecutive_clustered >= CONSECUTIVE_DAYS_THRESHOLD:
            alerts.append(_alert(
                code="persistent_direction_clustering",
                severity=CRITICAL,
                headline="PERSISTENT direction clustering",
                detail=(f"{consecutive_clustered} consecutive days above the "
                        f"{DIRECTION_CLUSTER_THRESHOLD:.0%} same-direction floor"),
                cause="the model has collapsed to a one-directional view — it has lost "
                      "cross-sectional signal, so its rankings carry no usable alpha",
                action=(("treat the model as degenerate; " + champion_action)
                        if champion_remedy_available else champion_action),
                consecutive_days=consecutive_clustered,
                date=date_str,
                **training_ctx,
            ))

    # ── Confidence checks ────────────────────────────────────────────────────
    # Two checks, both scale-free (see the CONFIDENCE_* block at module top for
    # why the retired absolute floor was neither):
    #   1. DEGENERATE — the batch mean sits at coin-flip, i.e. p_up ≡ 0.5 and
    #      there is no directional signal to size on at all. Absolute, but it is
    #      an identity (0 == no edge) rather than a tuned level, so it stays
    #      correct under any monotone re-parameterisation of confidence.
    #   2. RELATIVE DROP — today's mean is far below the book's own trailing
    #      median. This is what "the model got worse" actually looks like.
    confidences = [p["prediction_confidence"] for p in today
                   if p.get("prediction_confidence") is not None]
    if confidences:
        mean_conf = float(np.mean(confidences))
        daily = _daily_confidence_means(recent_preds)
        baseline_daily = _daily_confidence_means(window_preds)
        # Baseline excludes today so a collapsed today cannot lower the bar it
        # is being judged against.
        prior_means = [m for _, m in baseline_daily[1:]]

        if mean_conf <= CONFIDENCE_DEGENERATE_MEAN:
            alerts.append(_alert(
                code="confidence_degenerate",
                severity=CRITICAL,
                headline="Confidence degenerate",
                detail=(f"mean prediction confidence {mean_conf:.4f} is at coin-flip "
                        f"(<= {CONFIDENCE_DEGENERATE_MEAN}) — confidence is |p_up-0.5|*2, so "
                        f"the calibrator is mapping essentially every name to p_up = 0.5"),
                cause="the calibrator or the meta-model has collapsed — the batch carries no "
                      "directional edge, so nothing downstream can rank or size on it",
                action="check the served calibrator + meta-model artifacts for today's run; this "
                       "is an inference-side failure, not a slow model-quality decay",
                value=round(mean_conf, 4),
                threshold=CONFIDENCE_DEGENERATE_MEAN,
                date=date_str,
            ))
        elif len(prior_means) < CONFIDENCE_BASELINE_MIN_DAYS:
            # Not enough history to say whether today is low FOR THIS BOOK. Record
            # the absence rather than passing silently — an unrun check is not a
            # clean check.
            if skipped is not None:
                skipped.append({
                    "check": "confidence_relative_drop",
                    "severity": INFO,
                    "reason": (f"{len(prior_means)} prior trading days of predictions in the "
                               f"{CONFIDENCE_BASELINE_LOOKBACK_DAYS}-day lookback; the baseline "
                               f"needs {CONFIDENCE_BASELINE_MIN_DAYS}"),
                    "value": round(mean_conf, 4),
                    "date": date_str,
                })
        else:
            baseline = float(np.median(prior_means))
            bar = baseline * CONFIDENCE_RELATIVE_DROP
            if baseline > 0 and mean_conf < bar:
                deficit = (baseline - mean_conf) / baseline
                n_days = len(daily)
                days_below = sum(1 for _, m in daily if m < bar)
                lo = min((m for _, m in daily), default=mean_conf)
                hi = max((m for _, m in daily), default=mean_conf)

                # chronic = the recent window has been below the bar for days;
                # acute = only today dropped while the prior days held up.
                if n_days >= CONSECUTIVE_DAYS_THRESHOLD and days_below >= CONSECUTIVE_DAYS_THRESHOLD:
                    trend, trend_word = "chronic", "CHRONIC"
                elif n_days >= 2 and days_below == 1 and daily[0][1] < bar:
                    trend, trend_word = "acute", "ACUTE"
                else:
                    trend, trend_word = "indeterminate", "trend-indeterminate"

                severity = CRITICAL if trend == "acute" else WARN

                trend_detail = (f"below the bar {days_below}/{n_days} recent days "
                                f"(range {lo:.3f}–{hi:.3f})") if n_days else "no recent history"
                if trend == "chronic":
                    cause = ("conviction has been sitting well under this book's own recent norm "
                             "for days — a standing model-quality condition, NOT a same-day "
                             "regression")
                    action = champion_action
                elif trend == "acute":
                    cause = ("a sudden drop from a healthy recent baseline — points at today's "
                             "served-model version or feature inputs, not a slow decay")
                    action = ("check today's served model + inference feature inputs for a regression")
                else:
                    cause = "conviction below the trailing baseline; too little recent history to "\
                            "call chronic vs acute"
                    action = "watch the next few days to classify; check served-model health"

                alerts.append(_alert(
                    code="confidence_collapse",
                    severity=severity,
                    headline="Confidence collapse",
                    detail=(f"mean prediction confidence {mean_conf:.3f} is {deficit:.0%} below "
                            f"this book's own {len(prior_means)}-day median of {baseline:.3f} "
                            f"(bar {bar:.3f} = {CONFIDENCE_RELATIVE_DROP:.0%} of it) — "
                            f"{trend_word} ({trend_detail})"),
                    cause=cause,
                    action=action,
                    value=round(mean_conf, 3),
                    threshold=round(bar, 3),
                    baseline=round(baseline, 3),
                    baseline_days=len(prior_means),
                    pct_below_threshold=round(deficit, 3),
                    trend=trend,
                    recent_daily_means=[round(m, 3) for _, m in daily],
                    date=date_str,
                    **training_ctx,
                ))

    # Alpha degeneration — predictions nearly constant ⇒ no cross-sectional signal.
    alphas = [p.get("predicted_alpha", 0.0) for p in today
              if p.get("predicted_alpha") is not None]
    if alphas:
        alpha_std = float(np.std(alphas))
        if alpha_std < ALPHA_MIN_STDEV:
            alerts.append(_alert(
                code="alpha_degeneration",
                severity=CRITICAL,
                headline="Alpha degeneration",
                detail=(f"predicted-alpha stdev {alpha_std:.6f} is below the {ALPHA_MIN_STDEV} floor "
                        f"— predictions are nearly constant across the universe"),
                cause="the model emits a near-identical alpha for every name — there is no "
                      "cross-sectional ranking signal left to size on",
                action=(("treat as degenerate; investigate the meta-model output, then "
                         + champion_action)
                        if champion_remedy_available else champion_action),
                value=round(alpha_std, 6),
                threshold=ALPHA_MIN_STDEV,
                date=date_str,
                **training_ctx,
            ))

    return alerts


def check_drift(
    bucket: str = DEFAULT_BUCKET,
    date_str: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run all drift checks. Returns a structured, severity-aware result.

    ``dry_run=True`` skips the S3 write of ``drift_{date}.json`` — added so
    the deploy-time canary (``infrastructure/deploy.sh``, action=check_drift)
    can exercise this action's full read path without overwriting the real
    EOD SF's artifact for the current trading day on every deploy (the
    canary invokes a freshly-published, not-yet-live version; it has no
    business mutating production state). The genuine EOD SF invocation never
    passes dry_run, so its write behavior is unchanged (config#3025 dim8).

    Prediction-drift only (config#1853) — feature-distribution drift is a
    separate layer owned by ``monitoring/feature_drift.py``'s
    ``feature_drift_ks``, which already runs at daily inference time
    (config#859); duplicating it here would be redundant, not defense in
    depth.

    Backward-compatible: ``alerts`` remains a ``list[str]`` (each a rich,
    self-describing line). New additive fields: ``severity`` (overall max, or
    None when clean), ``alert_details`` (the structured dicts), and
    ``skipped_checks`` — checks that could not RUN today (currently only the
    confidence relative-drop check, which needs a trailing baseline before it
    has a verdict). A skipped check does NOT raise ``status`` to ``alert`` or
    contribute to ``severity``: it is not a finding. It is recorded so an
    absent verdict reads as absent rather than as a clean bill of health.

    ``date_str=None`` defaults to ``last_closed_trading_day()`` (NYSE-aware,
    NOT ``date.today()``) — alpha-engine-config-I2722 (2026-07-16): this
    handler is being re-homed off ``ne-preopen-trading-pipeline`` onto its own
    direct EventBridge trigger, which can no longer thread the SF's own
    ``$.trading_day_gate.Payload.check_date`` through the Payload, so a
    missing ``date`` must resolve correctly on its own. ``date.today()`` is a
    bare calendar date — on a weekend/holiday/pre-close weekday it silently
    scores drift against a day with no predictions yet (or the wrong day
    entirely), exactly the class of trade-decision-key bug this repo's Date
    Conventions rule (never ``date.today()`` for trade-decision keys) exists
    to prevent. See ``~/Development/CLAUDE.md`` "Date Conventions" +
    ``inference/stages/load_prices.py`` for the same helper's existing use in
    this repo."""
    import boto3
    from krepis.trading_calendar import last_closed_trading_day

    s3 = boto3.client("s3")

    if date_str is None:
        date_str = last_closed_trading_day().isoformat()

    details: list[dict] = []
    skipped: list[dict] = []
    details.extend(check_prediction_drift(s3, bucket, date_str, skipped))

    overall_severity = _max_severity([d["severity"] for d in details])
    status = "ok" if not details else "alert"
    result = {
        "date": date_str,
        "status": status,
        "severity": overall_severity,            # NEW: overall max (None when clean)
        "alerts": [d["line"] for d in details],  # backward-compatible list[str]
        "alert_details": details,                # NEW: structured per-alert context
        "skipped_checks": skipped,               # NEW: checks that could not run
        "n_alerts": len(details),
    }

    # Write results to S3 (skipped in dry_run — see docstring)
    if not dry_run:
        try:
            s3.put_object(
                Bucket=bucket,
                Key=f"predictor/metrics/drift_{date_str}.json",
                Body=json.dumps(result, indent=2).encode(),
                ContentType="application/json",
            )
        except Exception as e:
            logger.warning("Drift results S3 write failed: %s", e)

    if details:
        for d in details:
            log_fn = logger.error if d["severity"] == CRITICAL else logger.warning
            log_fn("DRIFT ALERT %s", d["line"])
    else:
        logger.info("No drift detected for %s", date_str)

    return result


def format_alert_report(result: dict) -> str:
    """Render a human-readable, severity-led report for SNS / email / stdout.

    Leads with the overall severity and a per-severity count so urgency is the
    first thing read, then each alert as a labeled block, then any skipped
    checks. This is what makes the alert self-explanatory."""
    details = result.get("alert_details", [])
    lines: list[str] = []
    overall = result.get("severity") or "none"
    counts = {CRITICAL: 0, WARN: 0, INFO: 0}
    for d in details:
        counts[d.get("severity", WARN)] = counts.get(d.get("severity", WARN), 0) + 1
    lines.append(f"SEVERITY: {overall}")
    lines.append(f"Date: {result.get('date')}")
    lines.append(
        f"Predictor drift check — {len(details)} alert(s) "
        f"({counts[CRITICAL]} CRITICAL, {counts[WARN]} WARN)"
    )
    lines.append("")
    for d in details:
        lines.append(f"[{d['severity']}] {d['headline']}")
        lines.append(f"  • {d['detail']}")
        if d.get("cause"):
            lines.append(f"  • Likely cause: {d['cause']}")
        if d.get("action"):
            lines.append(f"  • Action: {d['action']}")
        lines.append("")
    for s in result.get("skipped_checks", []):
        lines.append(f"[{s['severity']}] Skipped check — {s['check']}: {s['reason']}")
    return "\n".join(lines).rstrip()


def main():
    parser = argparse.ArgumentParser(description="Check feature and prediction drift")
    parser.add_argument("--date", default=None, help="Date to check (YYYY-MM-DD)")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--alert", action="store_true", help="Send SNS alert on drift")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = check_drift(bucket=args.bucket, date_str=args.date)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["alerts"]:
            print(format_alert_report(result))
        else:
            print(f"No drift detected for {result['date']}")
            for s in result.get("skipped_checks", []):
                print(f"  (skipped: {s['check']} — {s['reason']})")

    if result["alerts"] and args.alert:
        try:
            import boto3
            sns = boto3.client("sns", region_name="us-east-1")
            topic_arn = os.environ.get("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts")
            # Severity-led subject so the inbox conveys urgency + the headline
            # without opening the mail.
            headlines = "; ".join(dict.fromkeys(d["headline"] for d in result["alert_details"]))
            subject = f"Alpha Engine — Drift [{result.get('severity')}]: {headlines}"
            sns.publish(
                TopicArn=topic_arn,
                Subject=subject[:100],  # SNS subject hard limit
                Message=format_alert_report(result),
            )
        except Exception as e:
            logger.warning("SNS alert failed: %s", e)

    # Exit 1 on any drift, 0 when clean — unchanged contract for the SF
    # wrapper (which collapses all non-zero to one path). Severity is conveyed
    # in the alert content / SNS subject / drift_{date}.json `severity`, not the
    # exit code, so no unconsumed cross-repo coupling is introduced.
    sys.exit(1 if result["alerts"] else 0)


if __name__ == "__main__":
    main()
