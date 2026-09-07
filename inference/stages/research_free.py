"""Stage: research_free — daily research-free counterfactual producer
(alpha-engine-config-I10067; module implemented under config#2365).

Runs ``inference/research_free_inference.py::run_research_free_inference``
once per trading day, AFTER ``write_output`` has written the live
``predictor/predictions/{date}.json``. Writes
``predictor/predictions_research_free/{date}.json`` + ``latest.json``.

Why this stage exists
---------------------
``inference/research_free_inference.py`` was merged with ZERO callers
(measured 2026-09-05: grep of ``inference/stages/`` and ``handler.py``
returned nothing, and the S3 prefix ``predictor/predictions_research_free/``
was empty). Its live consumer — crucible-research's
``producers/filling_arms.py::load_research_free_pool``, the
``scanner_predictor_direct`` filling arm — was therefore reading the
BACKTESTER's post-join offline parquet instead, a file written later in the
same weekly Step Function than the arm that reads it. That arm consequently
failed every canonical Saturday. Wiring the intended daily producer here is
what lets the consumer read a live producer contract
(alpha-engine-config-I10067).

Placement and failure posture
-----------------------------
Registered AFTER ``write_output`` and, like ``shadow_versions``, marked
non-critical in ``inference/pipeline.py::STAGES``: this artifact is a
counterfactual ("what would the champion predict with the 4 research
meta-features zeroed?"). Nothing on the live trading path consumes it today
— the executor consumes ``predictor/predictions/{date}.json``, which this
stage never touches — so a research-free failure must not fail the weekday
PredictorInference Lambda and halt trading.

Non-critical is NOT silent, though. The generic non-critical handler in
``run_pipeline`` only logs a warning, which is exactly the detection
blindness I10067 is about. So this stage handles its own failure and
publishes an ops alert before returning:

- (a) failure mode swallowed: any exception from
  ``run_research_free_inference`` (missing/malformed candidates.json,
  unloadable MetaModel, unreachable ArcticDB, S3 write failure).
- (b) why the primary deliverable survives: the live predictions artifact
  was already written by ``write_output``, upstream of this stage, and is
  not read or mutated here. The executor's input is unaffected.
- (c) recording surface: ``ops_alerts.publish_ops_alert`` (SNS +
  flow-doctor forum topic) plus the ERROR log line. The weekly consumer
  provides the second detection layer: a missing artifact raises
  ``FillingShadowError`` naming this producer.

Alert registration and shape
----------------------------
The alert source is ``alpha-engine-predictor-inference`` — the EXISTING
``predictor_inference`` row in nousergon-data
``infrastructure/overseer/playbooks.yaml::alert_classes`` (operator ruling
2026-08-21, alpha-engine-config-I7740), which is deliberately ONE
``severities: [dynamic]`` row covering every failure shape inside this
Lambda. A failure here is one of those shapes, so it needs no new class and
no companion registry PR.

Severity is ``warning`` with the row's ``drain-queue`` response, not
``critical``/operator: this is a MEASUREMENT-COVERAGE signal (the weekly
``scanner_predictor_direct`` arm goes unmeasurable), never a trading-halt
condition — the same pairing, for the same reason, as
``predictor_shadow_leaderboard_unmeasurable_arm``.

The dedup key is the CONDITION, not the run — a producer broken for a week
must page once, not once per weekday (ALERT003).
"""
from __future__ import annotations

import logging

import config as cfg
from inference.pipeline import PipelineContext

log = logging.getLogger(__name__)

# Registered class ``predictor_inference`` (nousergon-data
# infrastructure/overseer/playbooks.yaml::alert_classes) — see the module
# docstring. Do NOT replace this with a new literal without landing the
# companion registry row first; the alert-class PR guard fails on an
# unregistered PR-added source.
_ALERT_SOURCE = "alpha-engine-predictor-inference"

# Condition-keyed, NOT run-keyed: a producer broken every weekday must page
# once for the outage, not once per run (ALERT003).
_DEDUP_KEY = "predictor_research_free_producer_failed"


def _alert(message: str) -> None:
    """Publish the degradation to the ops surface. Never raises — an alerting
    failure must not take down a stage that is already only reporting."""
    try:
        from ops_alerts import publish_ops_alert

        publish_ops_alert(
            message,
            severity="warning",
            source=_ALERT_SOURCE,
            dedup_key=_DEDUP_KEY,
        )
    except Exception:  # noqa: BLE001 - see (c) above; the ERROR log survives
        log.error(
            "research_free: ops alert publish FAILED — the degradation below is "
            "recorded in this log line only.", exc_info=True,
        )


def run(ctx: PipelineContext) -> None:
    if not getattr(cfg, "RESEARCH_FREE_INFERENCE_ENABLED", True):
        log.info("research_free: disabled (RESEARCH_FREE_INFERENCE_ENABLED=False) — skip.")
        return
    if ctx.dry_run or ctx.local:
        # Mirrors inference.stages.regime_fast_signal's dry_run/local guard.
        # The deploy-time canary (infrastructure/deploy.sh's predict(dry_run)
        # action) invokes this same pipeline with dry_run=True, off the Step
        # Function, off any trading-day gate, and off any calendar. This
        # producer's own precondition is "today's scanner-passing pool exists
        # at candidates/{date}/candidates.json" — an artifact this stage never
        # writes and has no way to conjure for a canary date, a non-trading
        # day, or any date the weekly Scanner has not (yet) run for. Loading
        # it anyway and raising RuntimeError/NoSuchKey on the expected miss
        # paged twice from a single Labor Day (2026-09-07) deploy canary
        # invocation whose OWN contract, stated at the daily_predict.py
        # heartbeat call site, is "no S3 writes, no email"
        # (alpha-engine-config-I10141 investigation). A dry_run invocation
        # produces no live artifact for any consumer to read, so there is
        # nothing here for the read to be correct ABOUT.
        log.info("research_free: skipped (dry_run/local)")
        return
    if getattr(ctx, "weights_prefix_override", None):
        # A cloned shadow context re-scores with a challenger's weights; the
        # research-free counterfactual is defined against the CHAMPION only.
        log.info("research_free: skipped on a shadow (challenger-weights) context.")
        return
    if getattr(ctx, "explicit_tickers", None):
        # Coverage-gap re-invocation scores a subset and merges into the live
        # predictions artifact. The research-free artifact is whole-pool by
        # construction (its pool is scanner_eval_log, not ctx.tickers) and was
        # already written by the first invocation of the day — rewriting it
        # here would be identical work, so skip rather than duplicate it.
        log.info("research_free: skipped on supplemental (explicit_tickers) run.")
        return

    from inference.research_free_inference import run_research_free_inference

    try:
        summary = run_research_free_inference(
            ctx.date_str, bucket=ctx.bucket, dry_run=ctx.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 - see the module docstring, (a)/(b)/(c)
        log.error(
            "research_free: producer FAILED for %s: %s — live predictions/%s.json "
            "is already written and unaffected; predictions_research_free/%s.json "
            "will be ABSENT and crucible-research's scanner_predictor_direct arm "
            "will raise FillingShadowError naming this producer.",
            ctx.date_str, exc, ctx.date_str, ctx.date_str, exc_info=True,
        )
        _alert(
            f"Predictor research-free producer FAILED for {ctx.date_str}\n"
            f"{type(exc).__name__}: {exc}\n"
            f"predictor/predictions_research_free/{ctx.date_str}.json was NOT written. "
            f"Live predictor/predictions/{ctx.date_str}.json is unaffected. "
            f"crucible-research's scanner_predictor_direct filling arm consumes this "
            f"artifact weekly and will fail loud on the next canonical Saturday "
            f"(alpha-engine-config-I10067).",
        )
        return

    log.info(
        "research_free: %s → %d prediction(s), %d error(s), artifact=%s",
        ctx.date_str,
        summary.get("n_written", 0),
        summary.get("n_errors", 0),
        summary.get("artifact_key") or "(dry-run, not written)",
    )
