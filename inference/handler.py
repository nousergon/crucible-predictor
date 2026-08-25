"""
inference/handler.py — AWS Lambda handler (inference only).

Lambda runs daily GBM + CatBoost inference. Training has moved to EC2 spot
instance (infrastructure/spot_train.sh) due to CatBoost + multi-horizon
training exceeding Lambda's 15-minute timeout.

  action == "predict" (default, or omitted):
    Triggered by EventBridge Mon–Fri at 6:15am PT. Loads LightGBM + CatBoost
    models from S3, runs blended inference on the research watchlist, writes
    predictions to S3. Sends predictor email.

  action == "check_coverage":
    Compute buy_candidates - predictions delta and return the missing tickers.
    Used by the weekday Step Function's coverage-gap Choice state to decide
    whether to re-invoke `action=predict` with a `tickers` payload.

  action == "check_deploy_drift":
    Compare the deployed Step Function DEFINITION BODY against the
    definition the last deploy published, and the CloudFormation stack SHA
    against alpha-engine-data @main HEAD. Optional "sf_name" selects which
    state machine (default ne-preopen-trading-pipeline; must be one of
    deploy_drift.INVOKABLE_SF_NAMES). Used by the weekday Step Function's
    DeployDriftCheck state to halt when the orchestration about to run
    differs from what main describes (alpha-engine-config-I7799).

  action == "check_trading_day":
    NYSE holiday gate for the weekday SF (pure calendar, no preflight).
    Returns {is_trading_day, check_date, day_name, marker, ...} computed
    from nousergon_lib.trading_calendar BEFORE StartExecutorEC2. Replaces
    the prior on-box SSM trading_calendar check whose stdout was unreliably
    captured on a cold-booted instance (config#1430).

  action == "check_market_hours":
    NYSE regular-session gate for BOTH trading pipelines' first state
    (alpha-engine-config-I7111; pure calendar, no preflight). Judges
    `now` (both pipelines pass $$.Execution.StartTime) against the
    half-open [09:30, 16:00) ET session and validates the
    `market_hours_override` carried in `execution_input`. Returns
    {is_market_hours, verdict, override, reason, ...} where verdict is
    PROCEED | BLOCKED | PROCEED_OVERRIDE | OVERRIDE_MALFORMED.

  action == "check_weekly_run_day":
    Weekly-SF run-day gate (pure calendar, no preflight; config#1824).
    Returns {is_weekly_run_day, check_date, day_name, marker, ...} — true
    iff the date is one calendar day after the LAST trading session of its
    Mon-Fri week (normally Saturday; Friday/Thursday on holiday-shortened
    weeks). Lets the weekly pipeline's THU-SAT cron self-select the single
    correct firing.

  action == "check_pipeline_contract":
    Validate PIPELINE_CONTRACT.yaml's self-consistency + every artifact_id ∈
    ARTIFACT_REGISTRY.yaml, fetched from GitHub raw. Used by the Saturday Step
    Function as an early pre-spend gate (mirrors check_lib_pin_drift) so a
    contract break halts before any spot launch. Fail-open on a fetch/parse
    miss; halts only on a confirmed structural violation. Also invoked as a
    deploy-time canary (infrastructure/deploy.sh) with dry_run=true, which
    skips only the preflight's deploy-drift assertion (not the contract
    check itself) — see PredictorPreflight.run_for_drift_gate (config#2731).

  action == "train":
    DEPRECATED — returns error directing to spot_train.sh.

Lambda configuration:
  - Runtime: container image (public.ecr.aws/lambda/python:3.12)
  - Memory: 3072 MB  (inference with LightGBM + CatBoost + multi-horizon)
  - Timeout: 900 seconds  (inference takes ~3–4 min)
  - Environment variables:
      S3_BUCKET          — override default bucket (optional)
      EMAIL_SENDER       — from-address for notification emails
      EMAIL_RECIPIENTS   — comma-separated recipient list
      GMAIL_APP_PASSWORD — Gmail App Password (enables SMTP path)
      AWS_REGION         — SES fallback region (default: us-east-1)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path so sibling modules can be
# imported below. Cheap; safe at module-top.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Structured logging + flow-doctor singleton via alpha-engine-lib (shared
# pattern across all 5 entrypoints; see executor/main.py for reference).
# When FLOW_DOCTOR_ENABLED=1, attaches a FlowDoctorHandler at ERROR so
# every log.error() call routes through flow-doctor's dispatch (email +
# optional GitHub issue) without explicit fd.report() plumbing.
#
# Path resolution: LAMBDA_TASK_ROOT (=/var/task in the Lambda image,
# where Dockerfile COPYs flow-doctor.yaml) takes precedence; falls back
# to two-dirs-up from this file for local dev (inference/handler.py →
# repo root). Mirrors alpha-engine-research/lambda/handler.py.
#
# exclude_patterns starts empty by deliberate convention: add patterns
# only after observing real ERROR-level noise from a Lambda invocation.
#
# flow-doctor.yaml references EMAIL_SENDER / EMAIL_RECIPIENTS /
# GMAIL_APP_PASSWORD, all populated by Lambda's `--environment` block
# BEFORE the Python interpreter starts. Other secrets are pulled lazily
# from SSM via krepis.secrets.get_secret() (per-process cached).
from krepis.logging import setup_logging, monitor_handler
from stage_coverage_safety import is_synthetic_invocation, safe_assert_stage_coverage
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "predictor",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

log = logging.getLogger(__name__)

# Every `action` this handler dispatches. An omitted action means "predict",
# the historical default, which is why `None` is a member.
#
# alpha-engine-config-I7111: an UNRECOGNISED action used to fall through to
# that same default, so a caller asking a question this build cannot answer
# got a full prediction run as the answer. On 2026-08-13 the market-hours gate
# — the first state of both trading pipelines — invoked
# action=check_market_hours against a `live` alias frozen on a version
# predating it, and received {"statusCode": 200, "body": "Predictions written
# for today"}. The Step Function then died with States.Runtime on the absent
# verdict, 48s in, with no orders placed and no alert.
#
# Falling through was wrong in both directions. It answered a gate with an
# unrelated side-effecting job, and it made a version skew — the one condition
# a gate most needs to detect — look like a successful invoke. Raising instead
# surfaces as a FunctionError, which every SF Task that calls this handler
# already Catches into its own fail-closed path.
KNOWN_ACTIONS = frozenset({
    None,
    "predict",
    "check_coverage",
    "check_drift",
    "check_deploy_drift",
    "check_trading_day",
    "check_market_hours",
    "check_weekly_run_day",
    "check_pipeline_contract",
    "check_lib_pin_drift",
    # The M slot's numeric correctness verdict (alpha-engine-config-I7262).
    "check_self_test",
    "train",
})


class UnknownAction(ValueError):
    """Raised when `event["action"]` names something this build cannot do.

    Deliberately raised rather than returned: the caller is a Step Function
    Task whose Catch routes a FunctionError to that pipeline's fail-closed
    path. A returned error dict would be indistinguishable from a successful
    invoke to a Choice state keyed on a domain key.
    """


@monitor_handler
def handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point.

    event may contain:
        action    (str)        : "predict" (default) | "train"
        date      (str)        : Override date YYYY-MM-DD.
        dry_run   (bool)       : If True, skip S3 writes and email (for testing).
        tickers   (list[str])  : Supplemental-scoring mode. When non-empty,
                                 score ONLY these tickers and merge into the
                                 existing predictions/{date}.json. Used by the
                                 weekday Step Function's coverage-gap re-invoke.
    """
    # Stage-coverage window (config-I7214): captured at handler ENTRY so the
    # three weekly-SF gate actions below (WeeklyRunDayGate, LibPinDriftCheck,
    # PipelineContractCheck) can assert against it before they return.
    _started = datetime.now(timezone.utc)
    os.environ.setdefault("S3_BUCKET", "alpha-engine-research")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    # setup_logging already ran at module-top (see comment near the
    # krepis.logging import). Apply the standard log level here.
    logging.getLogger().setLevel(logging.INFO)

    # Reject an unrecognised action BEFORE any dispatch, preflight or S3 write
    # — see KNOWN_ACTIONS. This runs first precisely because the fall-through
    # it replaces was side-effecting.
    _requested = event.get("action")
    if _requested not in KNOWN_ACTIONS:
        # The build's identity is the operative field here — this error means
        # "the deployed image is older than the caller", so a message that
        # cannot name the image is a message that cannot be acted on.
        # `GIT_SHA` is a Docker BUILD-ARG stamped to /var/task/GIT_SHA.txt, NOT
        # an environment variable: measured 2026-08-13 against the live Lambda,
        # the env var does not exist and this field always read "unknown".
        # model.registry.resolve_code_sha() is the fleet's existing resolver for
        # exactly this (file, then env, then `git rev-parse` for local dev, with
        # the Dockerfile's "unknown" ARG default normalised to None). Imported
        # inside the failure path so the guard stays free on every good call.
        from model.registry import resolve_code_sha

        _sha = resolve_code_sha() or "unresolved"
        raise UnknownAction(
            f"action={_requested!r} is not implemented by this build "
            f"(code_sha={_sha}). Known actions: "
            f"{sorted(a for a in KNOWN_ACTIONS if a)}. If a Step Function state "
            f"invokes this action, the deployed image predates it — check that "
            f"the `live` alias points at the newest published version "
            f"(alpha-engine-config-I7111)."
        )

    # ── Trading-day gate (weekday SF, runs BEFORE StartExecutorEC2) ─────────
    # Pure NYSE-calendar math — deliberately returns BEFORE preflight so the
    # gate cannot be broken by env / connectivity / ArcticDB / model issues.
    # Replaces the prior on-box SSM trading_calendar check (config#1430).
    if event.get("action") == "check_trading_day":
        from inference.trading_day_gate import check_trading_day
        _r = check_trading_day(event.get("date"))
        log.info(
            "Trading-day gate: %s (%s) -> %s%s",
            _r["check_date"], _r["day_name"], _r["marker"],
            "" if _r["is_trading_day"] else f" (next: {_r.get('next_trading_day')})",
        )
        return _r

    # ── Market-hours gate (FIRST state of both trading pipelines) ──────────
    # alpha-engine-config-I7111. Same zero-infra posture as
    # check_trading_day, and for a sharper reason: this action decides
    # whether a trading pipeline may start at all, so it must not be
    # capable of failing for any reason other than its own calendar math.
    # A preflight dependency here would let an ArcticDB or S3 problem
    # take out the boundary AND the pipeline together.
    if event.get("action") == "check_market_hours":
        from inference.trading_day_gate import check_market_hours
        _r = check_market_hours(event.get("now"), event.get("execution_input"))
        log.info(
            "Market-hours gate: %s (%s) -> %s [%s] override=%s%s",
            _r["now_et"], _r["day_name"], _r["verdict"], _r["reason"],
            _r["override"]["present"],
            f" rejected: {_r['override']['rejection']}"
            if _r["override"].get("rejection") else "",
        )
        return _r

    # ── Weekly run-day gate (weekly SF THU-SAT cron self-selection) ─────────
    # Same zero-infra posture as check_trading_day (config#1824).
    if event.get("action") == "check_weekly_run_day":
        from inference.trading_day_gate import check_weekly_run_day
        _r = check_weekly_run_day(event.get("date"))
        log.info(
            "Weekly run-day gate: %s (%s) -> %s%s",
            _r["check_date"], _r["day_name"], _r["marker"],
            "" if _r["is_weekly_run_day"] else f" ({_r.get('reason')})",
        )
        # Stage-coverage (config-I7214): WeeklyRunDayGate is a
        # positively-declared no-output gate stage — the lib returns
        # COVERED_NO_OUTPUT for it. Loud, not silent, if the module is
        # absent at the pinned nousergon-lib SHA: the handler's own outcome
        # is unchanged (observe mode).
        #
        # alpha-engine-config-I8155: run_date is the event's own `date`
        # (see `stage_coverage_safety.resolve_event_run_date`), never
        # `_r["check_date"]` — when `event.get("date")` is absent,
        # `check_weekly_run_day` computes `check_date` from
        # `datetime.now(_NYSE)`, which is exactly the self-computed
        # stand-in this fix removes: it would report a real-looking
        # run_date for an execution whose own identity was never supplied.
        _verdict = safe_assert_stage_coverage(
            "WeeklyRunDayGate", event=event, window_start=_started, log=log,
        )
        if _verdict is not None:
            _r["stage_coverage"] = _verdict
        return _r

    # Preflight — fail fast on env / connectivity / ArcticDB freshness
    # before loading models or touching inference. See PR #5 and
    # inference/preflight.py.
    #
    # Action-aware dispatch: drift-check is a Step Function gate that only
    # needs env + image-SHA validation. Running the full preflight here
    # (200s universe scan + 5 macro reads + model-weights check) caused
    # the 2026-05-01 SF DeployDriftCheck timeout cascade. Other actions
    # need the full preflight before doing real work.
    from inference.preflight import PredictorPreflight
    _bucket = os.environ.get("S3_BUCKET", "alpha-engine-research")
    _action = event.get("action", "predict")
    # dry_run is re-parsed at its canonical site below; read it early here so
    # the canary (dry_run=true) can skip the deploy-drift assertion. A canary
    # writes/emails nothing, so drift-vs-main is the wrong invariant and a
    # false-failure source during merge bursts (config#1073). Production
    # (dry_run=false) and the SF drift gate below are unaffected.
    _dry_run = bool(event.get("dry_run", False))
    _pf = PredictorPreflight(bucket=_bucket)
    if _action == "check_deploy_drift":
        # The dedicated SF first-state gate. Its entire job is to assert
        # image/SF/CF drift vs origin/main HEAD unconditionally — never
        # thread a dry_run skip into this action's path (config#2731).
        _pf.run_for_drift_gate()
    elif _action in ("check_lib_pin_drift", "check_pipeline_contract"):
        # Both are lightweight pre-spend SF gates (GitHub reads only) — run
        # the minimal preflight, not the full predictor bootstrap. Two
        # distinct call contexts reach this branch:
        #   - The Step Function's own pre-spend invocation (no dry_run) —
        #     retains its existing drift belt-and-suspenders, unchanged.
        #   - infrastructure/deploy.sh's deploy-time canary invocation
        #     (dry_run=true) — writes/emails nothing, so (exactly like the
        #     predict(dry_run) canary fixed by config#1073) asserting its
        #     image SHA against live main HEAD is the wrong invariant and a
        #     false-failure source during a merge burst (config#2731,
        #     2026-07-16). Only THIS dry_run=true path skips drift.
        _pf.run_for_drift_gate(skip_deploy_drift=_dry_run)
    elif _action == "check_drift":
        # Drift detection is a post-inference monitoring step: it reads the
        # day's predictions/features and writes drift_{date}.json. It needs env
        # + S3 reachability only — NOT model weights (it scores nothing) and NOT
        # the deploy-drift gate (a stale-code halt must never silence the
        # monitoring/observability arm). config#1282.
        _pf.check_env_vars("AWS_REGION")
        _pf.check_s3_bucket()
    else:
        _pf.run(skip_deploy_drift=_dry_run)

    fd = None

    action  = event.get("action", "predict")
    date_str = event.get("date", None)
    dry_run  = bool(event.get("dry_run", False))
    raw_tickers = event.get("tickers") or []
    if isinstance(raw_tickers, str):
        raw_tickers = [t.strip() for t in raw_tickers.split(",") if t.strip()]
    explicit_tickers = [t.upper() for t in raw_tickers if t]

    log.info(
        "Lambda invocation: action=%s  date=%s  dry_run=%s  tickers=%s  function=%s",
        action,
        date_str or "today",
        dry_run,
        f"{len(explicit_tickers)} supplemental" if explicit_tickers else "full universe",
        getattr(context, "function_name", "local"),
    )

    bucket = os.environ.get("S3_BUCKET", "alpha-engine-research")

    # ── Coverage check (Step Function coverage-gap Choice state) ────────────
    if action == "check_coverage":
        from inference.coverage_check import compute_coverage_delta
        result = compute_coverage_delta(bucket=bucket, date_str=date_str)
        log.info(
            "Coverage check: %d buy_candidates, %d predictions, %d missing → %s",
            result["n_buy_candidates"], result["n_predictions"],
            result["missing_count"],
            ", ".join(result["missing_tickers"][:10]) + (
                "…" if len(result["missing_tickers"]) > 10 else ""
            ) if result["missing_tickers"] else "none",
        )
        return result

    # ── Drift detection (daily EOD Step Function state, config#1282) ─────────
    # Runs AFTER predictions are written so drift_{date}.json is produced every
    # weekday EOD — matching the artifact's registered `eod_sf` (daily) cadence.
    # Decouples the drift OUTPUT from the skippable, Saturday-only DriftDetection
    # state (which froze the artifact at drift_2026-06-13.json). check_drift
    # already writes predictor/metrics/drift_{date}.json and never raises on a
    # detected-drift condition (it returns a structured result); a real error
    # propagates so the SF Catch sees a true failure.
    if action == "check_drift":
        from monitoring.drift_detector import check_drift
        result = check_drift(bucket=bucket, date_str=date_str, dry_run=dry_run)
        log.info(
            "Drift check %s: status=%s severity=%s n_alerts=%d → drift_%s.json",
            result["date"], result["status"], result.get("severity"),
            result.get("n_alerts", 0), result["date"],
        )
        return result

    # ── Deploy-drift check (Step Function first state) ──────────────────────
    if action == "check_deploy_drift":
        from inference.deploy_drift import INVOKABLE_SF_NAMES, check_deploy_drift
        account_id = (
            getattr(context, "invoked_function_arn", "").split(":")[4]
            if context is not None and getattr(context, "invoked_function_arn", "")
            else os.environ.get("AWS_ACCOUNT_ID", "")
        )
        # alpha-engine-config-I7799 closes-when, clause 1: the canonical-JSON
        # comparison is declared for step_function_daily.json AND
        # step_function_eod.json, but until now no caller could ask for the
        # second — this handler hardcoded the preopen default, so the postclose
        # half of _SF_DEFINITION_PATHS was reachable only from unit tests. A
        # verification path nothing can invoke is not coverage; it is code that
        # will drift unobserved until the day it is wired to a gate.
        #
        # Validated against the declared set rather than passed through: an
        # unrecognised name reaching check_deploy_drift() falls through to
        # STAMP semantics (_SF_DEFINITION_PATHS.get -> None), which would
        # answer confidently about a state machine that does not exist. Fail
        # loud at the edge instead — the SF Task's Catch routes it to
        # HandleFailure, and an absent sf_drift halts anyway (fail closed).
        sf_name = event.get("sf_name") or "ne-preopen-trading-pipeline"
        if sf_name not in INVOKABLE_SF_NAMES:
            raise ValueError(
                f"check_deploy_drift: unknown sf_name {sf_name!r}. Declared "
                f"state machines: {sorted(INVOKABLE_SF_NAMES)}. Add the name "
                f"(and, for a definition-compared pipeline, its repo path) to "
                f"inference/deploy_drift._SF_DEFINITION_PATHS — never pass a "
                f"definition path in alongside the name."
            )
        result = check_deploy_drift(
            region=os.environ.get("AWS_REGION", "us-east-1"),
            account_id=account_id,
            sf_name=sf_name,
        )
        # alpha-engine-config-I7048: sf_drift/cf_drift are OMITTED (not
        # False) when unmeasured (upstream fetch failed) — .get() keeps
        # this log line from KeyError-ing on that exact path.
        log.info(
            "Deploy-drift check [%s]: upstream=%s  sf=%s(drift=%s, %s, "
            "compared=%s via %s)  stamp_stale=%s  cf=%s(drift=%s)",
            sf_name,
            (result["upstream_sha"] or "?")[:12],
            (result["sf_sha"] or "missing")[:12], result.get("sf_drift", "unmeasured"),
            result.get("sf_drift_reason"), result.get("sf_definition_compared"),
            result.get("sf_definition_source"),
            result.get("deploy_stamp_stale", "unmeasured"),
            (result["stack_sha"] or "missing")[:12], result.get("cf_drift", "unmeasured"),
        )
        return result

    # ── Lib-pin drift check (Saturday SF early state, L4517) ────────────────
    if action == "check_lib_pin_drift":
        from inference.lib_pin_drift import check_lib_pin_drift
        # probe: the deploy-time canary invokes this action only to exercise
        # its wiring and gates on the PRESENCE of has_drift. A true finding
        # from a synthetic invocation is still logged (at WARNING) but must
        # not page — a cross-repo lockstep pin bump is two merges, so a canary
        # landing between them reports a real, useless, self-clearing parity
        # break (alpha-engine-config-I7954). The SF's own invocation is not
        # synthetic and keeps ERROR.
        #
        # alpha-engine-config-I8155: `probe=dry_run` alone did not deliver
        # that — measured on `infrastructure/deploy.sh`, THIS action's canary
        # payload is `{"action": "check_lib_pin_drift"}` with no `dry_run`, so
        # probe was False on every canary and I7954's suppression never
        # applied to the call site its own comment names. `dry_run` is a
        # per-payload literal and drifted; `invocation_kind` is stamped by
        # `run_canary_action` itself and cannot. Keep the `dry_run` term so an
        # explicit hand-invoked dry run still suppresses.
        result = check_lib_pin_drift(probe=dry_run or is_synthetic_invocation(event))
        # alpha-engine-config-I7048: has_drift is OMITTED (not False) on a
        # fetch/parse miss — .get() with an "unmeasured" sentinel keeps this
        # log line from KeyError-ing on the exact degraded path it exists to
        # report, while still logging the honest state.
        #
        # alpha-engine-config-I7316: `offenders` and `pins` need the SAME
        # treatment and did not get it. I7048 guarded the VERDICT key and left
        # its siblings as bare subscripts; alpha-engine-config-I7277 then made
        # the unmeasured payload omit the evidence list too, for the same
        # reason (an empty list is as much a claim as a false verdict). From
        # that merge on, every unmeasured invocation raised KeyError:
        # 'offenders' HERE — in the line written to report it. Measured on the
        # 2026-08-14T03:10 deploy canary: FunctionError: Unhandled,
        # payload: 'offenders', which refused the promote and froze the :live
        # alias at v474.
        offenders = result.get("offenders") or ()
        log.info(
            "Lib-pin drift check: has_drift=%s reason=%s pins=%s%s",
            result.get("has_drift", "unmeasured"),
            result.get("reason", "unreported"),
            result.get("pins", {}),
            (" offenders=" + "; ".join(offenders)) if offenders else "",
        )
        # Stage-coverage (config-I7214): LibPinDriftCheck is a
        # positively-declared no-output gate stage — the lib returns
        # COVERED_NO_OUTPUT for it. Observe mode — the handler's own
        # outcome is unchanged on an absent module.
        #
        # alpha-engine-config-I8155: `date_str or datetime.now(...)` was the
        # forbidden silent fall-back — measured against
        # nousergon-data/infrastructure/step_function.json, the SF's
        # LibPinDriftCheck Task Payload carries only `action`, so date_str
        # is always None on the live path and the `now()` half always fired,
        # writing every real invocation's verdict under a self-computed date
        # nobody could join back to the execution. Skip the assertion loudly
        # instead when the event carries no run_date.
        _verdict = safe_assert_stage_coverage(
            "LibPinDriftCheck", event=event, window_start=_started, log=log,
        )
        if _verdict is not None:
            result["stage_coverage"] = _verdict
        return result

    # ── Pipeline-contract preflight (Saturday SF early state, L4595) ─────────
    if action == "check_pipeline_contract":
        from inference.pipeline_contract_check import check_pipeline_contract
        # probe — same canary-must-not-page contract as check_lib_pin_drift
        # above (alpha-engine-config-I7954), now keyed on the centrally
        # stamped `invocation_kind` as well as `dry_run`
        # (alpha-engine-config-I8155). This action's canary payload DOES carry
        # `dry_run: true` today; the OR is what stops that from being a fact
        # anyone has to keep true by hand.
        result = check_pipeline_contract(
            probe=dry_run or is_synthetic_invocation(event)
        )
        # alpha-engine-config-I7048: has_violation is OMITTED (not False) on
        # a fetch/parse miss — same .get() rationale as check_lib_pin_drift.
        # alpha-engine-config-I7316: and so are `violations` and
        # `boundary_count`, which were bare subscripts here until the
        # 2026-08-14T03:10 canary raised KeyError: 'violations' on them.
        violations = result.get("violations") or ()
        log.info(
            "Pipeline-contract preflight: has_violation=%s reason=%s boundaries=%s%s",
            result.get("has_violation", "unmeasured"),
            result.get("reason", "unreported"),
            result.get("boundary_count"),
            (" violations=" + "; ".join(violations)) if violations else "",
        )
        # Stage-coverage (config-I7214): PipelineContractCheck is a
        # positively-declared no-output gate stage — the lib returns
        # COVERED_NO_OUTPUT for it. Observe mode — the handler's own
        # outcome is unchanged on an absent module.
        #
        # alpha-engine-config-I8155: same fix as LibPinDriftCheck above —
        # the SF's PipelineContractCheck Task Payload also carries only
        # `action`, so `date_str or datetime.now(...)` always took the
        # `now()` half in production. Skip loudly instead when the event
        # carries no run_date.
        _verdict = safe_assert_stage_coverage(
            "PipelineContractCheck", event=event, window_start=_started, log=log,
        )
        if _verdict is not None:
            result["stage_coverage"] = _verdict
        return result

    # ── Numeric self-test (alpha-engine-config-I7262) ───────────────────────
    if action == "check_self_test":
        # The M slot's CORRECTNESS verdict (`sf-pipeline-policy.md` §2.3a),
        # computed where the numbers are computed: in the deployed interpreter,
        # on the deployed wheels. CI proves the code is right on a runner; this
        # proves the INSTRUMENT is right on the image that produced the week's
        # predictions.
        #
        # `run_self_test` never raises and this action adds no hard-fail path,
        # no new SF state and no topology change: the verdict is RETURNED for a
        # Choice state to read, exactly as check_lib_pin_drift and
        # check_pipeline_contract are. An accuracy instrument that can take down
        # the pipeline is a worse defect than the one it detects.
        from inference.self_test import (
            publish_console_row,
            run_self_test,
            self_test_key,
            write_self_test,
        )

        # `date_str` is the handler's already-resolved event date (None => today,
        # the same convention every other action here uses). Resolved to a
        # concrete ISO date for the artifact key, because a key containing
        # "None" is a key nothing can ever look up.
        import datetime as _dt

        run_date = date_str or _dt.date.today().isoformat()
        result = run_self_test(run_date=run_date)

        # Both emissions are isolated: the verdict is already in `result` and in
        # the logs, so a failed S3 write degrades the EVIDENCE, never the
        # verdict, and must not turn a measured PASS into a stage error.
        artifact_key = None
        try:
            artifact_key = write_self_test(bucket, run_date, result)
        except Exception as exc:  # noqa: BLE001 — evidence emission never blocks
            log.error(
                "self-test artifact emission failed for %s (verdict=%s is still "
                "carried in this response and in the logs): %s",
                run_date, result.get("verdict"), exc, exc_info=True,
            )
        # `principles.md` §2.7 — a check that reports nowhere is unobserved.
        console_uri = publish_console_row(result)

        log.info(
            "Predictor self-test: verdict=%s cases=%s failed=%s errored=%s "
            "known_gaps=%s artifact=%s console=%s libs=%s",
            result.get("verdict", "unmeasured"), result.get("n_cases"),
            result.get("n_failed"), result.get("n_errored"),
            result.get("n_known_gaps"), artifact_key or "<not written>",
            console_uri or "<not published>", result.get("libraries"),
        )
        # The full case list is in the artifact; the response carries the
        # verdict and the counts a Choice state can branch on without parsing
        # a 33-element body out of the Step Functions payload size limit.
        return {
            "verdict": result.get("verdict"),
            "n_cases": result.get("n_cases"),
            "n_failed": result.get("n_failed"),
            "n_errored": result.get("n_errored"),
            "n_known_gaps": result.get("n_known_gaps"),
            "run_date": run_date,
            "self_test_key": artifact_key,
            "expected_key": self_test_key(run_date),
            "console_row": console_uri,
        }

    # ── Train (DEPRECATED — moved to EC2 spot instance) ─────────────────────
    if action == "train":
        log.warning(
            "action=train is deprecated on Lambda. Training now runs on EC2 spot "
            "via infrastructure/spot_train.sh (CatBoost + multi-horizon exceeds "
            "Lambda's 15-minute timeout)."
        )
        return {
            "statusCode": 400,
            "body": (
                "Training has moved to EC2 spot instance. "
                "Use infrastructure/spot_train.sh or the Saturday cron. "
                "Lambda is inference-only."
            ),
        }

    # ── Predict (default) ──────────────────────────────────────────────────────
    # Any failure must raise so Step Functions sees a real task failure and the
    # Catch branch blocks downstream executor. Returning statusCode:500 would
    # look like a successful Lambda response and let executor proceed on stale
    # predictions — the exact silent-failure mode that hit production on
    # 2026-04-13.
    from inference.daily_predict import main
    main(
        date_str=date_str,
        dry_run=dry_run,
        local=False,
        model_type="gbm",
        watchlist_path="auto",
        explicit_tickers=explicit_tickers,
    )
    log.info("Predictor Lambda completed successfully")

    # ── Flow-doctor end-of-run heartbeat (config#646) ───────────────────────
    # Write the flow's end-of-run status() snapshot to the research bucket so
    # the dashboard System Health consumer can read it from
    # s3://alpha-engine-research/_flow_doctor/heartbeat/predictor/{date}.json.
    # `bucket` resolves to alpha-engine-research (S3_BUCKET default set above).
    # emit_heartbeat soft-fails (returns None, never raises), so the run's
    # success is unaffected by a heartbeat write miss.
    # Guard on the method's presence too: the producing repos deploy
    # independently and emit_heartbeat only exists in flow-doctor >=0.6.2, so a
    # version-skewed lib pin would AttributeError at end-of-run without the
    # hasattr check (mirrors flow-doctor's own soft-fail philosophy).
    # dry_run also skips the write: dry_run's whole contract (and the
    # deploy-time canary's, which invokes action=predict with dry_run=true)
    # is "no S3 writes, no email" — an unconditional heartbeat PUT here broke
    # that contract silently and made the deploy canary a non-read-only
    # action (config#3025 dim8).
    if not dry_run:
        from krepis.logging import get_flow_doctor
        fd = get_flow_doctor()
        if fd and hasattr(fd, "emit_heartbeat"):
            fd.emit_heartbeat(bucket=bucket)

    return {
        "statusCode": 200,
        "body": (
            f"Supplemental predictions written for {date_str or 'today'} "
            f"({len(explicit_tickers)} tickers)"
            if explicit_tickers else
            f"Predictions written for {date_str or 'today'}"
        ),
    }
