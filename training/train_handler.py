"""
training/train_handler.py — Weekly GBM retraining pipeline for Lambda.

Called by inference/handler.py when event["action"] == "train".

Pipeline:
  1. Load the universe (OHLCV + pre-computed feature columns) from the
     ArcticDB universe library. ArcticDB is the single, canonical source of
     training data — the legacy S3-parquet / yfinance-refresh fallback was
     removed by PR #6 (training cutover 2026-04-16). No data is fetched over
     yfinance and no parquets are written back to S3 by this handler.
  2. Build regression arrays (29 features) and apply 70/15/15 time-based split
     with purge gaps of FORWARD_DAYS between train/val and val/test.
  3. Train GBMScorer with Optuna-tuned params from config.GBM_TUNED_PARAMS
     (n_estimators=2000, early_stopping=50).
  4. Evaluate on test set: IC, IC IR, positive-period rate.
  5. Upload dated backup unconditionally; promote to gbm_latest.txt only if IC gate passes.
  5b. Write slim 2-year price cache to predictor/price_cache_slim/ for the
      daily inference Lambda's price reads.
  6. Send training summary email with results.

S3 layout:
  predictor/weights/gbm_latest.txt      — (output) active inference weights
  predictor/weights/gbm_{date}.txt      — (output) dated backup

  predictor/price_cache/*.parquet       — LEGACY 10y per-ticker OHLCV parquets.
                                          No longer read or written by training —
                                          ArcticDB is canonical since PR #6
                                          (2026-04-16). Retained on S3 only as a
                                          historical fallback artifact; orphaned.
  predictor/daily_closes/{date}.parquet — Backward-split-adjusted OHLCV snapshot per
                                          trading day.  Written by DataPhase1 in
                                          alpha-engine-data/collectors/daily_closes.py
                                          as the first step of the weekday Step Function,
                                          before the predictor inference Lambda runs.
                                          Historically read by inference as the Mon–Fri
                                          delta to the slim cache; inference now reads
                                          directly from ArcticDB (Phase 7a, 2026-04-16).
                                          These parquets are now orphaned — candidates
                                          for removal in Phase 7e.
  predictor/price_cache_slim/*.parquet  — 2-year slice of each ticker (written by
                                          write_slim_cache() after each weekly training run)
                                          for the daily inference Lambda's price reads.

Lambda environment variables (same as predictor email):
  EMAIL_SENDER / EMAIL_RECIPIENTS / GMAIL_APP_PASSWORD / AWS_REGION
"""

from __future__ import annotations

import json
import logging
import os
import re
import smtplib
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from krepis.secrets import get_secret
from krepis.console import console_url as build_console_url

# Deep-link target for the slimmed training email → console Predictor Training
# page (config#856). The slug is pinned in crucible-dashboard app.py
# (url_path="predictor-training") and guarded by
# tests/test_predictor_training_page.py; the page honors ?date=YYYY-MM-DD, keyed
# by this cycle's date_str, so the email link lands on the exact training cycle
# it summarizes. The base URL is built by the krepis.console.console_url
# chokepoint (config#1300); only the slug (the cross-repo contract with the
# dashboard url_path) stays local. Mirrors model_zoo.py's MODEL_ZOO_SLUG.
TRAINING_PAGE_SLUG = "predictor-training"

# Structured logging + flow-doctor singleton via alpha-engine-lib (shared
# pattern across all 5 entrypoints; see executor/main.py for reference).
# Module-top so import-time errors in numpy / lightgbm / catboost below
# (heavy GBM imports) are also captured by flow-doctor's ERROR handler.
# Training runs on EC2 spot (NOT Lambda) — no LAMBDA_TASK_ROOT, so the
# yaml path resolves via two-dirs-up from this file (training/train_handler.py
# → repo root, where flow-doctor-training.yaml lives).
#
# exclude_patterns starts empty by deliberate convention; add patterns
# only after observing real ERROR-level noise during training runs.
from krepis.logging import setup_logging, guard_entrypoint
from nousergon_lib.phase_registry import PhaseRegistry
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = str(
    Path(__file__).resolve().parent.parent / "flow-doctor-training.yaml"
)
setup_logging(
    "predictor-training",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

import numpy as np

log = logging.getLogger(__name__)


_SUBSAMPLE_NOISE_FLOOR_PCT = 30.0


def _annotate_subsample_noise_floor(
    result: dict, s3, bucket: str,
    latest_key: str = "predictor/metrics/training_summary_latest.json",
) -> None:
    """Annotate `result["short_history_subsample"][*]` with `n_pct_change_vs_prior`
    and WARN-log on >|`_SUBSAMPLE_NOISE_FLOOR_PCT`| swing.

    Closes 5/23-SF P0 sweep (L980 noise-floor WARN). Reads the prior cycle's
    `training_summary_latest.json` from S3 BEFORE the current cycle's write
    overwrites it, computes the per-component subsample-`n` percent change,
    and surfaces a WARN when the swing is large enough that a per-week
    margin comparison is no longer apples-to-apples (the 5/24 incident:
    `train_start_date` clamp shrunk vol subsample n 821→263 = -68%,
    ratcheting baseline IC +0.057 while no underlying capability changed).

    Best-effort: any failure (no prior, parse error, S3 hiccup) logs DEBUG
    and returns without raising. The current-cycle write proceeds normally.
    """
    try:
        sh = result.get("short_history_subsample") or {}
        if not sh:
            return

        try:
            prior_obj = s3.get_object(
                Bucket=bucket,
                Key=latest_key,
            )
            prior = json.loads(prior_obj["Body"].read())
        except Exception as e:
            log.debug("subsample noise-floor: no prior summary (%s); skipping", e)
            return

        prior_sh = prior.get("short_history_subsample") or {}
        for component, current in sh.items():
            if not isinstance(current, dict):
                continue
            current_n = current.get("n")
            prior_component = prior_sh.get(component) or {}
            prior_n = prior_component.get("n")
            if current_n is None or prior_n is None:
                continue
            if not isinstance(current_n, (int, float)) or not isinstance(
                prior_n, (int, float)
            ) or prior_n == 0:
                continue
            pct_change = float((current_n - prior_n) / prior_n * 100.0)
            current["n_pct_change_vs_prior"] = round(pct_change, 2)
            current["prior_n"] = int(prior_n)
            if abs(pct_change) > _SUBSAMPLE_NOISE_FLOOR_PCT:
                log.warning(
                    "subsample noise-floor: %s.n changed by %+.1f%% vs prior cycle "
                    "(prior=%d, current=%d). Per-week margin comparison on this "
                    "component is no longer apples-to-apples — baseline IC may have "
                    "ratcheted mechanically rather than from a real capability shift. "
                    "See ROADMAP L980 / 5/24 train_start_date clamp incident.",
                    component, pct_change, int(prior_n), int(current_n),
                )
    except Exception as e:
        log.debug("subsample noise-floor annotation failed (non-blocking): %s", e)


# ── Training email ─────────────────────────────────────────────────────────────


def _build_shadow_calibration_html(sm: dict | None) -> str:
    """Render the Shadow Calibrator observability row.

    Returns the empty string when no shadow calibrator was fit (shadow
    disabled via ``calibration.shadow_method: null`` OR the shadow fit
    failed — both surface as ``fitted: False`` in the dict). Mirrors the
    Confidence Calibration row's style; adds shadow-specific knobs
    (``class_weight``, ``C``) and gate-pass fingerprints so the operator
    can tell at a glance whether shadow is ready to promote → live.
    """
    if not sm or not sm.get("fitted"):
        return ""

    def _gate(passed_key: str, failed_key: str) -> str:
        passed = sm.get(passed_key)
        label = "PASS" if passed else ("FAIL" if passed is False else "n/a")
        failed_check = sm.get(failed_key)
        suffix = f" ({failed_check})" if failed_check else ""
        return f"<b>{label}{suffix}</b>"

    ece_before = sm["ece_before"]
    ece_after = sm["ece_after"]
    return (
        '<h3 style="margin-top:16px; margin-bottom:4px;">'
        'Shadow Calibrator (observability)</h3>'
        '<p style="font-size:12px;">'
        f'Method: <b>{sm["method"]}</b> &nbsp;|&nbsp; '
        f'class_weight: <b>{sm.get("class_weight")}</b> &nbsp;|&nbsp; '
        f'C: <b>{sm.get("C")}</b> &nbsp;|&nbsp; '
        f'Samples: <b>{sm["n_samples"]:,}</b> &nbsp;|&nbsp; '
        f'ECE: <b>{ece_before:.4f} → {ece_after:.4f}</b> &nbsp;|&nbsp; '
        'output_dist: '
        + _gate("output_distribution_gate_passed",
                "output_distribution_gate_failed_check")
        + ' &nbsp;|&nbsp; stratified: '
        + _gate("stratified_per_regime_passed",
                "stratified_per_regime_failed_check")
        + '</p>'
    )


def send_training_email(result: dict, date_str: str) -> bool:
    """
    Send GBM training summary email via Gmail SMTP (primary) or SES (fallback).
    Returns True on success. Never raises.
    """
    import config as cfg

    sender     = cfg.EMAIL_SENDER
    recipients = cfg.EMAIL_RECIPIENTS

    if not sender or not recipients:
        log.info("Training email skipped — EMAIL_SENDER/EMAIL_RECIPIENTS not set")
        return False

    promoted     = result.get("promoted", False)
    promoted_mode = result.get("promoted_mode")
    # `passes_ic` is now the QUALITY verdict (all gates passed), decoupled from
    # the promotion decision (L4540). Training is ALWAYS challenger-first since
    # config#1052/#679 (the dead TRAINING_AUTO_PROMOTE_ENABLED flag was retired):
    # a run can pass every gate and NOT promote — that is a registered challenger,
    # NOT a failure; the model-zoo `select_winner` step owns promotion. Legacy
    # pre-challenger-first archive manifests may carry `auto_promote_enabled=True`
    # (where promoted == gate_passed); honor that for old-archive reads, but a
    # current run is challenger-registered whenever it passes the gate without
    # promoting.
    passes_ic    = result.get("passes_ic_gate", False)
    auto_promote = result.get("auto_promote_enabled", False)
    challenger_registered = bool(passes_ic and not promoted and not auto_promote)
    val_ic       = result.get("val_ic", 0.0)
    test_ic      = result.get("test_ic", 0.0)
    mse_ic       = result.get("mse_ic", test_ic)
    rank_ic      = result.get("rank_ic")
    ic_ir        = result.get("ic_ir", 0.0)
    version      = result.get("model_version", "unknown")
    is_meta      = "meta" in str(version).lower()
    elapsed_s    = result.get("elapsed_s", 0)
    n_train      = result.get("n_train", 0)
    ic_pos       = result.get("ic_positive_20", 0)

    ic_color    = "#2e7d32" if passes_ic else "#c62828"
    ic_label    = "PASS ✓" if passes_ic else "FAIL ✗"
    promo_color = "#2e7d32" if promoted else "#c62828"

    # Build the promotion label. Previously this was hardcoded to
    # "NOT promoted (IC gate failed ✗)" for all non-promotion cases, which
    # was actively misleading: the 2026-04-11 v3.0-meta run produced
    # Meta-Model IC 0.0525 (well above the 0.03 gate) but was blocked by
    # the walk-forward validation on the momentum base model. The email
    # said "IC gate failed" when the IC gate actually passed.
    def _build_failure_reason() -> str:
        # Prefer the authoritative gate-resolution block (added 2026-05-14
        # to close ROADMAP L1668). `promotion_gate_detail.promoted_blocker_reason`
        # is computed off the SAME boolean values the live `promoted` formula
        # uses, so the email's failure-reason can never disagree with the
        # actual blocker. Falls back to the legacy walk_forward scan only
        # for older training_summary readers (pre-2026-05-14 archive runs).
        gate_detail = result.get("promotion_gate_detail") or {}
        blocker = gate_detail.get("promoted_blocker_reason")
        if blocker:
            # Decorate each named gate with the most informative scalar so
            # the email reader doesn't have to S3-grep training_summary for
            # context. The blocker string is +-joined when multiple gates
            # fired (e.g. "meta_ic+output_dist").
            parts: list[str] = []
            for name in blocker.split("+"):
                if name == "meta_ic":
                    meta_ic_val = result.get("meta_model_ic")
                    threshold = gate_detail.get("meta_ic_threshold")
                    if meta_ic_val is not None and threshold is not None:
                        parts.append(
                            f"meta IC {meta_ic_val:+.4f} < gate {threshold:+.4f}"
                        )
                    else:
                        parts.append("meta IC gate")
                elif name == "subsample":
                    sh = result.get("short_history_subsample") or {}
                    vol_sh = (sh.get("volatility") or {})
                    cic = vol_sh.get("component_ic")
                    bic = vol_sh.get("baseline_ic")
                    if cic is not None and bic is not None:
                        parts.append(
                            f"volatility subsample (component {cic:+.4f} ≤ baseline {bic:+.4f})"
                        )
                    else:
                        parts.append("volatility subsample gate")
                elif name == "output_dist":
                    parts.append("output-distribution gate (calibrator shape)")
                elif name == "stratified":
                    parts.append("stratified per-regime gate")
                else:
                    parts.append(name)
            return "promotion blocked: " + "; ".join(parts)
        # Legacy fallback path for pre-2026-05-14 archive runs that lack
        # the gate_detail block. Same heuristic as before — accurate only
        # when walk_forward has a negative volatility median.
        wf = result.get("walk_forward") or {}
        if not wf:
            return "IC gate failed"  # fallback — no wf info available
        vol = wf.get("volatility_median_ic")
        reasons: list[str] = []
        if vol is not None and vol <= 0:
            reasons.append(f"volatility median IC {vol:+.4f}")
        if reasons:
            return "walk-forward failed: " + ", ".join(reasons)
        return "IC gate failed"

    promo_label = (
        f"Promoted → weights/meta/ ✓" if promoted and is_meta
        else f"Promoted → gbm_latest ({promoted_mode}) ✓" if promoted
        else "Registered challenger (gates passed; model-zoo select_winner decides promotion) ◆"
        if challenger_registered
        else f"NOT promoted ({_build_failure_reason()}) ✗"
    )
    status_str  = "PASS" if passes_ic else "FAIL"

    # Subject IC: prefer OOS Spearman over in-sample Ridge fit. PR #133's
    # meta_model_oos_ic field is the honest measurement; fall back to the
    # horizon_diagnostic curve at the active forward horizon, then to
    # in-sample as a last resort (with explicit "(in-sample)" suffix so
    # the deprecation is visible in subject-line search results).
    if is_meta:
        _subj_oos = result.get("meta_model_oos_ic")
        if _subj_oos is None:
            _subj_oos = (
                (result.get("horizon_diagnostic") or {}).get("curve") or {}
            ).get("21d", {}).get("spearman")
        if _subj_oos is not None:
            _subj_ic_text = f"Meta OOS IC {_subj_oos:+.4f}"
        else:
            _subj_ic_text = f"Meta IC {test_ic:+.4f} (in-sample)"
    else:
        _subj_ic_text = f"IC {test_ic:.4f}"
    _subj_promo = (
        f"Promoted ({promoted_mode})" if promoted
        else "Registered challenger" if challenger_registered
        else "Not promoted"
    )
    subject = (
        f"Alpha Engine Training | {date_str} | "
        f"{_subj_ic_text} {status_str} | "
        f"{_subj_promo}"
    )

    # ── Slim body (config#856: pull-for-state, push-on-transition) ──────────
    # The per-step content email is replaced by the console Predictor Training
    # page: the full detail (walk-forward folds, feature-importance bars, gain-
    # vs-SHAP comparison, feature health, calibration + multi-horizon tables) now
    # lives on …/predictor-training?date=YYYY-MM-DD, rendered from the SAME
    # training_summary_{date}.json this run already writes (via
    # _write_training_summary — untouched, no data loss). The email keeps only
    # the operator-actionable headline: model + IC-gate verdict + promotion
    # status + the shadow-calibrator readiness row, plus any data warnings and
    # the deep-link.
    console_link = build_console_url(TRAINING_PAGE_SLUG, date=date_str or None)

    # One-line walk-forward verdict (the full fold table is on the console page).
    # Fail-soft: only touches median + fold count, so a malformed fold (a fold
    # dict missing n_train / ic — config#1083) can never crash the email.
    wf_data = result.get("walk_forward") or {}
    if wf_data.get("folds"):
        _wf_median = (
            wf_data.get("volatility_median_ic") if is_meta
            else (wf_data.get("median_ic") or wf_data.get("momentum_median_ic"))
        )
        _wf_pass = wf_data.get("passes_wf")
        _wf_median_txt = (
            f"{_wf_median:+.4f}" if isinstance(_wf_median, (int, float)) else "n/a"
        )
        wf_line = (
            f"Walk-forward ({'volatility ' if is_meta else ''}median IC "
            f"{_wf_median_txt}): {'PASS' if _wf_pass else 'FAIL'} "
            f"({len(wf_data['folds'])} folds)"
        )
    else:
        wf_line = "Walk-forward: no folds recorded"

    # Data warnings — feature-noise candidates are operator-actionable, so they
    # stay inline; the per-feature IC table itself is on the console page.
    noise_cands = result.get("noise_candidates") or []
    warn_line = (
        f"Noise candidates ({len(noise_cands)}): {', '.join(noise_cands)}"
        if noise_cands else ""
    )

    _ic_summary = (
        f"Val IC {val_ic:+.4f} · Test IC {test_ic:+.4f} · MSE IC {mse_ic:+.4f}"
        + (f" · Rank IC {rank_ic:+.4f}" if isinstance(rank_ic, (int, float)) else "")
        + f" · IC IR {ic_ir:.3f} ({ic_pos}/20 positive)"
    )
    _oos_line = (
        f"Meta OOS IC (Spearman): {result['meta_model_oos_ic']:+.4f}"
        if is_meta and isinstance(result.get("meta_model_oos_ic"), (int, float))
        else ""
    )

    html_body = (
        f'<html><body style="font-family:sans-serif; font-size:13px; color:#222; max-width:600px;">'
        f'<h2 style="margin-bottom:4px;">Alpha Engine Training — {date_str}</h2>'
        f'<p style="color:#555; font-size:12px; margin-top:0;">'
        f'Model: <b>{version}</b> &nbsp;|&nbsp; Training samples: <b>{n_train:,}</b> '
        f'&nbsp;|&nbsp; Elapsed: <b>{elapsed_s:.0f}s ({elapsed_s/60:.1f} min)</b></p>'
        f'<p style="margin:6px 0;">IC gate: <b style="color:{ic_color};">{ic_label}</b> '
        f'&nbsp;|&nbsp; Promotion: <b style="color:{promo_color};">{promo_label}</b></p>'
        f'<p style="margin:6px 0; font-size:12px;">{_ic_summary}</p>'
        + (f'<p style="margin:4px 0; font-size:12px;">{_oos_line}</p>' if _oos_line else "")
        + f'<p style="margin:6px 0; font-size:12px;">{wf_line}</p>'
        + (f'<p style="margin:6px 0; font-size:12px; color:#c62828;">{warn_line}</p>' if warn_line else "")
        # Shadow calibrator readiness row — operator-actionable (tells the
        # operator whether shadow is safe to promote → live); kept in-email.
        + _build_shadow_calibration_html(result.get("shadow_calibration"))
        + f'<p style="margin-top:18px;">'
        f'<a href="{console_link}" style="color:#1976d2; font-weight:bold;">'
        f'View full training detail on the console →</a></p>'
        f'<p style="font-size:11px; color:#aaa; margin-top:6px;">'
        f'Walk-forward folds, feature importance (gain + SHAP), feature health, '
        f'calibration and multi-horizon detail are on the console Predictor '
        f'Training page (config#856).</p>'
        f'</body></html>'
    )

    plain_body = (
        f"Alpha Engine Training — {date_str}\n"
        f"Model: {version}  Samples: {n_train:,}  Elapsed: {elapsed_s:.0f}s\n"
        f"\nIC gate: {ic_label}"
        f"\nPromotion: {promo_label}"
        f"\n{_ic_summary}"
        + (f"\n{_oos_line}" if _oos_line else "")
        + f"\n{wf_line}"
        + (f"\n{warn_line}" if warn_line else "")
        + f"\n\nFull training detail (walk-forward folds, feature importance, "
        f"feature health, calibration, multi-horizon):\n{console_link}\n"
    )

    # SMTP/SES dispatch via the krepis.email_sender chokepoint
    # (L4356 — Gmail SMTP primary, SES fallback). Same semantics as the
    # pre-consolidation inline path.
    from krepis.email_sender import send_email as _send_email
    return _send_email(
        subject, plain_body,
        recipients=recipients, html=html_body,
        sender=sender, region=cfg.AWS_REGION,
    )


# ── Orchestrator ───────────────────────────────────────────────────────────────

_TRAINING_SUMMARY_KEY = "predictor/metrics/training_summary_{date}.json"
# config#1170 — per-challenger-spec training summary (the meta_training phase's
# L4524 reload artifact). Namespaced under the spec's MODEL_VERSION_LABEL so a
# zoo rerun reloads ITS OWN summary, never a sibling spec's (consequence #3).
_TRAINING_SUMMARY_KEY_SPEC = "predictor/model_zoo/{spec}/metrics/training_summary_{date}.json"


def _spec_namespace() -> "str | None":
    """The current zoo-challenger spec namespace, or ``None`` for the base champion.

    config#1170 — phase markers + the training-summary reload artifact were
    keyed at DATE granularity (``predictor/{date}/.phases/meta_training.json``),
    a SINGLE file shared by every model-zoo spec for that date. That made
    same-date reruns skip ALL specs (any spec's marker auto-skipped every spec)
    and blocked a FAILED spec from retrying same-day (it saw a sibling's success
    marker). The fix namespaces the markers + reload artifact per challenger spec.

    The namespace is derived from ``cfg.MODEL_VERSION_LABEL`` — the model_zoo
    sets this per spec via ``spec_overrides`` around the ``main()`` call
    (model_zoo.train_spec). When it equals the base champion default
    (``v3.0-meta`` / config ``model_version_label``) the run IS the base champion,
    so we return None and the legacy spec-less paths are preserved (the issue
    requires the base champion keeps its current path). A challenger returns a
    filesystem-safe slug of its label so each spec's skip/reload is independent.
    """
    try:
        import config as _cfg
    except Exception:  # config unreadable (dev/CI w/o predictor.yaml)
        return None
    label = getattr(_cfg, "MODEL_VERSION_LABEL", None)
    base = _cfg_default_label(_cfg)
    if not label or label == base:
        return None
    # Slugify so an arbitrary spec label is a safe single S3 path segment.
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_")
    return slug or None


def _cfg_default_label(_cfg) -> str:
    """The base champion's model_version_label (the spec-less identity).

    config sets ``MODEL_VERSION_LABEL`` from ``model_version_label`` (default
    ``v3.0-meta``); a zoo spec then OVERRIDES the module attr at train time. So
    the live attr can't tell us the base — we read the underlying yaml default
    via ``config._cfg`` when present, else fall back to the documented default.
    """
    try:
        raw = getattr(_cfg, "_cfg", None)
        if isinstance(raw, dict):
            return raw.get("model_version_label", "v3.0-meta") or "v3.0-meta"
    except Exception:
        pass
    return "v3.0-meta"


def _training_summary_key(
    date_str: str, spec: "str | None", io: "TrainingIOSpec | None" = None,
) -> str:
    """The training-summary S3 key for this run.

    PR7-7b — a SHADOW run (``io.is_shadow``) writes to the io spec's isolated
    ``predictor/metrics_shadow/{basis}/`` key (shadow is never simultaneously a
    zoo spec). Otherwise: spec-namespaced for a zoo challenger (config#1170),
    the legacy date-only key for the base champion."""
    if io is not None and io.is_shadow:
        return io.summary_key(date_str)
    if spec:
        return _TRAINING_SUMMARY_KEY_SPEC.format(spec=spec, date=date_str)
    return _TRAINING_SUMMARY_KEY.format(date=date_str)


def _build_registry(
    bucket: str,
    date_str: str,
    dry_run: bool,
    *,
    skip_phases: str = "",
    force_phases: str = "",
    force: bool = False,
    spec: "str | None" = None,
    marker_prefix_override: "str | None" = None,
) -> "PhaseRegistry | None":
    """Construct the training :class:`PhaseRegistry` (markers under
    ``predictor/{date}/.phases/`` + watchdog), or ``None`` in dry-run.

    Predictor is the 3rd consumer of the lib phase framework (after the
    backtester + alpha-engine-data). Dry-run returns None so the smoke pass
    (``spot_train.sh`` calls ``main(dry_run=True)``) writes no markers. Recovery
    controls come from explicit args OR, on the spot, the ``PREDICTOR_SKIP_PHASES``
    / ``PREDICTOR_FORCE_PHASES`` / ``PREDICTOR_FORCE`` env vars (args win).
    Watchdog caps read from ``config.FULL_RUN_HARD_CAPS_SECONDS`` — best-effort,
    so a dev/CI run with no ``predictor.yaml`` degrades to an unwatchdogged
    registry (logged) rather than failing the import.

    config#1170 — ``spec`` namespaces the markers per zoo challenger:
    ``predictor/model_zoo/{spec}/{date}/.phases/{phase}.json`` instead of the
    date-only ``predictor/{date}/.phases/{phase}.json``. The base champion
    (spec is None) keeps its legacy path, so each challenger's skip/auto-skip is
    independent: a same-date rerun re-runs only the not-yet-completed / failed
    specs, and a failed spec can retry same-day without a sibling's marker
    short-circuiting it.
    """
    if dry_run:
        return None
    import os as _os

    def _csv(s: str) -> list[str]:
        return [p.strip() for p in (s or "").split(",") if p.strip()]

    skip = _csv(skip_phases) or _csv(_os.environ.get("PREDICTOR_SKIP_PHASES", ""))
    fphases = _csv(force_phases) or _csv(_os.environ.get("PREDICTOR_FORCE_PHASES", ""))
    force_all = bool(force) or _os.environ.get("PREDICTOR_FORCE", "").lower() in ("1", "true", "yes")
    try:
        import config as _cfg
        hard_caps = getattr(_cfg, "FULL_RUN_HARD_CAPS_SECONDS", {}) or {}
    except Exception as _cap_err:  # config unreadable (dev/CI w/o predictor.yaml)
        log.warning("phase watchdog caps unavailable (%s) — running unwatchdogged", _cap_err)
        hard_caps = {}
    # config#1170 — per-spec marker prefix for zoo challengers; the base champion
    # (spec is None) keeps the spec-less "predictor" prefix. PR7-7b — a shadow run
    # passes marker_prefix_override so its phase markers never collide with the
    # live run's markers for the same date.
    if marker_prefix_override:
        marker_prefix = marker_prefix_override
    else:
        marker_prefix = f"predictor/model_zoo/{spec}" if spec else "predictor"
    return PhaseRegistry(
        date=date_str,
        bucket=bucket,
        marker_prefix=marker_prefix,
        skip_phases=skip,
        force=force_all,
        force_phases=fphases,
        hard_caps=hard_caps,
    )


def _maybe_phase(reg: "PhaseRegistry | None", name: str, **log_ctx):
    """Marker-only phase wrapper (watchdog + START/END marker, no auto-skip) for
    stages whose body manages its own hard-fail/best-effort posture inline.
    Returns :func:`contextlib.nullcontext` in dry-run (reg is None)."""
    if reg is None:
        return nullcontext()
    return reg.phase(name, supports_auto_skip=False, **log_ctx)


def _load_training_result(
    bucket: str, date_str: str, spec: "str | None" = None,
    io: "TrainingIOSpec | None" = None,
) -> "dict | None":
    """Reload a prior run's ``result`` dict from the persisted training summary
    so a recovery that AUTO-SKIPS the ~90-min ``meta_training`` phase can still
    drive the downstream summary/risk-model/gate/email/health stages. The summary
    (``predictor/metrics/training_summary_{date}.json``, or the spec-namespaced
    key for a zoo challenger — config#1170) IS the json-serialized ``result`` and
    is the meta_training phase's recorded artifact — so L4524 only auto-skips when
    it exists, and this reload then succeeds. Mirrors the backtester's
    ``_load_predictor_artifacts``. 404 → None (caller refuses to proceed on a
    half-trained state); any other S3 error → raise (fail loud)."""
    import boto3
    from botocore.exceptions import ClientError

    key = _training_summary_key(date_str, spec, io=io)
    try:
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NoSuchBucket", "NotFound"):
            log.warning(
                "meta_training auto-skip: training summary s3://%s/%s absent — "
                "cannot reload result", bucket, key,
            )
            return None
        raise


def _write_training_summary(
    result: dict, bucket: str, date_str: str, spec: "str | None" = None,
    io: "TrainingIOSpec | None" = None,
) -> None:
    """Annotate + persist the training ``result`` to S3 (dated + latest). Extracted
    from the inline Step-2d block so the ``meta_training`` phase can record the
    dated summary as its L4524 artifact (the reload source).

    **Fail-loud (config#1234).** The dated/latest ``training_summary`` is the
    training SSOT: the executor and dashboard read it, and the ``meta_training``
    phase records the dated key as its L4524 artifact — so a swallowed write
    failure is a *ghost success* (the SF/phase reports OK with no artifact). This
    mirrors the research signals.json producer fix (crucible-research#312) and the
    repo's own read-side convention (``_load_training_result``: 404→None, any other
    S3 error→raise). The actual S3 PUT now RE-RAISES on failure; only the
    best-effort prior-cycle ``_annotate_subsample_noise_floor`` enrichment (a
    *read* of last cycle's latest.json, purely for a percent-change reference) is
    kept non-fatal. The enforce-mode freshness monitor (config#1231) is the
    registered-artifact backstop, but the producer must not silently lose the SSOT.

    config#1170 — for a zoo challenger (``spec`` set) the dated summary is written
    to the spec-namespaced key so it's that spec's OWN reload artifact, and the
    shared ``training_summary_latest.json`` champion pointer is NOT overwritten
    (only the base champion advances "latest")."""
    import boto3 as _b3_sum
    _s3_sum = _b3_sum.client("s3")
    # Annotate subsample-noise-floor BEFORE the write — reads the PRIOR cycle's
    # latest.json so the percent-change reference is genuinely prior-cycle rather
    # than this-cycle. This enrichment is best-effort: a failure to read the
    # prior cycle must NOT block persisting the current (load-bearing) summary.
    # PR7-7b — a shadow run reads/writes its OWN isolated latest pointer so it
    # never reads the live champion's prior cycle for the noise-floor reference
    # nor overwrites the live "latest".
    _latest_key = (
        io.summary_latest_key if (io is not None and io.is_shadow)
        else "predictor/metrics/training_summary_latest.json"
    )
    try:
        _annotate_subsample_noise_floor(result, _s3_sum, bucket, latest_key=_latest_key)
    except Exception as _ann_err:  # noqa: BLE001 — prior-cycle read enrichment only
        log.warning(
            "subsample-noise-floor annotation skipped (non-fatal prior-cycle read): %s",
            _ann_err,
        )
    # SSOT write — fail-loud (config#1234): a swallowed failure here is a ghost
    # success for every downstream reader (executor/dashboard) + the phase artifact.
    _sum_body = json.dumps(result, indent=2, default=str).encode()
    _s3_sum.put_object(
        Bucket=bucket,
        Key=_training_summary_key(date_str, spec, io=io),
        Body=_sum_body, ContentType="application/json",
    )
    # The "latest" pointer advances for the base champion AND for a shadow run
    # (its own isolated latest); a zoo challenger (spec set, non-shadow) never
    # touches a "latest".
    _shadow = io is not None and io.is_shadow
    if _shadow or not spec:
        _s3_sum.put_object(
            Bucket=bucket,
            Key=_latest_key,
            Body=_sum_body, ContentType="application/json",
        )
    log.info(
        "Training summary written to S3 (%s)",
        "shadow dated + latest" if _shadow
        else ("dated, spec-namespaced" if spec else "dated + latest"),
    )


def _env_truthy(val: Optional[str]) -> bool:
    """A standard truthy-env reading: set + not one of {"", "0", "false", "no",
    "off"} (case-insensitive). Used by the PREDICTOR_DEFER_TRAINING_EMAIL gate."""
    if val is None:
        return False
    return val.strip().lower() not in ("", "0", "false", "no", "off")


def main(
    bucket: str,
    date_str: Optional[str] = None,
    dry_run: bool = False,
    *,
    skip_phases: str = "",
    force_phases: str = "",
    force: bool = False,
    send_email: bool = True,
    shadow_basis: Optional[str] = None,
) -> dict:
    """Top-level training entrypoint.

    ``shadow_basis`` (PR7-7b): when set (e.g. ``"crsp"``) — or when the
    ``CRSP_SHADOW_ENABLED`` env is truthy (``SHADOW_BASIS`` selects the basis,
    default ``crsp``) — the run becomes an EVIDENCE-ONLY shadow: it reads the
    scratch ``universe_crsp`` library, labels off ``total_return_close``,
    writes every artifact under disjoint ``*_shadow/{basis}/`` prefixes, and is
    hard-blocked from promoting the live champion or entering the model-zoo
    pool. ``None`` + no env (the default) = the live production run, byte-
    identical to before. See :class:`training.io_spec.TrainingIOSpec`.

    Thin wrapper around :func:`_main_impl` that runs the body inside
    ``guard_entrypoint()`` so an uncaught ``raise`` from the spot-train
    run is reported to flow-doctor (email) and re-raised. ``guard_entrypoint``
    no-ops when flow-doctor is inactive (dev / CI / pytest), so this is
    safe everywhere; it activates only on the deployed spot (where
    ``ALPHA_ENGINE_DEPLOYED=1`` is exported by ``infrastructure/spot_train.sh``).
    The spot driver calls ``train_handler.main()`` directly, so wrapping
    here is the true top-level entrypoint.

    ``send_email`` gates the per-run training summary email. The model-zoo
    rotation trains every challenger via ``train_spec`` → ``main()``; without
    this gate each challenger emails, producing the per-challenger email storm
    the consolidated zoo digest replaces. The zoo passes ``send_email=False``;
    the base ``--full-only`` retrain defers its own email to the digest via the
    ``PREDICTOR_DEFER_TRAINING_EMAIL`` env (honored below) so the SF/spot can
    defer without threading the arg through the spot heredoc. The dated
    ``training_summary_{date}.json`` S3 write is ALWAYS performed regardless of
    ``send_email`` (the email is observability over an artifact that already
    persisted).
    """
    # Env override so the spot/SF can defer the base retrain's email to the
    # zoo digest without arg-threading: when PREDICTOR_DEFER_TRAINING_EMAIL is
    # set/truthy, default send_email to False. An explicit send_email=False
    # caller (the zoo) is already covered; the env only flips the default-True
    # callers (the base --full-only spot heredoc, which passes no send_email).
    if send_email and _env_truthy(os.environ.get("PREDICTOR_DEFER_TRAINING_EMAIL")):
        log.info(
            "PREDICTOR_DEFER_TRAINING_EMAIL set — deferring the per-run training "
            "email to the model-zoo digest (training_summary S3 write unaffected)."
        )
        send_email = False
    # Resolve the data-basis + output-path + promotion-eligibility spec from the
    # explicit arg, falling back to CRSP_SHADOW_ENABLED/SHADOW_BASIS env (the
    # spot heredoc flips shadow mode with an env export, mirroring
    # PREDICTOR_DEFER_TRAINING_EMAIL). live() when neither selects shadow.
    from training.io_spec import TrainingIOSpec
    io = TrainingIOSpec.resolve(shadow_basis)
    if io.is_shadow:
        log.info(
            "SHADOW TRAINING RUN (basis=%s) — universe_lib=%s close_col=%s; "
            "outputs isolated under %s / %s; promotion + model-zoo registration "
            "DISABLED (evidence-only).",
            io.shadow_basis, io.universe_lib, io.close_col,
            io.weights_prefix, io.summary_key_tmpl,
        )
    with guard_entrypoint():
        return _main_impl(
            bucket, date_str=date_str, dry_run=dry_run,
            skip_phases=skip_phases, force_phases=force_phases, force=force,
            send_email=send_email, io=io,
        )


def _main_impl(
    bucket: str,
    date_str: Optional[str] = None,
    dry_run: bool = False,
    *,
    skip_phases: str = "",
    force_phases: str = "",
    force: bool = False,
    send_email: bool = True,
    io: "TrainingIOSpec | None" = None,
) -> dict:
    """
    Entry point called from inference/handler.py for the "train" action.

    1.  Load the universe (OHLCV + features) from the ArcticDB universe
        library to /tmp. ArcticDB is canonical since PR #6 (2026-04-16);
        there is no yfinance fetch and no S3-parquet refresh/upload.
    2.  Run GBM training pipeline on the loaded cache.
    2b. Write a 2-year slim cache to S3 (predictor/price_cache_slim/)
        for the daily inference Lambda's price reads.
    3.  Send training summary email.

    Returns the result dict from run_meta_training().
    """
    if date_str is None:
        # Trading-day axis (config#1015): the Saturday SF runs on a calendar
        # date that is NOT a trading day, so a calendar now() keyed the training
        # summary / meta-archive / oos_rows / risk_model artifacts on Saturday
        # instead of Friday's close — contaminating the predictor artifact chain.
        # Mirror model_zoo's default: resolve the trading_day via now_dual,
        # falling back to calendar only if the lib lookup fails.
        try:
            from krepis.dates import now_dual
            _td = now_dual().trading_day
            date_str = _td.isoformat() if hasattr(_td, "isoformat") else str(_td)
            log.info("train_handler: no date passed — defaulting to trading_day %s", date_str)
        except Exception:  # noqa: BLE001 — date defaulting must not block training
            date_str = datetime.now().strftime("%Y-%m-%d")
            log.warning("train_handler: could not resolve trading_day via now_dual; "
                        "fell back to calendar date %s", date_str, exc_info=True)

    # setup_logging already ran at module-top (see comment near the
    # krepis.logging import). Apply standard log level here.
    logging.getLogger().setLevel(logging.INFO)

    # PR7-7b: default to the live spec when called directly (e.g.
    # inference/handler.py "train" action) without an explicit io.
    from training.io_spec import TrainingIOSpec
    if io is None:
        io = TrainingIOSpec.live()

    # config#1170 — the zoo-challenger spec namespace (None for the base
    # champion). Derived from cfg.MODEL_VERSION_LABEL, which model_zoo sets per
    # spec via spec_overrides around this main() call. Namespaces the phase
    # markers + the meta_training reload artifact PER SPEC so a same-date rerun
    # re-runs only the not-yet-completed / failed specs (and a failed spec can
    # retry same-day) instead of every spec skipping on the first spec's marker.
    spec_ns = _spec_namespace()
    if spec_ns:
        log.info("model-zoo challenger run: phase markers namespaced under spec=%s", spec_ns)

    # Phase registry: markers under predictor/{date}/.phases/ (or, for a zoo
    # challenger, predictor/model_zoo/{spec}/{date}/.phases/) + watchdog +
    # meta_training auto-skip-reload (L4528, 3rd lib consumer). None in dry-run.
    reg = _build_registry(
        bucket, date_str, dry_run,
        skip_phases=skip_phases, force_phases=force_phases, force=force,
        spec=spec_ns,
        marker_prefix_override=(
            f"predictor/shadow/{io.shadow_basis}" if io.is_shadow else None
        ),
    )

    # Preflight — fail fast on env / connectivity / ArcticDB staleness
    # before a 90-minute training run commits to stale data. See PR #6
    # and training/preflight.py.
    from training.preflight import TrainingPreflight
    with _maybe_phase(reg, "preflight"):
        TrainingPreflight(bucket=bucket).run()

    log.info("GBM training run: date=%s  bucket=%s  dry_run=%s", date_str, bucket, dry_run)

    # Step 1: Load price cache from ArcticDB. PR #6 removed the S3 parquet
    # fallback — it masked the PR #3 miswiring for a week (same root cause
    # as the inference-path issue fixed in PR #5). Preflight above already
    # verified macro/SPY freshness, so ArcticDB is known reachable here.
    # supports_auto_skip=False (via _maybe_phase): the cache lands in local
    # /tmp, which is empty on a fresh recovery spot, so it must always re-run.
    tmp_cache = Path(tempfile.mkdtemp()) / "cache"
    from store.arctic_reader import download_from_arctic
    log.info("[data_source=arcticdb] Loading universe from ArcticDB...")
    with _maybe_phase(reg, "data_load"):
        n_files = download_from_arctic(
            bucket=bucket, local_dir=tmp_cache, universe_lib=io.universe_lib,
        )
        if n_files == 0:
            raise RuntimeError(
                f"ArcticDB returned zero files for training — "
                f"universe library is empty or unreachable at bucket={bucket}. "
                "Check Saturday DataPhase1 + weekly backfill ran cleanly."
            )

    # Step 1b: Price cache refresh now handled by alpha-engine-data (Phase 1).
    # The data repo runs weekly_collector.py --phase 1 before training,
    # so S3 parquets are already current.
    log.info("Price cache refresh: skipped (handled by alpha-engine-data)")

    # Step 2 (+ 2d): Train + upload (v3 meta-model only — v2 single-GBM and
    # multi-horizon dispatch branches removed 2026-04-13; v2 machinery itself
    # still in-tree, full rip-out tracked in ROADMAP) AND the training-summary
    # write, folded into ONE `meta_training` phase. The dated summary
    # (predictor/metrics/training_summary_{date}.json) IS this phase's recorded
    # L4524 artifact AND the reload source — so on a recovery the phase
    # AUTO-SKIPS only when training fully completed (weights + summary on S3),
    # and `result` is reloaded from the summary instead of re-running the
    # ~90-min long pole. A crash between training and the summary write leaves
    # the artifact absent → the marker is invalid → a clean re-train. The
    # watchdog hard-cap on this phase converts an OOM (today an opaque SSM
    # TimedOut, L4511) into a PhaseTimeoutError + faulthandler dump.
    from training.meta_trainer import run_meta_training

    def _train_and_summarize() -> dict:
        _r = run_meta_training(
            data_dir=str(tmp_cache), bucket=bucket, date_str=date_str, dry_run=dry_run,
            io=io,
        )
        # Slim cache write + feature-store registry upload are handled by
        # alpha-engine-data (Phase 1) — nothing to do here.
        log.info("Slim cache write: skipped (handled by alpha-engine-data)")
        if not dry_run:
            _write_training_summary(_r, bucket, date_str, spec_ns, io=io)
        return _r

    if reg is None:
        result = _train_and_summarize()
    else:
        with reg.phase("meta_training", supports_auto_skip=True) as ctx:
            if ctx.skipped:
                result = _load_training_result(bucket, date_str, spec_ns, io=io)
                if result is None:
                    raise RuntimeError(
                        "meta_training marker is ok but the training summary is absent "
                        "— refusing to proceed on a half-trained state. Re-run with "
                        "--force-phases=meta_training (PREDICTOR_FORCE_PHASES=meta_training)."
                    )
                log.info(
                    "meta_training auto-skipped (%s) — reloaded result from summary",
                    ctx.skip_reason,
                )
            else:
                result = _train_and_summarize()
                ctx.record_artifact(_training_summary_key(date_str, spec_ns, io=io))

    # Step 2d2: Factor-risk-model F + D weekly persistence (ROADMAP C.2b).
    # Reads the per-ticker parquets the train_handler already populated
    # to ``tmp_cache`` from ArcticDB, extracts close → log returns + the
    # 8 ``*_zscore`` factor-loading columns (C.1, alpha-engine-data
    # #324), runs the Fama-MacBeth cross-sectional regression
    # (C.2a, ``risk_model.build_factor_risk_model``), and persists F + D
    # parquets to ``s3://{bucket}/risk_model/{date}/``. Non-blocking:
    # failure here doesn't abort the training pipeline. Auto-skips
    # when loading-column coverage is insufficient (pre-#324 cache).
    #
    # The actual wiring of Σ = B · F · Bᵀ + D into
    # ``executor.portfolio_optimizer.solve_target_weights`` is C.3 —
    # under a hard sequencing constraint per the plan doc: do NOT
    # wire until B.5 (α̂-uncertainty cutover gate) passes. Two
    # simultaneous Σ-substrate changes make backtester regressions
    # untraceable. C.2b ships the weekly persistence so by the time
    # C.3 reads ``risk_model/{date}/``, there are ≥4 weeks of F + D
    # accumulated.
    # PR7-7b: skipped on a shadow run — risk_model_persist writes to the SHARED
    # live ``risk_model/{date}/`` keys, which an evidence-only run must not
    # touch. It is also irrelevant to the champion-vs-shadow IC comparison.
    if not dry_run and not io.write_side_artifacts:
        log.info(
            "Shadow run (basis=%s) — skipping risk_model_persist (writes shared "
            "live risk_model/ keys; out of scope for the IC comparison).",
            io.shadow_basis,
        )
    elif not dry_run:
        try:
            from training.risk_model_persist import (
                build_and_persist_risk_model,
            )
            with _maybe_phase(reg, "risk_model_persist"):
                _rm_payload = build_and_persist_risk_model(
                    data_dir=str(tmp_cache),
                    bucket=bucket,
                    date_str=date_str,
                )
            result["risk_model"] = _rm_payload
            log.info(
                "risk_model_persist: status=%s",
                _rm_payload.get("status"),
            )
        except Exception as _rm_err:
            log.warning(
                "risk_model_persist failed (non-blocking): %s",
                _rm_err, exc_info=True,
            )
            # config#2443 — a swallowed exception here previously left NO
            # alertable trace: training completes green, the training summary
            # has no "risk_model" key, and the ARTIFACT_REGISTRY freshness
            # monitor is a `severity: warning` row on a 10-calendar-day
            # recency window (config#1297) — it does not page on a single
            # missed saturday_sf cycle, only flags the dashboard. A whole week
            # could silently pass with zero human-visible signal. Route the
            # failure through the same best-effort ops-alert channel
            # model_zoo.py's rotation alerts use, so an operator actually
            # sees it. Alert failure must never fail training (mirrors the
            # try/except posture around every other publish_ops_alert call
            # site in this repo).
            try:
                from ops_alerts import publish_ops_alert

                publish_ops_alert(
                    message=(
                        f"risk_model_persist failed for date={date_str} "
                        f"(non-blocking — training completed normally): "
                        f"{_rm_err}"
                    ),
                    severity="warning",
                    source="alpha-engine-predictor/training/train_handler.py::risk_model_persist",
                    dedup_key=f"risk_model_persist_failed_{date_str}",
                )
            except Exception:  # noqa: BLE001 — alert failure must not fail the SF
                log.warning("risk_model_persist: failure alert itself failed", exc_info=True)

    # Step 2e: Triple-barrier cutover gate (Stage 3 PR 5 SF wiring).
    # Runs after model upload + training summary so the gate has access
    # to the freshly-promoted meta_model_tb.pkl and the most recent ~6
    # weeks of inference predictions. Result lands at
    # ``s3://{bucket}/predictor/variant_gates/triple_barrier_{date}.json``
    # and is merged into the training summary dict so the email and
    # downstream consumers can surface it.
    #
    # Failure is non-blocking — the gate is observe-only until
    # ``triple_barrier.enforce_cutover`` flips in alpha-engine-config.
    # Until ~6-9 weeks of parallel-observation data accumulate (PR 3
    # inference path needs to write meta_alpha_tb fields + the 21d
    # forward window needs to close), the gate naturally returns an
    # insufficient-data verdict per ``min_pairs=100``.
    # PR7-7b: skipped on a shadow run — the cutover gate writes to the shared
    # live ``predictor/variant_gates/`` key and reads the live inference
    # prediction stream; neither applies to an evidence-only basis comparison.
    if not dry_run and not io.write_side_artifacts:
        log.info(
            "Shadow run (basis=%s) — skipping triple-barrier cutover gate "
            "(writes shared live variant_gates/ keys).",
            io.shadow_basis,
        )
    elif not dry_run:
        try:
            from analysis.triple_barrier_cutover_runner import run_gate as _run_tb_gate
            with _maybe_phase(reg, "triple_barrier_gate"):
                _gate_payload = _run_tb_gate(
                    bucket=bucket,
                    n_days=42,
                    horizon_days=21,
                    threshold=1.15,
                    write_to_s3=True,
                )
            result["triple_barrier_cutover_gate"] = {
                "passed": _gate_payload.get("passed"),
                "n_pairs": _gate_payload.get("n_pairs"),
                "baseline_ic": _gate_payload.get("baseline_ic"),
                "variant_ic": _gate_payload.get("variant_ic"),
                "relative_lift": _gate_payload.get("relative_lift"),
                "n_realized_filled": _gate_payload.get("n_realized_filled"),
                "reason": _gate_payload.get("reason"),
            }
            log.info(
                "Stage 3 cutover gate: passed=%s n_pairs=%d lift=%s",
                _gate_payload.get("passed"),
                _gate_payload.get("n_pairs", 0),
                _gate_payload.get("relative_lift"),
            )
        except Exception as _gate_err:
            log.warning(
                "Stage 3 cutover gate failed (non-blocking, observe-only): %s",
                _gate_err,
            )

    # Step 3: Email — gated on send_email (model-zoo passes False so each
    # challenger does NOT email; the base --full-only retrain defers via
    # PREDICTOR_DEFER_TRAINING_EMAIL → the consolidated zoo digest). The dated
    # training_summary_{date}.json S3 write happened above regardless, so a
    # suppressed email loses no DATA — only the per-run notification.
    if not dry_run and send_email:
        send_training_email(result, date_str)
    elif not dry_run:
        log.info(
            "Per-run training email suppressed (send_email=False) — summary "
            "persisted to predictor/metrics/training_summary_%s.json; the "
            "model-zoo digest reports champion + challengers + promotion.",
            date_str,
        )

    # Step 4: Health status
    if not dry_run:
        try:
            from nousergon_lib.health import Deliverable, write_health
            write_health(
                module_name="predictor_training",
                deliverables=[
                    Deliverable(name="training_run", required=True, produced=True),
                ],
                run_date=date_str,
                duration_seconds=result.get("train_time_seconds", 0),
                summary={
                    "promoted": result.get("promoted", False),
                    "ic_30d": result.get("ic_30d"),
                    "n_train": result.get("n_train"),
                    "slim_cache_tickers": result.get("slim_cache_tickers"),
                    "slim_cache_failed": result.get("slim_cache_failed", 0),
                },
                bucket=bucket,
            )
        except Exception as _he:
            log.warning("Health status write failed: %s", _he)

        # Data manifest
        try:
            from data_manifest import write_data_manifest
            write_data_manifest(
                bucket=bucket,
                module_name="predictor_training",
                run_date=date_str,
                manifest={
                    "promoted": result.get("promoted", False),
                    "promoted_mode": result.get("promoted_mode"),
                    "test_ic": result.get("test_ic"),
                    "n_train": result.get("n_train"),
                    "n_test": result.get("n_test"),
                    "slim_cache_tickers": result.get("slim_cache_tickers"),
                    "slim_cache_failed": result.get("slim_cache_failed", 0),
                },
            )
        except Exception as _me:
            log.warning("Data manifest write failed: %s", _me)

    return result

