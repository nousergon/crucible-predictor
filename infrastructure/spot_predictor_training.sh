#!/usr/bin/env bash
# infrastructure/spot_predictor_training.sh — Run GBM champion retraining on a
# dedicated spot EC2 instance (PredictorTraining SF state).
#
# Sources infrastructure/_spot_common.sh for the shared spot-launch, SSM,
# cleanup, bootstrap, and dependency infrastructure.
#
# Supports:
#   --shadow-basis <basis>   — evidence-only total-return shadow retrain
#   --preflight-only         — boot + import/ArcticDB probe, exit 0 (NO writes)
#   --smoke-only             — boot + dry-run training, exit 0 (NO promotion)
#   --smoke-first            — dry-run smoke, THEN full training (manual only)
#   --instance-type <type>   — override instance type
#
# Usage:
#   ./infrastructure/spot_predictor_training.sh                        # full training
#   ./infrastructure/spot_predictor_training.sh --preflight-only         # Friday dry path
#   ./infrastructure/spot_predictor_training.sh --smoke-only             # smoke only
#   ./infrastructure/spot_predictor_training.sh --smoke-first            # smoke, then full
#   ./infrastructure/spot_predictor_training.sh --shadow-basis crsp      # shadow retrain
#   ./infrastructure/spot_predictor_training.sh --instance-type c5.2xlarge
#
# No mode flag is needed — this script IS the PredictorTraining runner. Use
# spot_train_spec_dispatch.sh for --model-zoo-spec and spot_model_zoo_select.sh
# for --model-zoo-weekly / --model-zoo-select.

set -euo pipefail

# Captured BEFORE the source: `_spot_common.sh` still assigns a default to
# MAX_RUNTIME_SECONDS, so a post-source `${MAX_RUNTIME_SECONDS:-...}` here is a
# no-op against a parameter that is already non-empty and an operator override
# would be indistinguishable from the shared default. Capture, then decide.
_ENV_MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_spot_common.sh"

# ── Override stage-specific defaults ──────────────────────────────────────────
# (Must happen AFTER source so they take effect before the functions below use them)
_SPOT_NAME="${_SPOT_NAME:-predictor-training}"
_SSM_SLUG="${_SSM_SLUG:-spot-full-training}"
_PROCESS_NAME="${_PROCESS_NAME:-predictor-training}"

# ── Timeout ordering: the inner ceiling must be STRICTLY LESS than the outer ──
#
# This launcher runs INSIDE the weekly SF's PredictorTraining sendCommand, whose
# `executionTimeout` is SF_STAGE_EXECUTION_TIMEOUT_SECONDS below. That outer
# budget covers EVERYTHING this script does: spot launch, instance-running wait,
# SSM-agent wait, config staging, bootstrap, pip install, and only then the
# workload. The inner `run_ssm "full-training" ... "$MAX_RUNTIME_SECONDS"`
# ceiling used to be set to the SAME 5400s.
#
# An inner ceiling EQUAL to the outer can never bind, so it can never name
# itself. The run dies at the outer boundary instead — SSM reports the outer
# command Failed/DeliveryTimedOut with no message attributing it to training,
# no partial artifact, and no line in the log saying which step was still
# running. The whole point of a per-step ceiling is that the step that ran out
# of time is the step that says so.
#
# So the inner ceiling is DERIVED, never a literal: outer minus a stated
# overhead reserve. The reserve covers the pre-workload phases above; measured
# boot alone is ~7 min and the pip install adds several more, so 20 min is the
# reserve with headroom. If SF_EXECUTION_TIMEOUT is exported by the caller
# (the #883 spot-reclaim coupling variable, already read by _spot_common.sh) it
# wins over the mirrored literal — that is the path by which this stops being a
# mirror at all, once the SF exports its own budget.
#
# Consequence to know: if champion retraining ever legitimately needs more than
# SPOT_WORKLOAD_MAX_RUNTIME_SECONDS, the fix is to raise the OUTER budget in
# nousergon-data `infrastructure/step_function.json` FIRST and let this derive
# upward. Raising this alone re-creates the exact non-binding ceiling above.
#
# Mirror source of truth: nousergon-data `infrastructure/step_function.json`,
# state ResearchPredictorParallel/PredictorTraining,
# Parameters.Parameters.executionTimeout (and Parameters.TimeoutSeconds).
SF_STAGE_EXECUTION_TIMEOUT_SECONDS="${SF_EXECUTION_TIMEOUT:-5400}"
# Pre-workload reserve: launch + instance-running wait + SSM-agent wait +
# config staging + bootstrap + pip install.
SPOT_PREWORKLOAD_RESERVE_SECONDS="${SPOT_PREWORKLOAD_RESERVE_SECONDS:-1200}"
SPOT_WORKLOAD_MAX_RUNTIME_SECONDS=$((SF_STAGE_EXECUTION_TIMEOUT_SECONDS - SPOT_PREWORKLOAD_RESERVE_SECONDS))

if [ "$SPOT_WORKLOAD_MAX_RUNTIME_SECONDS" -le 0 ]; then
  echo "ERROR: pre-workload reserve (${SPOT_PREWORKLOAD_RESERVE_SECONDS}s) >= the outer stage budget (${SF_STAGE_EXECUTION_TIMEOUT_SECONDS}s) — no time is left for training." >&2
  exit 2
fi

# An explicit operator override still wins; the derived value is the default.
MAX_RUNTIME_SECONDS="${_ENV_MAX_RUNTIME_SECONDS:-$SPOT_WORKLOAD_MAX_RUNTIME_SECONDS}"

# ── Parse flags ──────────────────────────────────────────────────────────────
# This script has no --mode flags — it IS the PredictorTraining runner.
# Sub-modes: --preflight-only, --smoke-only.
MODE="full-only"  # full-only | preflight-only | smoke-only | smoke-first
SHADOW_BASIS=""

_ORIG_ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --preflight-only) MODE="preflight-only"; PREFLIGHT_ONLY=1 ;;
    --smoke-only) MODE="smoke-only" ;;
    --smoke-first) MODE="smoke-first" ;;
    --shadow-basis) shift; SHADOW_BASIS="$1" ;;
    --instance-type) shift; INSTANCE_TYPE="$1" ;;
    *) echo "ERROR: unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ -n "$SHADOW_BASIS" ]; then
  if [ "$SHADOW_BASIS" != "crsp" ]; then
    echo "ERROR: --shadow-basis: only 'crsp' is supported (got '$SHADOW_BASIS')" >&2
    exit 2
  fi
fi

if [ -n "$SHADOW_BASIS" ]; then
  SHADOW_EXPORT="export CRSP_SHADOW_ENABLED=true SHADOW_BASIS=${SHADOW_BASIS}"$'\n'
else
  SHADOW_EXPORT=""
fi

# Defer-email propagation
if [ -n "${PREDICTOR_DEFER_TRAINING_EMAIL:-}" ]; then
  DEFER_EMAIL_EXPORT="export PREDICTOR_DEFER_TRAINING_EMAIL=${PREDICTOR_DEFER_TRAINING_EMAIL}"$'\n'
else
  DEFER_EMAIL_EXPORT=""
fi

# Collapse --instance-type to single value
if [ -n "$INSTANCE_TYPE" ]; then
  INSTANCE_TYPES="$INSTANCE_TYPE"
fi

# ── Preflight checks ─────────────────────────────────────────────────────────
resolve_or_stage_predictor_config "$(cd "$SCRIPT_DIR/.." && pwd)/config/predictor.yaml"

echo "═══════════════════════════════════════════════════════════════"
echo "  GBM Predictor Training — $(date +%Y-%m-%d)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Instance types: $INSTANCE_TYPES"
echo "  Subnets       : $SUBNETS"
echo "  AMI           : $AMI_ID"
echo "  Region        : $AWS_REGION"
echo "  Branch        : $BRANCH"
echo "  Mode          : $MODE"
echo "  Shadow basis  : ${SHADOW_BASIS:-<none> (live)}"
echo "  S3 bucket     : $S3_BUCKET"
echo "  Spot attempt  : $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS"
echo ""

# ── Launch spot + wait + config staging ──────────────────────────────────────
spot_launch
trap 'cleanup "$_SSM_SLUG"' EXIT

aws ec2 wait instance-running --instance-ids "$_INSTANCE_ID" --region "$AWS_REGION"
stage_config "$(cd "$SCRIPT_DIR/.." && pwd)/config/predictor.yaml" "predictor.yaml"
wait_ssm_agent

# ── Bootstrap + deps ─────────────────────────────────────────────────────────
bootstrap_spot
install_deps

# ── Preflight-only (Friday shell_run dry path) ───────────────────────────────
# Shared across all three per-stage scripts — see _spot_common.sh.
maybe_run_preflight_only_and_exit

# ── Smoke test (dry_run=True) ────────────────────────────────────────────────
#
# GUARDED — it was not, and that is what put the dry-run path on the critical
# path of the Saturday SF.
#
# The monolith `spot_train.sh` skipped this step whenever MODE was `full-only`
# (its line: `if [ "$MODE" != "full-only" ] && ... ; then`), and the SF invokes
# the champion retrain in exactly that mode. When I4442/I4497 split the monolith
# into per-stage scripts, `full-only` became this script's DEFAULT mode and the
# guard was dropped, so every SF run performed a complete `dry_run=True`
# training pass and THEN a complete `dry_run=False` one: double the spot wall
# time, double the ArcticDB read cost, and — the part that actually broke the
# pipeline — a code path that was previously UNREACHABLE from the SF became a
# hard prerequisite for reaching the real training. The 2026-08-11 run died
# here, in the dry run, having never attempted the work it exists to do.
#
# Modes that run the smoke step:
#   smoke-only   — smoke, then exit 0 (the standalone dry check)
#   smoke-first  — smoke, then full training (the monolith's old bare/`both`
#                  invocation, kept for manual use so nothing is lost)
# Modes that skip it:
#   full-only    — the DEFAULT, and the mode the weekly SF uses
if [ "$MODE" = "smoke-only" ] || [ "$MODE" = "smoke-first" ]; then
print_banner "SMOKE TEST (dry_run=True)"
_heartbeat_start "spot-${_SSM_SLUG}" 300
run_ssm "smoke" "${_RUN_TOKEN_EXPORT}$(cat <<'SMOKE'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
export ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
cat > /tmp/spot-smoke.py <<'PYEOF'
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', os.environ.get('S3_BUCKET', 'alpha-engine-research'))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s')

from training.train_handler import main as train_main
result = train_main(bucket=os.environ.get('S3_BUCKET', 'alpha-engine-research'), dry_run=True)

print()
print('=' * 60)
print('  SMOKE TEST RESULTS')
print('=' * 60)
v = result.get('model_version', '')
is_meta = 'meta' in str(v).lower()
if is_meta:
    print('  Architecture:   v3.0 Meta-Model')
    print('  Meta-Model IC:  %s' % result.get('meta_model_ic', result.get('test_ic', 'n/a')))
    wf = result.get('walk_forward', {})
    wf_status = 'PASS' if wf.get('passes_wf') else 'FAIL'
    print('  WF Status:      %s' % wf_status)
else:
    print('  Architecture:   v2.0 Single/Ensemble GBM')
    print('  Test IC:        %s' % result.get('test_ic', 'n/a'))
    wf = result.get('walk_forward', {})
    wf_status = 'PASS' if wf.get('passes_wf') else 'FAIL/skipped'
    print('  Walk-forward:   %s' % wf_status)
print('  Promoted:       %s' % result.get('promoted', 'n/a'))
print('  Elapsed:        %ss' % result.get('elapsed_s', 'n/a'))
print('=' * 60)
PYEOF
$PY -m krepis.ssm_log_capture run --slug spot-smoke --log /var/log/spot-smoke.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-smoke.py
SMOKE
)" 1800
_heartbeat_stop

if [ "$MODE" = "smoke-only" ]; then
  echo "==> Smoke-only mode — skipping full training."
  exit 0
fi
else
  echo "==> Mode '$MODE' — skipping the dry_run=True smoke step (see the guard above)."
fi

# ── Full training (dry_run=False) ────────────────────────────────────────────
print_banner "FULL TRAINING (dry_run=False)"
_heartbeat_start "spot-${_SSM_SLUG}" 300
run_ssm "full-training" "${_RUN_TOKEN_EXPORT}${DEFER_EMAIL_EXPORT}${SHADOW_EXPORT}$(cat <<'TRAIN'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
export ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
cat > /tmp/spot-full-training.py <<'PYEOF'
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', os.environ.get('S3_BUCKET', 'alpha-engine-research'))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s')

from training.train_handler import main as train_main
result = train_main(bucket=os.environ.get('S3_BUCKET', 'alpha-engine-research'), dry_run=False)

print()
print('=' * 60)
print('  FULL TRAINING RESULTS')
print('=' * 60)
v = result.get('model_version', '')
is_meta = 'meta' in str(v).lower()
if is_meta:
    print('  Architecture:   v3.0 Meta-Model')
    print('  Meta-Model IC:  %s' % result.get('meta_model_ic', result.get('test_ic', 'n/a')))
    wf = result.get('walk_forward', {})
    wf_status = 'PASS' if wf.get('passes_wf') else 'FAIL/skipped'
    print('  WF Status:      %s' % wf_status)
    coefs = result.get('meta_coefficients', {})
    if coefs:
        print('  Meta-model coefficients:')
        for name, val in sorted(coefs.items(), key=lambda x: -abs(x[1])):
            if name != 'intercept' and abs(val) > 0.0001:
                print('    %-30s %+.4f' % (name, val))
else:
    print('  Architecture:   v2.0 Single/Ensemble GBM')
    print('  Test IC:        %s' % result.get('test_ic', 'n/a'))
    wf = result.get('walk_forward', {})
    wf_status = 'PASS' if wf.get('passes_wf') else 'FAIL/skipped'
    print('  Walk-forward:   %s' % wf_status)
print('  Promoted:       %s' % result.get('promoted', 'n/a'))
print('  Promoted mode:  %s' % result.get('promoted_mode', 'n/a'))
print('  Elapsed:        %ss' % result.get('elapsed_s', 'n/a'))
print('=' * 60)
PYEOF
$PY -m krepis.ssm_log_capture run --slug spot-full-training --log /var/log/spot-full-training.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-full-training.py
TRAIN
)" "${MAX_RUNTIME_SECONDS}"
_heartbeat_stop

# ── Completion ───────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Training complete."
echo "═══════════════════════════════════════════════════════════════"
emit_heartbeat
