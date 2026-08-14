#!/usr/bin/env bash
# infrastructure/spot_train_spec_dispatch.sh — Train ONE challenger model-zoo
# spec on a dedicated spot EC2 (TrainSpecDispatch SF Map iteration).
#
# Sources infrastructure/_spot_common.sh for shared spot-launch, SSM, cleanup,
# bootstrap, and dependency infrastructure.
#
# The SF ModelZooTrainMap launches one spot per spec id; this trains exactly
# that spec as a challenger (challenger-first + live-contract restore enforced
# inside model_zoo.train_one_spec) and exits NON-ZERO only on a real training
# failure — so the Map iteration records THIS spec's failure without aborting
# siblings.
#
# No selection/promotion happens here — that is the separate ModelZooSelect
# state (spot_model_zoo_select.sh --select-only).
#
# Usage:
#   ./infrastructure/spot_train_spec_dispatch.sh --spec-id <id>       # train one challenger
#   ./infrastructure/spot_train_spec_dispatch.sh --spec-id <id> --instance-type c5.2xlarge
#   ./infrastructure/spot_train_spec_dispatch.sh --spec-id <id> --preflight-only  # Friday dry path
#
# Required: --spec-id <id> (the MODEL_ZOO_SPEC_ID from the SF Map payload)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_spot_common.sh"

# ── Override stage-specific defaults ──────────────────────────────────────────
_SPOT_NAME="${_SPOT_NAME:-train-spec-dispatch}"
_SSM_SLUG="${_SSM_SLUG:-spot-model-zoo-spec}"
_PROCESS_NAME="${_PROCESS_NAME:-predictor-model-zoo-spec}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-5400}"

# Stage-coverage identity (config-I7214). This script IS the SF's
# TrainSpecDispatch Map-iteration Task (nousergon-data
# infrastructure/step_function.json,
# ResearchPredictorParallel/ModelZooTrainMap/TrainSpecDispatch) — NOT
# "ResolveZooSpecs", which is a separate inline `python -m training.model_zoo
# list-rotation-specs` SSM command with no launcher script of its own.
_COVERAGE_STAGE="TrainSpecDispatch"

# ── Parse flags ──────────────────────────────────────────────────────────────
MODEL_ZOO_SPEC_ID=""

_ORIG_ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --spec-id) shift; MODEL_ZOO_SPEC_ID="$1" ;;
    --instance-type) shift; INSTANCE_TYPE="$1" ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    *) echo "ERROR: unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ -z "$MODEL_ZOO_SPEC_ID" ]; then
  echo "ERROR: --spec-id is required (got empty)" >&2
  exit 2
fi

# Collapse --instance-type to single value
if [ -n "$INSTANCE_TYPE" ]; then
  INSTANCE_TYPES="$INSTANCE_TYPE"
fi

# compute spec-id export line for the heredoc prefix
MZ_SPEC_EXPORT="export MODEL_ZOO_SPEC_ID=${MODEL_ZOO_SPEC_ID}"$'\n'

# ── Preflight checks ─────────────────────────────────────────────────────────
resolve_or_stage_predictor_config "$(cd "$SCRIPT_DIR/.." && pwd)/config/predictor.yaml"

echo "═══════════════════════════════════════════════════════════════"
echo "  MODEL-ZOO TRAIN ONE SPEC: ${MODEL_ZOO_SPEC_ID}"
echo "═══════════════════════════════════════════════════════════════"
echo "  Instance types: $INSTANCE_TYPES"
echo "  Subnets       : $SUBNETS"
echo "  Region        : $AWS_REGION"
echo "  Branch        : $BRANCH"
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
maybe_run_preflight_only_and_exit

# ── Train one spec ───────────────────────────────────────────────────────────
print_banner "MODEL-ZOO TRAIN ONE SPEC: ${MODEL_ZOO_SPEC_ID}"

run_ssm "model-zoo-spec" "${_RUN_TOKEN_EXPORT}${MZ_SPEC_EXPORT}$(cat <<'ZOOSPEC'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
export ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
cat > /tmp/spot-model-zoo-spec.py <<'PYEOF'
import os, sys
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', os.environ.get('S3_BUCKET', 'alpha-engine-research'))
bucket = os.environ.get('S3_BUCKET', 'alpha-engine-research')

import logging
import os.path as _osp
_FD_YAML = _osp.join(_osp.abspath("."), "flow-doctor-model-zoo.yaml")
from krepis.logging import setup_logging
setup_logging("predictor-model-zoo", flow_doctor_yaml=_FD_YAML, exclude_patterns=[])

import config as cfg
from training.model_zoo import train_one_spec

log = logging.getLogger('model_zoo.spot')
spec_id = os.environ.get('MODEL_ZOO_SPEC_ID', '')
log.info('model_zoo train-spec probe: spec=%s  MODEL_SPECS=%d', spec_id, len(getattr(cfg, 'MODEL_SPECS', [])))
if not spec_id:
    raise SystemExit('MODEL_ZOO_SPEC_ID not set on the spot')

try:
    from krepis.dates import now_dual
    _td = now_dual().trading_day
    date_str = _td.isoformat() if hasattr(_td, 'isoformat') else str(_td)
except Exception:
    log.warning('model_zoo spot: now_dual unavailable', exc_info=True)
    date_str = None

train_one_spec(spec_id, bucket, date_str=date_str)
print()
print('=' * 60)
print('  MODEL-ZOO TRAIN-SPEC ' + spec_id + ' COMPLETE')
print('=' * 60)
PYEOF
$PY -m krepis.ssm_log_capture run --slug spot-model-zoo-spec --log /var/log/spot-model-zoo-spec.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-model-zoo-spec.py
ZOOSPEC
)" "${MAX_RUNTIME_SECONDS}"

emit_heartbeat

# Per-stage output assertion (config-I7214, sf-pipeline-policy.md §2.1):
# assert THIS stage wrote what it declared, at the boundary where the fact
# becomes knowable. OBSERVE MODE — it can never fail the stage.
"$LIB_PYTHON" -m nousergon_lib.stage_coverage assert --stage "$_COVERAGE_STAGE" --window-start "$_STAGE_WINDOW_START" || echo "WARNING: stage-coverage assertion did not run for $_COVERAGE_STAGE (rc=$?) — observe mode, stage NOT failed (config-I7214)" >&2

echo ""
echo "==> Model-zoo train-spec ${MODEL_ZOO_SPEC_ID} complete. Instance will be terminated."
