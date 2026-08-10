#!/usr/bin/env bash
# infrastructure/spot_model_zoo_select.sh — Run model-zoo rotation, selection,
# and leaderboard promotion on a dedicated spot EC2 (ModelZooSelect SF state).
#
# Sources infrastructure/_spot_common.sh for the shared spot-launch, SSM,
# cleanup, bootstrap, and dependency infrastructure.
#
# Two modes:
#   --weekly        — train N stalest challenger specs, rank by CPCV,
#                     promote winner (ModelZooWeekly SF state)
#   --select-only   — select over already-trained specs, write leaderboard,
#                     promote winner (ModelZooSelect SF state)
#
# Usage:
#   ./infrastructure/spot_model_zoo_select.sh --weekly              # rotation + selection
#   ./infrastructure/spot_model_zoo_select.sh --select-only         # select only (after Map joins)
#   ./infrastructure/spot_model_zoo_select.sh --weekly --instance-type c5.2xlarge
#   ./infrastructure/spot_model_zoo_select.sh --select-only --instance-type c5.xlarge
#   ./infrastructure/spot_model_zoo_select.sh --select-only --preflight-only  # Friday dry path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_spot_common.sh"

# ── Override stage-specific defaults ──────────────────────────────────────────
_SPOT_NAME="${_SPOT_NAME:-model-zoo-select}"
_PROCESS_NAME="${_PROCESS_NAME:-predictor-model-zoo-select}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-5400}"

# ── Parse flags ──────────────────────────────────────────────────────────────
MODE=""  # weekly | select-only

_ORIG_ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --weekly) MODE="weekly" ; _SPOT_NAME="model-zoo-weekly" ; _SSM_SLUG="spot-model-zoo-weekly" ; _PROCESS_NAME="predictor-model-zoo" ;;
    --select-only) MODE="select-only" ; _SPOT_NAME="model-zoo-select" ; _SSM_SLUG="spot-model-zoo-select" ; _PROCESS_NAME="predictor-model-zoo-select" ;;
    --instance-type) shift; INSTANCE_TYPE="$1" ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    *) echo "ERROR: unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ -z "$MODE" ]; then
  echo "ERROR: one of --weekly or --select-only is required" >&2
  exit 2
fi

# Collapse --instance-type to single value
if [ -n "$INSTANCE_TYPE" ]; then
  INSTANCE_TYPES="$INSTANCE_TYPE"
fi

# ── Preflight checks ─────────────────────────────────────────────────────────
check_config_exists "$(cd "$SCRIPT_DIR/.." && pwd)/config/predictor.yaml"

echo "═══════════════════════════════════════════════════════════════"
echo "  MODEL-ZOO $(echo "$MODE" | tr 'a-z' 'A-Z') — $(date +%Y-%m-%d)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Instance types: $INSTANCE_TYPES"
echo "  Subnets       : $SUBNETS"
echo "  Region        : $AWS_REGION"
echo "  Branch        : $BRANCH"
echo "  Mode          : $MODE"
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

# ── Weekly rotation ──────────────────────────────────────────────────────────
if [ "$MODE" = "weekly" ]; then
  print_banner "MODEL-ZOO WEEKLY ROTATION + SELECT (observe-first by default)"
  run_ssm "model-zoo-weekly" "${_RUN_TOKEN_EXPORT}$(cat <<'ZOO'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
export ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
cat > /tmp/spot-model-zoo-weekly.py <<'PYEOF'
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
from training.model_zoo import run_rotation_and_select

log = logging.getLogger('model_zoo.spot')
log.info('model_zoo spot probe: MODEL_SPECS=%d', len(getattr(cfg, 'MODEL_SPECS', [])))
try:
    from krepis.dates import now_dual
    _td = now_dual().trading_day
    date_str = _td.isoformat() if hasattr(_td, 'isoformat') else str(_td)
except Exception:
    log.warning('model_zoo spot: now_dual unavailable', exc_info=True)
    date_str = None

budget = int(os.environ.get('MODEL_ZOO_WEEKLY_BUDGET', getattr(cfg, 'MODEL_ZOO_WEEKLY_BUDGET', 3)))
board = run_rotation_and_select(bucket, budget=budget, date_str=date_str)
print()
print('=' * 60)
print('  MODEL-ZOO ROTATION + SELECT')
print('=' * 60)
champ = board.get('champion', {})
print('  Champion CPCV:  %s fwd=%s' % (champ.get('cpcv_mean_ic'), champ.get('forward_days')))
for c in board.get('candidates', []):
    print('    %-18s cpcv=%s fwd=%s gate=%s eligible=%s (%s)' % (
        c.get('spec_id'), c.get('cpcv_mean_ic'), c.get('forward_days'),
        c.get('passes_gate'), c.get('eligible'), c.get('reason')))
print('  Winner:         %s' % board.get('winner_version_id'))
print('  Promoted:       %s' % board.get('promoted'))
print('=' * 60)
PYEOF
$PY -m krepis.ssm_log_capture run --slug spot-model-zoo-weekly --log /var/log/spot-model-zoo-weekly.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-model-zoo-weekly.py
ZOO
)" "${MAX_RUNTIME_SECONDS}"

  emit_heartbeat
  echo ""
  echo "==> Model-zoo rotation complete. Instance will be terminated."
  exit 0
fi

# ── Select-only (after Map joins) ────────────────────────────────────────────
if [ "$MODE" = "select-only" ]; then
  print_banner "MODEL-ZOO SELECT (observe-first by default)"
  run_ssm "model-zoo-select" "${_RUN_TOKEN_EXPORT}$(cat <<'ZOOSEL'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
export ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
cat > /tmp/spot-model-zoo-select.py <<'PYEOF'
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
from training.model_zoo import run_select_only

log = logging.getLogger('model_zoo.spot')
log.info('model_zoo select probe: MODEL_SPECS=%d', len(getattr(cfg, 'MODEL_SPECS', [])))

try:
    from krepis.dates import now_dual
    _td = now_dual().trading_day
    date_str = _td.isoformat() if hasattr(_td, 'isoformat') else str(_td)
except Exception:
    log.warning('model_zoo spot: now_dual unavailable', exc_info=True)
    date_str = None

board = run_select_only(bucket, date_str=date_str)
print()
print('=' * 60)
print('  MODEL-ZOO SELECT')
print('=' * 60)
champ = board.get('champion', {})
print('  Champion CPCV:  %s fwd=%s' % (champ.get('cpcv_mean_ic'), champ.get('forward_days')))
for c in board.get('candidates', []):
    print('    %s cpcv=%s fwd=%s eligible=%s (%s)' % (
        c.get('spec_id'), c.get('cpcv_mean_ic'), c.get('forward_days'),
        c.get('eligible'), c.get('reason')))
print('  Winner:         %s' % board.get('winner_version_id'))
print('  Promoted:       %s' % board.get('promoted'))
print('=' * 60)
PYEOF
$PY -m krepis.ssm_log_capture run --slug spot-model-zoo-select --log /var/log/spot-model-zoo-select.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-model-zoo-select.py
ZOOSEL
)" "${MAX_RUNTIME_SECONDS}"

  emit_heartbeat
  echo ""
  echo "==> Model-zoo select complete. Instance will be terminated."
  exit 0
fi
