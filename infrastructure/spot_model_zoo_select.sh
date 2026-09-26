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

# Stage-coverage identity (config-I7214). Only --select-only is reachable
# from the live weekly SF (nousergon-data infrastructure/step_function.json
# state "ModelZooSelect" runs exactly `spot_model_zoo_select.sh
# --select-only`, verified 2026-08-13). --weekly has no corresponding state
# in the live definition — "ModelZooWeekly" does not exist there — so it is
# NOT asserted below; asserting an undeclared registry stage would be a
# guaranteed-false MISSING, not a real signal. Set per-mode in the flag
# parser.
_COVERAGE_STAGE=""

# ── Parse flags ──────────────────────────────────────────────────────────────
MODE=""  # weekly | select-only

_ORIG_ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --weekly) MODE="weekly" ; _SPOT_NAME="model-zoo-weekly" ; _SSM_SLUG="spot-model-zoo-weekly" ; _PROCESS_NAME="predictor-model-zoo" ;;
    --select-only) MODE="select-only" ; _SPOT_NAME="model-zoo-select" ; _SSM_SLUG="spot-model-zoo-select" ; _PROCESS_NAME="predictor-model-zoo-select" ; _COVERAGE_STAGE="ModelZooSelect" ;;
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
resolve_or_stage_predictor_config "$(cd "$SCRIPT_DIR/.." && pwd)/config/predictor.yaml"

echo "═══════════════════════════════════════════════════════════════"
echo "  MODEL-ZOO $(echo "$MODE" | tr 'a-z' 'A-Z') — $(stage_run_date)"
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
spot_launch   # arms the cleanup EXIT trap before it launches (alpha-engine-config-I11573)

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
command -v python3.12 >/dev/null 2>&1 || { echo "ERROR: python3.12 not found — bootstrap_spot() should have installed it; refusing to fall back to a different interpreter (alpha-engine-config-I7380)" >&2; exit 1; }
PY=python3.12
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
  # No stage-coverage assertion here — see the "_COVERAGE_STAGE" comment
  # above the flag parser: --weekly has no live SF state to assert against
  # (config-I7214).
  echo ""
  echo "==> Model-zoo rotation complete. Instance will be terminated."
  exit 0
fi

# ── Select-only (after Map joins) ────────────────────────────────────────────
if [ "$MODE" = "select-only" ]; then
  print_banner "MODEL-ZOO SELECT (observe-first by default)"
  # The SSM output is teed so the launcher can read the workload's outcome
  # line back out of it (MODEL_ZOO_SELECT_OUTCOME v1, below). `set -o pipefail`
  # is in force (_spot_common.sh), so a failed run_ssm still fails the stage.
  _ZOO_SELECT_OUT="$(mktemp -t spot-model-zoo-select.XXXXXX)"
  run_ssm "model-zoo-select" "${_RUN_TOKEN_EXPORT}$(cat <<'ZOOSEL'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
export ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null 2>&1 || { echo "ERROR: python3.12 not found — bootstrap_spot() should have installed it; refusing to fall back to a different interpreter (alpha-engine-config-I7380)" >&2; exit 1; }
PY=python3.12
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

# MODEL_ZOO_SELECT_OUTCOME v1 (alpha-engine-config-I11106) — the M slot's
# verdict, handed to the launcher on stdout. 'unservable' means the serving
# preconditions ran and excluded every arm: the slot DECIDED not to promote,
# which is a successful stage with a negative result. The DURABLE source of
# truth is arena/model/{date}.json::decision.status; this line is only a
# cheap transport for the weekly SF's Choice (moving that Choice onto a
# structured read of the artifact is alpha-engine-config-I11101 deliverable
# 4). A missing/unknown outcome is NOT defaulted — it exits non-zero.
_outcome = board.get('arena_outcome')
if _outcome not in ('decided', 'unservable'):
    raise SystemExit(
        'model_zoo select: no usable arena outcome on the leaderboard (%r) — '
        'refusing to report an outcome-less select as a successful stage '
        '(alpha-engine-config-I11106)' % (_outcome,)
    )
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

# LAST line the workload prints, deliberately, and the SSM body below hands
# the launcher only the TAIL of the output, so this line always reaches it.
print('MODEL_ZOO_SELECT_OUTCOME_RAW=%s' % _outcome)
PYEOF
# SSM keeps only the FIRST ~24,000 characters of StandardOutputContent and
# appends "--output truncated--"; it does not rotate. The launcher reads the
# outcome line out of that inline copy, and the outcome line is the workload's
# LAST, so once the select log passed 24KB the line was never seen and a
# correct 'unservable' verdict failed the stage (rehearsal-2026-09-25-1: the
# spot log ended in MODEL_ZOO_SELECT_OUTCOME_RAW=unservable, the launcher saw
# none). So the workload's output goes to a file and only its tail reaches
# stdout. Nothing is lost: ssm_log_capture ships the FULL log to
# _ssm_logs/spot-model-zoo-select/ in S3 either way.
_ZOO_RC=0
$PY -m krepis.ssm_log_capture run --slug spot-model-zoo-select --log /var/log/spot-model-zoo-select.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-model-zoo-select.py >/tmp/spot-model-zoo-select.stdout 2>&1 || _ZOO_RC=$?
tail -c 12000 /tmp/spot-model-zoo-select.stdout
exit "$_ZOO_RC"
ZOOSEL
)" "${MAX_RUNTIME_SECONDS}" | tee "$_ZOO_SELECT_OUT"

  # ── MODEL_ZOO_SELECT_OUTCOME v1 (alpha-engine-config-I11106) ───────────────
  # Three distinguishable states, not two:
  #   unservable — every M arm failed a hard serving precondition. The stage
  #                EVALUATED the arms, applied the §5.3 preconditions, and
  #                correctly concluded that nothing may be promoted. A
  #                VERDICT, exit 0. Nothing is weakened: no arm becomes
  #                promotable and the pointer does not move.
  #   decided    — the slot has an arm it may serve. Exit 0.
  #   anything else (exception, resource kill, timeout, no outcome line at
  #                all) — the stage BROKE. Exit non-zero, unchanged.
  # The weekly SF's Choice matches this exact string on
  # StandardOutputContent, so it is a single anchored line and must not be
  # reworded. The durable source of truth stays
  # arena/model/{date}.json::decision.status — this is a transport only, and
  # moving the Choice onto a structured read of that artifact is tracked as
  # alpha-engine-config-I11101 deliverable 4.
  if grep -qx 'MODEL_ZOO_SELECT_OUTCOME_RAW=unservable' "$_ZOO_SELECT_OUT"; then
    _ZOO_OUTCOME="unservable"
  elif grep -qx 'MODEL_ZOO_SELECT_OUTCOME_RAW=decided' "$_ZOO_SELECT_OUT"; then
    _ZOO_OUTCOME="decided"
  else
    rm -f "$_ZOO_SELECT_OUT"
    echo "ERROR: model-zoo select produced NO MODEL_ZOO_SELECT_OUTCOME_RAW line — the workload exited 0 without reporting an outcome. Refusing to report an outcome-less select as a successful stage (alpha-engine-config-I11106)." >&2
    exit 1
  fi
  rm -f "$_ZOO_SELECT_OUT"

  emit_heartbeat

  # Per-stage output assertion (config-I7214, sf-pipeline-policy.md §2.1):
  # assert THIS stage wrote what it declared, at the boundary where the fact
  # becomes knowable. OBSERVE MODE — it can never fail the stage.
# alpha-engine-config-I8155: pass the SF execution's own run_date via
# EXECUTION_RUN_DATE, never $RUN_DATE — RUN_DATE is reassigned to the
# trading day elsewhere in the fleet (crucible-backtester _spot_common.sh),
# so it is not a reliable carrier of the execution identity. No fallback:
# an unset EXECUTION_RUN_DATE must reach the CLI empty so it exits loudly
# under the observe-mode guard below rather than writing under run_date="".
  "$LIB_PYTHON" -m krepis.stage_coverage assert --stage "$_COVERAGE_STAGE" --window-start "$_STAGE_WINDOW_START" --run-date "${EXECUTION_RUN_DATE:-}" || echo "WARNING: stage-coverage assertion did not run for $_COVERAGE_STAGE (rc=$?) — observe mode, stage NOT failed (config-I7214)" >&2

  echo ""
  echo "==> Model-zoo select complete. Instance will be terminated."
  echo "MODEL_ZOO_SELECT_OUTCOME: ${_ZOO_OUTCOME}"
  exit 0
fi
