#!/usr/bin/env bash
# infrastructure/_spot_common.sh — Shared spot-instance infrastructure for
# crucible-predictor per-stage launcher scripts.
#
# Source this file from per-stage scripts before declaring per-script functions:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/_spot_common.sh"
#
# Provides:
#   - Default config variables (AWS_REGION, S3_BUCKET, SUBNETS, AMI_ID, etc.)
#   - cleanup() — terminate instance + S3 staging teardown + spot-reclaim relaunch
#   - _heartbeat_stop() / _heartbeat_start() — krepis heartbeat pid management
#   - run_ssm() — SSM send-command wrapper via krepis.ssm_dispatcher
#   - spot_launch() — capacity-resilient EC2 spot launch via krepis.ec2_spot
#   - wait_ssm_agent() — poll SSM until instance agent is Online
#   - stage_config() — upload config yaml to S3 staging prefix
#   - bootstrap_spot() — watchdog + python + git clone + staged config fetch
#   - install_deps() — pip install -r requirements.txt, keeping the log
#   - emit_heartbeat() — CloudWatch Heartbeat metric
#   - print_banner() / check_config_exists() — utilities
#   - maybe_run_preflight_only_and_exit() — Friday shell_run dry path: boot +
#     import/lib-pin + read-only ArcticDB probe, then `exit 0`. Every
#     per-stage script MUST call this (with PREFLIGHT_ONLY=1 set by its own
#     --preflight-only flag) right after install_deps, BEFORE its
#     stage-specific work — config#4497 incident: two of the three split
#     scripts (spot_train_spec_dispatch.sh, spot_model_zoo_select.sh)
#     initially treated --preflight-only as an unrecognized flag and fell
#     through into real training/selection, which would have made the
#     Friday shell-run dry path spend real training compute + write real
#     S3 artifacts every week. Lifted here so the invariant is enforced in
#     ONE place instead of copy-pasted three times.
#
# Each per-stage script MUST set the following BEFORE calling spot_launch:
#   _SPOT_NAME     — human-readable name for the spot instance
#   _SSM_SLUG      — log-capture slug for krepis.ssm_log_capture
#   _PROCESS_NAME  — CloudWatch dimension Process name
#   MAX_RUNTIME_SECONDS — SSM command timeout for this stage's workload
#   _ORIG_ARGS     — array of original CLI args (for spot-reclaim re-exec)

set -euo pipefail

# ── Global defaults (overridable via env or per-stage script before source) ──

AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-alpha-engine-research}"
ALPHA_ENGINE_EXPERIMENT_ID="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"
BRANCH="${BRANCH:-main}"

# Instance launch defaults
INSTANCE_TYPES="${INSTANCE_TYPES:-r5.large,r5a.large,r6i.large,m5.large}"
INSTANCE_TYPE=""  # --instance-type X collapses INSTANCE_TYPES to single value
AMI_ID="ami-0c421724a94bba6d6"  # Amazon Linux 2023 x86_64
KEY_NAME="alpha-engine-key"
SECURITY_GROUP="sg-03cd3c4bd91e610b0"
SUBNETS="${SUBNETS:-subnet-a61ec0fb,subnet-1e58307a,subnet-789d3857,subnet-c670118d,subnet-7cff7c43,subnet-e07166ec}"
IAM_PROFILE="alpha-engine-executor-profile"

# LIB_PYTHON — path to python binary with krepis/nousergon-lib installed
LIB_PYTHON="${LIB_PYTHON:-/home/ec2-user/alpha-engine-dashboard/.venv/bin/python}"
REPO_URL="https://github.com/nousergon/crucible-predictor.git"

# Spot-reclaim relaunch (#883)
MAX_SPOT_ATTEMPTS="${MAX_SPOT_ATTEMPTS:-2}"
SPOT_ATTEMPT="${SPOT_ATTEMPT:-1}"
SF_EXECUTION_TIMEOUT="${SF_EXECUTION_TIMEOUT:-}"

# Per-stage overrides (set these BEFORE sourcing _spot_common.sh if needed)
_SPOT_NAME="${_SPOT_NAME:-predictor-training}"
_SSM_SLUG="${_SSM_SLUG:-spot-training}"
_PROCESS_NAME="${_PROCESS_NAME:-predictor-training}"
# MAX_RUNTIME_SECONDS must be set per-stage (default 5400 = 90 min)
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-5400}"

# Derived at launch time
_INSTANCE_ID=""
_S3_STAGING_PREFIX=""
_S3_STAGING=""
_RUN_ID=""

# krepis RUN_TOKEN forwarding (fleet §116 rule 6)
if [ -n "${RUN_TOKEN:-}" ]; then
  _RUN_TOKEN_EXPORT="export RUN_TOKEN=${RUN_TOKEN}"$'\n'
else
  _RUN_TOKEN_EXPORT="export RUN_TOKEN=spot-${ALPHA_ENGINE_EXPERIMENT_ID:-default}-$(date -u +%Y%m%d)"$'\n'
fi

# ── Heartbeat pid management ─────────────────────────────────────────────────

_HEARTBEAT_PID=""

_heartbeat_stop() {
  if [ -n "$_HEARTBEAT_PID" ]; then
    kill "$_HEARTBEAT_PID" 2>/dev/null || true
    _HEARTBEAT_PID=""
  fi
}

_heartbeat_start() {
  _heartbeat_stop
  local _slug="$1" _interval="${2:-300}"
  "$LIB_PYTHON" -m krepis.heartbeat emit --slug "$_slug" --interval "$_interval" &
  _HEARTBEAT_PID=$!
}

# ── SSM dispatch primitive ───────────────────────────────────────────────────

run_ssm() {
  local description="$1" script="$2" timeout_s="${3:-3600}"
  printf '%s' "$script" | "$LIB_PYTHON" -m krepis.ssm_dispatcher run \
    --instance-id "$_INSTANCE_ID" \
    --description "${_PROCESS_NAME}: $description" \
    --timeout "$timeout_s" \
    --output-bucket "$S3_BUCKET" \
    --output-key-prefix "${_S3_STAGING_PREFIX}/ssm-output" \
    --region "$AWS_REGION" \
    --diagnostics-bucket "$S3_BUCKET" \
    --diagnostics-prefix "_spot_diagnostics/ae-predictor" \
    --script-stdin
}

# ── Spot launch (capacity-resilient) ─────────────────────────────────────────

spot_launch() {
  echo "==> Requesting spot instance (lib CLI rotation: types=[$INSTANCE_TYPES], subnets=[$SUBNETS])..."

  _INSTANCE_ID=$("$LIB_PYTHON" -m krepis.ec2_spot launch \
    --types "$INSTANCE_TYPES" \
    --subnets "$SUBNETS" \
    --image-id "$AMI_ID" \
    --key-name "$KEY_NAME" \
    --security-group "$SECURITY_GROUP" \
    --iam-profile "$IAM_PROFILE" \
    --name "alpha-engine-${_SPOT_NAME}-$(date +%Y%m%d)" \
    --region "$AWS_REGION")
  local ec2_spot_rc=$?

  if [ "$ec2_spot_rc" -ne 0 ] || [ -z "$_INSTANCE_ID" ]; then
    if [ "$ec2_spot_rc" -eq 64 ]; then
      echo "ERROR: capacity exhausted across all instance_type x subnet combinations" >&2
    fi
    if [ "$ec2_spot_rc" -eq 0 ]; then
      echo "ERROR: ec2_spot launch exited 0 without an instance id — failing loud (config#1646)" >&2
      ec2_spot_rc=1
    fi
    exit "$ec2_spot_rc"
  fi

  echo "  Instance ID: $_INSTANCE_ID"

  _RUN_ID="$(date +%Y%m%dT%H%M%SZ)-${_INSTANCE_ID}"
  _S3_STAGING_PREFIX="tmp/spot_train/${_RUN_ID}"
  _S3_STAGING="s3://${S3_BUCKET}/${_S3_STAGING_PREFIX}"

  echo "  S3 staging: ${_S3_STAGING}/"
}

# ── Cleanup + spot-reclaim trap ──────────────────────────────────────────────
# Installed AFTER spot_launch so _INSTANCE_ID / _S3_STAGING are populated.

cleanup() {
  local exit_code=$?
  echo ""

  # Belt-and-suspenders: confirm spot-side workload log landed in S3
  local _logdate_now _hit
  _logdate_now="$(date -u +%Y-%m-%d)"
  echo "==> Confirming spot-side workload logs in s3://${S3_BUCKET}/_ssm_logs/ ..."
  for _slug in "$@"; do
    _hit="$(aws s3 ls "s3://${S3_BUCKET}/_ssm_logs/${_slug}/${_logdate_now}/" --region "$AWS_REGION" 2>/dev/null | tail -1 || true)"
    if [ -n "$_hit" ]; then
      echo "    ${_slug}: s3://${S3_BUCKET}/_ssm_logs/${_slug}/${_logdate_now}/$(echo "$_hit" | awk '{print $NF}')"
    fi
  done
  echo "    (spot logs above are the FULL workload stdout/stderr — primary diagnostic on RC=-1/OOM)"
  echo ""

  # Spot-reclaim DECISION (lib chokepoint) — run BEFORE terminate-instances
  local _spot_relaunch=0
  if [ "$exit_code" -ne 0 ] && [ -n "${_INSTANCE_ID:-}" ] && [ "$SPOT_ATTEMPT" -lt "$MAX_SPOT_ATTEMPTS" ]; then
    local _decide_out _decide_rc
    _decide_out="$("$LIB_PYTHON" -m krepis.ec2_spot relaunch-decision \
      --instance-id "$_INSTANCE_ID" \
      --region "$AWS_REGION" \
      --attempt "$SPOT_ATTEMPT" \
      --max-attempts "$MAX_SPOT_ATTEMPTS" \
      ${SF_EXECUTION_TIMEOUT:+--sf-execution-timeout "$SF_EXECUTION_TIMEOUT" --per-attempt-seconds "$MAX_RUNTIME_SECONDS"} \
      2>/dev/null)"
    _decide_rc=$?
    echo "    spot relaunch-decision (attempt $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS): rc=$_decide_rc ${_decide_out:+[$_decide_out]}"
    if [ "$_decide_rc" -eq 0 ]; then
      _spot_relaunch=1
      aws cloudwatch put-metric-data \
        --namespace "AlphaEngine" \
        --metric-name "SpotInterruptionRetry" \
        --dimensions "Process=${_PROCESS_NAME}" \
        --value 1 --unit "Count" \
        --region "$AWS_REGION" 2>/dev/null || true
    fi
  fi

  echo "==> Terminating spot instance $_INSTANCE_ID..."
  aws ec2 terminate-instances --instance-ids "$_INSTANCE_ID" --region "$AWS_REGION" --output text > /dev/null 2>&1 || true
  aws s3 rm "$_S3_STAGING" --recursive --quiet 2>/dev/null || true
  echo "  Instance terminated; S3 staging cleaned."

  # Relaunch on classified reclaim
  if [ "$_spot_relaunch" = "1" ]; then
    echo "==> Spot RECLAIMED by AWS mid-run — relaunching on a fresh spot (attempt $((SPOT_ATTEMPT + 1))/$MAX_SPOT_ATTEMPTS)"
    trap - EXIT
    SPOT_ATTEMPT=$((SPOT_ATTEMPT + 1)) exec bash "$0" ${_ORIG_ARGS[@]+"${_ORIG_ARGS[@]}"}
  fi

  exit "$exit_code"
}

# ── SSM agent wait ───────────────────────────────────────────────────────────

wait_ssm_agent() {
  echo "==> Waiting for SSM agent to come Online..."
  for i in $(seq 1 36); do
    local ping
    ping=$(aws ssm describe-instance-information \
      --filters "Key=InstanceIds,Values=$_INSTANCE_ID" \
      --query 'InstanceInformationList[0].PingStatus' \
      --output text --region "$AWS_REGION" 2>/dev/null || true)
    if [ "$ping" = "Online" ]; then
      echo "  SSM agent Online."
      return 0
    fi
    if [ "$i" -eq 36 ]; then
      echo "ERROR: SSM agent not Online after 180s (instance $_INSTANCE_ID)" >&2
      exit 1
    fi
    sleep 5
  done
}

# ── Config staging ────────────────────────────────────────────────────────────

stage_config() {
  local src="$1" dest_key="${2:-predictor.yaml}"
  echo "==> Staging ${src} → ${_S3_STAGING}/${dest_key}"
  aws s3 cp "$src" "${_S3_STAGING}/${dest_key}" --region "$AWS_REGION" --quiet
}

# ── Bootstrap (watchdog + python + clone + config) ───────────────────────────
# NOTE: single-quoted heredoc body is literal on the spot; the launcher-side
# export prefix sets S3_STAGING/BRANCH/REPO_URL so the spot can fetch config.

bootstrap_spot() {
  echo "==> Bootstrapping spot (watchdog, python, clone, config)..."
  local _spot_env_export
  # REPO_URL was named in the NOTE above but never actually exported: the
  # heredoc is single-quoted (literal on the spot), so ${REPO_URL} resolved to
  # the empty string there and the clone died with
  # `fatal: repository '' does not exist` — ne-weekly-freshness-pipeline
  # watch-rerun-2026-08-10-7, 2026-08-11. Third defect in this bootstrap
  # exposed by fixing the two ahead of it (#461 watchdog hang, #462 missing
  # python3.12); a step that has never completed hides the next failure behind
  # the current one. tests/test_spot_bootstrap_env_closure.py now fails when a
  # variable the heredoc reads is neither exported here nor defaulted inline.
  _spot_env_export="export S3_STAGING=${_S3_STAGING} BRANCH=${BRANCH} REPO_URL=${REPO_URL} ALPHA_ENGINE_EXPERIMENT_ID=${ALPHA_ENGINE_EXPERIMENT_ID}"$'\n'
  run_ssm "bootstrap" "${_spot_env_export}$(cat <<'BOOTSTRAP'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1

# systemd watchdog — self-terminates on SSM-agent stoppage (config#2693).
# Type=simple, NOT oneshot: ExecStart is an endless supervision loop, and
# `systemctl start` on a Type=oneshot unit BLOCKS until ExecStart exits
# (TimeoutStartSec defaults to infinity for oneshot). This unit declared
# Type=oneshot + RemainAfterExit=yes, so every bootstrap hung here until SSM
# gave up at its budget — status=TimedOut with stderr ending on the
# `systemctl enable` symlink line and nothing after it. Measured on
# ne-weekly-freshness-pipeline watch-rerun-2026-08-10-5 (2026-08-11): the
# PredictorTraining stage died exactly this way, taking Branch B down.
# nousergon-data hit the identical defect in its twin of this file and fixed
# it in nousergon-data#1294; this is the mirror (AGENTS.md: when a SOTA
# pattern exists in another alpha-engine repo, mirror it).
if ! systemctl is-enabled ec2-spot-watchdog 2>/dev/null; then
  cat > /tmp/ec2-spot-watchdog.service <<'UNIT'
[Unit]
Description=EC2 Spot Watchdog — self-terminate on SSM agent stoppage
After=amazon-ssm-agent.service
Requires=amazon-ssm-agent.service

[Service]
Type=simple
ExecStart=/usr/local/bin/ec2-spot-watchdog.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
UNIT
  cat > /usr/local/bin/ec2-spot-watchdog.sh <<'WDSH'
#!/usr/bin/env bash
set -euo pipefail
while true; do
  if ! systemctl is-active amazon-ssm-agent >/dev/null 2>&1; then
    sleep 60
    if ! systemctl is-active amazon-ssm-agent >/dev/null 2>&1; then
      shutdown -h now
    fi
  fi
  sleep 60
done
WDSH
  chmod +x /usr/local/bin/ec2-spot-watchdog.sh
  cp /tmp/ec2-spot-watchdog.service /etc/systemd/system/
  # `timeout` so a future unit-shape regression fails in seconds WITH a message,
  # instead of consuming the whole SSM budget and dying with no output — that
  # silence is what made the hang read as "bootstrap is slow" rather than
  # "systemctl is blocked forever".
  timeout 60 systemctl enable --now ec2-spot-watchdog || {
    echo "ERROR: enabling ec2-spot-watchdog did not return within 60s — the unit is misdeclared (an endless ExecStart under Type=oneshot blocks systemctl start forever)" >&2
    exit 1
  }
fi

# Install the interpreter — the AL2023 spot AMI does not ship python3.12.
# This was a bare assertion, encoding an AMI contract nothing provides. It was
# latent behind the watchdog hang (#461): once `systemctl` stopped blocking,
# the very next bootstrap died here — ne-weekly-freshness-pipeline
# watch-rerun-2026-08-10-6, 2026-08-11, "ERROR: python3.12 not found".
# gcc + devel are needed by source-built wheels in requirements.txt; git for
# the clone below. nousergon-data hit and fixed the identical defect in its
# twin of this file (nousergon-data#1296) — this is the mirror.
dnf install -y -q python3.12 python3.12-pip python3.12-devel git gcc 2>/dev/null || \
    dnf install -y -q python3 python3-pip python3-devel git gcc

# Post-condition, not a precondition: the install above is what makes this
# true, and a silent fallback to a system python3 is exactly the drift this
# bootstrap must not inherit (requirements.txt is resolved against 3.12).
command -v python3.12 >/dev/null || { echo "ERROR: python3.12 not found after dnf install" >&2; exit 1; }
echo "Using: $(python3.12 --version)"

if [ ! -d /home/ec2-user/predictor/.git ]; then
  rm -rf /home/ec2-user/predictor
  git clone --depth 1 --branch "${BRANCH:-main}" "${REPO_URL}" /home/ec2-user/predictor
fi

mkdir -p "/home/ec2-user/predictor/config"
mkdir -p "/home/ec2-user/predictor/experiments/${ALPHA_ENGINE_EXPERIMENT_ID:-reference}/predictor"
aws s3 cp "${S3_STAGING}/predictor.yaml" "/home/ec2-user/predictor/config/predictor.yaml" --region "${AWS_REGION:-us-east-1}" --quiet
aws s3 cp "${S3_STAGING}/predictor.yaml" "/home/ec2-user/predictor/experiments/${ALPHA_ENGINE_EXPERIMENT_ID:-reference}/predictor/predictor.yaml" --region "${AWS_REGION:-us-east-1}" --quiet
BOOTSTRAP
)" 300
  echo "  Bootstrap complete."
}

# ── Dependency installation ──────────────────────────────────────────────────

install_deps() {
  # config-I6949: this step used to pipe pip through `tail -1`, so a run that
  # exited 0 having silently skipped an extra was indistinguishable from a
  # clean one — and pip reports a dropped extra as a WARNING on a SUCCESSFUL
  # exit, which is precisely the line `tail -1` could not keep. The fleet copy
  # is krepis.spot_bootstrap.render_install_deps; keep the two in step.
  echo "==> Installing python deps..."
  run_ssm "deps" "$(cat <<'DEPS'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
_pip_log=/tmp/pip-install-deps.log
if ! $PY -m pip install --no-warn-script-location -r requirements.txt > "$_pip_log" 2>&1; then
  echo "ERROR: pip install -r requirements.txt failed" >&2
  tail -80 "$_pip_log" >&2
  exit 1
fi
grep -E "^Successfully installed" "$_pip_log" || true
# A dropped extra is a BROKEN ENVIRONMENT, not a note. pip reports it as a
# WARNING on a SUCCESSFUL exit, so nothing downstream fails until an import
# does — in another process, minutes later, with the install log long gone.
# That is exactly how config#6963 reached production: the training smoke died
# on `ModuleNotFoundError: No module named 'flow_doctor'` out of
# krepis.logging.setup_logging, from an install pip had called a success.
# AL2023 ships pip 23.2.1, which predates PEP 685 extras normalisation
# (measured boundary: 23.2.1 drops, 23.3.2 honours), so this bootstrap sits on
# the broken side by default and cannot rely on the resolver to be strict.
# Fail here, where the log is still in hand and the cause is one line.
if grep -E "^WARNING: .*does not provide the extra" "$_pip_log" >&2; then
  echo "ERROR: pip dropped a requested extra (above) — the environment is incomplete." >&2
  echo "       Extras must be HYPHENATED; pip <23.3 does not normalise '_' to '-'." >&2
  exit 1
fi
# Non-fatal on purpose: an inconsistent environment is reported, not raised.
# (a) The failure mode left unraised is a pre-existing AMI-baked conflict
# unrelated to this checkout, which would otherwise fail every lane on every
# run. (b) It is recorded on stdout of this SSM step, captured with the rest
# of the deps output.
$PY -m pip check || echo "WARNING: pip check reports an inconsistent environment (above)"
DEPS
)" 600
  echo "  Deps installed."
}

# ── CloudWatch heartbeat on completion ───────────────────────────────────────

emit_heartbeat() {
  aws cloudwatch put-metric-data \
    --namespace "AlphaEngine" \
    --metric-name "Heartbeat" \
    --dimensions "Process=${_PROCESS_NAME}" \
    --value 1 --unit "Count" \
    --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
    && echo "Heartbeat emitted: ${_PROCESS_NAME}" \
    || echo "WARNING: Failed to emit heartbeat (non-fatal)"
}

# ── Print summary banner ─────────────────────────────────────────────────────

print_banner() {
  local title="$1"
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  ${title}"
  echo "═══════════════════════════════════════════════════════════════"
}

# ── Preflight checks ─────────────────────────────────────────────────────────

check_config_exists() {
  local config_path="$1"
  if [ ! -f "$config_path" ]; then
    echo "ERROR: ${config_path} not found" >&2
    exit 1
  fi
}

# PREFLIGHT_ONLY — unconditionally initialized (empty = disabled). Each
# per-stage script's flag parser sets this to "1" on --preflight-only.
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-}"

# maybe_run_preflight_only_and_exit — Friday shell_run dry path (weekly-sf-
# policy.md §7.1). Boot + import/lib-pin + read-only ArcticDB connectivity
# probe, then `exit 0` — NO training, NO promotion, ZERO S3/config writes.
# No-op (returns 0) when PREFLIGHT_ONLY is unset, so a real run is
# byte-behavior-identical to before this function existed. MUST be called
# by every per-stage script immediately after install_deps and BEFORE any
# stage-specific work.
maybe_run_preflight_only_and_exit() {
  if [ -z "$PREFLIGHT_ONLY" ]; then
    return 0
  fi
  print_banner "PREFLIGHT-ONLY (no training, no promotion, no writes)"
  run_ssm "preflight-only" "${_RUN_TOKEN_EXPORT}$(cat <<'PREFLIGHT'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
export ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
$PY - <<'PYEOF'
import os, sys
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', os.environ.get('S3_BUCKET', 'alpha-engine-research'))
bucket = os.environ.get('S3_BUCKET', 'alpha-engine-research')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s')
log = logging.getLogger('preflight-only')

log.info('[1/3] Importing training package...')
import nousergon_lib
from training import train_handler
from training import model_zoo
from training.preflight import TrainingPreflight
log.info('       OK — nousergon_lib + training.train_handler + model_zoo import clean')

log.info('[2/3] Running TrainingPreflight (env + S3 connectivity)...')
TrainingPreflight(bucket=bucket).run()
log.info('       OK — env vars present, S3 bucket reachable')

log.info('[3/3] ArcticDB connectivity + universe-freshness probe...')
from nousergon_lib.arcticdb import open_arctic
arctic = open_arctic(bucket)
universe = arctic.get_library('universe')
symbols = universe.list_symbols()
n = len(symbols)
if n == 0:
    raise RuntimeError('ArcticDB universe library is empty/unreachable')
probe = sorted(symbols)[0]
df_tail = universe.read(probe).data.tail(1)
latest = df_tail.index.max() if not df_tail.empty else 'n/a'
log.info('       OK — universe has %d symbols; %s latest index=%s', n, probe, latest)

print()
print('=' * 60)
print('  PREFLIGHT-ONLY RESULT: PASS')
print('=' * 60)
print('  Imports:        nousergon_lib + training stack clean')
print('  TrainingPreflight: PASS (env + S3 reachable)')
print('  ArcticDB:       %d universe symbols (probe %s latest=%s)' % (n, probe, latest))
print('  Training:       SKIPPED (no run_meta_training call)')
print('  Promotion:      SKIPPED (no weights/meta write)')
print('  S3/config writes: NONE')
print('=' * 60)
PYEOF
PREFLIGHT
)" 600
  echo "==> Preflight-only mode — PASS. Exiting 0 BEFORE stage-specific work."
  exit 0
}
