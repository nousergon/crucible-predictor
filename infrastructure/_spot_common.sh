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
#   - install_deps() — pip install -r requirements.txt
#   - emit_heartbeat() — CloudWatch Heartbeat metric
#   - print_banner() / check_config_exists() — utilities
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
  _spot_env_export="export S3_STAGING=${_S3_STAGING} BRANCH=${BRANCH} ALPHA_ENGINE_EXPERIMENT_ID=${ALPHA_ENGINE_EXPERIMENT_ID}"$'\n'
  run_ssm "bootstrap" "${_spot_env_export}$(cat <<'BOOTSTRAP'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1

# systemd watchdog — self-terminates on SSM-agent stoppage (config#2693)
if ! systemctl is-enabled ec2-spot-watchdog 2>/dev/null; then
  cat > /tmp/ec2-spot-watchdog.service <<'UNIT'
[Unit]
Description=EC2 Spot Watchdog — self-terminate on SSM agent stoppage
After=amazon-ssm-agent.service
Requires=amazon-ssm-agent.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/ec2-spot-watchdog.sh
RemainAfterExit=yes

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
  systemctl enable ec2-spot-watchdog
  systemctl start ec2-spot-watchdog
fi

command -v python3.12 >/dev/null || { echo "ERROR: python3.12 not found" >&2; exit 1; }

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
  echo "==> Installing python deps..."
  run_ssm "deps" "$(cat <<'DEPS'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
$PY -m pip install --quiet --no-warn-script-location -r requirements.txt 2>&1 | tail -1
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
