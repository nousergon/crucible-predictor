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

# Lib CLI path: every spot launcher on the dispatcher box resolves its
# interpreter through the ops-owned guard /opt/nousergon/bin/lib-python
# (nous-ergon-ops: alpha-engine-dashboard/live/infrastructure/bin/lib-python).
# That guard execs the box's DECLARED krepis venv and aborts with EX_CONFIG
# (78), naming the version it found, when the venv is absent or below the
# launcher floor. It never falls back to a co-tenant checkout — the silent
# fallback is exactly the defect alpha-engine-config-I6931/I7343 removes.
# Do NOT add a guard block here: the contract lives ONCE, in the repo that
# owns this box's provisioning (nine copies across five repos is I6922).
LIB_PYTHON="${LIB_PYTHON:-/opt/nousergon/bin/lib-python}"
REPO_URL="https://github.com/nousergon/crucible-predictor.git"

# Spot-reclaim relaunch (#883)
MAX_SPOT_ATTEMPTS="${MAX_SPOT_ATTEMPTS:-2}"
SPOT_ATTEMPT="${SPOT_ATTEMPT:-1}"
SF_EXECUTION_TIMEOUT="${SF_EXECUTION_TIMEOUT:-}"

# Stage-coverage window (config-I7214): the instant this launcher started.
# An artifact older than this is a leftover from a previous cycle, not this
# run's output — an existence-only probe cannot tell those apart.
_STAGE_WINDOW_START="${_STAGE_WINDOW_START:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

# Per-stage identity — DECLARED HERE, NEVER DEFAULTED HERE.
#
# This block used to carry `predictor-training` / `spot-training` defaults, and
# every per-stage script then set its own with the SAME `${VAR:-...}` form
# AFTER sourcing this file — where the parameter is already non-empty, so the
# expansion is a NO-OP and the stage identity silently stayed whatever this
# file said. Measured 2026-08-11 on `watch-rerun-2026-08-10-9`: the heartbeat
# emitted under slug `spot-spot-training` although `spot_predictor_training.sh`
# asks for `spot-full-training`, and `cleanup`'s log-pointer probe therefore
# listed `_ssm_logs/spot-training/<date>/` — a prefix nothing ever writes — and
# printed no pointer while the real log sat under a different slug. The same
# no-op made every model-zoo spot instance Name-tagged `predictor-training` and
# published `SpotInterruptionRetry` under the wrong `Process` dimension.
#
# Declared EMPTY so `set -u` is satisfied at source time and the per-stage
# `${VAR:-<stage default>}` lines are load-bearing again (an explicit env
# override still wins, because it is set before this file runs). `spot_launch`
# asserts all four are non-empty before any instance exists, so a stage that
# forgets one fails loud and free rather than inheriting a sibling's identity.
_SPOT_NAME="${_SPOT_NAME:-}"
_SSM_SLUG="${_SSM_SLUG:-}"
_PROCESS_NAME="${_PROCESS_NAME:-}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-}"

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
  # Stage identity must be real BEFORE anything is billable. See the
  # "Per-stage identity" block above for why these can no longer be defaulted
  # in this file: a default here silently swallows the per-stage assignment.
  local _unset=""
  [ -n "$_SPOT_NAME" ] || _unset="${_unset} _SPOT_NAME"
  [ -n "$_SSM_SLUG" ] || _unset="${_unset} _SSM_SLUG"
  [ -n "$_PROCESS_NAME" ] || _unset="${_unset} _PROCESS_NAME"
  [ -n "$MAX_RUNTIME_SECONDS" ] || _unset="${_unset} MAX_RUNTIME_SECONDS"
  if [ -n "$_unset" ]; then
    echo "ERROR: per-stage variable(s) unset before spot_launch:${_unset}" >&2
    echo "       Set them AFTER sourcing _spot_common.sh — see its header." >&2
    exit 2
  fi

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

  # FIRST, before anything that can itself fail or block. _heartbeat_start
  # backgrounds `krepis.heartbeat emit`, and every per-stage script stops it
  # only on the SUCCESS path — a `set -e` abort between start and stop lands
  # here with the heartbeat still running. That process inherited this script's
  # stdout, which is the pipe `krepis.ssm_log_capture` reads until EOF, so it
  # held the whole SSM command open long after the workload had died.
  #
  # Measured on ne-weekly-freshness-pipeline/watch-rerun-2026-08-10-9
  # (2026-08-11): the smoke step failed 142 seconds in on a flow-doctor
  # ImportError (config-I6963); the command was SIGKILLed at its full 5400s
  # executionTimeout — ResponseCode 137, ExecutionTimedOut — and no log was
  # ever shipped, because the S3 ship runs after that read loop. The stage was
  # recorded as a 90-minute overrun (config-I6948) when it had in fact failed
  # almost immediately.
  #
  # krepis 0.50.0 breaks the hang at the chokepoint (krepis-PR132) and kills
  # the orphan group. This is the launcher's own half: not leaking it in the
  # first place, so the breaker never has to fire and the heartbeat's absence
  # remains a true signal that the phase ended.
  _heartbeat_stop

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
    # See alpha-engine-config-I7009 — migrated off the exit-code contract to --json.
    local _decide_json="" _decide_rc=0
    _decide_json="$("$LIB_PYTHON" -m krepis.ec2_spot relaunch-decision \
      --instance-id "$_INSTANCE_ID" \
      --region "$AWS_REGION" \
      --attempt "$SPOT_ATTEMPT" \
      --max-attempts "$MAX_SPOT_ATTEMPTS" \
      ${SF_EXECUTION_TIMEOUT:+--sf-execution-timeout "$SF_EXECUTION_TIMEOUT" --per-attempt-seconds "$MAX_RUNTIME_SECONDS"} \
      --json \
      2>/dev/null)" || _decide_rc=$?
    # alpha-engine-config-I7009: --json puts the verdict on a field, not the
    # exit code. Non-zero here means the CLI could not answer (bad input /
    # AWS error), not a verdict — treated explicitly as hold below.
    if [ "$_decide_rc" -ne 0 ]; then
      echo "    spot relaunch-decision: CLI failed to answer (rc=$_decide_rc) — treating as hold"
    else
      local _relaunch=""
      _relaunch="$(printf '%s' "$_decide_json" | "$LIB_PYTHON" -c 'import json,sys; print("1" if json.load(sys.stdin).get("relaunch") else "0")')"
      echo "    spot relaunch-decision (attempt $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS): $_decide_json"
      if [ "$_relaunch" = "1" ]; then
        _spot_relaunch=1
        aws cloudwatch put-metric-data \
          --namespace "AlphaEngine" \
          --metric-name "SpotInterruptionRetry" \
          --dimensions "Process=${_PROCESS_NAME}" \
          --value 1 --unit "Count" \
          --region "$AWS_REGION" 2>/dev/null || true
      fi
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
# The heredoc this repo carried through crucible-predictor#461/#462/#463 is
# GONE — this is the cutover to krepis.spot_bootstrap (alpha-engine-config-
# I4992/I6922), the fleet's canonical renderer for the shared, non-repo-
# specific part of a spot bootstrap (watchdog + interpreter + clone). It
# became a deletion once the launcher's LIB_PYTHON resolution stopped
# falling back to a co-tenant venv with no declared floor (config-I6931,
# cleared by config-I7343): every launcher interpreter that can run this repo
# at all now satisfies krepis's own pin. tests/test_spot_bootstrap_*.py assert
# the deletion changed nothing, against the RENDERED script rather than a
# restated heredoc.
#
# REPO_URL is passed as a --repo-url LITERAL, not interpolated into a
# single-quoted heredoc: that interpolation was the exact defect that shipped
# an empty string and killed the clone with `fatal: repository '' does not
# exist` (crucible-predictor#463, ne-weekly-freshness-pipeline
# watch-rerun-2026-08-10-7). Baking it in at render time removes that class
# rather than re-guarding it.
#
# ALPHA_ENGINE_EXPERIMENT_ID is deliberately NOT passed through: it was
# exported into the old heredoc's environment but nothing in the heredoc body
# ever read it (the second `aws s3 cp` that once consumed it was removed
# 2026-08-13 as a destination no resolver searched). Carrying an unread export
# forward would be restating dead surface area, not preserving behaviour.
bootstrap_spot() {
  echo "==> Bootstrapping spot (watchdog, python, clone, config)..."
  local _script
  # --max-runtime-seconds (krepis 0.59.6, alpha-engine-config-I7372): this
  # render call previously carried no hard runtime cap at all — PR504 shipped
  # before the flag existed on the pinned krepis version. The SSM-liveness
  # ec2-spot-watchdog unit (self-terminate on SSM-agent stoppage) is a
  # SEPARATE guarantee and does not cover a live-but-hung workload; only a
  # transient systemd-run timer does. `spot_launch()` above asserts
  # MAX_RUNTIME_SECONDS non-empty before any instance exists, and
  # bootstrap_spot() only ever runs after spot_launch() has populated
  # `_INSTANCE_ID`/`_S3_STAGING`, so the value is guaranteed here for every
  # caller of this shared helper.
  _script="$("$LIB_PYTHON" -m krepis.spot_bootstrap render \
    --repo-url "$REPO_URL" \
    --checkout /home/ec2-user/predictor \
    --branch "${BRANCH:-main}" \
    --region "$AWS_REGION" \
    --export "S3_STAGING=${_S3_STAGING}" \
    --config-copy "predictor.yaml:/home/ec2-user/predictor/config/predictor.yaml" \
    --max-runtime-seconds "$MAX_RUNTIME_SECONDS")"
  run_ssm "bootstrap" "$_script" 300
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
# No silent fallback to the AMI's system python3 (crucible-predictor#462
# class, alpha-engine-config-I7372): bootstrap_spot() installs python3.12 and
# asserts it is present before any per-step run_ssm body ever runs, so an
# absence here means that postcondition was violated — resolving against a
# different interpreter than requirements.txt was pinned for is worse than
# failing loud.
command -v python3.12 >/dev/null 2>&1 || { echo "ERROR: python3.12 not found — bootstrap_spot() should have installed it; refusing to fall back to a different interpreter" >&2; exit 1; }
PY=python3.12
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

# resolve_or_stage_predictor_config — self-heal predictor.yaml at the DISPATCH
# box before declaring it missing (alpha-engine-config-I7216, crucible-
# backtester mirror: spot_common_resolve_predictor_config).
#
# All three predictor-family launchers (spot_predictor_training.sh,
# spot_model_zoo_select.sh, spot_train_spec_dispatch.sh) called plain
# check_config_exists() against this repo's local `config/predictor.yaml`,
# on the DISPATCH box. That file is gitignored (the .example pattern), so a
# fresh dispatcher only has it because something copied it there — and the
# only thing that does, today, is the weekly SF's own command array
# (`sudo -u ec2-user cp --remove-destination
# ~/alpha-engine-config/experiments/reference/predictor/predictor.yaml
# ~/alpha-engine-predictor/config/predictor.yaml`), run by PredictorTraining
# and ModelZooSelect immediately BEFORE this script's own SSM command.
#
# Audited 2026-08-13 against the live step_function.json: every path that
# reaches spot_train_spec_dispatch.sh or spot_model_zoo_select.sh --select-
# only passes through PredictorTraining first (skip_predictor_training's
# validated-skip terminal, PredictorTrainingSkipped, ends the WHOLE branch
# before ResolveZooSpecs/ModelZooTrainMap is ever reached) — so this is not
# observed failing today. But that guarantee lives entirely in SF TOPOLOGY,
# never asserted by this script, exactly the shape that let crucible-
# backtester's PredictorBacktest silently inherit a missing predictor.yaml
# the moment a mechanical rerun's skip-set diverged from the graph's
# assumption (config#7216 live failure, 2026-08-13). A future skip flag
# scoped to ONLY the model-zoo fan-out, or a reordered Map, reintroduces the
# identical defect here with the identical blast radius.
#
# Staging here — the layer that DECLARES the requirement — rather than a
# fourth `cp` in the SF's command array keeps the fix where PR657 put it:
# the stage that needs the file obtains it, instead of every caller
# remembering to.
resolve_or_stage_predictor_config() {
  local dest="$1"
  if [ ! -f "$dest" ]; then
    local experiment_id="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"
    local _ref="$HOME/alpha-engine-config/experiments/${experiment_id}/predictor/predictor.yaml"
    if [ -f "$_ref" ]; then
      echo "  predictor.yaml absent at ${dest} — staging from alpha-engine-config reference (config-I7216)"
      mkdir -p "$(dirname "$dest")"
      # `rm -f` then plain `cp`, NOT `cp --remove-destination`: the SF's own
      # copy uses the GNU flag, which exists only to replace a SYMLINKED
      # destination and is unsupported by BSD cp. This form is equivalent on
      # Linux and also runs on a developer's machine, which is what lets this
      # staging path be exercised before it ships.
      if rm -f "$dest" && cp "$_ref" "$dest"; then
        echo "  staged predictor.yaml from ${_ref}"
      else
        echo "  WARNING: staging predictor.yaml from ${_ref} failed" >&2
      fi
    fi
  fi
  check_config_exists "$dest"
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
# No silent fallback to the AMI's system python3 (crucible-predictor#462
# class, alpha-engine-config-I7372): bootstrap_spot() installs python3.12 and
# asserts it is present before any per-step run_ssm body ever runs, so an
# absence here means that postcondition was violated — resolving against a
# different interpreter than requirements.txt was pinned for is worse than
# failing loud.
command -v python3.12 >/dev/null 2>&1 || { echo "ERROR: python3.12 not found — bootstrap_spot() should have installed it; refusing to fall back to a different interpreter" >&2; exit 1; }
PY=python3.12
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
