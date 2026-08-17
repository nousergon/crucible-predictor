#!/usr/bin/env bash
# infrastructure/spot_train.sh — Run GBM retraining on a spot EC2 instance.
#
# STATUS (alpha-engine-config-I6998 deliverable 3, 2026-08-13): this monolith
# is NOT invoked by the current SF definition. `ne-weekly-freshness-pipeline`
# -> `PredictorTraining` sends `bash infrastructure/spot_predictor_training.sh`
# (nousergon-data/infrastructure/step_function.json, PredictorTraining state
# Command). It IS deliberately retained, unchanged, as the rollback path for
# the whole spot_train.sh -> per-stage split (crucible-predictor#436,
# alpha-engine-config-I4442/I4497, 2026-08-09) — every sendCommand state
# across the weekly SF carries that same "the monolith is retained unchanged
# as the rollback path" comment. Roll back by repointing the SF Command back
# to this file if the split proves unstable; do not delete it while that
# comment stands in the SF definition.
#
# The one other repo-external reference found by grep,
# nous-ergon-ops/alpha-engine-predictor/infrastructure/add-training-cron.sh
# (a crontab-registration helper, last touched 2026-07-30 — one day before
# #436 merged), still names this script by its pre-split path and was never
# updated for the split. It is STALE, not a second live caller: the SF is the
# sole scheduling authority for the weekly training run (policy-sf-pipeline),
# and nothing re-runs that installer today. Tracked to correct/retire it so a
# future re-run cannot fire a second, conflicting training pass:
# alpha-engine-config-I7155.
#
# G16 spot-bootstrap cutover (alpha-engine-config-I4992/I6922/I7372): this
# monolith's own inline bootstrap heredoc — carrying crucible-predictor#461
# (watchdog Type=oneshot hang), #462 (bare python3.12 assertion with nothing
# installing it) and #463 (an interpolated ${REPO_URL}/${BRANCH} git-clone
# line) — is GONE. It now sources _spot_common.sh and reuses its
# bootstrap_spot() (the fleet's canonical krepis.spot_bootstrap renderer
# call) instead of restating those three defects a second time in this
# retained rollback path. Every per-step heredoc's silent interpreter
# fallback (bare `command -v python3.12` gating a `python3` fallback) is
# also gone — the same class fixed in _spot_common.sh's install_deps() and
# preflight-only step in this same PR: a fallback resolves requirements.txt
# against a different interpreter than it was pinned for. See the `source`
# line below for why this file keeps its own run_ssm()/cleanup() rather than
# adopting _spot_common.sh's wholesale.
#
# Launches a c5.large spot instance, syncs code, runs training via the
# same train_handler.main() pipeline that Lambda uses (S3 price cache
# download → refresh → train → promote → slim cache → email).
#
# Communication is via `aws ssm send-command` (IAM-authenticated, CloudTrail-
# audited) — NOT SSH/SCP. Config is staged through S3; secrets are read on
# the spot via krepis.secrets.get_secret() (SSM Parameter Store),
# so there is no `.env` SCP and no `~/.ssh/alpha-engine-key.pem` dependency
# in the workflow. (PR 2 of the spot-train-260512 SSH/SCP→SSM migration;
# canonical plan: alpha-engine-docs/private/spot-train-260512.md.)
#
# Usage:
#   ./infrastructure/spot_train.sh                  # smoke (dry_run) then full
#   ./infrastructure/spot_train.sh --full-only       # full training only (Saturday SF)
#   ./infrastructure/spot_train.sh --smoke-only      # smoke only, then terminate
#   ./infrastructure/spot_train.sh --preflight-only  # boot + import/lib-pin +
#                                                    # ArcticDB connectivity probe,
#                                                    # then exit 0 — NO training,
#                                                    # NO promotion, ZERO S3/config
#                                                    # writes (Friday shell_run dry path)
#   ./infrastructure/spot_train.sh --instance-type c5.2xlarge  # override type
#
# Prerequisites:
#   - AWS CLI configured (alpha-engine-executor-profile — S3 + SSM + email).
#     The instance profile carries AmazonSSMManagedInstanceCore so the spot
#     registers with SSM; this script polls SSM for readiness (no port 22).
#   - Code committed + pushed to origin/$BRANCH (the spot clones HTTPS).
#   - config/predictor.yaml present locally (gitignored — staged to S3).
#
# The script will:
#   1. Request a spot instance (r5.large ≈ $0.04/hr spot; ≥8 GiB RAM)
#   2. Wait for the SSM agent to register (no SSH)
#   3. Stage config/predictor.yaml to S3; spot bootstraps + fetches it
#   4. Run smoke (dry_run=True), then full training (dry_run=False)
#      — OR, under --preflight-only, run the import/lib-pin + read-only
#        ArcticDB connectivity probe and exit 0 (no training, no promotion,
#        no S3/config writes; Friday shell_run dry path)
#   5. Terminate the spot instance + clean the S3 staging prefix
#
# Rollback: `git revert` this commit restores the SSH/SCP script. Port 22
# ingress on the SG is intentionally left in place until the migration's
# PR 3 (SG cleanup), so emergency `ssh`/`aws ssm start-session` remains
# available during the validation window.

set -euo pipefail

# SSM RunCommand executes as root with a minimal env — set HOME/cache dirs
# explicitly wherever the workload runs (done per-step below too).
export HOME="${HOME:-/home/ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# #883 — captured BEFORE the flag-parse loop consumes "$@" so the mid-run
# spot-reclaim relaunch can re-exec this script with the IDENTICAL argv
# (--full-only / --model-zoo-spec <id> / --instance-type X / etc.). A
# relaunched attempt MUST re-run under the same mode it was invoked with.
_ORIG_ARGS=("$@")

# ── Configuration ──────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-alpha-engine-research}"
# config#1066 — experiment package. config.py searches
# ~/alpha-engine-config/experiments/$ALPHA_ENGINE_EXPERIMENT_ID/predictor/predictor.yaml
# FIRST (the 2026-06-12 experiment-package adoption), then the legacy
# config/predictor.yaml. The spot is a bare predictor clone with NO
# alpha-engine-config tree, so the experiment path is absent and resolution
# silently falls through to config/predictor.yaml — a coincidence that was the
# 6/13 inert-rotation fragility (MODEL_SPECS empty → 0 challengers trained).
# We pin the id here, EXPORT it into every spot heredoc, and stage the yaml to
# BOTH the experiment-package path AND config/predictor.yaml on the spot so
# config.py resolves DETERMINISTICALLY to the staged, MODEL_SPECS-populated yaml
# via the SAME path it uses on the always-on box. Default "reference" matches
# config.py's own _EXPERIMENT_ID default.
ALPHA_ENGINE_EXPERIMENT_ID="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"
BRANCH="${BRANCH:-main}"
# Defer-email propagation: the Saturday SF exports PREDICTOR_DEFER_TRAINING_EMAIL
# before invoking this script for the --full-only champion retrain, so the base
# retrain defers its per-run training email to the consolidated model-zoo digest
# (the --model-zoo-weekly workload sends the digest). The full-training heredoc
# is a single-quoted heredoc and cannot interpolate a bash var, so we compute an
# export line HERE and prepend it to the heredoc body via string concatenation
# (an interpolating prefix + the quoted body). Empty when the var is unset, so a
# bare full-only run is byte-equivalent to before. Keep this single-line and
# paren/apostrophe-free per the bash 3.2 run_ssm note.
if [ -n "${PREDICTOR_DEFER_TRAINING_EMAIL:-}" ]; then
  DEFER_EMAIL_EXPORT="export PREDICTOR_DEFER_TRAINING_EMAIL=${PREDICTOR_DEFER_TRAINING_EMAIL}"$'\n'
else
  DEFER_EMAIL_EXPORT=""
fi
# krepis 0.18.8+ fleet §116 rule 6: --correlation-id (or $RUN_TOKEN) is required.
# The dashboard box now sets RUN_TOKEN via systemd; forward it to every spot-side
# heredoc so krepis calls on the spot instance pick it up automatically.
if [ -n "${RUN_TOKEN:-}" ]; then
  RUN_TOKEN_EXPORT="export RUN_TOKEN=${RUN_TOKEN}"$'\n'
else
  # Fallback: if RUN_TOKEN is somehow unset (e.g. a developer running spot_train.sh
  # directly), derive one from the experiment id + UTC date so the spot doesn't fail.
  RUN_TOKEN_EXPORT="export RUN_TOKEN=spot-${ALPHA_ENGINE_EXPERIMENT_ID:-default}-$(date -u +%Y%m%d)"$'\n'
fi
# Capacity-resilient instance-type fallback set (2026-05-22 incident:
# spot launches in single-AZ subnet-e07166ec/us-east-1f hit
# InsufficientInstanceCapacity). Order = preference; the lib CLI tries
# each in turn until one launches.
#
# 2026-06-06 — memory-optimized (≥8 GiB, all 2 vCPU). The prior set
# (c5.large/c6i.large/c5a.large = 4 GiB) OOM-killed full-training on the
# Saturday SF: the meta-trainer's peak RSS now exceeds 4 GiB (universe +
# history growth plus the observe-only canonical-alpha matrix), so the
# rotation picked c5.large (4 GiB) and the kernel SIGKILL'd the process
# right after regime-data load. This is the SECOND OOM on a 4 GiB box
# (first: 2026-04-28, addressed by the meta_trainer.py streaming refactor;
# data growth since re-crossed 4 GiB). Lead with r5.large (16 GiB) for
# ~4× headroom over the failing footprint; m5.large (8 GiB) is the
# last-resort capacity fallback. The old "steady-state ~1-1.5 GB" note
# was stale — see test_meta_trainer_streaming.py for the peak-RSS context.
INSTANCE_TYPES="${INSTANCE_TYPES:-r5.large,r5a.large,r6i.large,m5.large}"
INSTANCE_TYPE=""  # backward-compat: --instance-type X collapses INSTANCE_TYPES to single value
AMI_ID="ami-0c421724a94bba6d6"  # Amazon Linux 2023 x86_64 (Python 3.12, SSM agent preinstalled)
# Spot-side watchdog budget: meta-trainer typically completes 40-70 min;
# include pip install + smoke + full run. 90 min with headroom.
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-5400}"
# #883 — bounded mid-run spot-reclaim relaunch. When AWS reclaims the spot
# *mid-workload* (the 2026-05-30 DataPhase1 incident class:
# instance-terminated-no-capacity / Server.SpotInstanceTermination), the
# cleanup() EXIT trap relaunches a FRESH spot up to MAX_SPOT_ATTEMPTS, re-using
# the S3-staged config + the same argv. The classify→decide DECISION is the lib
# chokepoint `python -m krepis.ec2_spot relaunch-decision` (lib v0.65.0+
# nousergon_lib.ec2_spot re-export, now invoked directly as krepis.ec2_spot
# per config#1649 — the re-export shim is guard-less under `python -m` on
# lib >=0.81.0 and silently no-ops): ONLY a confirmed reclaim relaunches; a genuine workload
# failure (OOM / crash / timeout) classifies as "other"/"unknown" and fails loud
# — a blind retry would mask a real training bug. SPOT_ATTEMPT is threaded across
# re-execs via the env (first run = 1).
#
# MAX_SPOT_ATTEMPTS ↔ per-attempt-budget coupling (#883 requirement): each
# attempt costs ~7-min boot + up to MAX_RUNTIME_SECONDS of workload. The lib's
# --sf-execution-timeout/--per-attempt-seconds guard refuses to advise a relaunch
# the OUTER budget cannot absorb. The predictor Saturday retrain runs from a weekly
# cron (`spot_train.sh --full-only`), NOT under a Step-Functions executionTimeout,
# so there is no outer SF budget to couple to — SF_EXECUTION_TIMEOUT defaults empty
# (guard inert; bound is MAX_SPOT_ATTEMPTS only). If this launcher is ever wired
# under an SF state with an executionTimeout, set SF_EXECUTION_TIMEOUT to that
# budget and the lib guard activates ((attempt+1)*MAX_RUNTIME_SECONDS must fit).
MAX_SPOT_ATTEMPTS="${MAX_SPOT_ATTEMPTS:-2}"
SPOT_ATTEMPT="${SPOT_ATTEMPT:-1}"
SF_EXECUTION_TIMEOUT="${SF_EXECUTION_TIMEOUT:-}"
# KEY_NAME is still passed to run-instances so emergency SSH stays possible
# during the validation window (the SG's port 22 ingress is dropped only in
# the migration's PR 3, after this PR validates against a Saturday SF).
KEY_NAME="alpha-engine-key"
SECURITY_GROUP="sg-03cd3c4bd91e610b0"
# All 6 default-VPC subnets across us-east-1{a..f}. The lib CLI rotates
# across this list on capacity error. Same VPC + same SG as the data +
# backtester spots; lockstep with their launchers.
SUBNETS="${SUBNETS:-subnet-a61ec0fb,subnet-1e58307a,subnet-789d3857,subnet-c670118d,subnet-7cff7c43,subnet-e07166ec}"
IAM_PROFILE="alpha-engine-executor-profile"
# Lib CLI path. The ops-owned guard /opt/nousergon/bin/lib-python
# (nous-ergon-ops: alpha-engine-dashboard/live/infrastructure/bin/lib-python)
# execs the box's DECLARED krepis venv and aborts with EX_CONFIG (78), naming
# the version it found, rather than silently falling back to a co-tenant
# checkout — the defect alpha-engine-config-I6931/I7343 removes.
#
# It is NOT the default here, because THIS SCRIPT DOES NOT RUN ON THAT BOX
# (alpha-engine-config-I7386). The guard is installed by
# nous-ergon-ops/alpha-engine-dashboard/live/infrastructure/bin/install-box-config.sh,
# whose whole tree provisions the DASHBOARD BOX. This file is sourced by the
# spot_*.sh scripts the weekly SF delivers as ssm:sendCommand payloads to
# $.ec2_instance_id — an ephemeral spot, bootstrapped by
# nousergon-data/infrastructure/lambdas/weekly-freshness-spot-dispatcher/index.py
# (_bootstrap_command), which builds
# /home/ec2-user/alpha-engine-dashboard/.venv and never creates
# /opt/nousergon. Measured on execution
# friday-shell-2026-08-14-validate-i7382 (nousergon-data's copy of this same
# line, MorningEnrich): "No such file or directory", exit 127.
#
# So the default names the interpreter that host actually has. The
# ${LIB_PYTHON:-...} override is preserved, so a caller ON a box that does
# have the guard still names it explicitly and gets the declared floor.
# Do NOT add a guard block here: the contract lives ONCE, in the repo that
# owns the box's provisioning (nine copies across five repos is I6922). The
# SOTA close — install the guard on the spot too, then restore this default —
# is alpha-engine-config-I7383.
LIB_PYTHON="${LIB_PYTHON:-/home/ec2-user/alpha-engine-dashboard/.venv/bin/python}"
REPO_URL="https://github.com/nousergon/crucible-predictor.git"  # public repo, no auth

# infrastructure/_spot_common.sh — sourced ONLY for its bootstrap_spot()
# (alpha-engine-config-I4992/I6922/I7372 cutover): this monolith's own
# "Bootstrap" step below now calls that shared function instead of restating
# the watchdog + interpreter-install + clone heredoc that carried
# crucible-predictor#461/#462/#463. Every variable bootstrap_spot() reads
# (AWS_REGION, REPO_URL, BRANCH, LIB_PYTHON, MAX_RUNTIME_SECONDS) is already
# set above with the SAME `${VAR:-default}` literals _spot_common.sh itself
# declares, so sourcing it here is a no-op for all of them — confirmed by
# diffing the two files' preambles before this change. This script keeps its
# OWN run_ssm()/cleanup()/heartbeat functions (defined further below, using
# un-prefixed INSTANCE_ID/S3_STAGING rather than _spot_common.sh's
# underscore-prefixed _INSTANCE_ID/_S3_STAGING globals): those definitions
# come AFTER this source line, so they override _spot_common.sh's
# same-named functions and nothing here changes. _spot_common.sh's OTHER
# functions (spot_launch, install_deps, wait_ssm_agent, stage_config, etc.)
# are pulled in but never called — dormant, zero behavior change. Only
# bootstrap_spot() is exercised, and its one un-prefixed dependency
# (`_S3_STAGING`) is bridged immediately before the call, at the call site
# below. A full rewrite of this legacy, non-SF-invoked rollback script onto
# _spot_common.sh's own `_`-prefixed variable convention is out of scope for
# this cutover — it is retained unchanged as the split's rollback path
# (see the file header), and reusing bootstrap_spot() via a narrow bridge
# achieves the collapse this arc requires without that rewrite's risk.
source "$SCRIPT_DIR/_spot_common.sh"

# Parse flags
MODE="both"  # both | full-only | smoke-only | preflight-only | model-zoo-weekly | model-zoo-spec | model-zoo-select
MODEL_ZOO_SPEC_ID=""  # set by --model-zoo-spec <id>
# PR7-7b — evidence-only total-return shadow basis (e.g. "crsp"). When set, the
# full-training run reads the scratch universe_crsp lib + labels off
# total_return_close + isolates all outputs under *_shadow/{basis}/ + is
# hard-blocked from promoting the live champion. Empty = live run (unchanged).
SHADOW_BASIS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --full-only) MODE="full-only" ;;
    --smoke-only) MODE="smoke-only" ;;
    --preflight-only) MODE="preflight-only" ;;
    # L4544: train the weekly model-zoo rotation + immediate CPCV selection
    # (challenger-first; champion retrain is the separate --full-only run).
    --model-zoo-weekly) MODE="model-zoo-weekly" ;;
    # config#1083 PARALLEL fan-out: train exactly ONE challenger spec on this
    # spot (the SF ModelZooTrainMap launches one spot per spec id).
    --model-zoo-spec) shift; MODE="model-zoo-spec"; MODEL_ZOO_SPEC_ID="$1" ;;
    # config#1083 PARALLEL fan-out: run ONLY the selection over whatever specs
    # registered for the date (the SF ModelZooSelect joins after the Map).
    --model-zoo-select) MODE="model-zoo-select" ;;
    # PR7-7b: evidence-only shadow retrain on an alternate basis (crsp).
    --shadow-basis) shift; SHADOW_BASIS="$1" ;;
    --instance-type) shift; INSTANCE_TYPE="$1" ;;
  esac
  shift
done

# PR7-7b — the shadow retrain is the base champion-arch pipeline on an alternate
# basis, so it only composes with the full-training path. Reject combining it
# with the model-zoo modes (those have their own per-spec namespacing) up front.
if [ -n "$SHADOW_BASIS" ]; then
  case "$MODE" in
    both|full-only) : ;;
    *) echo "ERROR: --shadow-basis only composes with --full-only (got MODE=$MODE)" >&2; exit 2 ;;
  esac
  if [ "$SHADOW_BASIS" != "crsp" ]; then
    echo "ERROR: --shadow-basis: only 'crsp' is supported (got '$SHADOW_BASIS')" >&2
    exit 2
  fi
fi

# The full-training heredoc is single-quoted and cannot interpolate a bash var,
# so compute the shadow env-export line HERE and prepend it to the heredoc body
# (interpolating prefix + quoted body), exactly like DEFER_EMAIL_EXPORT below.
# train_handler.main() reads CRSP_SHADOW_ENABLED / SHADOW_BASIS (TrainingIOSpec.
# resolve). Empty when --shadow-basis unset, so a live run is byte-equivalent.
if [ -n "$SHADOW_BASIS" ]; then
  SHADOW_EXPORT="export CRSP_SHADOW_ENABLED=true SHADOW_BASIS=${SHADOW_BASIS}"$'\n'
else
  SHADOW_EXPORT=""
fi

# config#1083 — fail loud if --model-zoo-spec was given without a spec id (the
# Map iteration must carry an explicit spec; a blank id is a wiring bug).
if [ "$MODE" = "model-zoo-spec" ] && [ -z "$MODEL_ZOO_SPEC_ID" ]; then
  echo "ERROR: --model-zoo-spec requires a spec id (got empty)" >&2
  exit 2
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  GBM Spot Training — $(date +%Y-%m-%d)  (SSM transport)"
echo "═══════════════════════════════════════════════════════════════"
if [ -n "$INSTANCE_TYPE" ]; then
  INSTANCE_TYPES="$INSTANCE_TYPE"  # --instance-type X collapses to single value
fi
echo "  Instance types: $INSTANCE_TYPES"
echo "  Subnets       : $SUBNETS"
echo "  AMI           : $AMI_ID"
echo "  Region        : $AWS_REGION"
echo "  Branch        : $BRANCH"
echo "  Mode          : $MODE"
echo "  Shadow basis  : ${SHADOW_BASIS:-<none> (live)}"
echo "  S3 bucket     : $S3_BUCKET"
echo "  Spot attempt  : $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS  (#883 — relaunch on confirmed mid-run reclaim)"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
if [ ! -f "$REPO_ROOT/config/predictor.yaml" ]; then
  echo "ERROR: config/predictor.yaml not found — copy from predictor.sample.yaml"
  exit 1
fi

# Uncommitted-changes check — WARN only (non-interactive: this runs under the
# Saturday Step Function with no TTY). The spot clones origin/$BRANCH, so
# uncommitted local changes simply won't be included.
cd "$REPO_ROOT"
if ! git diff --quiet HEAD -- config.py config/predictor.sample.yaml training/train_handler.py model/ data/ README.md 2>/dev/null; then
  echo "WARNING: uncommitted changes in key files — the spot clones origin/$BRANCH,"
  echo "         so those changes will NOT be included. Commit + push first if intended."
  echo ""
fi

# ── Launch spot instance ──────────────────────────────────────────────────────
# Capacity-resilient launch via krepis.ec2_spot (lib v0.26.0+ as
# nousergon_lib.ec2_spot; invoked directly via krepis per config#1649 — the
# nousergon_lib re-export shim is guard-less under `python -m` on lib
# >=0.81.0 and silently no-ops, the 2026-07-03 incident class).
# Rotates (instance_type × subnet) on InsufficientInstanceCapacity etc.
# Replaces the broken-by-design hardcoded single-subnet + single-instance-type
# pattern (2026-05-22 incident — Evaluator failed in sibling backtester spot).
echo "==> Requesting spot instance (lib CLI rotation: types=[$INSTANCE_TYPES], subnets=[$SUBNETS])..."
INSTANCE_ID=$("$LIB_PYTHON" -m krepis.ec2_spot launch \
  --types "$INSTANCE_TYPES" \
  --subnets "$SUBNETS" \
  --image-id "$AMI_ID" \
  --key-name "$KEY_NAME" \
  --security-group "$SECURITY_GROUP" \
  --iam-profile "$IAM_PROFILE" \
  --name "alpha-engine-gbm-train-$(date +%Y%m%d)" \
  --region "$AWS_REGION")
ec2_spot_rc=$?
if [ "$ec2_spot_rc" -ne 0 ] || [ -z "$INSTANCE_ID" ]; then
  if [ "$ec2_spot_rc" -eq 64 ]; then
    echo "ERROR: capacity exhausted across all instance_type × subnet combinations" >&2
  fi
  if [ "$ec2_spot_rc" -eq 0 ]; then
   # rc=0 with an EMPTY instance id = the launch layer produced nothing
   # (e.g. the guard-less `-m nousergon_lib.ec2_spot` shim no-op,
   # config#1646 — closed at this launcher's transport by the krepis
   # migration, config#1649). `${ec2_spot_rc:-1}` defaults only when UNSET — a
   # captured 0 passed through and the SF recorded a silent success
   # on 2026-07-03. An empty id must always fail loud.
   echo "ERROR: ec2_spot launch exited 0 without an instance id — failing loud (config#1646)" >&2
   ec2_spot_rc=1
  fi
  exit "$ec2_spot_rc"
fi
echo "  Instance ID: $INSTANCE_ID"

RUN_ID="$(date +%Y%m%dT%H%M%SZ)-${INSTANCE_ID}"
S3_STAGING_PREFIX="tmp/spot_train/${RUN_ID}"
S3_STAGING="s3://${S3_BUCKET}/${S3_STAGING_PREFIX}"

# Cleanup — always terminate the instance + remove the S3 staging prefix.
# (S3 lifecycle on tmp/ is the belt-and-suspenders if the trap never fires.)
cleanup() {
  # #883 — capture the dispatcher's exit status FIRST (a non-zero exit is what
  # a mid-run spot reclaim surfaces as, once the SSM workload step fails). Every
  # command below (echo / aws ... || true) would otherwise overwrite $? and a
  # later `exit "$exit_code"` is required so the EXIT trap never masks a real
  # failure as rc=0 (the L4485 class the sibling backtester also guards).
  local exit_code=$?

  # Stop the background heartbeat FIRST — the comment above _heartbeat_stop has
  # said "or in the cleanup EXIT trap" since it was written, and nothing did it.
  # `_heartbeat_start` backgrounds `krepis.heartbeat emit`, which inherits this
  # script's stdout; that stdout is the pipe `krepis.ssm_log_capture` reads
  # until EOF, and EOF needs every writer to close. A `set -e` abort between
  # start and stop therefore left the SSM command held open long after the
  # workload had died — measured as a full 5400s executionTimeout, SIGKILL,
  # ResponseCode 137, and no log shipped, on a stage that had actually failed
  # 142 seconds in (config-I6948, root cause config-I6963).
  #
  # `declare -F` guard: the trap is installed before this function is defined,
  # so an early abort would otherwise hit command-not-found INSIDE the trap and
  # skip the instance-termination call below — leaving a spot instance billing.
  declare -F _heartbeat_stop >/dev/null 2>&1 && _heartbeat_stop

  echo ""
  # Belt-and-suspenders (STEP 3): BEFORE terminating the spot, confirm where
  # each workload's spot-side log landed in S3. The spot SELF-SHIP via
  # krepis.ssm_log_capture (each workload heredoc) is PRIMARY — this
  # is only a bounded best-effort confirmation + a one-hop pointer in the
  # dispatcher log so an operator triaging a failure (esp. an OOM RC=-1 where
  # SSM get-command-invocation returns empty) can find the full log immediately.
  # Bounded: a single short `aws s3 ls` per slug, all failures swallowed, never
  # blocks teardown. Key shape: _ssm_logs/{slug}/{YYYY-MM-DD}/{host}-{HHMMSSZ}.log
  # (krepis.ssm_log_capture._exit_key). The exit-time UTC date is the
  # key component; on a run straddling UTC midnight the log lands under the exit
  # date, so probe today's date.
  local _logdate_now _hit
  _logdate_now="$(date -u +%Y-%m-%d)"
  echo "==> Confirming spot-side workload logs in s3://${S3_BUCKET}/_ssm_logs/ ..."
  for _slug in spot-smoke spot-model-zoo-weekly spot-model-zoo-spec spot-model-zoo-select spot-full-training; do
    _hit="$(aws s3 ls "s3://${S3_BUCKET}/_ssm_logs/${_slug}/${_logdate_now}/" --region "$AWS_REGION" 2>/dev/null | tail -1 || true)"
    if [ -n "$_hit" ]; then
      echo "    ${_slug}: s3://${S3_BUCKET}/_ssm_logs/${_slug}/${_logdate_now}/$(echo "$_hit" | awk '{print $NF}')"
    fi
  done
  echo "    (spot logs above are the FULL workload stdout/stderr — primary diagnostic on RC=-1/OOM)"
  echo "    Failure diagnostics record (if any): s3://${S3_BUCKET}/_spot_diagnostics/ae-predictor/${_logdate_now}.json"
  echo ""

  # #883 — mid-run spot-reclaim relaunch DECISION (lib chokepoint). On a non-zero
  # exit with a provisioned instance, ask the lib whether this was a confirmed AWS
  # reclaim that warrants a fresh-spot relaunch. The lib's classify_termination
  # (describe-instances) MUST run while the instance still exists, so decide HERE,
  # BEFORE terminate-instances. --json's "relaunch" field carries the verdict
  # (alpha-engine-config-I7009); a CLI failure (non-zero exit) is treated as
  # hold (fail loud). The actual `exec` happens AFTER teardown so the dead
  # worker + its S3 staging are already cleaned when the fresh attempt starts.
  local _spot_relaunch=0
  if [ "$exit_code" -ne 0 ] && [ -n "${INSTANCE_ID:-}" ] && [ "$SPOT_ATTEMPT" -lt "$MAX_SPOT_ATTEMPTS" ]; then
    # See alpha-engine-config-I7009 — migrated off the exit-code contract to --json.
    local _decide_json="" _decide_rc=0
    _decide_json="$("$LIB_PYTHON" -m krepis.ec2_spot relaunch-decision \
      --instance-id "$INSTANCE_ID" \
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
        # Fail-loud-but-recovering: record the absorbed interruption on a named
        # CloudWatch surface so the retry is observable, never silent (mirrors the
        # #349 data launcher's AlphaEngine/SpotInterruptionRetry metric).
        aws cloudwatch put-metric-data \
          --namespace "AlphaEngine" \
          --metric-name "SpotInterruptionRetry" \
          --dimensions "Process=predictor-training" \
          --value 1 --unit "Count" \
          --region "$AWS_REGION" 2>/dev/null || true
      fi
    fi
  fi

  echo "==> Terminating spot instance $INSTANCE_ID..."
  aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --output text > /dev/null 2>&1 || true
  # alpha-engine-config-I7442: this file provisions its own un-prefixed
  # S3_STAGING (see the `source _spot_common.sh` header comment above for why
  # this monolith keeps its own cleanup() rather than adopting the shared
  # one wholesale) — bridge it into _spot_common.sh's `spot_common_teardown_
  # staging()`, the SAME chokepoint every other launcher in this repo now
  # goes through, rather than restating its retain-before-delete logic here.
  # "spot_train" matches this file's own S3_STAGING_PREFIX="tmp/spot_train/..."
  # above — there is no separate per-substage slug var in this rollback path.
  _S3_STAGING="$S3_STAGING" _SSM_SLUG="spot_train" spot_common_teardown_staging "$exit_code"
  echo "  Instance terminated."

  # #883 — on a classified reclaim, relaunch a FRESH spot with the SAME argv,
  # threading the incremented SPOT_ATTEMPT via the env. `trap - EXIT` first so the
  # exec'd process installs its own trap cleanly; exec replaces this PID so the
  # relaunch is bounded by SPOT_ATTEMPT<MAX_SPOT_ATTEMPTS (re-checked above). The
  # exec MUST stay in bash — it replaces the launcher's own PID and cannot be lifted.
  if [ "$_spot_relaunch" = "1" ]; then
    echo "==> Spot RECLAIMED by AWS mid-run — relaunching on a fresh spot (attempt $((SPOT_ATTEMPT + 1))/$MAX_SPOT_ATTEMPTS)"
    trap - EXIT
    SPOT_ATTEMPT=$((SPOT_ATTEMPT + 1)) exec bash "$0" ${_ORIG_ARGS[@]+"${_ORIG_ARGS[@]}"}
  fi

  # #883 — CRITICAL: re-exit with the captured status so a recovered cleanup path
  # (the echos / `|| true` teardown above all succeed) can never mask a real
  # failure as rc=0 to the cron/orchestration wrapper.
  exit "$exit_code"
}
trap cleanup EXIT

echo "==> Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

# Stage config/predictor.yaml to S3 (spot fetches via its IAM role).
echo "==> Staging config/predictor.yaml → ${S3_STAGING}/predictor.yaml"
aws s3 cp "$REPO_ROOT/config/predictor.yaml" "${S3_STAGING}/predictor.yaml" --region "$AWS_REGION" --quiet

# ── Wait for the SSM agent to register ────────────────────────────────────────
# Replaces the old SSH-readiness poll. AL2023 ships the SSM agent; with the
# instance profile's AmazonSSMManagedInstanceCore it registers within ~1 min.
echo "==> Waiting for SSM agent to come Online..."
for i in $(seq 1 36); do  # 36 × 5s = 180s budget
  ping=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text --region "$AWS_REGION" 2>/dev/null || true)
  if [ "$ping" = "Online" ]; then
    echo "  SSM agent Online."
    break
  fi
  if [ "$i" -eq 36 ]; then
    echo "ERROR: SSM agent not Online after 180s (instance $INSTANCE_ID)"
    exit 1
  fi
  sleep 5
done

# ── SSM command primitive ─────────────────────────────────────────────────────
# run_ssm "<description>" "<bash script>" [timeout_seconds]
#
# **2026-05-27 — Lib chokepoint lift (ROADMAP L342 PR 4).** This helper
# was the 54-line inline ``aws ssm send-command`` + poll + stream + S3
# capture bash function that L342 was explicitly chartered to retire.
# The lib equivalent ships in ``krepis.ssm_dispatcher`` (lib
# v0.35.0+ as ``nousergon_lib.ssm_dispatcher``; invoked directly via
# ``krepis`` per config#1649 — the nousergon_lib re-export shim is
# guard-less under ``python -m`` on lib >=0.81.0 and silently no-ops,
# [#73](https://github.com/nousergon/nousergon-lib/pull/73))
# with identical contract: base64-wrap → SendCommand → poll → stream
# StandardOutputContent delta → fetch StandardErrorContent on terminal
# non-Success → propagate exit. Adds InvocationDoesNotExist
# registration-grace handling (2026-05-23 SF event-16 substrate weakness)
# that the pre-lift inline form lacked.
#
# The calling convention is unchanged so the existing 5 call sites
# (bootstrap / deps / preflight-only / smoke / full-training) need no
# rewrite. The body is piped to the lib CLI's ``--script-stdin``, which
# reads it verbatim (no command-substitution scanning) — matching the
# pattern alpha-engine-data PR 2 (#330) and alpha-engine-backtester
# PR 3 (#251) adopted for their migrations.
# L394 cascade: --diagnostics-bucket + --diagnostics-prefix activate the
# lib v0.39.0 chokepoint that writes a JSON failure record (status +
# command_id + 4KB stdout/stderr tails + instance_id) to
# s3://${S3_BUCKET}/_spot_diagnostics/ae-predictor/{YYYY-MM-DD}.json on
# terminal non-Success. Best-effort write inside the lib — S3 failure
# swallowed; inner SSM exit always preserved. Substrate is failure-only
# (no-op on Success). Per-repo subprefix discriminates cascade A
# (ae-data) + cascade B (ae-backtester) sibling writes — lib's
# {date}.json key shape would otherwise clobber within a shared prefix.
# S116 rule 5: heartbeat pid for in-phase progress signals (krepis.heartbeat).
# Start via _heartbeat_start before each long-run_ssm call; stop via _heartbeat_stop
# after the phase completes or in the cleanup EXIT trap.
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

run_ssm() {
  local description="$1" script="$2" timeout_s="${3:-3600}"
  printf '%s' "$script" | "$LIB_PYTHON" -m krepis.ssm_dispatcher run \
    --instance-id "$INSTANCE_ID" \
    --description "predictor-training: $description" \
    --timeout "$timeout_s" \
    --output-bucket "$S3_BUCKET" \
    --output-key-prefix "${S3_STAGING_PREFIX}/ssm-output" \
    --region "$AWS_REGION" \
    --diagnostics-bucket "$S3_BUCKET" \
    --diagnostics-prefix "_spot_diagnostics/ae-predictor" \
    --script-stdin
}

# Each run_ssm step is a fresh SSM shell with a minimal env. The
# .env-deprecation arc deleted the sourced .env, so AWS_REGION/
# AWS_DEFAULT_REGION (which boto3 + training/preflight.py's
# check_env_vars("AWS_REGION") require) are no longer set unless each
# step's export line sets them. Same #247 regression as alpha-engine-data's
# spot scripts; spot_train.sh is a sibling repo the original arc missed.
# System is single-region us-east-1 (matches this file's own
# ${AWS_REGION:-us-east-1} defaults). Origin: 2026-05-16 Saturday SF
# PredictorTraining preflight failure.
# ── Bootstrap (watchdog + deps + clone + staged config) ───────────────────────
# GONE: the inline heredoc that carried crucible-predictor#461 (watchdog
# Type=oneshot hang), #462 (bare `command -v python3.12` assertion with
# nothing installing it) and #463 (an interpolated `${REPO_URL}`/`${BRANCH}`
# git-clone line — cutover to infrastructure/_spot_common.sh's bootstrap_spot()
# (alpha-engine-config-I4992/I6922/I7372), the fleet's canonical renderer for
# the shared, non-repo-specific part of a spot bootstrap. See the `source`
# line above for why this script can reuse it without a full rewrite.
#
# The old dual config-copy (experiment-package path PLUS repo-local fallback)
# is now a single copy to the repo-local fallback only: the experiment-package
# destination was audited (config#6846 / alpha-engine-config-I6922) as a dead
# write nothing on the spot ever reads — config.py's experiment-package
# candidates are rooted at the alpha-engine-config checkout, which does not
# exist on a bare predictor clone — and removed from _spot_common.sh's
# bootstrap_spot() ahead of this cutover; carrying it forward here would
# restate dead surface area, not preserve behaviour.
#
# --max-runtime-seconds is threaded into bootstrap_spot() itself (see
# _spot_common.sh), so the spot-side hard-timeout timer this heredoc used to
# arm inline via `systemd-run --on-active=... shutdown -h now` is PRESERVED,
# not dropped — MAX_RUNTIME_SECONDS is already set (default 5400) above.
echo "==> Bootstrapping spot (watchdog, python, clone, config)..."
_S3_STAGING="$S3_STAGING"  # bridge to _spot_common.sh's underscore-prefixed global
bootstrap_spot

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "==> Installing Python dependencies..."
run_ssm "deps" "$(cat <<'DEPS'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference
cd /home/ec2-user/predictor
# No silent fallback (crucible-predictor#462 class, alpha-engine-config-I7372)
# — see the deps/smoke/model-zoo steps below for the same guard.
command -v python3.12 >/dev/null 2>&1 || { echo "ERROR: python3.12 not found — bootstrap_spot() should have installed it; refusing to fall back to a different interpreter" >&2; exit 1; }
PIP="python3.12 -m pip"
$PIP install --upgrade pip -q
# alpha-engine-lib is public (git+https in requirements.txt, no auth).
# CORRECTED 2026-08-13 (alpha-engine-config-I6998 deliverable 4): flow-doctor
# IS on public PyPI (latest 0.11.0 measured 2026-08-12) — this was never true.
# The real defect this line's `grep -v` was masking: pip <23.3 (AL2023 ships
# 23.2.1) predates PEP 685 extras normalisation, so an underscored
# `krepis[flow_doctor]` extra was silently dropped with only a WARNING on a
# SUCCESSFUL exit (config-I6963). The filter below has been a no-op self-fix
# for that — `flow-doctor` is not itself a requirements.txt line, it is an
# EXTRA on the `krepis[...]` line, so `grep -v '^flow-doctor'` never matched
# anything and never filtered anything out. See
# infrastructure/_spot_common.sh install_deps() (the live path this monolith
# is a rollback for) for the real fix: hyphenated extras + a hard fail on any
# "does not provide the extra" pip warning, so the defect surfaces at install
# time instead of at import time in a later process.
grep -v '^flow-doctor' requirements.txt | $PIP install -q -r /dev/stdin
echo "Dependencies installed."
$PIP list --format=columns | grep -iE 'numpy|pandas|lightgbm|scikit-learn|scipy|shap|pyyaml|alpha-engine-lib' || true
DEPS
)" 900

# ── Preflight-only (Friday shell_run dry path) ────────────────────────────────
# Boot + lib-pin/import + read-only ArcticDB/universe-freshness probe, then
# exit 0. This runs the SAME bootstrap+deps steps the real Saturday run uses
# (so it catches lib-pin drift, sys.path breakage, image gaps, SSM timeouts,
# stale ArcticDB) but stops HERE — before the smoke step and before the
# full-training step.
#
# Hard invariant under this mode:
#   • run_meta_training() is NEVER invoked → NO model training, NO walk-forward.
#   • The `if not dry_run:` upload/promote block in meta_trainer.py is never
#     reached → NO weights/meta/* write, NO manifest, NO dated archive.
#   • train_handler.main()'s training_summary / triple-barrier-gate / email /
#     health-status writes are never reached (they live after run_meta_training).
#   • The probe imports the training package + runs TrainingPreflight (env +
#     S3-bucket *reachability* check — no object writes) + a read-only
#     ArcticDB `list_symbols()` / latest-index probe. No put_object, no
#     config write, no external API (yfinance/Anthropic) call.
# The `exit 0` is a clean dispatcher exit; `trap cleanup EXIT` still fires
# (terminates the spot, clears the S3 staging prefix — staging cleanup only).
if [ "$MODE" = "preflight-only" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  PREFLIGHT-ONLY (no training, no promotion, no writes)"
  echo "═══════════════════════════════════════════════════════════════"
  run_ssm "preflight-only" "$(cat <<'PREFLIGHT'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference
cd /home/ec2-user/predictor
# No silent fallback to the AMI's system python3 (crucible-predictor#462
# class, alpha-engine-config-I7372): bootstrap_spot() installs python3.12 and
# asserts it is present before this step ever runs, so an absence here means
# the bootstrap's own postcondition was violated — resolving against a
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

# 1. Import the training package (catches sys.path / lib-pin / image gaps).
#    Importing train_handler transitively imports the lib + training stack
#    WITHOUT invoking main(), so no training runs.
log.info('[1/3] Importing training package...')
import nousergon_lib  # lib-pin presence (version asserted by requirements.txt pin)
from training import train_handler  # noqa: F401  (import-only; main() NOT called)
from training import model_zoo  # noqa: F401  (L4544 rotation path; import-only)
from training.preflight import TrainingPreflight
log.info('       OK — nousergon_lib + training.train_handler + model_zoo import clean')

# 2. Reuse the EXISTING training preflight (env vars + S3 bucket
#    *reachability*; check_s3_bucket is a read/head, no object write).
log.info('[2/3] Running TrainingPreflight (env + S3 connectivity)...')
TrainingPreflight(bucket=bucket).run()
log.info('       OK — env vars present, S3 bucket reachable')

# 3. Read-only ArcticDB connectivity + universe-freshness probe.
#    list_symbols() + a single read().tail(1) — NO download_from_arctic(),
#    NO parquet writes, NO training array build. Mirrors the connectivity
#    the real run depends on without doing any work.
log.info('[3/3] ArcticDB connectivity + universe-freshness probe...')
from nousergon_lib.arcticdb import open_arctic
arctic = open_arctic(bucket)
universe = arctic.get_library('universe')
symbols = universe.list_symbols()
n = len(symbols)
if n == 0:
    raise RuntimeError(
        'ArcticDB universe library is empty/unreachable — '
        'Saturday DataPhase1 + weekly backfill have not run cleanly.'
    )
probe = sorted(symbols)[0]
df_tail = universe.read(probe).data.tail(1)
latest = df_tail.index.max() if not df_tail.empty else 'n/a'
log.info('       OK — universe has %d symbols; %s latest index=%s', n, probe, latest)

print()
print('=' * 60)
print('  PREFLIGHT-ONLY RESULT: PASS')
print('=' * 60)
print(f'  Imports:        nousergon_lib + training stack clean')
print(f'  TrainingPreflight: PASS (env + S3 reachable)')
print(f'  ArcticDB:       {n} universe symbols (probe {probe} latest={latest})')
print(f'  Training:       SKIPPED (no run_meta_training call)')
print(f'  Promotion:      SKIPPED (no weights/meta write)')
print(f'  S3/config writes: NONE')
print('=' * 60)
PYEOF
PREFLIGHT
)" 600
  echo ""
  echo "==> Preflight-only mode — PASS. No training, no promotion, no writes."
  echo "    Exiting 0 BEFORE smoke + full-training steps."
  exit 0
fi

# ── Smoke test (dry_run=True) ─────────────────────────────────────────────────
# model-zoo modes skip the champion smoke (they train/select challenger variants).
if [ "$MODE" != "full-only" ] && [ "$MODE" != "model-zoo-weekly" ] && [ "$MODE" != "model-zoo-spec" ] && [ "$MODE" != "model-zoo-select" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  SMOKE TEST (dry_run=True)"
  echo "═══════════════════════════════════════════════════════════════"
  run_ssm "smoke" "${RUN_TOKEN_EXPORT}$(cat <<'SMOKE'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
# No silent fallback to the AMI's system python3 (crucible-predictor#462
# class, alpha-engine-config-I7372): bootstrap_spot() installs python3.12 and
# asserts it is present before this step ever runs, so an absence here means
# the bootstrap's own postcondition was violated — resolving against a
# different interpreter than requirements.txt was pinned for is worse than
# failing loud.
command -v python3.12 >/dev/null 2>&1 || { echo "ERROR: python3.12 not found — bootstrap_spot() should have installed it; refusing to fall back to a different interpreter" >&2; exit 1; }
PY=python3.12
# Spot-side log durability — the python workload below ran inline via $PY - so
# its stdout/stderr lived ONLY in SSM get-command-invocation, which returns
# EMPTY when the spot dies mid-run e.g. OOM RC=-1 and is destroyed when the
# dispatcher cleanup EXIT trap terminates the box. Route the workload through
# the lib chokepoint krepis.ssm_log_capture: it tees combined
# stdout+stderr to a spot-local logfile AND ships that logfile to S3 on EXIT
# including SIGKILL of the workload BEFORE the dispatcher tears the box down,
# then propagates the workload exit code verbatim so set -eo pipefail and the
# SF still see the real failure. The wrapper is a lightweight separate process
# so the kernel OOM-killer reaps the heavy workload subprocess, not the shipper.
# NOTE keep this region free of apostrophes and parens: bash 3.2 scans even a
# quoted heredoc body for the closing paren of the enclosing run_ssm command
# substitution.
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
    print(f'  Architecture:   v3.0 Meta-Model')
    print(f'  Meta-Model IC:  {result.get("meta_model_ic", result.get("test_ic", "n/a"))}')
    print(f'  Momentum IC:    {result.get("momentum_test_ic", "n/a")}')
    print(f'  Volatility IC:  {result.get("volatility_test_ic", "n/a")}')
    print(f'  Regime Acc:     {(result.get("regime_accuracy", 0) * 100):.1f}%')
    rc = result.get('research_calibrator_metrics', {})
    if rc:
        print(f'  Research Cal:   {rc.get("n_samples", 0)} samples, overall hit={rc.get("overall_hit_rate", "n/a")}')
        for bucket, info in rc.get('buckets', {}).items():
            if info.get('n', 0) > 0:
                print(f'    Score {bucket}: hit_rate={info["hit_rate"]:.1%} (n={info["n"]})')
    wf = result.get('walk_forward', {})
    print(f'  WF Momentum:    median_IC={wf.get("momentum_median_ic", "n/a")}')
    print(f'  WF Volatility:  median_IC={wf.get("volatility_median_ic", "n/a")}')
    wf_status = 'PASS' if wf.get('passes_wf') else 'FAIL'
    print(f'  WF Status:      {wf_status}')
    coefs = result.get('meta_coefficients', {})
    if coefs:
        print(f'  Meta-model coefficients:')
        for name, val in sorted(coefs.items(), key=lambda x: -abs(x[1])):
            if name != 'intercept' and abs(val) > 0.0001:
                print(f'    {name:<30} {val:+.4f}')
        print(f'    {"intercept":<30} {coefs.get("intercept", 0):+.4f}')
    if wf.get('folds'):
        print(f'  Per-fold ICs (momentum / volatility):')
        for f in wf['folds']:
            print(f'    Fold {f["fold"]:>2}: mom={f["mom_ic"]:+.4f}  vol={f["vol_ic"]:+.4f}  [{f["test_start"]} -> {f["test_end"]}]')
else:
    print(f'  Architecture:   v2.0 Single/Ensemble GBM')
    print(f'  Test IC:        {result.get("test_ic", "n/a")}')
    print(f'  MSE IC:         {result.get("mse_ic", "n/a")}')
    print(f'  Rank IC:        {result.get("rank_ic", "n/a")}')
    print(f'  Ensemble IC:    {result.get("ensemble_ic", "n/a")}')
    if result.get('catboost_enabled'):
        print(f'  CatBoost IC:    {result.get("catboost_ic", "n/a")}')
        print(f'  LGB-Cat Blend:  {result.get("lgb_cat_blend_ic", "n/a")}  weights={result.get("blend_weights", "n/a")}')
    print(f'  IC IR:          {result.get("ic_ir", "n/a")}')
    wf = result.get('walk_forward', {})
    wf_status = 'PASS' if wf.get('passes_wf') else 'FAIL/skipped'
    print(f'  Walk-forward:   {wf_status}  (median_IC={wf.get("median_ic", "n/a")})')
    fics = result.get('feature_ics', {})
    if fics:
        sorted_fics = sorted(fics.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f'  Top 5 feature ICs:')
        for name, ic in sorted_fics[:5]:
            print(f'    {name:<22} {ic:+.4f}')

print(f'  Promoted:       {result.get("promoted", "n/a")}')
print(f'  Elapsed:        {result.get("elapsed_s", "n/a")}s')
noise = result.get('noise_candidates', [])
if noise:
    print(f'  Noise features: {noise}')
print('=' * 60)
PYEOF
$PY -m krepis.ssm_log_capture run --slug spot-smoke --log /var/log/spot-smoke.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-smoke.py
SMOKE
)" 1800
  echo "Smoke test complete."
  if [ "$MODE" = "smoke-only" ]; then
    echo "==> Smoke-only mode — skipping full training."
    exit 0
  fi
fi

# ── Model-zoo weekly rotation + immediate CPCV selection (L4544) ──────────────
# Trains the N stalest challenger specs, ranks them by leak-free CPCV, writes a
# leaderboard, and (only if MODEL_ZOO_AUTO_PROMOTE_WINNER) promotes the winner.
# Challenger-first + live-contract-restore are enforced inside model_zoo, so this
# never disturbs the live champion. Runs INSTEAD OF the champion retrain (that's
# the separate --full-only state); exits 0 when done.
if [ "$MODE" = "model-zoo-weekly" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  MODEL-ZOO WEEKLY ROTATION + SELECT (observe-first by default)"
  echo "═══════════════════════════════════════════════════════════════"
  run_ssm "model-zoo-weekly" "${RUN_TOKEN_EXPORT}$(cat <<'ZOO'
set -eo pipefail
# config#1066 — pin ALPHA_ENGINE_EXPERIMENT_ID so config.py loads the staged
# experiment-package yaml, MODEL_SPECS populates, and the rotation trains
# challengers. The probe below logs the resolved path + count for diagnosis.
# NOTE keep this heredoc free of apostrophes and parens: bash 3.2 scans even a
# quoted heredoc body for the closing paren of the enclosing run_ssm command
# substitution.
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
# No silent fallback to the AMI's system python3 (crucible-predictor#462
# class, alpha-engine-config-I7372): bootstrap_spot() installs python3.12 and
# asserts it is present before this step ever runs, so an absence here means
# the bootstrap's own postcondition was violated — resolving against a
# different interpreter than requirements.txt was pinned for is worse than
# failing loud.
command -v python3.12 >/dev/null 2>&1 || { echo "ERROR: python3.12 not found — bootstrap_spot() should have installed it; refusing to fall back to a different interpreter" >&2; exit 1; }
PY=python3.12
# Spot-side log durability — see the smoke step comment. Route the workload
# through krepis.ssm_log_capture so the model-zoo log reaches S3 on
# EXIT including OOM-kill before the dispatcher terminates the box. Paren-free
# and apostrophe-free per the bash 3.2 note above.
cat > /tmp/spot-model-zoo-weekly.py <<'PYEOF'
import os, sys
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', os.environ.get('S3_BUCKET', 'alpha-engine-research'))
bucket = os.environ.get('S3_BUCKET', 'alpha-engine-research')

import logging
# Flow-doctor wiring: importing training.model_zoo runs its module-top
# setup_logging (predictor-model-zoo + flow-doctor-model-zoo.yaml: email +
# S3 sink), which clears+reinstalls the root handler. We ALSO call it
# explicitly here BEFORE the import so this entrypoint stays wired even if
# the import order is later changed, and so a config-import crash inside the
# import is captured. Idempotent: setup_logging clears existing handlers.
# NOTE keep this heredoc free of apostrophes per the bash 3.2 note above.
import os.path as _osp
_FD_YAML = _osp.join(_osp.abspath("."), "flow-doctor-model-zoo.yaml")
from krepis.logging import setup_logging
setup_logging("predictor-model-zoo", flow_doctor_yaml=_FD_YAML, exclude_patterns=[])

import config as cfg
from training.model_zoo import run_rotation_and_select

# config#1051 logging probe: pin WHAT the child spot loaded so an empty
# MODEL_SPECS (the 6/13 inert-rotation root cause) is diagnosable from the log.
log = logging.getLogger('model_zoo.spot')
log.info('model_zoo spot probe: MODEL_SPECS=%d  config=%s  ALPHA_ENGINE_EXPERIMENT_ID=%s',
         len(getattr(cfg, 'MODEL_SPECS', [])),
         getattr(cfg, '_CONFIG_PATH', '?'),
         getattr(cfg, '_EXPERIMENT_ID', os.environ.get('ALPHA_ENGINE_EXPERIMENT_ID', 'reference')))

# config#1051: pass a real trading_day so leaderboard / trial_log key on a date,
# not null (the 6/13 leaderboard had date=null). now_dual is backward-looking.
try:
    from krepis.dates import now_dual
    _td = now_dual().trading_day
    date_str = _td.isoformat() if hasattr(_td, 'isoformat') else str(_td)
except Exception:
    log.warning('model_zoo spot: now_dual unavailable — run_rotation_and_select will self-default', exc_info=True)
    date_str = None

budget = int(os.environ.get('MODEL_ZOO_WEEKLY_BUDGET', getattr(cfg, 'MODEL_ZOO_WEEKLY_BUDGET', 3)))
board = run_rotation_and_select(bucket, budget=budget, date_str=date_str)

print()
print('=' * 60)
print('  MODEL-ZOO ROTATION + SELECT')
print('=' * 60)
print(f'  Mode:           {board.get("mode")}')
champ = board.get('champion', {})
print(f'  Champion CPCV:  {champ.get("cpcv_mean_ic")} (fwd={champ.get("forward_days")})')
for c in board.get('candidates', []):
    print(f'    {c.get("spec_id"):<18} cpcv={c.get("cpcv_mean_ic")} fwd={c.get("forward_days")} '
          f'gate={c.get("passes_gate")} eligible={c.get("eligible")} ({c.get("reason")})')
print(f'  Winner:         {board.get("winner_version_id")}')
print(f'  Promoted:       {board.get("promoted")}')
print('=' * 60)
PYEOF
$PY -m krepis.ssm_log_capture run --slug spot-model-zoo-weekly --log /var/log/spot-model-zoo-weekly.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-model-zoo-weekly.py
ZOO
)" "${MAX_RUNTIME_SECONDS}"

  aws cloudwatch put-metric-data \
    --namespace "AlphaEngine" \
    --metric-name "Heartbeat" \
    --dimensions "Process=predictor-model-zoo" \
    --value 1 --unit "Count" \
    --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
    && echo "Heartbeat emitted: predictor-model-zoo" \
    || echo "WARNING: Failed to emit heartbeat (non-fatal)"

  echo ""
  echo "==> Model-zoo rotation complete. Instance will be terminated."
  exit 0
fi

# ── Model-zoo PARALLEL: train ONE challenger spec (config#1083) ────────────────
# The SF ModelZooTrainMap launches one spot per spec id; this trains exactly that
# spec as a challenger (challenger-first + live-contract restore are enforced
# inside model_zoo.train_one_spec) and exits NON-ZERO only on a real training
# failure — so the Map iteration records THIS spec's failure without aborting
# siblings (the per-spec isolation property). Mirrors the model-zoo-weekly
# workload wrapping: ssm_log_capture ship-on-exit, flow-doctor setup_logging,
# experiment-package staging. No selection / promotion happens here — that's the
# separate ModelZooSelect state (--model-zoo-select).
if [ "$MODE" = "model-zoo-spec" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  MODEL-ZOO TRAIN ONE SPEC: ${MODEL_ZOO_SPEC_ID}"
  echo "═══════════════════════════════════════════════════════════════"
  # Interpolating export prefix so the quoted heredoc body (which must stay
  # paren/apostrophe-free per the bash 3.2 note) reads the spec id from the env.
  MZ_SPEC_EXPORT="export MODEL_ZOO_SPEC_ID=${MODEL_ZOO_SPEC_ID}"$'\n'
  run_ssm "model-zoo-spec" "${RUN_TOKEN_EXPORT}${MZ_SPEC_EXPORT}$(cat <<'ZOOSPEC'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
# No silent fallback to the AMI's system python3 (crucible-predictor#462
# class, alpha-engine-config-I7372): bootstrap_spot() installs python3.12 and
# asserts it is present before this step ever runs, so an absence here means
# the bootstrap's own postcondition was violated — resolving against a
# different interpreter than requirements.txt was pinned for is worse than
# failing loud.
command -v python3.12 >/dev/null 2>&1 || { echo "ERROR: python3.12 not found — bootstrap_spot() should have installed it; refusing to fall back to a different interpreter" >&2; exit 1; }
PY=python3.12
# Spot-side log durability + flow-doctor wiring — see the model-zoo-weekly step.
# Paren-free and apostrophe-free per the bash 3.2 note above.
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
log.info('model_zoo train-spec probe: spec=%s  MODEL_SPECS=%d  config=%s  ALPHA_ENGINE_EXPERIMENT_ID=%s',
         spec_id, len(getattr(cfg, 'MODEL_SPECS', [])),
         getattr(cfg, '_CONFIG_PATH', '?'),
         getattr(cfg, '_EXPERIMENT_ID', os.environ.get('ALPHA_ENGINE_EXPERIMENT_ID', 'reference')))
if not spec_id:
    raise SystemExit('MODEL_ZOO_SPEC_ID not set on the spot')

try:
    from krepis.dates import now_dual
    _td = now_dual().trading_day
    date_str = _td.isoformat() if hasattr(_td, 'isoformat') else str(_td)
except Exception:
    log.warning('model_zoo spot: now_dual unavailable', exc_info=True)
    date_str = None

# Raises on a real training failure → non-zero exit → the Map iteration records
# THIS spec failed without aborting siblings.
train_one_spec(spec_id, bucket, date_str=date_str)
print()
print('=' * 60)
print('  MODEL-ZOO TRAIN-SPEC ' + spec_id + ' COMPLETE')
print('=' * 60)
PYEOF
$PY -m krepis.ssm_log_capture run --slug spot-model-zoo-spec --log /var/log/spot-model-zoo-spec.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-model-zoo-spec.py
ZOOSPEC
)" "${MAX_RUNTIME_SECONDS}"

  aws cloudwatch put-metric-data \
    --namespace "AlphaEngine" \
    --metric-name "Heartbeat" \
    --dimensions "Process=predictor-model-zoo-spec" \
    --value 1 --unit "Count" \
    --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
    && echo "Heartbeat emitted: predictor-model-zoo-spec" \
    || echo "WARNING: Failed to emit heartbeat (non-fatal)"

  echo ""
  echo "==> Model-zoo train-spec ${MODEL_ZOO_SPEC_ID} complete. Instance will be terminated."
  exit 0
fi

# ── Model-zoo PARALLEL: SELECT over the registered specs (config#1083) ─────────
# Runs AFTER the ModelZooTrainMap joins (one spot). Selects over whatever spec-*
# challengers registered for the date (failed Map iterations are simply absent —
# tolerated), writes the leaderboard to BOTH the dated key AND latest.json,
# promotes the winner if MODEL_ZOO_AUTO_PROMOTE_WINNER, and sends the one
# consolidated digest. No training happens here.
if [ "$MODE" = "model-zoo-select" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  MODEL-ZOO SELECT (observe-first by default)"
  echo "═══════════════════════════════════════════════════════════════"
  run_ssm "model-zoo-select" "${RUN_TOKEN_EXPORT}$(cat <<'ZOOSEL'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
# No silent fallback to the AMI's system python3 (crucible-predictor#462
# class, alpha-engine-config-I7372): bootstrap_spot() installs python3.12 and
# asserts it is present before this step ever runs, so an absence here means
# the bootstrap's own postcondition was violated — resolving against a
# different interpreter than requirements.txt was pinned for is worse than
# failing loud.
command -v python3.12 >/dev/null 2>&1 || { echo "ERROR: python3.12 not found — bootstrap_spot() should have installed it; refusing to fall back to a different interpreter" >&2; exit 1; }
PY=python3.12
# Spot-side log durability + flow-doctor wiring — see the model-zoo-weekly step.
# Paren-free and apostrophe-free per the bash 3.2 note above.
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
log.info('model_zoo select probe: MODEL_SPECS=%d  config=%s  ALPHA_ENGINE_EXPERIMENT_ID=%s',
         len(getattr(cfg, 'MODEL_SPECS', [])),
         getattr(cfg, '_CONFIG_PATH', '?'),
         getattr(cfg, '_EXPERIMENT_ID', os.environ.get('ALPHA_ENGINE_EXPERIMENT_ID', 'reference')))

try:
    from krepis.dates import now_dual
    _td = now_dual().trading_day
    date_str = _td.isoformat() if hasattr(_td, 'isoformat') else str(_td)
except Exception:
    log.warning('model_zoo spot: now_dual unavailable - run_select_only will self-default', exc_info=True)
    date_str = None

board = run_select_only(bucket, date_str=date_str)
print()
print('=' * 60)
print('  MODEL-ZOO SELECT')
print('=' * 60)
champ = board.get('champion', {})
print('  Mode:           ' + str(board.get('mode')))
print('  Champion CPCV:  ' + str(champ.get('cpcv_mean_ic')) + ' fwd=' + str(champ.get('forward_days')))
for c in board.get('candidates', []):
    print('    ' + str(c.get('spec_id')) + ' cpcv=' + str(c.get('cpcv_mean_ic')) + ' fwd=' + str(c.get('forward_days')) + ' eligible=' + str(c.get('eligible')) + ' (' + str(c.get('reason')) + ')')
print('  Winner:         ' + str(board.get('winner_version_id')))
print('  Promoted:       ' + str(board.get('promoted')))
print('=' * 60)
PYEOF
$PY -m krepis.ssm_log_capture run --slug spot-model-zoo-select --log /var/log/spot-model-zoo-select.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-model-zoo-select.py
ZOOSEL
)" "${MAX_RUNTIME_SECONDS}"

  aws cloudwatch put-metric-data \
    --namespace "AlphaEngine" \
    --metric-name "Heartbeat" \
    --dimensions "Process=predictor-model-zoo-select" \
    --value 1 --unit "Count" \
    --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
    && echo "Heartbeat emitted: predictor-model-zoo-select" \
    || echo "WARNING: Failed to emit heartbeat (non-fatal)"

  echo ""
  echo "==> Model-zoo select complete. Instance will be terminated."
  exit 0
fi

# ── Full training (dry_run=False) ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  FULL TRAINING (dry_run=False)"
echo "═══════════════════════════════════════════════════════════════"
  _heartbeat_start "spot-full-training" 300
run_ssm "full-training" "${RUN_TOKEN_EXPORT}${DEFER_EMAIL_EXPORT}${SHADOW_EXPORT}$(cat <<'TRAIN'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
# No silent fallback to the AMI's system python3 (crucible-predictor#462
# class, alpha-engine-config-I7372): bootstrap_spot() installs python3.12 and
# asserts it is present before this step ever runs, so an absence here means
# the bootstrap's own postcondition was violated — resolving against a
# different interpreter than requirements.txt was pinned for is worse than
# failing loud.
command -v python3.12 >/dev/null 2>&1 || { echo "ERROR: python3.12 not found — bootstrap_spot() should have installed it; refusing to fall back to a different interpreter" >&2; exit 1; }
PY=python3.12
# Spot-side log durability — this is THE workload whose log was lost on the
# off-cycle full-only OOM RC=-1 incident the python ran inline via $PY - so its
# full training log lived only in SSM get-command-invocation which returns empty
# on instance death. Route it through krepis.ssm_log_capture: tee
# combined stdout+stderr to /var/log/spot-full-training.log AND ship to S3 on
# EXIT including SIGKILL BEFORE the dispatcher cleanup EXIT trap terminates the
# box, propagating the workload exit code so set -eo pipefail and the SF still
# see the real failure. Paren-free and apostrophe-free per the bash 3.2 note.
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
    print(f'  Architecture:   v3.0 Meta-Model')
    print(f'  Meta-Model IC:  {result.get("meta_model_ic", result.get("test_ic", "n/a"))}')
    print(f'  Momentum IC:    {result.get("momentum_test_ic", "n/a")}')
    print(f'  Volatility IC:  {result.get("volatility_test_ic", "n/a")}')
    print(f'  Regime Acc:     {(result.get("regime_accuracy", 0) * 100):.1f}%')
    rc = result.get('research_calibrator_metrics', {})
    if rc:
        print(f'  Research Cal:   {rc.get("n_samples", 0)} samples, overall hit={rc.get("overall_hit_rate", "n/a")}')
    wf = result.get('walk_forward', {})
    print(f'  WF Momentum:    median_IC={wf.get("momentum_median_ic", "n/a")}')
    print(f'  WF Volatility:  median_IC={wf.get("volatility_median_ic", "n/a")}')
    coefs = result.get('meta_coefficients', {})
    if coefs:
        print(f'  Meta-model coefficients:')
        for name, val in sorted(coefs.items(), key=lambda x: -abs(x[1])):
            if name != 'intercept' and abs(val) > 0.0001:
                print(f'    {name:<30} {val:+.4f}')
else:
    print(f'  Architecture:   v2.0 Single/Ensemble GBM')
    print(f'  Test IC:        {result.get("test_ic", "n/a")}')
    print(f'  MSE IC:         {result.get("mse_ic", "n/a")}')
    print(f'  Rank IC:        {result.get("rank_ic", "n/a")}')
    print(f'  Ensemble IC:    {result.get("ensemble_ic", "n/a")}')
    wf = result.get('walk_forward', {})
    wf_status = 'PASS' if wf.get('passes_wf') else 'FAIL/skipped'
    print(f'  Walk-forward:   {wf_status}  (median_IC={wf.get("median_ic", "n/a")})')

print(f'  Promoted:       {result.get("promoted", "n/a")}')
print(f'  Promoted mode:  {result.get("promoted_mode", "n/a")}')
print(f'  Elapsed:        {result.get("elapsed_s", "n/a")}s')
print(f'  Slim cache:     {result.get("slim_cache_tickers", "n/a")} tickers')
print('=' * 60)
PYEOF
$PY -m krepis.ssm_log_capture run --slug spot-full-training --log /var/log/spot-full-training.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-full-training.py
TRAIN
)" "${MAX_RUNTIME_SECONDS}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Training complete."
echo "═══════════════════════════════════════════════════════════════"

_heartbeat_stop
# CloudWatch heartbeat on successful completion (unchanged).
aws cloudwatch put-metric-data \
  --namespace "AlphaEngine" \
  --metric-name "Heartbeat" \
  --dimensions "Process=predictor-training" \
  --value 1 --unit "Count" \
  --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
  && echo "Heartbeat emitted: predictor-training" \
  || echo "WARNING: Failed to emit heartbeat (non-fatal)"
