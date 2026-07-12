#!/usr/bin/env bash
# infrastructure/deploy.sh — Build and deploy the predictor Lambda container image.
#
# Uses the container image pattern because PyTorch (~350MB CPU-only) exceeds
# the Lambda zip layer limit. The image is pushed to ECR and the Lambda
# function code is updated in-place.
#
# Prerequisites:
#   - Docker installed and running
#   - AWS CLI configured (or IAM role on EC2/CodeBuild)
#   - ECR repo 'alpha-engine-predictor' exists in your account
#   - Lambda function 'alpha-engine-predictor-inference' already created
#
# Usage:
#   ./infrastructure/deploy.sh                # full deploy
#   ./infrastructure/deploy.sh --dry-run      # build image only, skip AWS push
#
# Environment variables (auto-detected if not set):
#   AWS_ACCOUNT_ID   — 12-digit AWS account ID (auto-detected via aws sts)
#   AWS_REGION       — defaults to us-east-1

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
ECR_REPO="alpha-engine-predictor"
LAMBDA_FUNCTION="alpha-engine-predictor-inference"
# Second Lambda sharing the same image, with a per-function CMD override
# (set at Lambda creation time via --image-config Command=[...]) pointing
# at regime.handler.lambda_handler. The Saturday SF RegimeSubstrate state
# (alpha-engine-config) invokes this function. Mirrors the shared-image
# pattern alpha-engine-research uses for eval-judge + rationale-clustering.
# Skipped gracefully if the function does not yet exist (one-time create
# is a manual operator step — see setup-regime-lambda.sh).
REGIME_LAMBDA_FUNCTION="alpha-engine-predictor-regime-substrate"
REGIME_LAMBDA_CMD='regime.handler.lambda_handler'
REGIME_LAMBDA_MEMORY=1024
REGIME_LAMBDA_TIMEOUT=300
# Third Lambda sharing the same image: the T1 retrospective HMM smoothing
# eval (regime-v3-260514.md §5.3.3). CMD override points at
# regime.retrospective_eval_handler.lambda_handler. Saturday SF
# RegimeRetrospectiveEval state (alpha-engine-config) invokes this
# function. Auto-created on first deploy when not yet present (see
# Step 10 below); break-glass operator step via
# setup-regime-retrospective-eval-lambda.sh still works.
REGIME_EVAL_LAMBDA_FUNCTION="alpha-engine-predictor-regime-retrospective-eval"
REGIME_EVAL_LAMBDA_CMD='regime.retrospective_eval_handler.lambda_handler'
REGIME_EVAL_LAMBDA_MEMORY=2048
REGIME_EVAL_LAMBDA_TIMEOUT=600
# Lambda execution role for both regime Lambdas (and inference). Used by
# the auto-create paths in Steps 9 + 10. The deploy IAM role
# (github-actions-lambda-deploy) is granted iam:PassRole on
# arn:aws:iam::*:role/alpha-engine-* with PassedToService=lambda.amazonaws.com
# per alpha-engine-data/infrastructure/iam/github-actions-lambda-deploy.json.
LAMBDA_EXECUTION_ROLE_NAME="alpha-engine-predictor-role"
IMAGE_TAG="latest"
DRY_RUN=false

# ── Canary status contract (config#853) ──────────────────────────────────────
# Post-deploy canaries invoke the freshly-published version with a dry_run
# payload and gate promotion on the response's `statusCode`. The acceptance
# decision is centralised here so all three canary sites (inference, regime
# substrate, regime retrospective-eval) share ONE producer-status contract and
# a future skip-path added to any handler inherits correct rollback semantics
# by default.
#
# Producer-status contract (verified against the handlers in this repo):
#   inference/handler.py            — lambda_handler
#   regime/handler.py               — lambda_handler
#   regime/retrospective_eval_handler.py — lambda_handler
#
#   200          — SUCCESS. The dry_run path returns 200 for both a real run
#                  AND every cold-start / empty-history no-op (the handlers
#                  fall through to the same 200 return when there is nothing
#                  to process). This is the only success code emitted today.
#   204          — NO-OP success (allowlisted, defensive). Reserved for a
#                  future "nothing to do" skip-path (HTTP No Content) so such
#                  a path does NOT false-red the deploy and roll back a
#                  perfectly good image. No handler emits this yet.
#   skipped      — NO-OP success (allowlisted, defensive). String form of the
#                  same no-op contract, mirroring nousergon-data #295's
#                  status='SKIPPED' allowlist. Matched against the payload's
#                  `statusCode` field (a future handler may set
#                  statusCode="skipped" on a no-op). No handler emits this yet.
#
#   400 / 500 / any other value — REAL ERROR. 400 = bad request (e.g. the
#                  deprecated `action=train` route, unknown regime action);
#                  500 / unhandled exception (FunctionError, checked
#                  separately) = genuine failure. These MUST roll back.
#
# Mirrors the explicit-allowlist pattern in
# nousergon-data/infrastructure/deploy.sh (config "data #295"): enumerate the
# legitimate no-op statuses rather than blanket-accepting non-200, so a real
# error code can never masquerade as a no-op.
CANARY_NOOP_STATUSES=("200" "204" "skipped")

# canary_status_ok <statusCode>
#   Exit 0 (accept → promote) iff <statusCode> is in the allowlist above.
#   Exit 1 (reject → roll back) for every other value, including the empty
#   string / "0" that the payload parsers emit on a malformed response.
canary_status_ok() {
  local status="$1"
  local allowed
  for allowed in "${CANARY_NOOP_STATUSES[@]}"; do
    if [ "$status" = "$allowed" ]; then
      return 0
    fi
  done
  return 1
}

# canary_accept <expect> <status> <present_keys>
#   Central per-action acceptance decision, shared by every canary site. The
#   handler in this repo emits TWO distinct response contracts, and gating both
#   on `statusCode` is wrong for one of them:
#
#   <expect> = "statusCode"  → HTTP-shaped handler (the default `predict` path
#              returns {"statusCode": 200, "body": ...}). Accept iff <status> is
#              in the CANARY_NOOP_STATUSES allowlist (canary_status_ok).
#
#   <expect> = any other value → a Step-Function GATE action (check_trading_day,
#              check_weekly_run_day, check_drift, check_pipeline_contract). These
#              return a RAW DOMAIN DICT consumed by SF Choice states on named
#              keys (is_trading_day / has_violation / status …) and deliberately
#              DO NOT emit a `statusCode` (config#1430/#1824/#1282/#1853). Accept
#              iff <expect> — the action's declared domain key — is present in
#              the space-separated <present_keys> of the returned object. Key
#              presence proves the invoke dispatched to the RIGHT branch and
#              returned its contract; a broken import / missing branch instead
#              surfaces as a FunctionError (checked by the caller) or a response
#              lacking the key. Gating these on `statusCode` checks an invariant
#              they never satisfy — the 2026-07-12 v351 false-canary-fail that
#              refused to promote a healthy image (PR #362).
#
#   A FunctionError (unhandled exception / broken import) is checked by the
#   caller BEFORE this and always rejects, in every mode.
canary_accept() {
  local expect="$1"
  local status="$2"
  local present_keys="$3"
  if [ "$expect" = "statusCode" ]; then
    canary_status_ok "$status"
    return $?
  fi
  local k
  for k in $present_keys; do
    if [ "$k" = "$expect" ]; then
      return 0
    fi
  done
  return 1
}

# run_canary_action <function_name> <version> <action> <payload> <expect>
#   Invoke a single read-only action via krepis.aws invoke-canary and gate on
#   its contract. <expect> selects the acceptance mode (see canary_accept):
#   "statusCode" for the HTTP-shaped predict path, or the required domain key
#   for a Step-Function gate action.
#   Returns: 0 if accepted; 1 if FunctionError or the contract is unmet.
#   Outputs error details to stdout on failure.
run_canary_action() {
  local func_name="$1"
  local version="$2"
  local action="$3"
  local payload="$4"
  local expect="$5"

  local canary_out
  canary_out=$(mktemp)
  local canary_meta
  canary_meta=$(python3 -m krepis.aws invoke-canary \
    --function-name "${func_name}:${version}" \
    --payload "$payload" \
    --out "$canary_out" \
    --region "${AWS_REGION}" \
    --max-attempts 6 \
    --label "${func_name}-${action}-canary") || canary_meta=""

  local func_err
  local status
  local err_msg
  local present_keys
  func_err=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('FunctionError',''))" "$canary_meta" 2>/dev/null || echo "")
  status=$(python3 -c "import json; d=json.load(open('$canary_out')); print(d.get('statusCode', 0) if isinstance(d, dict) else 0)" 2>/dev/null || echo "0")
  err_msg=$(python3 -c "import json; d=json.load(open('$canary_out')); print((d.get('errorMessage','') or d.get('body','')) if isinstance(d, dict) else '')" 2>/dev/null || echo "")
  present_keys=$(python3 -c "import json; d=json.load(open('$canary_out')); print(' '.join(d.keys()) if isinstance(d, dict) else '')" 2>/dev/null || echo "")
  rm -f "$canary_out"

  if [ -n "$func_err" ] || ! canary_accept "$expect" "$status" "$present_keys"; then
    echo "  ${action}: FAILED"
    echo "    FunctionError: ${func_err:-<none>}"
    echo "    expect: ${expect}"
    echo "    statusCode: ${status}"
    echo "    keys: ${present_keys:-<none>}"
    echo "    payload: ${err_msg:-<empty>}"
    return 1
  fi

  if [ "$expect" = "statusCode" ]; then
    echo "  ${action}: OK (status=$status)"
  else
    echo "  ${action}: OK (${expect} present)"
  fi
  return 0
}

# When sourced for unit testing (tests/test_canary_status_allowlist.py sets
# CANARY_LIB_SOURCE_ONLY=1), stop here: expose the helper without running the
# deploy. Placed BEFORE flag parsing so the test can source the script without
# its positional args being misread as deploy flags. `return` is only valid in
# a sourced context, so guard it.
if [ "${CANARY_LIB_SOURCE_ONLY:-0}" = "1" ]; then
  return 0 2>/dev/null || exit 0
fi

# Parse flags
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Resolve AWS identity ─────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
if [ -z "${AWS_ACCOUNT_ID:-}" ] && [ "$DRY_RUN" = false ]; then
  AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION" 2>/dev/null) \
    || { echo "ERROR: Could not auto-detect AWS_ACCOUNT_ID. Set it manually or configure AWS CLI."; exit 1; }
  echo "Auto-detected AWS_ACCOUNT_ID: $AWS_ACCOUNT_ID"
fi

# Move to repo root (script may be called from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "Working directory: $REPO_ROOT"

# ── Stage proprietary config from alpha-engine-config ────────────────────────
# predictor.yaml is gitignored in this repo and lives in the private
# nousergon/alpha-engine-config repo alongside the configs for the other
# modules. Without staging it into the build context, the Dockerfile's
# `COPY config/ config/` only captures predictor.sample.yaml and the Lambda
# fails at import with FileNotFoundError — the silent 2026-04-13 regression.
# Falling back to the sample is explicitly rejected: shipping placeholder
# hyperparameters to production is worse than refusing to deploy.
#
# Local dev workflow is preserved: if config/predictor.yaml already exists
# (the dev has it in place in their laptop checkout), we use it as-is.
CONFIG_REPO_DIR="${CONFIG_REPO_DIR:-$(dirname "$REPO_ROOT")/alpha-engine-config}"
CONFIG_STAGED_FROM_REPO=0

if [ -f "config/predictor.yaml" ]; then
  echo "Using existing config/predictor.yaml (local dev workflow)"
else
  src="$CONFIG_REPO_DIR/predictor/predictor.yaml"
  if [ -f "$src" ]; then
    echo "Staging config/predictor.yaml from $src"
    cp "$src" config/predictor.yaml
    CONFIG_STAGED_FROM_REPO=1
  else
    echo "ERROR: config/predictor.yaml not found — tried:"
    echo "  config/predictor.yaml (local dev)"
    echo "  $src (config repo sibling)"
    echo "Hint: clone nousergon/alpha-engine-config as a sibling directory,"
    echo "      or set CONFIG_REPO_DIR=/path/to/alpha-engine-config"
    exit 1
  fi
fi

# ── Stage alpha-engine-lib into vendor/ ──────────────────────────────────────
# alpha-engine-lib is installed inside the Dockerfile via pip from public
# git+https (lib was flipped public 2026-05-03). No vendor staging needed.

# Cleanup staged artifacts on exit so a failed deploy doesn't leave stray
# files in a dev laptop checkout.
cleanup_staged_artifacts() {
  if [ "$CONFIG_STAGED_FROM_REPO" = "1" ] && [ -f config/predictor.yaml ]; then
    rm -f config/predictor.yaml
  fi
}
trap cleanup_staged_artifacts EXIT

# ── Step 1: Build Docker image ────────────────────────────────────────────────
# Stamp the source commit SHA into the image for the PredictorPreflight
# deploy-drift check. CI passes $GITHUB_SHA; local dev falls back to HEAD.
GIT_SHA="${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
echo "  Stamping image with GIT_SHA=${GIT_SHA}"

echo ""
echo "==> Building Docker image..."
docker build \
  --platform linux/amd64 \
  --provenance=false \
  --build-arg "GIT_SHA=${GIT_SHA}" \
  --tag "${ECR_REPO}:${IMAGE_TAG}" \
  --file Dockerfile \
  .

echo "  Image built: ${ECR_REPO}:${IMAGE_TAG}"

if [ "$DRY_RUN" = true ]; then
  echo ""
  echo "==> DRY RUN: Skipping ECR push and Lambda update."
  echo "    Image built successfully as ${ECR_REPO}:${IMAGE_TAG}"
  exit 0
fi

# ── Step 2: Authenticate to ECR ───────────────────────────────────────────────
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

echo ""
echo "==> Authenticating to ECR (${ECR_REGISTRY})..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# ── Step 3: Tag and push image ────────────────────────────────────────────────
echo ""
echo "==> Tagging image: ${ECR_IMAGE}"
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_IMAGE}"

echo "==> Pushing to ECR (this may take a few minutes for first push)..."
docker push "${ECR_IMAGE}"
echo "  Pushed: ${ECR_IMAGE}"

# ── Step 4: Update Lambda function code ──────────────────────────────────────
echo ""
echo "==> Updating Lambda function: ${LAMBDA_FUNCTION}"
aws lambda update-function-code \
  --function-name "${LAMBDA_FUNCTION}" \
  --image-uri "${ECR_IMAGE}" \
  --region "${AWS_REGION}" \
  --output json \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('  FunctionArn:', d.get('FunctionArn','?')); print('  LastModified:', d.get('LastModified','?'))"

# ── Step 5: Wait for update to complete ──────────────────────────────────────
echo ""
echo "==> Waiting for Lambda update to complete..."
aws lambda wait function-updated \
  --function-name "${LAMBDA_FUNCTION}" \
  --region "${AWS_REGION}"

# ── Step 6: Publish version ──────────────────────────────────────────────────
echo ""
echo "==> Publishing Lambda version..."
VERSION=$(aws lambda publish-version \
  --function-name "${LAMBDA_FUNCTION}" \
  --query "Version" --output text \
  --region "${AWS_REGION}")
echo "  Published version: ${VERSION}"

# ── Step 7: Canary against the new version (NOT live) ────────────────────────
# Invoke the version directly so a broken image cannot reach the live alias.
# If the canary fails, live keeps pointing at the prior good version.
#
# Test multiple read-only handler actions to ensure all wiring is intact:
#   - predict (action omitted / default) + dry_run
#   - check_drift (monitoring gate)
#   - check_trading_day (calendar gate)
#   - check_weekly_run_day (calendar gate)
#   - check_pipeline_contract (early validation gate)
#
# Each action's wiring is exercised independently. A broken import or a
# missing handler will surface as FunctionError or statusCode != 200.
#
# Invoke via the shared ``krepis.aws invoke-canary`` CLI (config#1494, published
# in krepis 0.7.0) instead of a bare ``aws lambda invoke``. The CLI retries ONLY
# on the throttle/concurrency signal with bounded backoff+jitter and writes two
# streams, exactly as ``aws lambda invoke`` did:
#   - the response *payload* to --out
#   - the invoke API *metadata* JSON (StatusCode / FunctionError / ExecutedVersion)
#     to stdout
# Unhandled Lambda exceptions set FunctionError="Unhandled" on the *metadata*
# stream — NOT on the payload. The 2026-05-11 v138 ImportError surfaced as
# `statusCode=0, FunctionError=` (empty) because the prior implementation
# parsed FunctionError from the payload file and discarded stdout. boto3 path —
# no ``--cli-binary-format``/base64. A non-zero CLI exit (non-throttle boto
# error or throttle exhaustion) leaves $CANARY_META empty, so the promote gate
# below refuses to promote — live stays on the prior good version.
echo ""
echo "==> Running canary invocation matrix against :${VERSION}..."
CANARY_FAILED=0

# Test predict with dry_run — HTTP-shaped handler, gate on statusCode.
if ! run_canary_action "${LAMBDA_FUNCTION}" "${VERSION}" "predict(dry_run)" '{"dry_run": true}' "statusCode"; then
  CANARY_FAILED=1
fi

# Step-Function gate actions below return raw domain dicts (no statusCode);
# gate each on the presence of the domain key its SF Choice state consumes.

# Test check_drift — drift_detector.check_drift returns {date,status,...}.
if ! run_canary_action "${LAMBDA_FUNCTION}" "${VERSION}" "check_drift" '{"action": "check_drift"}' "status"; then
  CANARY_FAILED=1
fi

# Test check_trading_day — trading_day_gate returns {is_trading_day,...}.
if ! run_canary_action "${LAMBDA_FUNCTION}" "${VERSION}" "check_trading_day" '{"action": "check_trading_day"}' "is_trading_day"; then
  CANARY_FAILED=1
fi

# Test check_weekly_run_day — trading_day_gate returns {is_weekly_run_day,...}.
if ! run_canary_action "${LAMBDA_FUNCTION}" "${VERSION}" "check_weekly_run_day" '{"action": "check_weekly_run_day"}' "is_weekly_run_day"; then
  CANARY_FAILED=1
fi

# Test check_pipeline_contract — returns {has_violation,...}.
if ! run_canary_action "${LAMBDA_FUNCTION}" "${VERSION}" "check_pipeline_contract" '{"action": "check_pipeline_contract"}' "has_violation"; then
  CANARY_FAILED=1
fi

if [ "$CANARY_FAILED" = "1" ]; then
  echo ""
  echo "ERROR: One or more canary actions failed — refusing to promote :${VERSION} to live."
  echo "       Live alias is unchanged. Investigate logs for function:${LAMBDA_FUNCTION} version ${VERSION}."
  # ROADMAP L221 — independent-channel surveillance per the alpha-engine-data
  # #274 retrospective. ``dedup_key`` collapses an image-wide rebuild that
  # breaks N Lambdas' canaries within the hour into one alert per
  # (Lambda, version) — lib v0.24.0 substrate (L221 retrofit 2026-05-22).
  # Best-effort; trailing || true never overrides exit 1.
  python3 -m krepis.alerts publish \
    --severity error \
    --source "alpha-engine-predictor/infrastructure/deploy.sh" \
    --dedup-key "canary-fail-${LAMBDA_FUNCTION}-v${VERSION}" \
    --message "Canary matrix failed — refused to promote ${LAMBDA_FUNCTION}:${VERSION} to live. See deployment logs for individual action failures. Live alias unchanged." \
    || true
  exit 1
fi
echo ""
echo "  Canary matrix passed — all read-only actions functional"

# ── Step 8: Promote version to 'live' (only after canary passes) ─────────────
echo "==> Updating 'live' alias → version ${VERSION}"
aws lambda update-alias \
  --function-name "${LAMBDA_FUNCTION}" \
  --name live \
  --function-version "${VERSION}" \
  --region "${AWS_REGION}" 2>/dev/null || \
aws lambda create-alias \
  --function-name "${LAMBDA_FUNCTION}" \
  --name live \
  --function-version "${VERSION}" \
  --region "${AWS_REGION}"

echo ""
echo "==> Deploy complete (inference Lambda)"
echo "    Function : ${LAMBDA_FUNCTION}"
echo "    Version  : ${VERSION}"
echo "    Alias    : live → ${VERSION}"
echo "    Image    : ${ECR_IMAGE}"
echo ""

# ── Step 9: Update (or auto-create) regime substrate Lambda ───────────────────
# The regime substrate Lambda shares the predictor ECR image but is invoked
# with regime.handler.lambda_handler as the CMD override (set at Lambda
# creation time via --image-config Command=[...], persists across
# update-function-code calls).
#
# Auto-create on first deploy when not yet present (deploy IAM was extended
# 2026-05-14 with CreateFunction + CreateAlias + PassRole). Break-glass
# operator step setup-regime-lambda.sh kept for manual provisioning.
LAMBDA_EXECUTION_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${LAMBDA_EXECUTION_ROLE_NAME}"

echo "==> Checking for regime substrate Lambda: ${REGIME_LAMBDA_FUNCTION}"
if ! aws lambda get-function \
       --function-name "${REGIME_LAMBDA_FUNCTION}" \
       --region "${AWS_REGION}" \
       --query "Configuration.FunctionName" \
       --output text >/dev/null 2>&1; then
  echo "  NOT FOUND — auto-creating with CMD=${REGIME_LAMBDA_CMD}..."
  aws lambda create-function \
    --function-name "${REGIME_LAMBDA_FUNCTION}" \
    --package-type Image \
    --code "ImageUri=${ECR_IMAGE}" \
    --role "${LAMBDA_EXECUTION_ROLE_ARN}" \
    --image-config "Command=[\"${REGIME_LAMBDA_CMD}\"]" \
    --memory-size "${REGIME_LAMBDA_MEMORY}" \
    --timeout "${REGIME_LAMBDA_TIMEOUT}" \
    --region "${AWS_REGION}" \
    --output json \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('  FunctionArn:', d.get('FunctionArn','?')); print('  State:', d.get('State','?'))"
  aws lambda wait function-active --function-name "${REGIME_LAMBDA_FUNCTION}" --region "${AWS_REGION}"
fi

# Now the function exists either way — fall through to update + canary +
# promote (idempotent on a freshly-created function: update-function-code
# just no-ops if the image-uri already matches).
echo "  Found (or freshly created) — updating..."

  aws lambda update-function-code \
    --function-name "${REGIME_LAMBDA_FUNCTION}" \
    --image-uri "${ECR_IMAGE}" \
    --region "${AWS_REGION}" \
    --output json \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('  FunctionArn:', d.get('FunctionArn','?')); print('  LastModified:', d.get('LastModified','?'))"

  aws lambda wait function-updated \
    --function-name "${REGIME_LAMBDA_FUNCTION}" \
    --region "${AWS_REGION}"

  REGIME_VERSION=$(aws lambda publish-version \
    --function-name "${REGIME_LAMBDA_FUNCTION}" \
    --query "Version" --output text \
    --region "${AWS_REGION}")
  echo "  Published regime version: ${REGIME_VERSION}"

  # Canary via the shared krepis.aws invoke-canary CLI (config#1494); metadata
  # JSON (incl. FunctionError) on stdout, payload to --out. Non-zero CLI exit
  # leaves $REGIME_CANARY_META empty → the gate below refuses to promote.
  # Test both handler actions (produce and dry_run) to ensure regime substrate
  # wiring is complete before promoting to live.
  echo "==> Running regime canary matrix against :${REGIME_VERSION}..."
  REGIME_CANARY_FAILED=0

  if ! run_canary_action "${REGIME_LAMBDA_FUNCTION}" "${REGIME_VERSION}" "dry_run" '{"action": "dry_run"}'; then
    REGIME_CANARY_FAILED=1
  fi

  if ! run_canary_action "${REGIME_LAMBDA_FUNCTION}" "${REGIME_VERSION}" "produce(dry_run_equivalent)" '{"action": "produce"}'; then
    REGIME_CANARY_FAILED=1
  fi

  if [ "$REGIME_CANARY_FAILED" = "1" ]; then
    echo ""
    echo "ERROR: Regime canary matrix failed — refusing to promote :${REGIME_VERSION} to live."
    echo "       Inference Lambda was already promoted; rollback regime separately if needed."
    # ROADMAP L221 — independent-channel surveillance. ``dedup_key``
    # collapses an image-wide rebuild that breaks N Lambdas' canaries
    # within the hour into one alert per (Lambda, version). Inference
    # Lambda is already live at this point so the alert MUST surface:
    # the operator may need to revert inference too if the regime
    # canary failure indicates an upstream image issue.
    python3 -m krepis.alerts publish \
      --severity error \
      --source "alpha-engine-predictor/infrastructure/deploy.sh" \
      --dedup-key "canary-fail-${REGIME_LAMBDA_FUNCTION}-v${REGIME_VERSION}" \
      --message "Regime canary matrix failed — refused to promote ${REGIME_LAMBDA_FUNCTION}:${REGIME_VERSION} to live. See deployment logs for action details. NOTE: inference Lambda was already promoted to :${VERSION} — operator may need to rollback inference too if this is an image-wide issue." \
      || true
    exit 1
  fi
  echo "  Regime canary matrix passed — all actions functional"

  aws lambda update-alias \
    --function-name "${REGIME_LAMBDA_FUNCTION}" \
    --name live \
    --function-version "${REGIME_VERSION}" \
    --region "${AWS_REGION}" 2>/dev/null || \
  aws lambda create-alias \
    --function-name "${REGIME_LAMBDA_FUNCTION}" \
    --name live \
    --function-version "${REGIME_VERSION}" \
    --region "${AWS_REGION}"

  echo ""
  echo "==> Deploy complete (regime Lambda)"
  echo "    Function : ${REGIME_LAMBDA_FUNCTION}"
  echo "    Version  : ${REGIME_VERSION}"
  echo "    Alias    : live → ${REGIME_VERSION}"
  echo "    Image    : ${ECR_IMAGE}"

echo ""

# ── Step 10: Update (or auto-create) regime retrospective eval Lambda ─────────
# Same shared-image pattern as Step 9 — third Lambda function backed by the
# same ECR image with regime.retrospective_eval_handler.lambda_handler as
# the CMD override. Auto-create on first deploy when not yet present.
echo "==> Checking for regime retrospective eval Lambda: ${REGIME_EVAL_LAMBDA_FUNCTION}"
if ! aws lambda get-function \
       --function-name "${REGIME_EVAL_LAMBDA_FUNCTION}" \
       --region "${AWS_REGION}" \
       --query "Configuration.FunctionName" \
       --output text >/dev/null 2>&1; then
  echo "  NOT FOUND — auto-creating with CMD=${REGIME_EVAL_LAMBDA_CMD}..."
  aws lambda create-function \
    --function-name "${REGIME_EVAL_LAMBDA_FUNCTION}" \
    --package-type Image \
    --code "ImageUri=${ECR_IMAGE}" \
    --role "${LAMBDA_EXECUTION_ROLE_ARN}" \
    --image-config "Command=[\"${REGIME_EVAL_LAMBDA_CMD}\"]" \
    --memory-size "${REGIME_EVAL_LAMBDA_MEMORY}" \
    --timeout "${REGIME_EVAL_LAMBDA_TIMEOUT}" \
    --region "${AWS_REGION}" \
    --output json \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('  FunctionArn:', d.get('FunctionArn','?')); print('  State:', d.get('State','?'))"
  aws lambda wait function-active --function-name "${REGIME_EVAL_LAMBDA_FUNCTION}" --region "${AWS_REGION}"
fi

echo "  Found (or freshly created) — updating..."

  aws lambda update-function-code \
    --function-name "${REGIME_EVAL_LAMBDA_FUNCTION}" \
    --image-uri "${ECR_IMAGE}" \
    --region "${AWS_REGION}" \
    --output json \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('  FunctionArn:', d.get('FunctionArn','?')); print('  LastModified:', d.get('LastModified','?'))"

  aws lambda wait function-updated \
    --function-name "${REGIME_EVAL_LAMBDA_FUNCTION}" \
    --region "${AWS_REGION}"

  REGIME_EVAL_VERSION=$(aws lambda publish-version \
    --function-name "${REGIME_EVAL_LAMBDA_FUNCTION}" \
    --query "Version" --output text \
    --region "${AWS_REGION}")
  echo "  Published regime-eval version: ${REGIME_EVAL_VERSION}"

  # Canary via the shared krepis.aws invoke-canary CLI (config#1494); metadata
  # JSON (incl. FunctionError) on stdout, payload to --out. Non-zero CLI exit
  # leaves $REGIME_EVAL_CANARY_META empty → the gate below refuses to promote.
  # Test both handler actions (produce and dry_run) to ensure regime-eval wiring
  # is complete before promoting to live.
  echo "==> Running regime-eval canary matrix against :${REGIME_EVAL_VERSION}..."
  REGIME_EVAL_CANARY_FAILED=0

  if ! run_canary_action "${REGIME_EVAL_LAMBDA_FUNCTION}" "${REGIME_EVAL_VERSION}" "dry_run" '{"action": "dry_run"}'; then
    REGIME_EVAL_CANARY_FAILED=1
  fi

  if ! run_canary_action "${REGIME_EVAL_LAMBDA_FUNCTION}" "${REGIME_EVAL_VERSION}" "produce(dry_run_equivalent)" '{"action": "produce"}'; then
    REGIME_EVAL_CANARY_FAILED=1
  fi

  if [ "$REGIME_EVAL_CANARY_FAILED" = "1" ]; then
    echo ""
    echo "ERROR: Regime-eval canary matrix failed — refusing to promote :${REGIME_EVAL_VERSION} to live."
    echo "       Inference + substrate Lambdas were already promoted; rollback regime-eval separately if needed."
    # ROADMAP L221 — independent-channel surveillance. ``dedup_key``
    # collapses an image-wide rebuild that breaks N Lambdas' canaries
    # within the hour into one alert per (Lambda, version). Two upstream
    # Lambdas already promoted; the alert is load-bearing for operator
    # triage.
    python3 -m krepis.alerts publish \
      --severity error \
      --source "alpha-engine-predictor/infrastructure/deploy.sh" \
      --dedup-key "canary-fail-${REGIME_EVAL_LAMBDA_FUNCTION}-v${REGIME_EVAL_VERSION}" \
      --message "Regime-eval canary matrix failed — refused to promote ${REGIME_EVAL_LAMBDA_FUNCTION}:${REGIME_EVAL_VERSION} to live. See deployment logs for action details. NOTE: inference (${LAMBDA_FUNCTION}:${VERSION}) + regime (${REGIME_LAMBDA_FUNCTION}:${REGIME_VERSION}) were already promoted to live — operator may need to rollback all three if this is an image-wide issue." \
      || true
    exit 1
  fi
  echo "  Regime-eval canary matrix passed — all actions functional"

  aws lambda update-alias \
    --function-name "${REGIME_EVAL_LAMBDA_FUNCTION}" \
    --name live \
    --function-version "${REGIME_EVAL_VERSION}" \
    --region "${AWS_REGION}" 2>/dev/null || \
  aws lambda create-alias \
    --function-name "${REGIME_EVAL_LAMBDA_FUNCTION}" \
    --name live \
    --function-version "${REGIME_EVAL_VERSION}" \
    --region "${AWS_REGION}"

  echo ""
  echo "==> Deploy complete (regime-eval Lambda)"
  echo "    Function : ${REGIME_EVAL_LAMBDA_FUNCTION}"
  echo "    Version  : ${REGIME_EVAL_VERSION}"
  echo "    Alias    : live → ${REGIME_EVAL_VERSION}"
  echo "    Image    : ${ECR_IMAGE}"

echo ""
echo "Rollback: bash infrastructure/rollback.sh"
