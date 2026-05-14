#!/usr/bin/env bash
# infrastructure/setup-regime-retrospective-eval-lambda.sh — One-time
# create of the regime retrospective eval Lambda (T1 of the three-tier
# eval framework per regime-v3-260514.md §5.3.3).
#
# Creates ``alpha-engine-predictor-regime-retrospective-eval`` from the
# existing predictor ECR image with a per-function CMD override pointing
# at ``regime.retrospective_eval_handler.lambda_handler``. Same image
# as the inference + substrate Lambdas — only the CMD differs
# (shared-image pattern).
#
# Idempotent: skips with a friendly message if the function already exists.
# After this runs, subsequent ``infrastructure/deploy.sh`` invocations
# will update this Lambda automatically via Step 10.
#
# Usage:
#   bash infrastructure/setup-regime-retrospective-eval-lambda.sh
#
# Prerequisites:
#   - AWS CLI configured
#   - IAM role ``alpha-engine-predictor-role`` already exists with log-group
#     access for this Lambda's group (per infrastructure/iam/apply.sh).
#   - ECR repo ``alpha-engine-predictor`` has at least one image tagged
#     ``:latest`` (the inference deploy has run at least once).
#   - The regime substrate Lambda's been running long enough that
#     ``signals/{date}/signals.json`` archive carries macro_regime calls
#     to pair against the retrospective labels — fresh deploys will
#     return n_pairings=0 until that archive accrues.
#
# Sizing:
#   Memory  : 2048 MB  (HMM smoother fit + S3 enumeration of ~104
#                       signals.json files; sized larger than substrate
#                       Lambda to absorb the agent-calls load)
#   Timeout : 600 sec  (smoother fit ~30s + S3 ListObjects + per-week
#                       GetObject batch + ample headroom)

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
ECR_REPO="alpha-engine-predictor"
FUNCTION_NAME="alpha-engine-predictor-regime-retrospective-eval"
ROLE_NAME="alpha-engine-predictor-role"
IMAGE_TAG="latest"
MEMORY_MB=2048
TIMEOUT_SEC=600
CMD_OVERRIDE='regime.retrospective_eval_handler.lambda_handler'

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION")}"

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"
ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"

# ── Idempotency check ─────────────────────────────────────────────────────────
if aws lambda get-function \
     --function-name "${FUNCTION_NAME}" \
     --region "${AWS_REGION}" \
     --query "Configuration.FunctionName" \
     --output text >/dev/null 2>&1; then
  echo "Function ${FUNCTION_NAME} already exists in region ${AWS_REGION} — nothing to do."
  echo "Run infrastructure/deploy.sh to update the image (Step 10 will auto-update this Lambda)."
  exit 0
fi

# ── Verify prerequisites ──────────────────────────────────────────────────────
echo "==> Verifying ECR image exists: ${ECR_IMAGE}"
aws ecr describe-images \
  --repository-name "${ECR_REPO}" \
  --image-ids imageTag="${IMAGE_TAG}" \
  --region "${AWS_REGION}" \
  --query "imageDetails[0].imageDigest" \
  --output text >/dev/null \
  || { echo "ERROR: No image at ${ECR_IMAGE}. Run infrastructure/deploy.sh first."; exit 1; }

echo "==> Verifying IAM role exists: ${ROLE_ARN}"
aws iam get-role --role-name "${ROLE_NAME}" --query "Role.RoleName" --output text >/dev/null \
  || { echo "ERROR: Role ${ROLE_NAME} does not exist. Run infrastructure/iam/apply.sh first."; exit 1; }

# ── Create the function ───────────────────────────────────────────────────────
echo "==> Creating Lambda function: ${FUNCTION_NAME}"
aws lambda create-function \
  --function-name "${FUNCTION_NAME}" \
  --package-type Image \
  --code "ImageUri=${ECR_IMAGE}" \
  --role "${ROLE_ARN}" \
  --image-config "Command=[\"${CMD_OVERRIDE}\"]" \
  --memory-size "${MEMORY_MB}" \
  --timeout "${TIMEOUT_SEC}" \
  --region "${AWS_REGION}" \
  --output json \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('  FunctionArn:', d.get('FunctionArn','?')); print('  State:', d.get('State','?'))"

echo "==> Waiting for function to reach Active state..."
aws lambda wait function-active --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}"

# ── Initial canary ────────────────────────────────────────────────────────────
echo "==> Running initial dry_run canary..."
CANARY_OUT=$(mktemp)
CANARY_META=$(aws lambda invoke \
  --function-name "${FUNCTION_NAME}" \
  --payload '{"action": "dry_run"}' \
  --cli-binary-format raw-in-base64-out \
  --cli-read-timeout 600 \
  --region "${AWS_REGION}" \
  "$CANARY_OUT")
CANARY_FUNC_ERR=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('FunctionError',''))" "$CANARY_META" 2>/dev/null || echo "")
CANARY_STATUS=$(python3 -c "import json; d=json.load(open('$CANARY_OUT')); print(d.get('statusCode', 0))" 2>/dev/null || echo "0")
rm -f "$CANARY_OUT"

if [ -n "$CANARY_FUNC_ERR" ] || [ "$CANARY_STATUS" != "200" ]; then
  echo ""
  echo "WARNING: Initial canary did not return statusCode=200."
  echo "         FunctionError : ${CANARY_FUNC_ERR:-<none>}"
  echo "         statusCode    : ${CANARY_STATUS}"
  echo ""
  echo "         This is expected on the very first deploy if the price-cache"
  echo "         parquets at s3://alpha-engine-research/predictor/price_cache/"
  echo "         are not yet fully populated, OR if signals/ archive has fewer"
  echo "         than lag_weeks of weekly agent calls. The function is still"
  echo "         created; subsequent deploys + Saturday SF runs will exercise"
  echo "         it on real data."
else
  echo "  Canary passed (status=$CANARY_STATUS)"
fi

# ── Create live alias ─────────────────────────────────────────────────────────
INITIAL_VERSION=$(aws lambda publish-version \
  --function-name "${FUNCTION_NAME}" \
  --query "Version" --output text \
  --region "${AWS_REGION}")
aws lambda create-alias \
  --function-name "${FUNCTION_NAME}" \
  --name live \
  --function-version "${INITIAL_VERSION}" \
  --region "${AWS_REGION}" >/dev/null

echo ""
echo "==> Setup complete!"
echo "    Function : ${FUNCTION_NAME}"
echo "    Image    : ${ECR_IMAGE}"
echo "    CMD      : ${CMD_OVERRIDE}"
echo "    Memory   : ${MEMORY_MB} MB"
echo "    Timeout  : ${TIMEOUT_SEC} sec"
echo "    Alias    : live → ${INITIAL_VERSION}"
echo ""
echo "Next steps:"
echo "  1. Wire a Saturday SF RegimeRetrospectiveEval state to invoke"
echo "     ${FUNCTION_NAME}:live (see alpha-engine-config PR)."
echo "  2. Subsequent predictor deploys via infrastructure/deploy.sh"
echo "     will auto-update this function (Step 10)."
