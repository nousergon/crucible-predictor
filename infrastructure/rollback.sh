#!/bin/bash
# rollback.sh — Shift the 'live' alias back to the previous Lambda version.
#
# Usage: bash infrastructure/rollback.sh
set -euo pipefail

LAMBDA_FUNCTION="alpha-engine-predictor-inference"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Resolve the version the 'live' alias currently points at via
# ``get-function --qualifier live`` (Configuration.Version), NOT
# ``get-alias``: the github-actions-lambda-deploy role is granted
# lambda:GetFunction (deploy.sh's live-alias verify step already relies on
# it) but NOT lambda:GetAlias — so the get-alias form crashes the rollback
# with AccessDeniedException, leaving a canary-failed version LIVE and
# un-rolled-back (config#3205; mirrors crucible-research's 2026-07-21
# incident fix, PR #481). GetFunction with an alias qualifier returns that
# alias's resolved numeric version identically.
CURRENT=$(aws lambda get-function \
    --function-name "$LAMBDA_FUNCTION" \
    --qualifier live \
    --query "Configuration.Version" --output text \
    --region "$AWS_REGION")

if [ "$CURRENT" -le 1 ]; then
    echo "Cannot rollback: current version is $CURRENT (no prior version)"
    exit 1
fi

PREV=$((CURRENT - 1))

aws lambda update-alias \
    --function-name "$LAMBDA_FUNCTION" \
    --name live \
    --function-version "$PREV" \
    --region "$AWS_REGION" > /dev/null

echo "Rolled back: live → version $PREV (was $CURRENT)"
echo "To verify: aws lambda get-alias --function-name $LAMBDA_FUNCTION --name live --query FunctionVersion --output text --region $AWS_REGION"
