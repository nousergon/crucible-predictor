# Dockerfile — Lambda container image for alpha-engine-predictor.
#
# LightGBM + CatBoost inference with Platt calibration.
# Training runs on EC2 spot (spot_train.sh); Lambda is inference-only.
# Container image is ~500MB (LightGBM + CatBoost + scikit-learn).
#
# Build:
#   docker build --platform linux/amd64 -t alpha-engine-predictor .
#
# Run locally (simulates Lambda):
#   docker run -p 9000:8080 alpha-engine-predictor
#   curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
#        -d '{"dry_run": true}'
#
# The CMD points to inference.handler.handler, matching the Lambda handler
# configuration in infrastructure/deploy.sh.

FROM public.ecr.aws/lambda/python:3.12

# Install libgomp (OpenMP runtime required by LightGBM) and git (required by
# the alpha-engine-lib install below, which uses pip's git+https:// scheme).
RUN dnf install -y libgomp git && dnf clean all

# Bake the source commit SHA into the image so PredictorPreflight can detect
# deploy drift (deployed SHA vs origin/main HEAD). Passed by deploy.sh via
# `--build-arg GIT_SHA=<sha>` (CI uses $GITHUB_SHA; local dev defaults to
# `git rev-parse HEAD`). A file is chosen over an env var so the stamp
# travels with the image artifact itself — you can't have a "deployed image"
# with a different stamp than what was baked.
ARG GIT_SHA=unknown
RUN echo "${GIT_SHA}" > /var/task/GIT_SHA.txt

# Copy and install Python requirements first for better layer caching.
# alpha-engine-lib is pinned in requirements-lambda.txt (single source of
# truth for the Lambda image — keep in lockstep with the project-root
# requirements.txt). The standalone install line previously here drifted
# behind requirements.txt and shipped v0.2.4 to prod even after the
# project pin moved to v0.5.5; consolidating prevents a repeat (see
# `feedback_two_doc_sources_two_staleness_vectors`).
COPY requirements-lambda.txt .

RUN pip install --no-cache-dir -r requirements-lambda.txt && \
    rm -rf /root/.cache/pip

# Copy application code
COPY retry.py .
COPY data_manifest.py .
COPY config.py .
# ops_alerts.py — inference/stages/write_output.py's fail-loud S3-write
# path (config#2333) deferred-imports ops_alerts.publish_ops_alert to
# page on primary/secondary predictions-write failures. Same
# config#1282/PR352 bug class as the monitoring/ comment below: a
# first-party module reachable from the Lambda entrypoint's import
# closure but missing its COPY line ModuleNotFoundErrors at runtime,
# undetected by CI unless caught by test_dockerfile_import_closure.py.
COPY ops_alerts.py .
COPY config/ config/
COPY data/ data/
COPY model/ model/
COPY inference/ inference/
COPY training/ training/
COPY store/ store/
# Drift monitoring — inference/handler.py's check_drift action imports
# monitoring.drift_detector (config#1282, PR #305). Missing this line let
# the module ship in the repo but not the image: check_drift 500'd with
# ModuleNotFoundError in prod for every invocation since #305 merged,
# undetected because deploy.sh's canary only exercises dry_run=true.
COPY monitoring/ monitoring/
# Regime substrate (v3) — standalone submodule independent of model/.
# Same image serves both inference Lambda (CMD=inference.handler.handler)
# and the regime-substrate Lambda (per-function CMD override =
# regime.handler.lambda_handler), mirroring the shared-image pattern
# alpha-engine-research uses for eval-judge + rationale-clustering.
COPY regime/ regime/

# flow-doctor.yaml at LAMBDA_TASK_ROOT is loaded by setup_logging() at
# module-top of inference/handler.py. The path resolves via:
#   os.environ.get("LAMBDA_TASK_ROOT", os.path.dirname(os.path.dirname(...)))
# Mirrors alpha-engine-research / alpha-engine-data Dockerfiles.
# (flow-doctor-training.yaml is NOT shipped here — training runs on EC2
# spot, not Lambda; that yaml is read from the repo root via the local
# checkout that spot_train.sh sets up.)
COPY flow-doctor.yaml ./
COPY flow-doctor-regime.yaml ./

# Lambda handler entry point
CMD ["inference.handler.handler"]
