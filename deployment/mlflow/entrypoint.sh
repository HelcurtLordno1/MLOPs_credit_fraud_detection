#!/usr/bin/env bash
set -euo pipefail

BACKEND_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}"
ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-/mlflow/artifacts}"

mkdir -p "${ARTIFACT_ROOT}"

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "${BACKEND_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}"
