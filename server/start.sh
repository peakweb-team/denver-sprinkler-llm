#!/usr/bin/env bash
# =============================================================================
# Inference server entrypoint
# 1. Download model from S3 if not already present
# 2. Start uvicorn
# =============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/ggml-model-i2_s.gguf}"
S3_BUCKET_NAME="${S3_BUCKET_NAME:-}"
S3_MODEL_PREFIX="${S3_MODEL_PREFIX:-models/denver-sprinkler-3b-1bit}"
MOCK_MODE="${MOCK_MODE:-false}"

# ---------- Download model from S3 if missing ----------
if [ "$MOCK_MODE" = "true" ] || [ "$MOCK_MODE" = "1" ]; then
    echo "[start.sh] Mock mode enabled — skipping model download."
elif [ -f "$MODEL_PATH" ]; then
    echo "[start.sh] Model already present at $MODEL_PATH"
else
    if [ -z "$S3_BUCKET_NAME" ]; then
        echo "[start.sh] ERROR: MODEL_PATH ($MODEL_PATH) does not exist and S3_BUCKET_NAME is not set."
        echo "[start.sh] Set MOCK_MODE=true for testing without a model."
        exit 1
    fi

    MODEL_DIR="$(dirname "$MODEL_PATH")"
    MODEL_FILENAME="$(basename "$MODEL_PATH")"
    mkdir -p "$MODEL_DIR"

    S3_URI="s3://${S3_BUCKET_NAME}/${S3_MODEL_PREFIX}/${MODEL_FILENAME}"
    echo "[start.sh] Downloading model from $S3_URI ..."
    aws s3 cp "$S3_URI" "$MODEL_PATH"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "[start.sh] ERROR: Model download failed."
        exit 1
    fi
    echo "[start.sh] Model downloaded successfully."
fi

# ---------- Start server ----------
echo "[start.sh] Starting uvicorn on 0.0.0.0:8000 ..."
exec uvicorn server.main:app --host 0.0.0.0 --port 8000
