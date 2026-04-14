# =============================================================================
# Denver Sprinkler Inference Server — Multi-stage Docker Build
# Stage 1: Build bitnet.cpp / llama-cli from source
# Stage 2: Slim Python runtime with server code
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — Builder: compile llama-cli from BitNet repo
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ git make python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone BitNet at pinned revision (v0.1, 2025-03-07)
RUN git clone https://github.com/microsoft/BitNet.git /bitnet \
    && cd /bitnet \
    && git checkout 58c005d97fafab4cdce7844649e85e1e08f81842

# Build llama-cli
RUN cd /bitnet \
    && mkdir -p build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && cmake --build . --config Release -j"$(nproc)"

# ---------------------------------------------------------------------------
# Stage 2 — Runtime: Python + llama-cli + server code
# ---------------------------------------------------------------------------
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled llama-cli binary from builder
COPY --from=builder /bitnet/build/bin/llama-cli /usr/local/bin/llama-cli
RUN chmod +x /usr/local/bin/llama-cli

# Python dependencies
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Server source code
COPY server/ ./server/

# Make start script executable
COPY server/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Defaults — override at runtime via env vars
ENV MODEL_PATH=/models/ggml-model-i2_s.gguf
ENV LLAMA_CLI_PATH=/usr/local/bin/llama-cli
ENV S3_BUCKET_NAME=""
ENV S3_MODEL_PREFIX="models/denver-sprinkler-3b-1bit"

EXPOSE 8000

ENTRYPOINT ["/app/start.sh"]
