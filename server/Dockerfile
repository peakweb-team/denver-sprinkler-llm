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

# Clone llama.cpp (official repo, supports GGUF inference)
RUN git clone https://github.com/ggml-org/llama.cpp.git /llama.cpp

# Build llama-cli with shared libs (needed for proper linking)
RUN cd /llama.cpp \
    && mkdir -p build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && cmake --build . --target llama-cli --config Release -j"$(nproc)"

# ---------------------------------------------------------------------------
# Stage 2 — Runtime: Python + llama-cli + server code
# ---------------------------------------------------------------------------
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled llama-cli binary from builder
COPY --from=builder /llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli
RUN chmod +x /usr/local/bin/llama-cli

# Copy all shared libraries from the build tree and update linker cache
RUN --mount=from=builder,source=/llama.cpp/build,target=/tmp/llama-build \
    find /tmp/llama-build -name '*.so*' -exec cp {} /usr/local/lib/ \; \
    && ldconfig

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
ENV MODEL_PATH=/models/model-Q4_K_M.gguf
ENV LLAMA_CLI_PATH=/usr/local/bin/llama-cli
ENV S3_BUCKET_NAME=""
ENV S3_MODEL_PREFIX="models/denver-sprinkler-3b-gguf"

EXPOSE 8000

ENTRYPOINT ["/app/start.sh"]
