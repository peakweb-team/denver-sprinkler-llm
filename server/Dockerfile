# =============================================================================
# Denver Sprinkler Inference Server — Multi-stage Docker Build
# Stage 1: Build bitnet.cpp from source (with optimized 1.58-bit kernels)
# Stage 2: Slim Python runtime with server code
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — Builder: compile bitnet.cpp
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake clang g++ git make python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone BitNet repo with submodules
RUN git clone --recurse-submodules https://github.com/microsoft/BitNet.git /bitnet

# Fix const-correctness error in ggml-bitnet-mad.cpp
RUN sed -i 's/int8_t \* y_col = y/const int8_t * y_col = y/' /bitnet/src/ggml-bitnet-mad.cpp

# Install BitNet's patched gguf package
RUN pip3 install --break-system-packages /bitnet/3rdparty/llama.cpp/gguf-py

# Build with clang (required for BitNet's optimized kernels)
RUN cd /bitnet && python3 setup_env.py --hf-repo microsoft/BitNet-b1.58-2B-4T -q i2_s \
    || (cd /bitnet && sed -i 's/int8_t \* y_col = y/const int8_t * y_col = y/' src/ggml-bitnet-mad.cpp \
    && cmake --build build --config Release)

# ---------------------------------------------------------------------------
# Stage 2 — Runtime: Python + bitnet.cpp + server code
# ---------------------------------------------------------------------------
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled llama-cli binary from builder
COPY --from=builder /bitnet/build/bin/llama-cli /usr/local/bin/llama-cli
RUN chmod +x /usr/local/bin/llama-cli

# Copy all shared libraries from the build tree
RUN --mount=from=builder,source=/bitnet/build,target=/tmp/bitnet-build \
    find /tmp/bitnet-build -name '*.so*' -exec cp {} /usr/local/lib/ \; \
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

# Defaults
ENV MODEL_PATH=/models/ggml-model-i2_s.gguf
ENV LLAMA_CLI_PATH=/usr/local/bin/llama-cli
ENV S3_BUCKET_NAME=""
ENV S3_MODEL_PREFIX="models/bitnet-2b"

EXPOSE 8000

ENTRYPOINT ["/app/start.sh"]
