#!/usr/bin/env bash
# =============================================================================
# Build bitnet.cpp / llama-cli from source (non-Docker environments)
#
# Usage:
#   ./scripts/build-bitnet.sh [--output-dir /usr/local/bin]
#
# Requires: cmake, g++, git, python3
# =============================================================================
set -euo pipefail

BITNET_REPO="https://github.com/microsoft/BitNet.git"
BITNET_REVISION="58c005d97fafab4cdce7844649e85e1e08f81842"  # v0.1 (2025-03-07)
CACHE_DIR=".bitnet-cache"
OUTPUT_DIR="/usr/local/bin"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "=== Building bitnet.cpp ==="
echo "  Repo:     $BITNET_REPO"
echo "  Revision: $BITNET_REVISION"
echo "  Cache:    $CACHE_DIR"
echo "  Output:   $OUTPUT_DIR"

# Clone or update the repo
if [ -d "$CACHE_DIR/BitNet" ]; then
    echo "Using cached BitNet repo at $CACHE_DIR/BitNet"
    cd "$CACHE_DIR/BitNet"
    git fetch origin
    git checkout "$BITNET_REVISION"
    cd -
else
    mkdir -p "$CACHE_DIR"
    git clone "$BITNET_REPO" "$CACHE_DIR/BitNet"
    cd "$CACHE_DIR/BitNet"
    git checkout "$BITNET_REVISION"
    cd -
fi

# Build
echo "Running cmake ..."
cd "$CACHE_DIR/BitNet"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j"$(nproc)"
cd -

# Copy binary
BINARY="$CACHE_DIR/BitNet/build/bin/llama-cli"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: llama-cli binary not found at $BINARY"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cp "$BINARY" "$OUTPUT_DIR/llama-cli"
chmod +x "$OUTPUT_DIR/llama-cli"

echo "=== Done: llama-cli installed to $OUTPUT_DIR/llama-cli ==="
