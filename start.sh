#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env file
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  echo "Error: .env file not found. Copy .env.example to .env and fill in your credentials."
  exit 1
fi

docker run -it \
  --env-file "$SCRIPT_DIR/.env" \
  -v "$SCRIPT_DIR":/workspace \
  -w /workspace \
  -u vscode \
  denver-sprinkler-llm \
  bash -c 'cd /workspace && claude --dangerously-skip-permissions'