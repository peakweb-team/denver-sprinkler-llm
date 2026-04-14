"""Server configuration — all tunables via environment variables."""

from __future__ import annotations

import json
import os
from typing import List

# ---------------------------------------------------------------------------
# Model / binary paths
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.getenv("MODEL_PATH", "/models/ggml-model-i2_s.gguf")
LLAMA_CLI_PATH: str = os.getenv("LLAMA_CLI_PATH", "/usr/local/bin/llama-cli")

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_default_origins = json.dumps([
    "https://denversprinklerservices.com",
    "https://www.denversprinklerservices.com",
    "https://*.vercel.app",
])
CORS_ORIGINS: List[str] = json.loads(os.getenv("CORS_ORIGINS", _default_origins))

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
RATE_LIMIT: str = os.getenv("RATE_LIMIT", "10/minute")

# ---------------------------------------------------------------------------
# Inference parameters
# ---------------------------------------------------------------------------
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "300"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
INFERENCE_TIMEOUT: int = int(os.getenv("INFERENCE_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# Mock mode — returns canned responses without llama-cli
# ---------------------------------------------------------------------------
MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Context limits
# ---------------------------------------------------------------------------
MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "8192"))

# ---------------------------------------------------------------------------
# System prompt — exact business details from CLAUDE.md / quantization.yaml
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = (
    "You are a helpful assistant for Denver Sprinkler and Landscape, "
    "a landscaping and sprinkler company in Englewood, Colorado. "
    "Phone: (303) 993-8717. Email: info@denversprinklerservices.com. "
    "Address: 3971 S Decatur St Unit A, Englewood, CO 80110. "
    "Hours: Mon-Fri 7am-5pm, Sat 8am-2pm, Sun Closed. "
    "Emergency service available 24/7."
)

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
MODEL_NAME: str = "denver-sprinkler-3b-1bit"
SERVER_VERSION: str = "0.1.0"

# ---------------------------------------------------------------------------
# S3 model download
# ---------------------------------------------------------------------------
S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
S3_MODEL_PREFIX: str = os.getenv("S3_MODEL_PREFIX", "models/denver-sprinkler-3b-1bit")
