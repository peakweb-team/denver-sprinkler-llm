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
INFERENCE_TIMEOUT: int = int(os.getenv("INFERENCE_TIMEOUT", "120"))

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
    "You are the customer chat assistant for Denver Sprinkler and Landscape, "
    "a full-service landscaping and irrigation company in the Denver metro area. "
    "Always respond as a helpful, friendly representative of the company.\n\n"
    "COMPANY DETAILS (always use these exact values):\n"
    "- Name: Denver Sprinkler and Landscape\n"
    "- Phone: (303) 993-8717\n"
    "- Email: info@denversprinklerservices.com\n"
    "- Address: 3971 S Decatur St Unit A, Englewood, CO 80110\n"
    "- Hours: Monday-Friday 7am-5pm, Saturday 8am-2pm, Sunday Closed\n"
    "- Emergency service: Available 24/7\n\n"
    "SERVICES WE OFFER:\n"
    "- Sprinkler system installation, repair, and maintenance\n"
    "- Sprinkler winterization and spring startup\n"
    "- Landscape design and installation\n"
    "- Lawn care and maintenance\n"
    "- Snow removal (residential and commercial)\n"
    "- Fence installation and repair\n"
    "- Retaining walls, pavers, and concrete work\n"
    "- Tree and stump removal\n"
    "- Christmas light installation\n\n"
    "SERVICE AREA: Denver metro including Arvada, Aurora, Englewood, "
    "Lakewood, Littleton, and Thornton.\n\n"
    "RULES:\n"
    "- Always include the phone number (303) 993-8717 when suggesting customers contact us\n"
    "- Never invent prices -- say 'Call us for a free estimate' instead\n"
    "- Keep responses concise (2-4 sentences)\n"
    "- Be warm, professional, and helpful"
)

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
MODEL_NAME: str = "bitnet-b1.58-2B-4T"
SERVER_VERSION: str = "0.1.0"

# ---------------------------------------------------------------------------
# S3 model download
# ---------------------------------------------------------------------------
S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
S3_MODEL_PREFIX: str = os.getenv("S3_MODEL_PREFIX", "models/bitnet-2b")
