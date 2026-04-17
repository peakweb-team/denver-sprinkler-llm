"""FastAPI inference server for the Denver Sprinkler chat assistant."""

import logging
import re
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from server.config import (
    CORS_ORIGINS,
    MODEL_NAME,
    RATE_LIMIT,
    SERVER_VERSION,
)
from server.inference import InferenceEngine
from server.models import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per-IP)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Custom CORS origin checker to support wildcard subdomains
# ---------------------------------------------------------------------------
def _build_origin_patterns(origins: list[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for origin in origins:
        # Escape dots and convert wildcard * to regex
        escaped = re.escape(origin).replace(r"\*", r"[a-zA-Z0-9\-]+")
        patterns.append(re.compile(f"^{escaped}$"))
    return patterns


_origin_patterns = _build_origin_patterns(CORS_ORIGINS)


def _is_allowed_origin(origin: str) -> bool:
    return any(p.match(origin) for p in _origin_patterns)


class CORSWildcardMiddleware(BaseHTTPMiddleware):
    """CORS middleware that supports wildcard subdomain patterns."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        origin = request.headers.get("origin", "")

        if request.method == "OPTIONS":
            if _is_allowed_origin(origin):
                return JSONResponse(
                    content={},
                    headers={
                        "Access-Control-Allow-Origin": origin,
                        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type, Authorization",
                        "Access-Control-Max-Age": "3600",
                    },
                )
            return JSONResponse(content={}, status_code=403)

        response = await call_next(request)

        if origin and _is_allowed_origin(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"

        return response


# ---------------------------------------------------------------------------
# Application lifespan — initialise the inference engine once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine = InferenceEngine()
    app.state.engine = engine
    logger.info("Inference engine ready")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Denver Sprinkler Chat API",
    version=SERVER_VERSION,
    lifespan=lifespan,
)

# Attach rate-limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Attach CORS middleware
app.add_middleware(CORSWildcardMiddleware)


# ---------------------------------------------------------------------------
# Global exception handler — never leak stack traces
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMIT)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Generate a chat response from the Denver Sprinkler assistant."""
    engine: InferenceEngine = request.app.state.engine

    try:
        text = await engine.generate(body.messages)
    except TimeoutError:
        return JSONResponse(  # type: ignore[return-value]
            status_code=504,
            content={"detail": "Inference timed out. Please try again."},
        )
    except FileNotFoundError as exc:
        return JSONResponse(  # type: ignore[return-value]
            status_code=503,
            content={"detail": str(exc)},
        )
    except Exception:
        logger.exception("Inference failed")
        return JSONResponse(  # type: ignore[return-value]
            status_code=503,
            content={"detail": "Inference service unavailable. Please try again later."},
        )

    return ChatResponse(response=text)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return server health status."""
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        version=SERVER_VERSION,
    )
