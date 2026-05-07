"""
DocumentLens FastAPI Service
Multi-Modal Document Analysis Microservice
"""

import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from document_analyser.api.routes import (
    academic_analysis,
    advanced_text,
    analyse,
    future_endpoints,
    health,
    semantic_analysis,
    text_analysis,
)
from document_analyser.core.config import settings

limiter = Limiter(key_func=get_remote_address)

# Create FastAPI document_analyser
document_analyser = FastAPI(
    title="DocumentLens API",
    description="Australian Document Analysis Microservice - Transform any content into actionable insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

document_analyser.state.limiter = limiter
document_analyser.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# CORS middleware.
#
# Two profiles:
#   - desktop mode (DOCUMENT_LENS_MODE=desktop): embedded in the document-lens
#     Electron document_analyser. The backend only listens on 127.0.0.1, reachable only by
#     the user's own processes, so we use a permissive regex to allow the
#     Vite dev server (any localhost port), the packaged renderer's
#     file:// origin, and the null-origin fallback some Chromium versions
#     emit for file://. Credentials are off — no auth between renderer and
#     local backend.
#   - web mode (default): strict allowlist from ALLOWED_ORIGINS, credentials
#     enabled. Used by docker-compose, web deployments, and shared hosting.
if os.getenv("DOCUMENT_LENS_MODE") == "desktop":
    document_analyser.add_middleware(
        CORSMiddleware,
        allow_origin_regex=(
            r"^(https?://localhost(:\d+)?"
            r"|https?://127\.0\.0\.1(:\d+)?"
            r"|file://.*"
            r"|null)$"
        ),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    document_analyser.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

# Include routers - Clean Australian microservice URLs
document_analyser.include_router(health.router, tags=["health"])
document_analyser.include_router(analyse.router, tags=["analyse"])
document_analyser.include_router(text_analysis.router, tags=["text-analysis"])
document_analyser.include_router(academic_analysis.router, tags=["academic-analysis"])
document_analyser.include_router(future_endpoints.router, tags=["file-processing"])
document_analyser.include_router(advanced_text.router, tags=["advanced-text"])
document_analyser.include_router(semantic_analysis.router, prefix="/semantic", tags=["semantic-analysis"])


@document_analyser.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint"""
    return {
        "service": "DocumentLens",
        "description": "Multi-Modal Document Analysis Microservice",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "available": {
                "health": "/health",
                "text_analysis": "/text",
                "academic_analysis": "/academic",
                "file_processing": "/files",
                "advanced_text": "/advanced",
                "semantic_analysis": "/semantic",
            },
            "description": {
                "text_analysis": "Analyse raw text (JSON input)",
                "academic_analysis": "Academic analysis of raw text (JSON input)",
                "file_processing": "Upload and analyse files (form data)",
                "advanced_text": "N-grams, NER, and keyword search",
                "semantic_analysis": "Domain mapping, structural mismatch, sentiment analysis",
            },
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("document_analyser.main:document_analyser", host="0.0.0.0", port=8000, reload=settings.DEBUG)
