"""
DocumentAnalyser FastAPI Service
Multi-Modal Document Analysis Microservice (lens family contract via lens-contract).

The FastAPI instance lives here (the `api` package's __init__) rather than an
`api.py` module, because this package already owns the route subpackage
(`document_analyser.api.routes`). The launch string is still the family-standard
`document_analyser.api:app`.
"""

from typing import Any

from fastapi import FastAPI
from lens_contract import add_auth, add_contract_routes, add_cors
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from document_analyser.api.routes import (
    academic_analysis,
    advanced_text,
    analyse,
    future_endpoints,
    images,
    semantic_analysis,
    text_analysis,
)
from document_analyser.manifest import MANIFEST

# Per-route rate limiting (slowapi). document-analyser keeps its own per-route
# @limiter.limit decorators (finer-grained than a single service-wide limit), so it
# deliberately does NOT use lens_contract.add_rate_limit. Toggle via RATE_LIMIT_ENABLED.
limiter = Limiter(key_func=get_remote_address)

# MANIFEST["version"] is the installed package version (resolved by lens-contract),
# so the service version always matches the package — no manual sync.
app = FastAPI(
    title="DocumentAnalyser API",
    description="Document text and signal extraction (PDF, DOCX, PPTX, MD) for the analyser family",
    version=MANIFEST["version"],
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# GET /health and GET /manifest (the family contract, via lens-contract).
add_contract_routes(app, MANIFEST)
# Optional bearer-token auth — no-op unless DOCUMENT_ANALYSER_AUTH_TOKEN is set.
# The desktop host (document-lens) generates a per-launch token, passes it via
# this env var, and sends it as `Authorization: Bearer …`; /health and /manifest
# stay open. Called BEFORE add_cors so CORS remains the outermost middleware.
add_auth(app, env_prefix="DOCUMENT_ANALYSER")
# CORS — env-driven: DOCUMENT_ANALYSER_MODE=desktop (Electron) or DOCUMENT_ANALYSER_ALLOWED_ORIGINS.
add_cors(app, env_prefix="DOCUMENT_ANALYSER")

# Include routers - Clean Australian microservice URLs
app.include_router(analyse.router, tags=["analyse"])
app.include_router(text_analysis.router, tags=["text-analysis"])
app.include_router(academic_analysis.router, tags=["academic-analysis"])
app.include_router(future_endpoints.router, tags=["file-processing"])
app.include_router(images.router, tags=["file-processing"])
app.include_router(advanced_text.router, tags=["advanced-text"])
app.include_router(semantic_analysis.router, prefix="/semantic", tags=["semantic-analysis"])


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint"""
    return {
        "service": "DocumentAnalyser",
        "description": "Multi-Modal Document Analysis Microservice",
        "version": MANIFEST["version"],
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

    uvicorn.run("document_analyser.api:app", host="0.0.0.0", port=8000)
