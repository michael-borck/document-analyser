"""
Health check endpoints
"""

import time
from importlib.metadata import version

from fastapi import APIRouter

from document_analyser.models.schemas import HealthResponse

router = APIRouter()

# Application start time for uptime calculation
START_TIME = time.time()

# Version sourced from pyproject.toml at install time so we don't drift
# between code-declared and packaged versions.
_VERSION = version("document-analyser")


def _embedding_model_state() -> tuple[bool, str | None]:
    """Report whether the shared embedding model loaded.

    Imported lazily so /health never fails if the semantic stack (torch /
    sentence-transformers) is unavailable — that's precisely the state we
    want to report, not crash on.
    """
    try:
        from document_analyser.api.routes.semantic_analysis import domain_mapper
    except Exception as e:  # noqa: BLE001 - import-time failure is itself the signal
        return False, f"semantic stack unavailable: {e}"
    return domain_mapper.model is not None, domain_mapper._load_error


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    uptime = time.time() - START_TIME
    model_loaded, model_error = _embedding_model_state()

    return HealthResponse(
        status="ok",
        version=_VERSION,
        uptime=uptime,
        embedding_model_loaded=model_loaded,
        embedding_model_error=model_error,
    )
