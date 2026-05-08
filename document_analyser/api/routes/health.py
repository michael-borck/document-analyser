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


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    uptime = time.time() - START_TIME

    return HealthResponse(
        status="ok",
        version=_VERSION,
        uptime=uptime
    )
