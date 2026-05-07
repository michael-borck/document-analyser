"""
Health check endpoints
"""

import time

from fastapi import APIRouter

from document_analyser.models.schemas import HealthResponse

router = APIRouter()

# Application start time for uptime calculation
START_TIME = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    uptime = time.time() - START_TIME

    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime=uptime
    )
