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


def _climate_framing_state() -> tuple[bool, str | None]:
    """Report whether the ClimateBERT framing models loaded (ADR-0039).

    Optional [nlp] extras — unavailable is a normal state, reported not
    raised (the app degrades to the deterministic framing rules).
    """
    try:
        from document_analyser.analyzers.climate_framing import climate_framing
    except Exception as e:  # noqa: BLE001 - import-time failure is itself the signal
        return False, f"climate framing stack unavailable: {e}"
    return climate_framing.available, climate_framing.load_error


def _suggestion_capability_state() -> tuple[str, str, str | None]:
    """(lens domain, loaded models, why this lens gets no suggestions).

    ADR-0043. The lens declares its own suggestion models; the health surface
    reports the correspondence, so a researcher can learn whose models proposed
    a value instead of inferring it from the absence of a warning.
    """
    try:
        from document_analyser.analyzers.lens_capability import capability_state, climate_capability
    except Exception as e:  # noqa: BLE001 - import-time failure is itself the signal
        return "", "", f"suggestion capability unavailable: {e}"
    cap = climate_capability()
    usable, reason = capability_state(cap)
    return cap.domain, (cap.model_summary if usable else ""), (None if usable else reason)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    uptime = time.time() - START_TIME
    model_loaded, model_error = _embedding_model_state()
    climate_loaded, climate_error = _climate_framing_state()

    lens_domain, models, refusal = _suggestion_capability_state()
    return HealthResponse(
        status="ok",
        version=_VERSION,
        uptime=uptime,
        embedding_model_loaded=model_loaded,
        embedding_model_error=model_error,
        climate_framing_loaded=climate_loaded,
        climate_framing_error=climate_error,
        lens_domain=lens_domain,
        suggestion_models=models,
        suggestion_refusal=refusal,
    )
