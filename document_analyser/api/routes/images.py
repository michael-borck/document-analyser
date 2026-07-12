"""
Embedded-image extraction endpoint.

Extraction only (find images + locating metadata) — analysis of image
content is out of scope for document-analyser; callers pipe the returned
images to image-analyser or their own vision layer. See the desktop app's
ADR-0027 for the family boundary this encodes.
"""

import os

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from document_analyser.core.config import settings
from document_analyser.services.image_extractor import extract_images

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Formats extract_images knows how to walk. Other supported document types
# (txt, md) simply return an empty result client-side; we reject them here
# so a caller mistake is visible rather than silently empty.
_EXTRACTABLE_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class ExtractImagesRequest(BaseModel):
    """Request model for path-based image extraction (localhost only)."""

    file_path: str = Field(..., description="Absolute path to the document on disk")
    min_dimension: int = Field(
        32, ge=1, le=4096, description="Skip images whose width or height is below this"
    )
    max_images: int = Field(
        200, ge=1, le=1000, description="Maximum number of images to return"
    )


class ExtractedImage(BaseModel):
    page_number: int | None = Field(
        None, description="1-based page (PDF). None for flow formats like DOCX."
    )
    image_index: int
    name: str
    width: int
    height: int
    format: str
    hash_sha256: str
    thumbnail_base64: str
    thumbnail_mime: str
    image_base64: str
    image_mime: str


class ExtractImagesResponse(BaseModel):
    filename: str
    content_type: str
    total_images: int
    images: list[ExtractedImage]
    skipped: dict[str, int]


@router.post("/files/extract-images", response_model=ExtractImagesResponse)
@limiter.limit(settings.RATE_LIMIT)
async def extract_images_by_path(
    request: Request,
    payload: ExtractImagesRequest = Body(...),
) -> ExtractImagesResponse:
    """
    Extract embedded images from a document on disk (PDF or DOCX).

    Returns each distinct image (deduplicated by content hash — repeated
    per-page logos collapse to one) with its page number, dimensions, and
    two base64 renditions: a thumbnail for grids and a display rendition
    capped at 1600px. Open the original document for full resolution.

    **Security:** localhost-only, same as /files/upload-path.
    """
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(
            status_code=403,
            detail="This endpoint is only available for local connections",
        )

    file_path = payload.file_path
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except PermissionError as e:
        raise HTTPException(
            status_code=400, detail=f"Permission denied reading: {file_path}"
        ) from e

    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max: {settings.MAX_FILE_SIZE} bytes",
        )

    filename = os.path.basename(file_path)
    from document_analyser.api.routes.future_endpoints import _detect_content_type_from_bytes

    content_type = _detect_content_type_from_bytes(content, filename, None)
    if content_type not in _EXTRACTABLE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Image extraction supports PDF and DOCX, got: {content_type}",
        )

    try:
        result = extract_images(
            content,
            content_type,
            min_dimension=payload.min_dimension,
            max_images=payload.max_images,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image extraction failed: {e!s}") from e

    return ExtractImagesResponse(
        filename=filename,
        content_type=content_type,
        **result,
    )
