"""Family-pattern /analyse endpoint — mirrors CLI output for single-file analysis."""

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File

router = APIRouter()

_SUPPORTED = {".pdf", ".docx", ".pptx", ".txt", ".md", ".rst"}


@router.post("/analyse")
async def analyse(file: UploadFile = File(...)) -> dict[str, Any]:
    """Analyse a single document. Mirrors the CLI output format."""
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if not suffix:
        raise HTTPException(status_code=422, detail="Cannot determine file type — include an extension in the filename.")
    if suffix not in _SUPPORTED:
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {suffix}. Supported: {', '.join(sorted(_SUPPORTED))}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="File is empty.")

    try:
        text = _extract(content, suffix, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not extract text: {e}") from e

    from document_analyser.analyzers.readability import ReadabilityAnalyzer
    analysis = ReadabilityAnalyzer().analyze(text)

    response: dict[str, Any] = {
        "filename": filename,
        "format": suffix.lstrip("."),
        "file_size": len(content),
        "word_count": analysis.word_count,
        "sentence_count": analysis.sentence_count,
        "paragraph_count": analysis.paragraph_count,
        "readability": {
            "flesch_reading_ease": analysis.flesch_score,
            "flesch_kincaid_grade": analysis.flesch_kincaid_grade,
            "gunning_fog": analysis.gunning_fog,
            "smog_index": analysis.smog_index,
            "automated_readability_index": analysis.automated_readability_index,
        },
    }

    # Additive: .pptx gets a slide-design block. python-pptx reads the deck
    # structure (slides, titles, layouts, images, bullets) that markitdown's
    # text extraction discards.
    if suffix == ".pptx":
        try:
            from document_analyser.analyzers.slide_design import analyse_pptx
            response["slide_design"] = analyse_pptx(content).to_dict()
        except Exception as e:
            response["slide_design_error"] = str(e)

    return response


def _extract(content: bytes, suffix: str, filename: str) -> str:
    # Canonical extractor: the single home for binary -> text in the family.
    from document_analyser.extraction import extract_text_from_bytes
    return extract_text_from_bytes(content, suffix)
