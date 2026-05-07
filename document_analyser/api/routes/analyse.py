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

    return {
        "filename": filename,
        "format": suffix.lstrip("."),
        "file_size": len(content),
        "word_count": analysis.word_count,
        "sentence_count": analysis.sentence_count,
        "paragraph_count": analysis.paragraph_count,
        "readability": {
            "flesch_reading_ease": analysis.flesch_reading_ease,
            "flesch_kincaid_grade": analysis.flesch_kincaid_grade,
            "gunning_fog": analysis.gunning_fog,
            "smog_index": analysis.smog_index,
            "automated_readability_index": analysis.automated_readability_index,
        },
    }


def _extract(content: bytes, suffix: str, filename: str) -> str:
    if suffix == ".pdf":
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages).strip()

    # markitdown needs a real file path
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        from markitdown import MarkItDown
        return MarkItDown().convert(tmp_path).text_content.strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
