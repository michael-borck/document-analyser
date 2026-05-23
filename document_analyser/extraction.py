"""Canonical text extraction for the lens family.

The single public home for "binary document -> plain text": PDF via pdfplumber,
plain-text formats read directly, everything else via markitdown. Other analysers
(e.g. conversation-analyser) import `extract_text` here rather than
re-implementing extraction. Heavy parsers are imported lazily so importing this
module stays cheap.

The richer async `services.document_processor.DocumentProcessor` (page-level
granularity + metadata inference) remains the web app's extractor; this module is
the simple, importable text-only path.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text", ""}


def extract_text(path: "str | Path") -> str:
    """Extract plain text from a document file (PDF, DOCX, PPTX, TXT, MD, ...)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages).strip()
    if suffix in _TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    from markitdown import MarkItDown

    return MarkItDown().convert(str(path)).text_content.strip()


def extract_text_from_bytes(content: bytes, suffix: str) -> str:
    """Extract text from in-memory bytes; `suffix` (e.g. '.pdf') selects the format."""
    suffix = (suffix or "").lower()
    if suffix == ".pdf":
        import io

        import pdfplumber

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages).strip()
    with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(content)
    try:
        from markitdown import MarkItDown

        return MarkItDown().convert(tmp_path).text_content.strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
