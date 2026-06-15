"""Canonical document analysis: extract text -> readability (+ slide design for .pptx).

This is the single shared implementation behind both the CLI
(`document-analyser <file> --json`) and the :class:`DocumentAnalyser` facade, so
the Python API and the CLI/JSON output never drift apart.

document-analyser's *core* contribution to the family is the canonical text
extractor (:func:`document_analyser.extract_text`); this module assembles the
prose/readability signal bundle on top of it.
"""

from __future__ import annotations

from pathlib import Path

from .extraction import extract_text


def analyse_document(path: str | Path) -> dict:
    """Analyse a document file and return its signal bundle (the CLI's JSON shape)."""
    path = Path(path)
    suffix = path.suffix.lower()
    text = extract_text(path)

    from .analyzers.readability import ReadabilityAnalyzer

    analysis = ReadabilityAnalyzer().analyze(text)
    result: dict = {
        "format": suffix.lstrip("."),
        "file_path": str(path.resolve()),
        "file_size": path.stat().st_size,
        "word_count": analysis.word_count,
        "sentence_count": analysis.sentence_count,
        "paragraph_count": analysis.paragraph_count,
        "text": text,
        "readability": {
            "flesch_reading_ease": analysis.flesch_score,
            "flesch_kincaid_grade": analysis.flesch_kincaid_grade,
            "gunning_fog": analysis.gunning_fog,
            "smog_index": analysis.smog_index,
            "automated_readability_index": analysis.automated_readability_index,
        },
    }

    # Additive: .pptx gets a slide-design block on top of prose/readability.
    if suffix == ".pptx":
        try:
            from .analyzers.slide_design import analyse_pptx

            result["slide_design"] = analyse_pptx(path).to_dict()
        except Exception as e:  # noqa: BLE001 - surfaced as a soft field, never fatal
            result["slide_design_error"] = str(e)

    return result


class DocumentAnalyser:
    """Family-canonical facade — ``DocumentAnalyser().analyse(path)``.

    Returns the same dict the CLI emits as JSON. (document-analyser predates the
    class convention; its extractor stays the canonical
    :func:`document_analyser.extract_text`, and this facade layers the readability
    bundle on top.)
    """

    def analyse(self, path: str | Path) -> dict:
        return analyse_document(path)
