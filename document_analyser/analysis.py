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

    # Readability is a soft signal layered on top of the canonical extraction.
    # It must never abort extraction — a missing NLTK resource (e.g. cmudict on
    # a packaged/offline build) or any textstat hiccup degrades to an empty
    # readability block, exactly like ai_tells / slide_design below.
    result: dict = {
        "format": suffix.lstrip("."),
        "file_path": str(path.resolve()),
        "file_size": path.stat().st_size,
        "text": text,
    }
    try:
        analysis = ReadabilityAnalyzer().analyze(text)
        result.update(
            word_count=analysis.word_count,
            sentence_count=analysis.sentence_count,
            paragraph_count=analysis.paragraph_count,
            readability={
                "flesch_reading_ease": analysis.flesch_score,
                "flesch_kincaid_grade": analysis.flesch_kincaid_grade,
                "gunning_fog": analysis.gunning_fog,
                "smog_index": analysis.smog_index,
                "automated_readability_index": analysis.automated_readability_index,
            },
        )
    except Exception as e:  # noqa: BLE001 - readability is advisory, never fatal
        result.update(
            word_count=0,
            sentence_count=0,
            paragraph_count=0,
            readability={},
            readability_error=str(e),
        )

    # Additive: AI-writing-tell signals (emojis, em-dashes, adverb ratio, …) —
    # advisory flags to read the document more closely, never fatal.
    try:
        from .analyzers.ai_tells import AiTellsAnalyzer

        result["ai_tells"] = AiTellsAnalyzer().analyze(text)
    except Exception as e:  # noqa: BLE001 - surfaced as a soft field, never fatal
        result["ai_tells_error"] = str(e)

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
