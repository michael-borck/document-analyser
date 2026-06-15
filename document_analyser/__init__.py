"""document-analyser: text + readability metrics for documents (lens family).

Canonical family surface:

    from document_analyser import DocumentAnalyser, analyse
    result = analyse("report.pdf")            # == DocumentAnalyser().analyse(...)

document-analyser is also the family's canonical text extractor, so it keeps
re-exporting that for other analysers to reuse:

    from document_analyser import extract_text
"""

from importlib.metadata import version as _v
from pathlib import Path

from .analysis import DocumentAnalyser, analyse_document
from .extraction import extract_text, extract_text_from_bytes
from .manifest import MANIFEST
from .models.schemas import DocumentAnalysis

__version__ = _v("document-analyser")
del _v


def analyse(path: str | Path) -> dict:
    """Analyse ``path`` and return its signal bundle (the CLI's JSON shape).

    Module-level convenience for the family's canonical call shape — equivalent
    to ``DocumentAnalyser().analyse(path)``.
    """
    return analyse_document(Path(path))


__all__ = [
    "DocumentAnalyser",
    "DocumentAnalysis",
    "analyse",
    "extract_text",
    "extract_text_from_bytes",
    "MANIFEST",
    "__version__",
]
