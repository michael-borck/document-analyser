"""The canonical public surface (see lens-analysers/CONVENTIONS.md).

document-analyser is a special case: it keeps exposing the canonical text
extractor alongside the standard analyser surface.
"""

from __future__ import annotations

import document_analyser


def test_canonical_surface_importable():
    from document_analyser import (  # noqa: F401
        MANIFEST,
        DocumentAnalyser,
        DocumentAnalysis,
        analyse,
    )

    assert callable(analyse)
    assert callable(DocumentAnalyser)
    assert MANIFEST["name"] == "document-analyser"
    assert isinstance(document_analyser.__version__, str)


def test_extractor_still_exported():
    # Back-compat: other analysers import the canonical extractor from here.
    from document_analyser import extract_text, extract_text_from_bytes  # noqa: F401

    assert callable(extract_text)


def test_surface_in_dunder_all():
    for name in (
        "DocumentAnalyser",
        "DocumentAnalysis",
        "analyse",
        "extract_text",
        "MANIFEST",
        "__version__",
    ):
        assert name in document_analyser.__all__
