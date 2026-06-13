"""Document embedding wiring (lens-embed) — field presence + graceful degradation."""

from __future__ import annotations

import importlib.util

import pytest

from document_analyser.analyzers.embedding import embed_document
from document_analyser.models.schemas import AnalysisResults

_TEXT_BACKEND = importlib.util.find_spec("lens_embed") is not None and (
    importlib.util.find_spec("sentence_transformers") is not None
)


def test_results_embedding_defaults_to_none():
    # The field is optional — results validate without it.
    assert "embedding" in AnalysisResults.model_fields
    assert AnalysisResults.model_fields["embedding"].default is None


def test_embed_document_empty_is_none():
    assert embed_document("") is None
    assert embed_document("   \n  ") is None


@pytest.mark.skipif(_TEXT_BACKEND, reason="embeddings extra is installed")
def test_embed_document_none_without_backend():
    # Without document-analyser[embeddings], embedding silently stays None.
    assert embed_document("A perfectly ordinary sentence about reports.") is None


@pytest.mark.skipif(not _TEXT_BACKEND, reason="needs document-analyser[embeddings]")
def test_embed_document_returns_vector_with_backend():
    vec = embed_document("Climate disclosure in corporate annual reports.\n\n" * 5)
    assert isinstance(vec, list) and len(vec) == 384
    assert all(isinstance(x, float) for x in vec)
