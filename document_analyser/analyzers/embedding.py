"""Document embedding via the family's shared helper (lens-embed).

A single pinned model across the family means this vector is directly
comparable to vectors from other members — the basis for cross-artefact
consistency and cohort-distinctiveness signals computed downstream.

Opt-in and degradable: install ``document-analyser[embeddings]`` to populate
``AnalysisResults.embedding``; without it (or on any failure) this returns
``None`` and the rest of the analysis is unaffected.
"""

from __future__ import annotations


def embed_document(text: str) -> list[float] | None:
    """Pooled, L2-normalised document vector, or None if embeddings are off."""
    if not text or not text.strip():
        return None
    try:
        from lens_embed import backend_available, embed_long_text
    except ImportError:
        return None  # the [embeddings] extra isn't installed
    if not backend_available("text"):
        return None
    try:
        return embed_long_text(text)
    except Exception:
        return None  # never let an embedding failure break the analysis
