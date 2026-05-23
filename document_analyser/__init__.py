# FastAPI backend for CiteSight
"""document-analyser: text + readability metrics for documents (lens family).

Re-exports the canonical text extractor so other analysers can reuse it:

    from document_analyser import extract_text
"""
from .extraction import extract_text, extract_text_from_bytes
from .manifest import MANIFEST

__all__ = ["extract_text", "extract_text_from_bytes", "MANIFEST"]
