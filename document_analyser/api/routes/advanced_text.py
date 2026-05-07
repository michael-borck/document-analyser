"""
Advanced text analysis endpoints: n-grams, NER, and contextual keyword search
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from document_analyser.analyzers.keyword_analyzer import KeywordAnalyzer
from document_analyser.analyzers.ner_analyzer import NerAnalyzer
from document_analyser.analyzers.ngram_analyzer import NgramAnalyzer
from document_analyser.models.schemas import (
    KeywordSearchResponse,
    MultiKeywordSearchResponse,
    NERResponse,
    NgramResponse,
)

router = APIRouter()

# Initialize analyzers
ngram_analyzer = NgramAnalyzer()
ner_analyzer = NerAnalyzer()
keyword_analyzer = KeywordAnalyzer()


class NgramRequest(BaseModel):
    text: str
    n: int = 2
    top_k: int = 20
    filter_terms: list[str] | None = None  # Optional: only return n-grams containing these terms


class NERRequest(BaseModel):
    text: str
    model: str | None = None


class KeywordSearchRequest(BaseModel):
    keyword: str
    documents: list[str]  # list of raw document texts
    document_names: list[str] | None = None


class MultiKeywordSearchRequest(BaseModel):
    """Request for batch keyword search across multiple keywords and documents"""

    keywords: list[str]  # list of keywords to search
    documents: list[str]  # list of raw document texts
    document_names: list[str] | None = None
    context_chars: int = 100  # characters around each match for context


@router.post("/ngrams", response_model=NgramResponse)
async def ngrams(req: NgramRequest) -> NgramResponse:
    """
    Extract n-grams (bigrams or trigrams) from text.

    Optionally filter results to only include n-grams containing specific terms.
    This is useful for focused analysis (e.g., sustainability keywords).

    Args:
        req.text: The text to analyze
        req.n: 2 for bigrams, 3 for trigrams
        req.top_k: Number of top results to return
        req.filter_terms: Optional list of terms to filter by

    Returns:
        List of n-grams with their counts
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if req.n not in (2, 3):
        raise HTTPException(
            status_code=400, detail="Only bigrams (2) and trigrams (3) are supported"
        )

    top = ngram_analyzer.extract_ngrams(
        req.text, n=req.n, top_k=req.top_k, filter_terms=req.filter_terms
    )
    return NgramResponse(n=req.n, top_ngrams=top)


@router.post("/ner", response_model=NERResponse)
async def ner(req: NERRequest) -> NERResponse:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # If user specified a model and spaCy is available, try to load it
    if req.model and isinstance(ner_analyzer := NerAnalyzer(req.model), NerAnalyzer):
        result = ner_analyzer.analyze(req.text)
        return result

    # Fallback to default analyzer
    result = ner_analyzer.analyze(req.text)
    return result


@router.post("/search/keyword", response_model=KeywordSearchResponse)
async def keyword_search(req: KeywordSearchRequest) -> KeywordSearchResponse:
    """
    Search for a single keyword across multiple documents.

    Returns all matches with context snippets.
    """
    if not req.keyword.strip():
        raise HTTPException(status_code=400, detail="Keyword cannot be empty")
    if not req.documents:
        raise HTTPException(status_code=400, detail="Documents cannot be empty")

    names = req.document_names or [f"doc_{i + 1}" for i in range(len(req.documents))]
    texts = list(zip(names, req.documents, strict=False))

    return keyword_analyzer.search(texts, req.keyword)


@router.post("/search/keywords", response_model=MultiKeywordSearchResponse)
async def multi_keyword_search(req: MultiKeywordSearchRequest) -> MultiKeywordSearchResponse:
    """
    Search for multiple keywords across multiple documents in a single request.

    This is optimized for batch analysis scenarios like sustainability research,
    where you need to search for many framework keywords (e.g., TCFD, GRI, SDGs)
    across a corpus of documents.

    Returns aggregated results with:
    - Match counts per keyword per document
    - Context snippets for each match
    - Summary statistics

    Example use case:
    - Search for 50 TCFD keywords across 10 annual report PDFs
    - Get counts and contexts for each keyword-document combination
    """
    if not req.keywords:
        raise HTTPException(status_code=400, detail="Keywords list cannot be empty")
    if not req.documents:
        raise HTTPException(status_code=400, detail="Documents list cannot be empty")

    # Filter out empty keywords
    valid_keywords = [k.strip() for k in req.keywords if k.strip()]
    if not valid_keywords:
        raise HTTPException(status_code=400, detail="No valid keywords provided")

    names = req.document_names or [f"doc_{i + 1}" for i in range(len(req.documents))]
    texts = list(zip(names, req.documents, strict=False))

    return keyword_analyzer.search_multiple(texts, valid_keywords, context_chars=req.context_chars)
