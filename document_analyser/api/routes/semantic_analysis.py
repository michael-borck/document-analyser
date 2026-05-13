"""Semantic analysis endpoints: domain mapping, structural mismatch, sentiment."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from document_analyser.analyzers.domain_mapper import DomainMapper
from document_analyser.analyzers.sentiment_analyzer import GranularSentimentAnalyzer
from document_analyser.analyzers.structural_mismatch import StructuralMismatchAnalyzer
from document_analyser.core.config import settings
from document_analyser.models.schemas import (
    DomainMappingResponse,
    GranularSentimentResponse,
    StructuralMismatchResponse,
)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Initialize analyzers at module level (efficient for PyInstaller)
domain_mapper = DomainMapper()
mismatch_analyzer = StructuralMismatchAnalyzer()
sentiment_analyzer = GranularSentimentAnalyzer()


# ===== Request Models =====
class DomainMappingRequest(BaseModel):
    """Request for domain mapping analysis."""

    text: str
    domains: list[str]


class SimilarTermsRequest(BaseModel):
    """Request for similar-terms ranking.

    Embeds source_terms and candidate_terms with the same sentence-
    transformers model the domain_mapper uses, then for each source
    term returns the top_n candidates by cosine similarity.
    """

    source_terms: list[str]
    candidate_terms: list[str]
    top_n: int = Field(default=20, ge=1, le=200)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class SimilarTermCandidate(BaseModel):
    """One candidate-term match for a source term."""

    candidate: str
    similarity: float


class SimilarTermsForSource(BaseModel):
    """Top-N candidates for one source term."""

    source: str
    candidates: list[SimilarTermCandidate]


class SimilarTermsResponse(BaseModel):
    """Response: top-N candidates per source term."""

    results: list[SimilarTermsForSource]


class StructuralMismatchRequest(BaseModel):
    """Request for structural mismatch detection."""

    text: str
    domains: list[str]
    threshold: float = 0.3  # Dislocation threshold


class SentimentAnalysisRequest(BaseModel):
    """Request for granular sentiment analysis."""

    text: str


class BatchSentimentRequest(BaseModel):
    """Batch sentiment analysis across multiple documents."""

    documents: list[str]


class BatchDomainMappingRequest(BaseModel):
    """Batch domain mapping across multiple documents."""

    documents: list[str]
    domains: list[str]


# ===== Endpoints =====
@router.post("/domain-mapping", response_model=DomainMappingResponse)
@limiter.limit(settings.RATE_LIMIT)
async def analyze_domain_mapping(
    request: Request, req: DomainMappingRequest
) -> DomainMappingResponse:
    """
    Map document sections to user-defined domains using semantic similarity.

    Uses sentence-transformers (all-MiniLM-L6-v2) to calculate cosine similarity
    between detected section headers and provided domains.

    Example domains: ["Teaching", "Research", "Service", "Administration"]
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if not req.domains or len(req.domains) < 2:
        raise HTTPException(
            status_code=400,
            detail="Must provide at least 2 domains for mapping"
        )

    try:
        result = domain_mapper.analyze(req.text, req.domains)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Domain mapping failed: {e!s}"
        ) from e


@router.post("/structural-mismatch", response_model=StructuralMismatchResponse)
@limiter.limit(settings.RATE_LIMIT)
async def analyze_structural_mismatch(
    request: Request, req: StructuralMismatchRequest
) -> StructuralMismatchResponse:
    """
    Detect thematic dislocation of sentences within sections.

    Compares the semantic domain of each sentence vs its parent section.
    If a sentence maps to a different domain (e.g., "Research") than its
    parent section (e.g., "Operations"), calculates a dislocation_score.

    Higher scores indicate content that may be misplaced or "stuffed" into
    the wrong section for keyword optimization purposes.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if not req.domains or len(req.domains) < 2:
        raise HTTPException(
            status_code=400,
            detail="Must provide at least 2 domains for analysis"
        )

    if not 0.0 <= req.threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Threshold must be between 0.0 and 1.0"
        )

    try:
        result = mismatch_analyzer.analyze(req.text, req.domains, req.threshold)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Structural mismatch analysis failed: {e!s}"
        ) from e


@router.post("/sentiment", response_model=GranularSentimentResponse)
@limiter.limit(settings.RATE_LIMIT)
async def analyze_sentiment(
    request: Request, req: SentimentAnalysisRequest
) -> GranularSentimentResponse:
    """
    Multi-level sentiment analysis: sentence, paragraph, and section levels.

    Returns sentiment scores simultaneously at all three levels:
    - Sentence-level: Individual sentence sentiment
    - Paragraph-level: Aggregated from sentences
    - Section-level: Aggregated from paragraphs
    - Document-level: Overall sentiment

    Each level includes positive/negative/neutral scores and a compound score (-1 to 1).
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        result = sentiment_analyzer.analyze(req.text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {e!s}"
        ) from e


@router.post("/sentiment/batch", response_model=list[GranularSentimentResponse])
@limiter.limit(settings.RATE_LIMIT)
async def analyze_sentiment_batch(
    request: Request, req: BatchSentimentRequest
) -> list[GranularSentimentResponse]:
    """Batch sentiment analysis across multiple documents."""
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    results = []
    for text in req.documents:
        if not text.strip():
            raise HTTPException(status_code=400, detail="One or more documents is empty")
        try:
            results.append(sentiment_analyzer.analyze(text))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e!s}") from e
    return results


@router.post("/domain-mapping/batch", response_model=list[DomainMappingResponse])
@limiter.limit(settings.RATE_LIMIT)
async def analyze_domain_mapping_batch(
    request: Request, req: BatchDomainMappingRequest
) -> list[DomainMappingResponse]:
    """Batch domain mapping across multiple documents."""
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    if not req.domains or len(req.domains) < 2:
        raise HTTPException(status_code=400, detail="Must provide at least 2 domains")
    results = []
    for text in req.documents:
        if not text.strip():
            raise HTTPException(status_code=400, detail="One or more documents is empty")
        try:
            results.append(domain_mapper.analyze(text, req.domains))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Domain mapping failed: {e!s}") from e
    return results


@router.post("/similar-terms", response_model=SimilarTermsResponse)
@limiter.limit(settings.RATE_LIMIT)
async def find_similar_terms(
    request: Request, req: SimilarTermsRequest
) -> SimilarTermsResponse:
    """
    Find candidate terms semantically similar to source terms.

    Embeds source_terms and candidate_terms with the sentence-transformers
    model the domain_mapper uses, then for each source term returns the
    top_n candidates by cosine similarity (above min_similarity).

    Designed for the synonym-discovery flow in Document Lens: source =
    keywords from a curated list, candidates = n-grams extracted from
    the user's corpus. Returns ranked, model-judged synonym candidates
    the user can accept or reject.
    """
    if not req.source_terms:
        raise HTTPException(status_code=400, detail="source_terms cannot be empty")
    if not req.candidate_terms:
        raise HTTPException(status_code=400, detail="candidate_terms cannot be empty")
    if domain_mapper.model is None or np is None:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding model unavailable: {domain_mapper._load_error or 'model not loaded'}"
        )

    try:
        # Embed both lists once. Encoding is the bulk of the cost; a single
        # call per list amortises model warm-up.
        source_embeddings = domain_mapper.model.encode(req.source_terms)
        candidate_embeddings = domain_mapper.model.encode(req.candidate_terms)

        # Normalise both matrices for cosine similarity = dot product.
        source_norms = source_embeddings / np.linalg.norm(source_embeddings, axis=1, keepdims=True)
        candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)

        # Similarity matrix: shape (n_sources, n_candidates).
        sims = np.dot(source_norms, candidate_norms.T)

        results: list[SimilarTermsForSource] = []
        for i, source in enumerate(req.source_terms):
            row = sims[i]
            # Indices of the top-N candidates (sorted desc).
            ranked_indices: Any = np.argsort(-row)[: req.top_n]
            candidates: list[SimilarTermCandidate] = []
            for idx in ranked_indices:
                score = float(row[idx])
                if score < req.min_similarity:
                    continue
                # Skip exact matches (a keyword finding itself in the
                # candidate list isn't a useful synonym).
                if req.candidate_terms[idx].lower() == source.lower():
                    continue
                candidates.append(SimilarTermCandidate(
                    candidate=req.candidate_terms[idx],
                    similarity=score,
                ))
            results.append(SimilarTermsForSource(source=source, candidates=candidates))

        return SimilarTermsResponse(results=results)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Similar-terms ranking failed: {e!s}"
        ) from e
