"""
Pydantic schemas for CiteSight API
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# Request models
class AnalysisOptions(BaseModel):
    citation_style: Literal["auto", "apa", "mla", "chicago"] = Field(default="auto")
    check_urls: bool = Field(default=True)
    check_doi: bool = Field(default=True)
    check_wayback: bool = Field(default=True)
    check_plagiarism: bool = Field(default=True)
    check_in_text: bool = Field(default=True)
    processing_mode: Literal["server", "local"] = Field(default="server")


# Result models
class Pattern(BaseModel):
    text: str
    severity: Literal["high", "medium", "low"]


class Issue(BaseModel):
    type: Literal["error", "warning"]
    title: str
    details: str


class WordFrequency(BaseModel):
    word: str
    count: int
    size: int


class PhraseCount(BaseModel):
    phrase: str
    count: int


class DocumentComparison(BaseModel):
    name: str
    word_count: int
    readability: float
    references: int
    issues: int


class SuspiciousPatterns(BaseModel):
    self_plagiarism: list[str] = Field(default_factory=list)
    citation_anomalies: list[str] = Field(default_factory=list)
    style_inconsistencies: list[str] = Field(default_factory=list)
    ai_indicators: dict[str, Any] = Field(default_factory=dict)
    integrity_score: float = Field(default=100.0)
    all_issues: list[str] = Field(default_factory=list)


class ReferenceResults(BaseModel):
    total: int
    broken_urls: int
    unresolved_dois: int
    missing_in_text: int
    orphaned_in_text: int
    issues: list[Issue] = Field(default_factory=list)


class DocumentAnalysis(BaseModel):
    word_count: int
    sentence_count: int
    avg_words_per_sentence: float
    paragraph_count: int
    flesch_score: float
    flesch_kincaid_grade: float


class WritingQuality(BaseModel):
    passive_voice_percentage: float
    sentence_variety: float
    transition_words: float
    hedging_language: float
    academic_tone: float


class WordAnalysis(BaseModel):
    most_frequent: list[WordFrequency] = Field(default_factory=list)
    unique_words: list[str] = Field(default_factory=list)  # Hapax legomena (sample)
    unique_phrases: list[PhraseCount] = Field(default_factory=list)
    # Vocabulary metrics
    unique_word_count: int = 0  # Total distinct words
    total_word_count: int = 0
    vocabulary_richness: float = 0.0  # Type-token ratio as percentage
    top_words: list[WordFrequency] = Field(default_factory=list)  # Alias for most_frequent
    bigrams: list[PhraseCount] = Field(default_factory=list)
    trigrams: list[PhraseCount] = Field(default_factory=list)


class AnalysisResults(BaseModel):
    references: ReferenceResults
    suspicious_patterns: SuspiciousPatterns
    document_analysis: DocumentAnalysis
    writing_quality: WritingQuality
    word_analysis: WordAnalysis
    comparison: list[DocumentComparison] | None = None
    processing_time: float
    file_count: int


# Page-level text extraction models
class PageText(BaseModel):
    """Text content from a single page"""

    page_number: int
    text: str


class ExtractedText(BaseModel):
    """Structured text extraction with page-level granularity"""

    full_text: str
    pages: list[PageText] = Field(default_factory=list)
    total_pages: int


class InferredMetadata(BaseModel):
    """Metadata inferred from document content"""

    probable_year: int | None = None
    probable_company: str | None = None
    probable_industry: str | None = None
    document_type: str | None = None
    confidence_scores: dict[str, float] = Field(
        default_factory=lambda: {
            "year": 0.0,
            "company": 0.0,
            "industry": 0.0,
            "document_type": 0.0,
        }
    )
    extraction_notes: list[str] = Field(default_factory=list)


# N-gram response
class Ngram(BaseModel):
    phrase: str
    count: int


class NgramResponse(BaseModel):
    n: int
    top_ngrams: list[Ngram] = Field(default_factory=list)


# NER response models
class NEREntity(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int


class NERResponse(BaseModel):
    entities: list[NEREntity] = Field(default_factory=list)


# Keyword search models
class KeywordMatch(BaseModel):
    document: str
    context: str
    start_char: int
    end_char: int


class KeywordSearchResponse(BaseModel):
    keyword: str
    matches: list[KeywordMatch] = Field(default_factory=list)


# Batch keyword search models
class DocumentKeywordResult(BaseModel):
    """Keyword results for a single document"""

    document: str
    count: int
    contexts: list[str] = Field(default_factory=list)


class KeywordResult(BaseModel):
    """Results for a single keyword across all documents"""

    keyword: str
    total_matches: int
    by_document: list[DocumentKeywordResult] = Field(default_factory=list)


class MultiKeywordSearchResponse(BaseModel):
    """Response for batch keyword search across multiple keywords and documents"""

    results: list[KeywordResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


# Response models
class ApiResponse(BaseModel):
    status: Literal["success", "error"]
    message: str | None = None
    data: dict | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float


class ErrorResponse(BaseModel):
    error: str
    details: str | None = None
    code: int | None = None


# ===== Domain Mapping Models =====
class DomainMapping(BaseModel):
    """Mapping of a section/segment to a domain"""

    section_text: str
    section_index: int
    primary_domain: str
    similarity_score: float  # 0.0-1.0
    all_domain_scores: dict[str, float] = Field(default_factory=dict)
    confidence: Literal["high", "medium", "low"]


class DomainMappingResponse(BaseModel):
    """Response for domain mapping analysis"""

    total_sections: int
    domains_analyzed: list[str] = Field(default_factory=list)
    mappings: list[DomainMapping] = Field(default_factory=list)
    domain_distribution: dict[str, int] = Field(default_factory=dict)  # domain -> count
    average_confidence: float = 0.0


# ===== Structural Mismatch Models =====
class SentenceDislocation(BaseModel):
    """A sentence that's thematically dislocated from its parent section"""

    sentence_index: int
    sentence_text: str
    sentence_domain: str
    parent_section_index: int
    parent_section_domain: str
    dislocation_score: float  # 0.0-1.0, higher = more dislocated
    severity: Literal["low", "medium", "high"]


class StructuralMismatchResponse(BaseModel):
    """Response for structural mismatch detection"""

    total_sentences_analyzed: int
    total_sections: int
    dislocations: list[SentenceDislocation] = Field(default_factory=list)
    overall_coherence_score: float = 0.0  # 0.0-1.0, higher = more coherent
    highly_dislocated_count: int = 0
    recommendations: list[str] = Field(default_factory=list)


# ===== Granular Sentiment Models =====
class SentimentScore(BaseModel):
    """Sentiment scores (0.0-1.0 for each)"""

    positive: float = 0.0
    negative: float = 0.0
    neutral: float = 0.0
    compound: float = 0.0  # -1.0 to 1.0


class SentenceSentiment(BaseModel):
    """Sentiment for a single sentence"""

    sentence_index: int
    text: str
    sentiment: SentimentScore
    dominant_sentiment: Literal["positive", "negative", "neutral"]


class ParagraphSentiment(BaseModel):
    """Sentiment for a paragraph (aggregated from sentences)"""

    paragraph_index: int
    sentence_count: int
    sentiment: SentimentScore  # Averaged from sentences
    sentences: list[SentenceSentiment] = Field(default_factory=list)
    dominant_sentiment: Literal["positive", "negative", "neutral"]


class SectionSentiment(BaseModel):
    """Sentiment for a document section"""

    section_name: str
    section_index: int
    paragraph_count: int
    sentiment: SentimentScore  # Aggregated
    paragraphs: list[ParagraphSentiment] = Field(default_factory=list)
    dominant_sentiment: Literal["positive", "negative", "neutral"]


class GranularSentimentResponse(BaseModel):
    """Multi-level sentiment analysis response"""

    document_sentiment: SentimentScore  # Overall document
    sections: list[SectionSentiment] = Field(default_factory=list)
    total_sentences: int = 0
    total_paragraphs: int = 0
    total_sections: int = 0
    sentiment_distribution: dict[str, float] = Field(default_factory=dict)  # positive/negative/neutral percentages
