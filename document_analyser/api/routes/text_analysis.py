"""
Text analysis endpoints - Core text metrics without academic features
"""

import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from document_analyser.analyzers.ner_analyzer import NerAnalyzer
from document_analyser.analyzers.readability import ReadabilityAnalyzer
from document_analyser.analyzers.word_analysis import WordAnalyzer
from document_analyser.analyzers.writing_quality import WritingQualityAnalyzer
from document_analyser.core.config import settings

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Initialize text analyzers only
readability_analyzer = ReadabilityAnalyzer()
writing_quality_analyzer = WritingQualityAnalyzer()
word_analyzer = WordAnalyzer()
ner_analyzer = NerAnalyzer()


class TextAnalysisRequest(BaseModel):
    text: str
    options: dict = {}


class TextAnalysisResponse(BaseModel):
    service: str = "DocumentLens"
    version: str = "1.0.0"
    content_type: str = "text"
    analysis: dict
    processing_time: float


@router.post("/text", response_model=TextAnalysisResponse)
@limiter.limit(settings.RATE_LIMIT)
async def analyse_text_only(
    request: Request, analysis_request: TextAnalysisRequest
) -> TextAnalysisResponse:
    """
    Core text analysis - readability, quality, and word metrics only.

    This endpoint provides essential text analysis without academic features:
    - Readability scores (Flesch, Flesch-Kincaid)
    - Writing quality metrics (passive voice, sentence variety)
    - Word analysis (frequency, n-grams, vocabulary richness)

    Perfect for blogs, articles, general writing assessment.
    """
    start_time = time.time()

    if not analysis_request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        text = analysis_request.text

        # Core text analysis
        document_analysis = readability_analyzer.analyze(text)
        writing_quality = writing_quality_analyzer.analyze(text)
        word_analysis = word_analyzer.analyze(text)
        ner_results = ner_analyzer.analyze(text)

        processing_time = time.time() - start_time

        return TextAnalysisResponse(
            analysis={
                "text_metrics": {
                    "word_count": document_analysis.word_count,
                    "sentence_count": document_analysis.sentence_count,
                    "paragraph_count": document_analysis.paragraph_count,
                    "avg_words_per_sentence": document_analysis.avg_words_per_sentence,
                },
                "readability": {
                    "flesch_score": document_analysis.flesch_score,
                    "flesch_kincaid_grade": document_analysis.flesch_kincaid_grade,
                    "interpretation": _interpret_readability(document_analysis.flesch_score),
                },
                "writing_quality": {
                    "passive_voice_percentage": writing_quality.passive_voice_percentage,
                    "sentence_variety": writing_quality.sentence_variety,
                    "academic_tone": writing_quality.academic_tone,
                    "transition_words": writing_quality.transition_words,
                    "hedging_language": writing_quality.hedging_language,
                },
                "word_analysis": {
                    "unique_words": word_analysis.unique_word_count,
                    "total_words": word_analysis.total_word_count,
                    "vocabulary_richness": word_analysis.vocabulary_richness,
                    "top_words": [w.model_dump() for w in word_analysis.top_words],
                    "bigrams": [p.model_dump() for p in word_analysis.bigrams],
                    "trigrams": [p.model_dump() for p in word_analysis.trigrams],
                },
                "ner": ner_results.model_dump()
                if hasattr(ner_results, "model_dump")
                else ner_results.__dict__,
            },
            processing_time=processing_time,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {e!s}") from e


def _interpret_readability(flesch_score: float) -> str:
    """Interpret Flesch Reading Ease score"""
    if flesch_score >= 90:
        return "Very Easy (5th grade level)"
    elif flesch_score >= 80:
        return "Easy (6th grade level)"
    elif flesch_score >= 70:
        return "Fairly Easy (7th grade level)"
    elif flesch_score >= 60:
        return "Standard (8th-9th grade level)"
    elif flesch_score >= 50:
        return "Fairly Difficult (10th-12th grade level)"
    elif flesch_score >= 30:
        return "Difficult (College level)"
    else:
        return "Very Difficult (Graduate level)"
