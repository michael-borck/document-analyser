"""
Document analysis endpoints
"""

import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from document_analyser.analyzers.integrity_checker import IntegrityChecker
from document_analyser.analyzers.readability import ReadabilityAnalyzer
from document_analyser.analyzers.word_analysis import WordAnalyzer
from document_analyser.analyzers.writing_quality import WritingQualityAnalyzer
from document_analyser.core.config import settings
from document_analyser.models.schemas import (
    AnalysisOptions,
    AnalysisResults,
    DocumentComparison,
    ReferenceResults,
)
from document_analyser.services.document_processor import DocumentProcessor
from document_analyser.services.doi_resolver import DOIResolver
from document_analyser.services.reference_extractor import ReferenceExtractor
from document_analyser.services.url_verifier import URLVerifier

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Initialize services
document_processor = DocumentProcessor()
reference_extractor = ReferenceExtractor()
url_verifier = URLVerifier()
doi_resolver = DOIResolver()
readability_analyzer = ReadabilityAnalyzer()
integrity_checker = IntegrityChecker()
writing_quality_analyzer = WritingQualityAnalyzer()
word_analyzer = WordAnalyzer()

@router.post("/analyze", response_model=AnalysisResults)
@limiter.limit(settings.RATE_LIMIT)
async def analyze_documents(  # type: ignore[no-untyped-def]
    request: Request,
    files: list[UploadFile] = File(..., description="Documents to analyze"),
    citation_style: str = Form(default="auto"),
    check_urls: bool = Form(default=True),
    check_doi: bool = Form(default=True),
    check_wayback: bool = Form(default=True),
    check_plagiarism: bool = Form(default=True),
    check_in_text: bool = Form(default=True),
    processing_mode: str = Form(default="server")
):
    """
    Analyze uploaded documents for citations, quality, and suspicious patterns.

    Args:
        files: List of uploaded files (PDF, DOCX, PPTX, TXT, MD, JSON)
        citation_style: Citation style to check for ('auto', 'apa', 'mla', 'chicago')
        check_urls: Whether to verify URLs in references
        check_doi: Whether to resolve DOIs
        check_wayback: Whether to check Wayback Machine for broken URLs
        check_plagiarism: Whether to check for self-plagiarism
        check_in_text: Whether to check in-text citations
        processing_mode: Processing mode ('server' or 'local')

    Returns:
        Complete analysis results including references, patterns, and quality metrics
    """
    start_time = time.time()

    # Validate input
    if len(files) > settings.MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {settings.MAX_FILES_PER_REQUEST} allowed."
        )

    # Validate file types and sizes
    for file in files:
        if file.content_type not in settings.SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )

        # Check file size (this is approximate as we haven't read the file yet)
        if hasattr(file, 'size') and file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )

    try:
        # Create analysis options
        from typing import Literal, cast
        options = AnalysisOptions(
            citation_style=cast("Literal['auto', 'apa', 'mla', 'chicago']", citation_style),
            check_urls=check_urls,
            check_doi=check_doi,
            check_wayback=check_wayback,
            check_plagiarism=check_plagiarism,
            check_in_text=check_in_text,
            processing_mode=cast("Literal['server', 'local']", processing_mode)
        )

        # Process documents
        all_texts = []
        document_names = []

        for file in files:
            # Extract text from document
            content = await file.read()
            text = await document_processor.extract_text(
                content,
                file.content_type or "application/octet-stream",
                file.filename or "unknown"
            )
            all_texts.append(text)
            document_names.append(file.filename or "unknown")

            # Reset file pointer for potential reuse
            await file.seek(0)

        # Combine all texts for analysis
        combined_text = "\n\n".join(all_texts)

        # Extract references
        references = reference_extractor.extract_references(combined_text, options.citation_style)

        # Verify URLs and DOIs if requested
        reference_results = await _analyze_references(
            references,
            combined_text,
            options
        )

        # Analyze document quality
        document_analysis = readability_analyzer.analyze(combined_text)
        writing_quality = writing_quality_analyzer.analyze(combined_text)
        word_analysis = word_analyzer.analyze(combined_text)

        # Detect suspicious patterns
        suspicious_patterns = integrity_checker.detect_patterns(
            combined_text,
            references,
            all_texts if options.check_plagiarism else []
        )

        # Document comparison if multiple files
        comparison = None
        if len(files) > 1:
            comparison = _create_document_comparison(
                document_names, all_texts, references
            )

        # Calculate processing time
        processing_time = time.time() - start_time

        return AnalysisResults(
            references=reference_results,
            suspicious_patterns=suspicious_patterns,
            document_analysis=document_analysis,
            writing_quality=writing_quality,
            word_analysis=word_analysis,
            comparison=comparison,
            processing_time=processing_time,
            file_count=len(files)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {e!s}"
        ) from e

async def _analyze_references(references: list, text: str, options: AnalysisOptions) -> ReferenceResults:
    """Analyze references for broken URLs, unresolved DOIs, etc."""
    # This will be implemented with the actual services
    # For now, return a placeholder

    return ReferenceResults(
        total=len(references),
        broken_urls=0,
        unresolved_dois=0,
        missing_in_text=0,
        orphaned_in_text=0,
        issues=[]
    )

def _create_document_comparison(names: list[str], texts: list[str], references: list) -> list[DocumentComparison]:
    """Create comparison data for multiple documents"""
    # This will be implemented to compare documents
    # For now, return a placeholder

    comparisons = []
    for _i, (name, text) in enumerate(zip(names, texts, strict=False)):
        comparisons.append(DocumentComparison(
            name=name,
            word_count=len(text.split()),
            readability=50.0,  # Placeholder
            references=len(references) // len(texts),  # Rough estimate
            issues=0
        ))

    return comparisons
