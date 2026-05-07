"""
Academic analysis endpoints - Citation, DOI, URL verification, and integrity checks
"""

import time
from typing import Literal, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from document_analyser.analyzers.integrity_checker import IntegrityChecker
from document_analyser.core.config import settings
from document_analyser.models.schemas import AnalysisOptions, ReferenceResults
from document_analyser.services.doi_resolver import DOIResolver
from document_analyser.services.reference_extractor import ReferenceExtractor
from document_analyser.services.url_verifier import URLVerifier

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Initialize academic services
reference_extractor = ReferenceExtractor()
url_verifier = URLVerifier()
doi_resolver = DOIResolver()
integrity_checker = IntegrityChecker()

class AcademicAnalysisRequest(BaseModel):
    text: str
    citation_style: str = "auto"
    check_urls: bool = True
    check_doi: bool = True
    check_wayback: bool = True
    check_plagiarism: bool = False
    check_in_text: bool = True

class AcademicAnalysisResponse(BaseModel):
    service: str = "DocumentLens"
    version: str = "1.0.0"
    content_type: str = "academic"
    analysis: dict
    processing_time: float

@router.post("/academic", response_model=AcademicAnalysisResponse)
@limiter.limit(settings.RATE_LIMIT)
async def analyse_academic_features(
    request: Request,
    analysis_request: AcademicAnalysisRequest
) -> AcademicAnalysisResponse:
    """
    Academic document analysis - citations, references, and integrity checks.

    This endpoint provides specialized academic analysis:
    - Citation extraction and style detection
    - DOI resolution and validation
    - URL verification with Wayback Machine fallback
    - Suspicious pattern detection (plagiarism, citation stuffing)
    - Reference integrity checking

    Perfect for research papers, academic documents, scholarly articles.
    """
    start_time = time.time()

    if not analysis_request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    try:
        text = analysis_request.text

        # Create analysis options
        options = AnalysisOptions(
            citation_style=cast("Literal['auto', 'apa', 'mla', 'chicago']", analysis_request.citation_style),
            check_urls=analysis_request.check_urls,
            check_doi=analysis_request.check_doi,
            check_wayback=analysis_request.check_wayback,
            check_plagiarism=analysis_request.check_plagiarism,
            check_in_text=analysis_request.check_in_text,
            processing_mode="server"
        )

        # Extract references
        references = reference_extractor.extract_references(text, options.citation_style)

        # Verify URLs and DOIs if requested
        reference_results = await _analyze_references(references, text, options)

        # Detect suspicious patterns
        suspicious_patterns = integrity_checker.detect_patterns(
            text,
            references,
            [text] if options.check_plagiarism else []
        )

        processing_time = time.time() - start_time

        return AcademicAnalysisResponse(
            analysis={
                "references": {
                    "total": reference_results.total,
                    "broken_urls": reference_results.broken_urls,
                    "unresolved_dois": reference_results.unresolved_dois,
                    "missing_in_text": reference_results.missing_in_text,
                    "orphaned_in_text": reference_results.orphaned_in_text,
                    "issues": [
                        {
                            "type": issue.type,
                            "title": issue.title,
                            "details": issue.details,
                            "severity": getattr(issue, 'severity', 'medium'),
                            "suggestion": getattr(issue, 'suggestion', None)
                        }
                        for issue in reference_results.issues
                    ]
                },
                "citations": {
                    "detected_style": options.citation_style,
                    "extracted": len(references),
                    "styles_found": _detect_citation_styles(references)
                },
                "integrity": {
                    "integrity_score": suspicious_patterns.integrity_score,
                    "ai_risk_level": suspicious_patterns.ai_indicators.get('risk_level', 'low'),
                    "ai_confidence": suspicious_patterns.ai_indicators.get('confidence', 0.0),
                    "issues_detected": len(suspicious_patterns.all_issues),
                    "issues": suspicious_patterns.all_issues,
                    "self_plagiarism": suspicious_patterns.self_plagiarism,
                    "citation_anomalies": suspicious_patterns.citation_anomalies,
                    "style_inconsistencies": suspicious_patterns.style_inconsistencies
                }
            },
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Academic analysis failed: {e!s}"
        ) from e

async def _analyze_references(references: list, text: str, options: AnalysisOptions) -> ReferenceResults:
    """Analyze references for broken URLs, unresolved DOIs, etc."""
    from document_analyser.models.schemas import Issue

    broken_urls = 0
    unresolved_dois = 0
    missing_in_text = 0
    orphaned_in_text = 0
    issues = []

    if not references:
        return ReferenceResults(
            total=0,
            broken_urls=0,
            unresolved_dois=0,
            missing_in_text=0,
            orphaned_in_text=0,
            issues=[]
        )

    # Extract URLs and DOIs from references
    urls = []
    dois = []

    for ref in references:
        ref_text = str(ref) if ref else ""

        # Extract URLs
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
        found_urls = re.findall(url_pattern, ref_text)
        urls.extend(found_urls)

        # Extract DOIs
        doi_pattern = r'(?:doi:|DOI:|\bdoi\s*[:=]\s*|https?://(?:dx\.)?doi\.org/)?\s*(10\.\d{4,}/[-._;()/:\w\[\]]+)'
        found_dois = re.findall(doi_pattern, ref_text, re.IGNORECASE)
        dois.extend(found_dois)

    # Verify URLs if requested
    if options.check_urls and urls:
        try:
            url_results = await url_verifier.verify_urls(urls)
            broken_urls = len(url_results.get("broken", []))

            if broken_urls > 0:
                issues.append(Issue(
                    type="warning",
                    title=f"{broken_urls} broken URL(s) found",
                    details=f"Found {broken_urls} inaccessible URLs in references"
                ))

        except Exception as e:
            issues.append(Issue(
                type="error",
                title="URL verification failed",
                details=f"Could not verify URLs: {e!s}"
            ))

    # Resolve DOIs if requested
    if options.check_doi and dois:
        try:
            doi_results = await doi_resolver.resolve_dois(dois)
            unresolved_dois = len(doi_results.get("unresolved", []))

            if unresolved_dois > 0:
                issues.append(Issue(
                    type="warning",
                    title=f"{unresolved_dois} unresolved DOI(s) found",
                    details=f"Found {unresolved_dois} DOIs that could not be resolved"
                ))

        except Exception as e:
            issues.append(Issue(
                type="error",
                title="DOI resolution failed",
                details=f"Could not resolve DOIs: {e!s}"
            ))

    # Check for in-text citation matching if requested
    if options.check_in_text:
        try:
            citation_analysis = _analyze_in_text_citations(references, text)
            missing_in_text = citation_analysis["missing_in_text"]
            orphaned_in_text = citation_analysis["orphaned_in_text"]

            if missing_in_text > 0:
                issues.append(Issue(
                    type="warning",
                    title=f"{missing_in_text} reference(s) not cited in text",
                    details="Some references in bibliography are not cited in the main text"
                ))

            if orphaned_in_text > 0:
                issues.append(Issue(
                    type="warning",
                    title=f"{orphaned_in_text} orphaned citation(s) found",
                    details="Some in-text citations do not have corresponding references"
                ))

        except Exception as e:
            issues.append(Issue(
                type="error",
                title="In-text citation analysis failed",
                details=f"Could not analyze in-text citations: {e!s}"
            ))

    return ReferenceResults(
        total=len(references),
        broken_urls=broken_urls,
        unresolved_dois=unresolved_dois,
        missing_in_text=missing_in_text,
        orphaned_in_text=orphaned_in_text,
        issues=issues
    )

def _analyze_in_text_citations(references: list, text: str) -> dict[str, int]:
    """
    Analyze in-text citations to find missing and orphaned citations

    Args:
        references: List of reference objects
        text: Main document text

    Returns:
        Dictionary with missing_in_text and orphaned_in_text counts
    """
    import re

    # Extract author names from references for matching
    reference_authors = set()
    for ref in references:
        ref_text = str(ref) if ref else ""

        # Common patterns for author extraction from different citation styles
        # APA: Author, A. B. (Year)
        # MLA: Author, First Last
        # Chicago: Author, First Last

        # Extract potential author names (simplified approach)
        # Look for patterns like "Smith, J." or "Smith, John" at start of reference
        author_patterns = [
            r'^([A-Z][a-z]+(?:-[A-Z][a-z]+)?),?\s+[A-Z]\.?(?:\s+[A-Z]\.?)*',  # Last, F. M.
            r'^([A-Z][a-z]+(?:-[A-Z][a-z]+)?),?\s+[A-Z][a-z]+',  # Last, First
            r'^([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s+et\s+al\.?',  # Smith et al.
        ]

        for pattern in author_patterns:
            matches = re.findall(pattern, ref_text)
            for match in matches:
                if isinstance(match, str):
                    reference_authors.add(match.lower())
                elif isinstance(match, tuple):
                    reference_authors.add(match[0].lower())

    # Find in-text citations
    in_text_citations = set()

    # Common in-text citation patterns
    citation_patterns = [
        r'\(([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s*,?\s*\d{4}\)',  # (Smith, 2023)
        r'\(([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s+et\s+al\.?\s*,?\s*\d{4}\)',  # (Smith et al., 2023)
        r'([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s+\(\d{4}\)',  # Smith (2023)
        r'([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s+et\s+al\.?\s+\(\d{4}\)',  # Smith et al. (2023)
        r'\[(\d+)\]',  # [1] - numbered citations
        r'\(([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s*&\s*[A-Z][a-z]+(?:-[A-Z][a-z]+)?\s*,?\s*\d{4}\)',  # (Smith & Jones, 2023)
    ]

    for pattern in citation_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if match.isdigit():
                # Numbered citation - harder to match without more context
                continue
            in_text_citations.add(match.lower())

    # Calculate missing and orphaned citations
    missing_in_text = 0
    orphaned_in_text = 0

    if reference_authors:
        # Count references not cited in text
        for ref_author in reference_authors:
            if ref_author not in in_text_citations:
                missing_in_text += 1

    if in_text_citations:
        # Count in-text citations without corresponding references
        for citation_author in in_text_citations:
            if citation_author not in reference_authors:
                orphaned_in_text += 1

    return {
        "missing_in_text": missing_in_text,
        "orphaned_in_text": orphaned_in_text
    }


def _detect_citation_styles(references: list) -> list[str]:
    """Detect citation styles present in the references"""
    import re

    if not references:
        return []

    styles_detected = set()

    for ref in references:
        ref_text = str(ref) if ref else ""

        # APA style patterns
        # Author, A. B. (Year). Title. Journal, Volume(Issue), pages.
        if re.search(r'[A-Z][a-z]+,\s+[A-Z]\.\s*[A-Z]?\.\s*\(\d{4}\)\.', ref_text):
            styles_detected.add("apa")

        # MLA style patterns
        # Author, First Last. "Title." Journal Volume.Issue (Year): pages.
        if re.search(r'[A-Z][a-z]+,\s+[A-Z][a-z]+\s+[A-Z][a-z]+\.\s*".*?"\s+.*?\s+\(\d{4}\):', ref_text):
            styles_detected.add("mla")

        # Chicago style patterns
        # Author, First Last. "Title." Journal Volume, no. Issue (Year): pages.
        if re.search(r'[A-Z][a-z]+,\s+[A-Z][a-z]+\s+[A-Z][a-z]+\.\s+".*?"\s+.*?\s+no\.\s+\d+\s+\(\d{4}\):', ref_text):
            styles_detected.add("chicago")

        # IEEE style patterns
        # [1] A. Author, "Title," Journal, vol. X, no. Y, pp. Z-W, Year.
        if re.search(r'\[\d+\]\s+[A-Z]\.\s+[A-Z][a-z]+.*?,\s+".*?",\s+.*?,\s+vol\.\s+\d+', ref_text):
            styles_detected.add("ieee")

    return list(styles_detected) if styles_detected else ["unknown"]
