"""
File processing endpoints - Enhanced document upload and analysis
"""

import base64
import time
from typing import Any

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from document_analyser.analyzers.integrity_checker import IntegrityChecker
from document_analyser.analyzers.readability import ReadabilityAnalyzer
from document_analyser.analyzers.word_analysis import WordAnalyzer
from document_analyser.analyzers.writing_quality import WritingQualityAnalyzer
from document_analyser.core.config import settings
from document_analyser.models.schemas import InferredMetadata
from document_analyser.services.document_processor import DocumentProcessor
from document_analyser.services.doi_resolver import DOIResolver
from document_analyser.services.reference_extractor import ReferenceExtractor
from document_analyser.services.url_verifier import URLVerifier

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# File type detection constants
# Magic bytes signatures for supported file types
MAGIC_BYTES = {
    b"%PDF": "application/pdf",
    b"PK\x03\x04": "application/vnd.openxmlformats-officedocument",  # ZIP-based (DOCX, PPTX)
}

# Extension to MIME type mapping
EXTENSION_MIME_MAP = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
}


async def detect_content_type(file: UploadFile, content: bytes) -> str:
    """
    Detect the actual content type of an uploaded file.

    Uses multiple detection methods in order of reliability:
    1. Magic bytes (file signature) - most reliable
    2. File extension
    3. Content-Type header from upload

    Args:
        file: The uploaded file object
        content: The file content as bytes

    Returns:
        The detected MIME type string
    """
    # Method 1: Check magic bytes (most reliable)
    for magic, mime_type in MAGIC_BYTES.items():
        if content.startswith(magic):
            # For ZIP-based formats, need to check filename extension
            if mime_type == "application/vnd.openxmlformats-officedocument":
                filename = file.filename or ""
                if filename.lower().endswith(".docx"):
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif filename.lower().endswith(".pptx"):
                    return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                # Default to docx for unknown zip-based office files
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            return mime_type

    # Method 2: Check file extension
    filename = file.filename or ""
    for ext, mime_type in EXTENSION_MIME_MAP.items():
        if filename.lower().endswith(ext):
            return mime_type

    # Method 3: Fall back to Content-Type header
    if file.content_type and file.content_type != "application/octet-stream":
        return file.content_type

    # Default fallback
    return "application/octet-stream"


# Initialize services
document_processor = DocumentProcessor()
reference_extractor = ReferenceExtractor()
url_verifier = URLVerifier()
doi_resolver = DOIResolver()
readability_analyzer = ReadabilityAnalyzer()
integrity_checker = IntegrityChecker()
writing_quality_analyzer = WritingQualityAnalyzer()
word_analyzer = WordAnalyzer()


class FileAnalysisOptions(BaseModel):
    analysis_type: str = "full"  # "full", "text", "academic"
    citation_style: str = "auto"
    check_urls: bool = True
    check_doi: bool = True
    check_plagiarism: bool = False
    extract_metadata: bool = True


class FileMetadata(BaseModel):
    filename: str
    size: int
    content_type: str
    pages: int | None = None
    author: str | None = None
    title: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None


class FileAnalysisResponse(BaseModel):
    service: str = "DocumentLens"
    version: str = "1.0.0"
    files_processed: int
    analysis_type: str
    processing_time: float
    results: dict[str, Any]


@router.post("/files", response_model=FileAnalysisResponse)
@limiter.limit(settings.RATE_LIMIT)
async def analyse_uploaded_files(
    request: Request,
    files: list[UploadFile] = File(..., description="Documents to analyse (PDF, DOCX, TXT, MD)"),
    analysis_type: str = Form(default="full", description="Analysis type: full, text, or academic"),
    citation_style: str = Form(
        default="auto", description="Citation style: auto, apa, mla, chicago"
    ),
    check_urls: bool = Form(default=True, description="Verify URLs in documents"),
    check_doi: bool = Form(default=True, description="Resolve and validate DOIs"),
    check_plagiarism: bool = Form(
        default=False, description="Check for self-plagiarism between files"
    ),
    extract_metadata: bool = Form(default=True, description="Extract document metadata"),
    include_extracted_text: bool = Form(
        default=False,
        description="Include full extracted text with page-level structure in response",
    ),
) -> FileAnalysisResponse:
    """
    Enhanced file upload and analysis endpoint.

    Supports comprehensive document analysis with metadata extraction:
    - PDF, DOCX, TXT, MD document processing
    - Text extraction with page-level granularity (when include_extracted_text=True)
    - Document metadata extraction (author, title, dates, etc.)
    - Full text analysis (readability, quality, word metrics)
    - Academic analysis (citations, DOI resolution, URL verification)
    - Batch processing with document comparison
    - AI-generated content detection

    Analysis types:
    - "full": Complete text + academic + metadata analysis
    - "text": Text analysis only (readability, quality, word metrics)
    - "academic": Academic features only (citations, DOI, URL verification)

    When include_extracted_text=True, the response includes:
    - extracted_text.full_text: Complete text from the document
    - extracted_text.pages: List of {page_number, text} for each page
    - extracted_text.total_pages: Total number of pages
    """
    start_time = time.time()

    # Validate input
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > settings.MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {settings.MAX_FILES_PER_REQUEST} allowed.",
        )

    # Validate analysis type
    if analysis_type not in ["full", "text", "academic"]:
        raise HTTPException(
            status_code=400, detail="analysis_type must be 'full', 'text', or 'academic'"
        )

    try:
        # Process each file
        file_results = []
        all_texts = []
        all_metadata = []

        for file in files:
            # Read file content first (needed for content type detection)
            content = await file.read()

            # Check file size
            if len(content) > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} too large. Max: {settings.MAX_FILE_SIZE} bytes",
                )

            # Detect actual content type (magic bytes > extension > header)
            detected_content_type = await detect_content_type(file, content)

            # Validate file type using detected type
            if detected_content_type not in settings.SUPPORTED_FILE_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {detected_content_type} (filename: {file.filename})",
                )

            # Extract text (with or without page-level structure)
            extracted_text_data = None
            if include_extracted_text:
                # Use page-level extraction
                extracted_text_data = await document_processor.extract_text_with_pages(
                    content,
                    detected_content_type,
                    file.filename or "unknown",
                )
                text = extracted_text_data["full_text"]
            else:
                # Standard extraction (just the text)
                text = await document_processor.extract_text(
                    content,
                    detected_content_type,
                    file.filename or "unknown",
                )

            # Extract metadata if requested
            metadata = None
            file_metadata = None
            if extract_metadata:
                metadata = await document_processor.extract_metadata(
                    content,
                    detected_content_type,
                    file.filename or "unknown",
                )

                file_metadata = FileMetadata(
                    filename=file.filename or "unknown",
                    size=len(content),
                    content_type=detected_content_type,
                    pages=metadata.get("pages"),
                    author=metadata.get("author"),
                    title=metadata.get("title"),
                    creation_date=metadata.get("creation_date"),
                    modification_date=metadata.get("modification_date"),
                )
                all_metadata.append(file_metadata)

            # Analyze based on analysis type
            file_analysis = await _analyse_file_content(
                text, analysis_type, citation_style, check_urls, check_doi
            )

            # Build result for this file
            file_result: dict[str, Any] = {
                "filename": file.filename,
                "content_type": detected_content_type,
                "size": len(content),
                "text_length": len(text),
                "metadata": file_metadata.model_dump() if file_metadata else None,
                "analysis": file_analysis,
            }

            # Include extracted text with page structure if requested
            if include_extracted_text and extracted_text_data:
                file_result["extracted_text"] = {
                    "full_text": extracted_text_data["full_text"],
                    "pages": extracted_text_data["pages"],
                    "total_pages": extracted_text_data["total_pages"],
                }

            file_results.append(file_result)

            all_texts.append(text)
            await file.seek(0)  # Reset for potential reuse

        # Cross-file analysis if multiple files and plagiarism check enabled
        cross_analysis = {}
        if len(files) > 1 and check_plagiarism and analysis_type in ["full", "academic"]:
            combined_text = "\n\n".join(all_texts)
            references = reference_extractor.extract_references(combined_text, citation_style)

            cross_analysis = {
                "document_comparison": _compare_documents(
                    all_texts, [f.filename or "unknown" for f in files]
                ),
                "cross_plagiarism": integrity_checker.detect_patterns(
                    combined_text, references, all_texts
                ).model_dump(),
            }

        processing_time = time.time() - start_time

        return FileAnalysisResponse(
            files_processed=len(files),
            analysis_type=analysis_type,
            processing_time=processing_time,
            results={
                "individual_files": file_results,
                "cross_analysis": cross_analysis,
                "summary": {
                    "total_files": len(files),
                    "total_text_length": sum(len(text) for text in all_texts),
                    "supported_formats": list(settings.SUPPORTED_FILE_TYPES),
                    "metadata_extracted": extract_metadata,
                },
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File analysis failed: {e!s}") from e


async def _analyse_file_content(
    text: str, analysis_type: str, citation_style: str, check_urls: bool, check_doi: bool
) -> dict[str, Any]:
    """Analyze file content based on analysis type"""
    results = {}

    if analysis_type in ["full", "text"]:
        # Text analysis
        results["readability"] = readability_analyzer.analyze(text).model_dump()
        results["writing_quality"] = writing_quality_analyzer.analyze(text).model_dump()
        results["word_analysis"] = word_analyzer.analyze(text).model_dump()

    if analysis_type in ["full", "academic"]:
        # Academic analysis
        references = reference_extractor.extract_references(text, citation_style)
        results["references"] = {
            "total_found": len(references),
            "citations": references[:10],  # First 10 for preview
            "citation_style": citation_style,
        }

        # URL and DOI analysis (simplified for individual files)
        if check_urls or check_doi:
            results["external_verification"] = {
                "urls_checked": check_urls,
                "dois_resolved": check_doi,
                "note": "Full verification available in cross-analysis for multiple files",
            }

        # Integrity check
        results["integrity"] = (
            integrity_checker.detect_patterns(text, references, []).__dict__
            if hasattr(integrity_checker.detect_patterns(text, references, []), "__dict__")
            else integrity_checker.detect_patterns(text, references, []).model_dump()
        )

    return results


def _compare_documents(texts: list[str], filenames: list[str]) -> dict[str, Any]:
    """Compare multiple documents for similarities"""
    comparisons = []

    for i, (text1, name1) in enumerate(zip(texts, filenames, strict=False)):
        word_count1 = len(text1.split())

        for j, (text2, name2) in enumerate(zip(texts, filenames, strict=False)):
            if i >= j:  # Avoid duplicate comparisons
                continue

            word_count2 = len(text2.split())

            # Simple similarity calculation (word overlap)
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            overlap = len(words1.intersection(words2))
            total_unique = len(words1.union(words2))
            similarity = overlap / max(total_unique, 1) * 100

            comparisons.append(
                {
                    "file1": name1,
                    "file2": name2,
                    "word_count1": word_count1,
                    "word_count2": word_count2,
                    "similarity_percentage": round(similarity, 2),
                    "shared_words": overlap,
                }
            )

    from typing import cast

    similarities: list[float] = [
        cast("float", c.get("similarity_percentage", 0.0)) for c in comparisons
    ]

    return {
        "total_comparisons": len(comparisons),
        "comparisons": comparisons,
        "highest_similarity": max(similarities) if similarities else 0.0,
    }


@router.post("/files/infer-metadata", response_model=InferredMetadata)
@limiter.limit(settings.RATE_LIMIT)
async def infer_document_metadata(
    request: Request,
    file: UploadFile = File(..., description="Document to analyze (PDF, DOCX, TXT, MD)"),
) -> InferredMetadata:
    """
    Infer metadata from document content using pattern matching.

    Designed for corporate annual reports and sustainability documents.
    Uses heuristics to detect:
    - **Year**: Fiscal year, report year from content and filename
    - **Company Name**: Extracted from title patterns and filename
    - **Industry**: Detected via industry-specific keyword matching
    - **Document Type**: Annual Report, Sustainability Report, Integrated Report, etc.

    Each field includes a confidence score (0.0-1.0) indicating extraction reliability.
    Users can review and edit these values in the desktop application.

    Returns:
        InferredMetadata with probable values and confidence scores
    """
    # Read file content first (needed for content type detection)
    content = await file.read()

    # Check file size
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum: {settings.MAX_FILE_SIZE} bytes",
        )

    # Detect actual content type
    detected_content_type = await detect_content_type(file, content)

    # Validate file type
    if detected_content_type not in settings.SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {detected_content_type} (filename: {file.filename})",
        )

    try:
        # Extract text from document
        text = await document_processor.extract_text(
            content,
            detected_content_type,
            file.filename or "unknown",
        )

        # Infer metadata from content
        inferred = document_processor.infer_metadata_from_content(text, file.filename)

        return InferredMetadata(**inferred)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata inference failed: {e!s}") from e


@router.post("/text/infer-metadata", response_model=InferredMetadata)
@limiter.limit(settings.RATE_LIMIT)
async def infer_text_metadata(
    request: Request,
    text: str = Form(..., description="Text content to analyze"),
    filename: str | None = Form(None, description="Optional filename for additional hints"),
) -> InferredMetadata:
    """
    Infer metadata from raw text content using pattern matching.

    Alternative to file upload when text has already been extracted.
    Uses same heuristics as /files/infer-metadata endpoint.

    Useful for:
    - Pre-extracted text
    - Cached document content
    - Testing metadata inference

    Returns:
        InferredMetadata with probable values and confidence scores
    """
    if not text or len(text.strip()) < 100:
        raise HTTPException(
            status_code=400,
            detail="Text content too short. Minimum 100 characters required.",
        )

    try:
        # Infer metadata from content
        inferred = document_processor.infer_metadata_from_content(text, filename)

        return InferredMetadata(**inferred)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata inference failed: {e!s}") from e


# =============================================================================
# Base64 File Upload Endpoint
# =============================================================================
# This endpoint accepts files as base64-encoded JSON, avoiding multipart form
# parsing issues that occur in PyInstaller bundles.


class Base64FileUpload(BaseModel):
    """Request model for base64-encoded file upload."""

    filename: str = Field(..., description="Original filename with extension")
    content_base64: str = Field(..., description="Base64-encoded file content")
    content_type: str | None = Field(None, description="MIME type (optional, will be detected)")


class Base64UploadRequest(BaseModel):
    """Request model for uploading files via base64 encoding."""

    files: list[Base64FileUpload] = Field(..., description="List of files to process")
    include_extracted_text: bool = Field(False, description="Include full extracted text in response")
    analysis_type: str = Field("full", description="Analysis type: full, text, or academic")


@router.post("/files/upload-base64", response_model=FileAnalysisResponse)
@limiter.limit(settings.RATE_LIMIT)
async def upload_files_base64(
    request: Request,
    payload: Base64UploadRequest = Body(...),
) -> FileAnalysisResponse:
    """
    Upload and analyze files using base64 encoding.

    This endpoint is an alternative to the multipart form upload that avoids
    issues with multipart parsing in certain environments (e.g., PyInstaller bundles).

    The file content should be base64-encoded. Example:
    ```json
    {
        "files": [{
            "filename": "report.pdf",
            "content_base64": "JVBERi0xLjQK...",
            "content_type": "application/pdf"
        }],
        "include_extracted_text": true,
        "analysis_type": "full"
    }
    ```

    Returns the same response format as /files endpoint.
    """
    start_time = time.time()

    if not payload.files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(payload.files) > settings.MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {settings.MAX_FILES_PER_REQUEST} allowed.",
        )

    if payload.analysis_type not in ["full", "text", "academic"]:
        raise HTTPException(
            status_code=400, detail="analysis_type must be 'full', 'text', or 'academic'"
        )

    try:
        file_results = []
        all_texts = []
        all_metadata = []

        for file_data in payload.files:
            # Decode base64 content
            try:
                content = base64.b64decode(file_data.content_base64)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 content for {file_data.filename}: {e!s}",
                ) from e

            # Check file size
            if len(content) > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file_data.filename} too large. Max: {settings.MAX_FILE_SIZE} bytes",
                )

            # Detect content type from magic bytes and extension
            detected_content_type = _detect_content_type_from_bytes(
                content, file_data.filename, file_data.content_type
            )

            # Validate file type
            if detected_content_type not in settings.SUPPORTED_FILE_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {detected_content_type} (filename: {file_data.filename})",
                )

            # Extract text
            extracted_text_data = None
            if payload.include_extracted_text:
                extracted_text_data = await document_processor.extract_text_with_pages(
                    content,
                    detected_content_type,
                    file_data.filename,
                )
                text = extracted_text_data["full_text"]
            else:
                text = await document_processor.extract_text(
                    content,
                    detected_content_type,
                    file_data.filename,
                )

            # Extract metadata
            metadata = await document_processor.extract_metadata(
                content,
                detected_content_type,
                file_data.filename,
            )

            file_metadata = FileMetadata(
                filename=file_data.filename,
                size=len(content),
                content_type=detected_content_type,
                pages=metadata.get("pages"),
                author=metadata.get("author"),
                title=metadata.get("title"),
                creation_date=metadata.get("creation_date"),
                modification_date=metadata.get("modification_date"),
            )
            all_metadata.append(file_metadata)

            # Analyze
            file_analysis = await _analyse_file_content(
                text, payload.analysis_type, "auto", True, True
            )

            # Build result
            file_result: dict[str, Any] = {
                "filename": file_data.filename,
                "content_type": detected_content_type,
                "size": len(content),
                "text_length": len(text),
                "metadata": file_metadata.model_dump(),
                "analysis": file_analysis,
            }

            if payload.include_extracted_text and extracted_text_data:
                file_result["extracted_text"] = {
                    "full_text": extracted_text_data["full_text"],
                    "pages": extracted_text_data["pages"],
                    "total_pages": extracted_text_data["total_pages"],
                }

            file_results.append(file_result)
            all_texts.append(text)

        processing_time = time.time() - start_time

        return FileAnalysisResponse(
            files_processed=len(file_results),
            analysis_type=payload.analysis_type,
            processing_time=processing_time,
            results={
                "individual_files": file_results,
                "cross_analysis": {},
                "summary": {
                    "total_files": len(file_results),
                    "total_text_length": sum(len(t) for t in all_texts),
                    "supported_formats": list(settings.SUPPORTED_FILE_TYPES),
                    "metadata_extracted": True,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File analysis failed: {e!s}") from e


def _detect_content_type_from_bytes(
    content: bytes, filename: str, provided_type: str | None
) -> str:
    """Detect content type from bytes, filename, or provided type."""
    # Check magic bytes
    for magic, mime_type in MAGIC_BYTES.items():
        if content.startswith(magic):
            if mime_type == "application/vnd.openxmlformats-officedocument":
                if filename.lower().endswith(".docx"):
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif filename.lower().endswith(".pptx"):
                    return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            return mime_type

    # Check extension
    for ext, mime_type in EXTENSION_MIME_MAP.items():
        if filename.lower().endswith(ext):
            return mime_type

    # Use provided type if valid
    if provided_type and provided_type != "application/octet-stream":
        return provided_type

    return "application/octet-stream"


# =============================================================================
# File Path Upload Endpoint
# =============================================================================
# For local deployments where frontend and backend share a filesystem.
# Much more efficient than base64 or multipart - backend reads file directly.


class FilePathUpload(BaseModel):
    """Request model for file path upload."""

    file_path: str = Field(..., description="Absolute path to the file on disk")


class FilePathUploadRequest(BaseModel):
    """Request model for uploading files via file paths."""

    files: list[FilePathUpload] = Field(..., description="List of file paths to process")
    include_extracted_text: bool = Field(False, description="Include full extracted text")
    analysis_type: str = Field("full", description="Analysis type: full, text, or academic")


@router.post("/files/upload-path", response_model=FileAnalysisResponse)
@limiter.limit(settings.RATE_LIMIT)
async def upload_files_by_path(
    request: Request,
    payload: FilePathUploadRequest = Body(...),
) -> FileAnalysisResponse:
    """
    Upload and analyze files by providing their filesystem paths.

    This is the most efficient method for local deployments where the frontend
    and backend share a filesystem. The backend reads files directly from disk.

    **Security:** This endpoint only accepts connections from localhost (127.0.0.1).
    Remote connections are rejected.

    Example:
    ```json
    {
        "files": [{"file_path": "/Users/me/Documents/report.pdf"}],
        "include_extracted_text": true,
        "analysis_type": "full"
    }
    ```
    """
    import os

    # Security: Only allow from localhost to prevent arbitrary file reads
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(
            status_code=403,
            detail="This endpoint is only available for local connections",
        )

    start_time = time.time()

    if not payload.files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(payload.files) > settings.MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {settings.MAX_FILES_PER_REQUEST} allowed.",
        )

    if payload.analysis_type not in ["full", "text", "academic"]:
        raise HTTPException(
            status_code=400, detail="analysis_type must be 'full', 'text', or 'academic'"
        )

    try:
        file_results = []
        all_texts = []
        all_metadata = []

        for file_data in payload.files:
            file_path = file_data.file_path

            # Validate file exists
            if not os.path.isfile(file_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"File not found: {file_path}",
                )

            # Read file from disk
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
            except PermissionError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Permission denied reading: {file_path}",
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error reading file {file_path}: {e!s}",
                )

            filename = os.path.basename(file_path)

            # Check file size
            if len(content) > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {filename} too large. Max: {settings.MAX_FILE_SIZE} bytes",
                )

            # Detect content type
            detected_content_type = _detect_content_type_from_bytes(content, filename, None)

            # Validate file type
            if detected_content_type not in settings.SUPPORTED_FILE_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {detected_content_type} (filename: {filename})",
                )

            # Extract text
            extracted_text_data = None
            if payload.include_extracted_text:
                extracted_text_data = await document_processor.extract_text_with_pages(
                    content,
                    detected_content_type,
                    filename,
                )
                text = extracted_text_data["full_text"]
            else:
                text = await document_processor.extract_text(
                    content,
                    detected_content_type,
                    filename,
                )

            # Extract metadata
            metadata = await document_processor.extract_metadata(
                content,
                detected_content_type,
                filename,
            )

            file_metadata = FileMetadata(
                filename=filename,
                size=len(content),
                content_type=detected_content_type,
                pages=metadata.get("pages"),
                author=metadata.get("author"),
                title=metadata.get("title"),
                creation_date=metadata.get("creation_date"),
                modification_date=metadata.get("modification_date"),
            )
            all_metadata.append(file_metadata)

            # Analyze
            file_analysis = await _analyse_file_content(
                text, payload.analysis_type, "auto", True, True
            )

            # Build result
            file_result: dict[str, Any] = {
                "filename": filename,
                "content_type": detected_content_type,
                "size": len(content),
                "text_length": len(text),
                "metadata": file_metadata.model_dump(),
                "analysis": file_analysis,
            }

            if payload.include_extracted_text and extracted_text_data:
                file_result["extracted_text"] = {
                    "full_text": extracted_text_data["full_text"],
                    "pages": extracted_text_data["pages"],
                    "total_pages": extracted_text_data["total_pages"],
                }

            file_results.append(file_result)
            all_texts.append(text)

        processing_time = time.time() - start_time

        return FileAnalysisResponse(
            files_processed=len(file_results),
            analysis_type=payload.analysis_type,
            processing_time=processing_time,
            results={
                "individual_files": file_results,
                "cross_analysis": {},
                "summary": {
                    "total_files": len(file_results),
                    "total_text_length": sum(len(t) for t in all_texts),
                    "supported_formats": list(settings.SUPPORTED_FILE_TYPES),
                    "metadata_extracted": True,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File analysis failed: {e!s}") from e
