"""
Tests for file upload and processing endpoints.

This module tests the /files endpoint with PDF documents.
Structured to easily add DOCX and other formats later.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


class TestFileUploadEndpoint:
    """Tests for the POST /files endpoint."""

    @pytest.mark.pdf
    def test_upload_single_pdf_returns_200(self, client: TestClient, sample_pdf_paths: list[Path]):
        """Uploading a single PDF should return 200 OK."""
        if not sample_pdf_paths:
            pytest.skip("No PDF files in test-data directory")

        pdf_path = sample_pdf_paths[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/files",
                files={"files": (pdf_path.name, f, "application/pdf")},
            )

        assert response.status_code == 200

    @pytest.mark.pdf
    def test_upload_pdf_returns_correct_structure(
        self, client: TestClient, sample_pdf_paths: list[Path]
    ):
        """PDF upload response should have expected structure."""
        if not sample_pdf_paths:
            pytest.skip("No PDF files in test-data directory")

        pdf_path = sample_pdf_paths[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/files",
                files={"files": (pdf_path.name, f, "application/pdf")},
            )

        data = response.json()

        # Check top-level structure
        assert data["service"] == "DocumentLens"
        assert data["version"] == "1.0.0"
        assert data["files_processed"] == 1
        assert data["analysis_type"] == "full"
        assert "processing_time" in data
        assert "results" in data

    @pytest.mark.pdf
    def test_upload_pdf_extracts_metadata(self, client: TestClient, sample_pdf_paths: list[Path]):
        """PDF upload should extract document metadata."""
        if not sample_pdf_paths:
            pytest.skip("No PDF files in test-data directory")

        pdf_path = sample_pdf_paths[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/files",
                files={"files": (pdf_path.name, f, "application/pdf")},
                data={"extract_metadata": "true"},
            )

        data = response.json()
        file_result = data["results"]["individual_files"][0]

        assert "metadata" in file_result
        metadata = file_result["metadata"]

        assert metadata["filename"] == pdf_path.name
        assert metadata["content_type"] == "application/pdf"
        assert "size" in metadata

    @pytest.mark.pdf
    def test_upload_pdf_with_text_analysis(self, client: TestClient, sample_pdf_paths: list[Path]):
        """PDF upload with text analysis should return readability metrics."""
        if not sample_pdf_paths:
            pytest.skip("No PDF files in test-data directory")

        pdf_path = sample_pdf_paths[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/files",
                files={"files": (pdf_path.name, f, "application/pdf")},
                data={"analysis_type": "text"},
            )

        data = response.json()
        file_result = data["results"]["individual_files"][0]
        analysis = file_result["analysis"]

        # Text analysis should include readability
        assert "readability" in analysis
        assert "writing_quality" in analysis
        assert "word_analysis" in analysis

    @pytest.mark.pdf
    def test_upload_pdf_with_academic_analysis(
        self, client: TestClient, sample_pdf_paths: list[Path]
    ):
        """PDF upload with academic analysis should check references."""
        if not sample_pdf_paths:
            pytest.skip("No PDF files in test-data directory")

        pdf_path = sample_pdf_paths[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/files",
                files={"files": (pdf_path.name, f, "application/pdf")},
                data={"analysis_type": "academic"},
            )

        data = response.json()
        file_result = data["results"]["individual_files"][0]
        analysis = file_result["analysis"]

        # Academic analysis should include references and integrity
        assert "references" in analysis
        assert "integrity" in analysis

    @pytest.mark.pdf
    def test_upload_pdf_with_extracted_text(self, client: TestClient, sample_pdf_paths: list[Path]):
        """PDF upload with include_extracted_text should return full text."""
        if not sample_pdf_paths:
            pytest.skip("No PDF files in test-data directory")

        pdf_path = sample_pdf_paths[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/files",
                files={"files": (pdf_path.name, f, "application/pdf")},
                data={"include_extracted_text": "true"},
            )

        data = response.json()
        file_result = data["results"]["individual_files"][0]

        assert "extracted_text" in file_result
        extracted = file_result["extracted_text"]

        assert "full_text" in extracted
        assert "pages" in extracted
        assert "total_pages" in extracted
        assert len(extracted["full_text"]) > 0

    @pytest.mark.pdf
    @pytest.mark.slow
    def test_upload_large_pdf(self, client: TestClient, sample_pdf_paths: list[Path]):
        """PDFs over 10MB but under 50MB should upload successfully."""
        # Find a PDF between 10MB and 50MB
        large_pdfs = [
            p for p in sample_pdf_paths if 10 * 1024 * 1024 < p.stat().st_size < 50 * 1024 * 1024
        ]

        if not large_pdfs:
            pytest.skip("No PDF files between 10MB and 50MB in test-data")

        pdf_path = large_pdfs[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/files",
                files={"files": (pdf_path.name, f, "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["files_processed"] == 1

    @pytest.mark.pdf
    @pytest.mark.slow
    def test_upload_multiple_pdfs(self, client: TestClient, sample_pdf_paths: list[Path]):
        """Uploading multiple PDFs should process all files (within size limits)."""
        # Filter to only PDFs under 50MB (the default MAX_FILE_SIZE)
        max_size = 50 * 1024 * 1024  # 50MB
        small_pdfs = [p for p in sample_pdf_paths if p.stat().st_size < max_size]

        if len(small_pdfs) < 2:
            pytest.skip("Need at least 2 PDF files under 50MB for this test")

        files = []
        file_handles = []
        for pdf_path in small_pdfs[:2]:
            f = open(pdf_path, "rb")
            file_handles.append(f)
            files.append(("files", (pdf_path.name, f, "application/pdf")))

        try:
            response = client.post("/files", files=files)
        finally:
            for f in file_handles:
                f.close()

        data = response.json()
        assert data["files_processed"] == 2
        assert len(data["results"]["individual_files"]) == 2


class TestFileUploadValidation:
    """Tests for file upload validation and error handling."""

    def test_upload_no_files_returns_400(self, client: TestClient):
        """Uploading with no files should return 400."""
        response = client.post("/files", files={})
        assert response.status_code == 422  # FastAPI validation error

    def test_upload_unsupported_type_returns_error(self, client: TestClient):
        """Uploading unsupported file type should return an error."""
        response = client.post(
            "/files",
            files={"files": ("test.exe", b"fake content", "application/x-msdownload")},
        )
        # API may return 400 or 500 depending on where validation occurs
        assert response.status_code in [400, 500]
        assert "Unsupported file type" in response.json()["detail"]

    def test_invalid_analysis_type_returns_400(
        self, client: TestClient, sample_pdf_paths: list[Path]
    ):
        """Invalid analysis_type should return 400."""
        if not sample_pdf_paths:
            pytest.skip("No PDF files in test-data directory")

        pdf_path = sample_pdf_paths[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/files",
                files={"files": (pdf_path.name, f, "application/pdf")},
                data={"analysis_type": "invalid_type"},
            )

        assert response.status_code == 400
        assert "analysis_type" in response.json()["detail"]


class TestFileAnalysisTypes:
    """Tests for different analysis types."""

    @pytest.mark.pdf
    @pytest.mark.parametrize("analysis_type", ["full", "text", "academic"])
    def test_all_analysis_types_work(
        self,
        client: TestClient,
        sample_pdf_paths: list[Path],
        analysis_type: str,
    ):
        """All analysis types should work without errors."""
        if not sample_pdf_paths:
            pytest.skip("No PDF files in test-data directory")

        pdf_path = sample_pdf_paths[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/files",
                files={"files": (pdf_path.name, f, "application/pdf")},
                data={"analysis_type": analysis_type},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["analysis_type"] == analysis_type


class TestMetadataInference:
    """Tests for the /files/infer-metadata endpoint."""

    @pytest.mark.pdf
    def test_infer_metadata_returns_200(self, client: TestClient, sample_pdf_paths: list[Path]):
        """Metadata inference should return 200 OK."""
        if not sample_pdf_paths:
            pytest.skip("No PDF files in test-data directory")

        pdf_path = sample_pdf_paths[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/files/infer-metadata",
                files={"file": (pdf_path.name, f, "application/pdf")},
            )

        assert response.status_code == 200

    @pytest.mark.pdf
    def test_infer_metadata_returns_expected_fields(
        self, client: TestClient, sample_pdf_paths: list[Path]
    ):
        """Metadata inference should return expected fields."""
        if not sample_pdf_paths:
            pytest.skip("No PDF files in test-data directory")

        pdf_path = sample_pdf_paths[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/files/infer-metadata",
                files={"file": (pdf_path.name, f, "application/pdf")},
            )

        data = response.json()

        # Check all expected fields are present
        assert "probable_year" in data
        assert "probable_company" in data
        assert "probable_industry" in data
        assert "document_type" in data
        assert "confidence_scores" in data
        assert "extraction_notes" in data

    @pytest.mark.pdf
    def test_infer_metadata_confidence_scores(
        self, client: TestClient, sample_pdf_paths: list[Path]
    ):
        """Confidence scores should be between 0 and 1."""
        if not sample_pdf_paths:
            pytest.skip("No PDF files in test-data directory")

        pdf_path = sample_pdf_paths[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/files/infer-metadata",
                files={"file": (pdf_path.name, f, "application/pdf")},
            )

        data = response.json()
        scores = data["confidence_scores"]

        for key, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Score for {key} out of range: {score}"


# === Parameterized Tests for Multiple File Types ===
# Uncomment and extend when adding DOCX support


# @pytest.mark.parametrize("file_path,content_type", [
#     pytest.param(p, t, marks=pytest.mark.pdf) if t == "application/pdf"
#     else pytest.param(p, t, marks=pytest.mark.docx)
#     for p, t in conftest.TEST_FILES
# ])
# def test_upload_various_file_types(
#     client: TestClient, file_path: str, content_type: str
# ):
#     """Test that various file types can be uploaded and processed."""
#     path = Path(file_path)
#     with open(path, "rb") as f:
#         response = client.post(
#             "/files",
#             files={"files": (path.name, f, content_type)},
#         )
#     assert response.status_code == 200


class TestDocxFileUpload:
    """Smoke tests for DOCX file upload and text extraction."""

    @pytest.mark.docx
    def test_upload_docx_returns_200(self, client: TestClient, sample_docx_paths: list[Path]):
        """Uploading a DOCX should return 200 OK."""
        if not sample_docx_paths:
            pytest.skip("No DOCX files in test-data directory")

        docx_path = sample_docx_paths[0]
        with open(docx_path, "rb") as f:
            response = client.post(
                "/files",
                files={"files": (docx_path.name, f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )

        assert response.status_code == 200

    @pytest.mark.docx
    def test_upload_docx_with_extracted_text(self, client: TestClient, sample_docx_paths: list[Path]):
        """DOCX upload with include_extracted_text should return non-empty text."""
        if not sample_docx_paths:
            pytest.skip("No DOCX files in test-data directory")

        docx_path = sample_docx_paths[0]
        with open(docx_path, "rb") as f:
            response = client.post(
                "/files",
                files={"files": (docx_path.name, f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                data={"include_extracted_text": "true"},
            )

        assert response.status_code == 200
        data = response.json()
        file_result = data["results"]["individual_files"][0]

        assert "extracted_text" in file_result
        extracted = file_result["extracted_text"]
        assert "full_text" in extracted
        assert len(extracted["full_text"]) > 0
        assert extracted["total_pages"] == 1
        assert len(extracted["pages"]) == 1
        assert extracted["pages"][0]["page_number"] == 1
