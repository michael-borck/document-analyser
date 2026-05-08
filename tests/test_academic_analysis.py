"""
Tests for academic analysis endpoints.

This module tests the /academic endpoint for academic document analysis.
"""

from importlib.metadata import version as _pkg_version

import pytest
from fastapi.testclient import TestClient

EXPECTED_VERSION = _pkg_version("document-analyser")


class TestAcademicAnalysisEndpoint:
    """Tests for the POST /academic endpoint."""

    def test_academic_analysis_returns_200(self, client: TestClient, sample_academic_text: str):
        """Academic analysis should return 200 OK."""
        response = client.post("/academic", json={"text": sample_academic_text})
        assert response.status_code == 200

    def test_academic_analysis_returns_correct_structure(
        self, client: TestClient, sample_academic_text: str
    ):
        """Academic analysis response should have expected structure."""
        response = client.post("/academic", json={"text": sample_academic_text})
        data = response.json()

        # Check top-level structure
        assert data["service"] == "DocumentAnalyser"
        assert data["version"] == EXPECTED_VERSION
        assert data["content_type"] == "academic"
        assert "analysis" in data
        assert "processing_time" in data


class TestAcademicAnalysisValidation:
    """Tests for academic analysis input validation."""

    def test_empty_text_returns_400(self, client: TestClient, empty_text: str):
        """Empty text should return 400 error."""
        response = client.post("/academic", json={"text": empty_text})
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_whitespace_only_returns_400(self, client: TestClient):
        """Whitespace-only text should return 400 error."""
        response = client.post("/academic", json={"text": "   \n\t  "})
        assert response.status_code == 400

    def test_missing_text_field_returns_422(self, client: TestClient):
        """Missing text field should return 422 validation error."""
        response = client.post("/academic", json={})
        assert response.status_code == 422


class TestAcademicAnalysisOptions:
    """Tests for academic analysis configuration options."""

    @pytest.mark.parametrize("citation_style", ["auto", "apa", "mla", "chicago"])
    def test_citation_style_options(
        self, client: TestClient, sample_academic_text: str, citation_style: str
    ):
        """All citation style options should be accepted and produce a real analysis."""
        response = client.post(
            "/academic",
            json={
                "text": sample_academic_text,
                "citation_style": citation_style,
            },
        )
        assert response.status_code == 200

        # Confirm the request was actually processed (references analysis ran)
        data = response.json()
        references = data["analysis"]["references"]
        assert "total" in references
        assert isinstance(references["total"], int)

    def test_disable_url_check(self, client: TestClient, sample_academic_text: str):
        """URL checking can be disabled."""
        response = client.post(
            "/academic",
            json={
                "text": sample_academic_text,
                "check_urls": False,
            },
        )
        assert response.status_code == 200

    def test_disable_doi_check(self, client: TestClient, sample_academic_text: str):
        """DOI checking can be disabled."""
        response = client.post(
            "/academic",
            json={
                "text": sample_academic_text,
                "check_doi": False,
            },
        )
        assert response.status_code == 200

    def test_enable_plagiarism_check(self, client: TestClient, sample_academic_text: str):
        """Plagiarism checking can be enabled."""
        response = client.post(
            "/academic",
            json={
                "text": sample_academic_text,
                "check_plagiarism": True,
            },
        )
        assert response.status_code == 200


class TestAcademicIntegrityScoring:
    """Tests for integrity scoring features."""

    def test_integrity_score_in_valid_range(self, client: TestClient, sample_academic_text: str):
        """Integrity score should be between 0 and 100."""
        response = client.post("/academic", json={"text": sample_academic_text})
        data = response.json()

        score = data["analysis"]["integrity"]["integrity_score"]
        assert 0 <= score <= 100

    def test_ai_risk_level_valid_values(self, client: TestClient, sample_academic_text: str):
        """AI risk level should be a valid value."""
        response = client.post("/academic", json={"text": sample_academic_text})
        data = response.json()

        risk_level = data["analysis"]["integrity"]["ai_risk_level"]
        valid_levels = ["low", "medium", "high", "very_high"]
        assert risk_level in valid_levels

    def test_ai_confidence_in_valid_range(self, client: TestClient, sample_academic_text: str):
        """AI confidence should be between 0 and 1."""
        response = client.post("/academic", json={"text": sample_academic_text})
        data = response.json()

        confidence = data["analysis"]["integrity"]["ai_confidence"]
        assert 0.0 <= confidence <= 1.0


class TestReferenceExtraction:
    """Tests for reference extraction functionality."""

    def test_extracts_references_from_academic_text(
        self, client: TestClient, sample_academic_text: str
    ):
        """Should extract references from academic text."""
        response = client.post("/academic", json={"text": sample_academic_text})
        data = response.json()

        # The sample academic text contains references
        references = data["analysis"]["references"]
        assert references["total"] > 0

    def test_detects_citation_styles(self, client: TestClient, sample_academic_text: str):
        """Should detect citation styles in text."""
        response = client.post("/academic", json={"text": sample_academic_text})
        data = response.json()

        styles_found = data["analysis"]["citations"]["styles_found"]
        assert isinstance(styles_found, list)

    def test_handles_text_without_references(self, client: TestClient):
        """Should handle text without any references gracefully."""
        text = """
        This is a simple document with no academic references or citations.
        It discusses various topics but does not cite any sources.
        The writing is informal and conversational in nature.
        """
        response = client.post("/academic", json={"text": text})
        assert response.status_code == 200

        data = response.json()
        references = data["analysis"]["references"]
        # Should return 0 or handle gracefully
        assert references["total"] >= 0
