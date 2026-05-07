"""
Tests for text analysis endpoints.

This module tests the /text endpoint for text analysis features.
"""

import pytest
from fastapi.testclient import TestClient


class TestTextAnalysisEndpoint:
    """Tests for the POST /text endpoint."""

    def test_text_analysis_returns_200(self, client: TestClient, sample_text: str):
        """Text analysis should return 200 OK."""
        response = client.post("/text", json={"text": sample_text})
        assert response.status_code == 200

    def test_text_analysis_returns_correct_structure(self, client: TestClient, sample_text: str):
        """Text analysis response should have expected structure."""
        response = client.post("/text", json={"text": sample_text})
        data = response.json()

        # Check top-level structure
        assert data["service"] == "DocumentLens"
        assert data["version"] == "1.0.0"
        assert data["content_type"] == "text"
        assert "analysis" in data
        assert "processing_time" in data

    def test_text_analysis_returns_text_metrics(self, client: TestClient, sample_text: str):
        """Text analysis should return text metrics."""
        response = client.post("/text", json={"text": sample_text})
        data = response.json()

        metrics = data["analysis"]["text_metrics"]
        assert "word_count" in metrics
        assert "sentence_count" in metrics
        assert "paragraph_count" in metrics
        assert "avg_words_per_sentence" in metrics

        # Values should be positive
        assert metrics["word_count"] > 0
        assert metrics["sentence_count"] > 0

    def test_text_analysis_returns_readability(self, client: TestClient, sample_text: str):
        """Text analysis should return readability scores."""
        response = client.post("/text", json={"text": sample_text})
        data = response.json()

        readability = data["analysis"]["readability"]
        assert "flesch_score" in readability
        assert "flesch_kincaid_grade" in readability
        assert "interpretation" in readability

    def test_text_analysis_returns_writing_quality(self, client: TestClient, sample_text: str):
        """Text analysis should return writing quality metrics."""
        response = client.post("/text", json={"text": sample_text})
        data = response.json()

        quality = data["analysis"]["writing_quality"]
        assert "passive_voice_percentage" in quality
        assert "sentence_variety" in quality
        assert "academic_tone" in quality
        assert "transition_words" in quality
        assert "hedging_language" in quality

    def test_text_analysis_returns_word_analysis(self, client: TestClient, sample_text: str):
        """Text analysis should return word analysis."""
        response = client.post("/text", json={"text": sample_text})
        data = response.json()

        word_analysis = data["analysis"]["word_analysis"]
        assert "unique_words" in word_analysis
        assert "vocabulary_richness" in word_analysis
        assert "top_words" in word_analysis
        assert "bigrams" in word_analysis
        assert "trigrams" in word_analysis

    def test_text_analysis_returns_ner(self, client: TestClient, sample_text: str):
        """Text analysis should return named entity recognition results."""
        response = client.post("/text", json={"text": sample_text})
        data = response.json()

        assert "ner" in data["analysis"]


class TestTextAnalysisValidation:
    """Tests for text analysis input validation."""

    def test_empty_text_returns_400(self, client: TestClient, empty_text: str):
        """Empty text should return 400 error."""
        response = client.post("/text", json={"text": empty_text})
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_whitespace_only_returns_400(self, client: TestClient):
        """Whitespace-only text should return 400 error."""
        response = client.post("/text", json={"text": "   \n\t  "})
        assert response.status_code == 400

    def test_missing_text_field_returns_422(self, client: TestClient):
        """Missing text field should return 422 validation error."""
        response = client.post("/text", json={})
        assert response.status_code == 422

    def test_invalid_json_returns_422(self, client: TestClient):
        """Invalid JSON should return 422 error."""
        response = client.post(
            "/text",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


class TestTextAnalysisMetrics:
    """Tests for correctness of text analysis metrics."""

    def test_word_count_accuracy(self, client: TestClient):
        """Word count should be reasonably accurate."""
        text = "This is a simple test sentence with exactly ten words here."
        response = client.post("/text", json={"text": text})
        data = response.json()

        # Allow some flexibility for tokenization differences
        word_count = data["analysis"]["text_metrics"]["word_count"]
        assert 9 <= word_count <= 12

    def test_sentence_count_accuracy(self, client: TestClient):
        """Sentence count should be reasonably accurate."""
        text = "First sentence. Second sentence. Third sentence."
        response = client.post("/text", json={"text": text})
        data = response.json()

        sentence_count = data["analysis"]["text_metrics"]["sentence_count"]
        assert sentence_count == 3

    def test_flesch_score_in_valid_range(self, client: TestClient, sample_text: str):
        """Flesch score should typically be between 0 and 100."""
        response = client.post("/text", json={"text": sample_text})
        data = response.json()

        flesch_score = data["analysis"]["readability"]["flesch_score"]
        # Flesch score can technically go below 0 or above 100 for extreme texts
        # but should typically be in a reasonable range
        assert -50 <= flesch_score <= 150

    def test_flesch_kincaid_grade_reasonable(self, client: TestClient, sample_text: str):
        """Flesch-Kincaid grade should be a reasonable value."""
        response = client.post("/text", json={"text": sample_text})
        data = response.json()

        grade = data["analysis"]["readability"]["flesch_kincaid_grade"]
        # Grade level should be between elementary and post-graduate
        assert 0 <= grade <= 25

    def test_percentages_in_valid_range(self, client: TestClient, sample_text: str):
        """Percentage values should be between 0 and 100."""
        response = client.post("/text", json={"text": sample_text})
        data = response.json()

        quality = data["analysis"]["writing_quality"]
        assert 0 <= quality["passive_voice_percentage"] <= 100


class TestTextInferMetadata:
    """Tests for the POST /text/infer-metadata endpoint."""

    def test_infer_metadata_returns_200(self, client: TestClient, sample_text: str):
        """Text metadata inference should return 200 OK."""
        response = client.post(
            "/text/infer-metadata",
            data={"text": sample_text},
        )
        assert response.status_code == 200

    def test_infer_metadata_returns_expected_fields(self, client: TestClient, sample_text: str):
        """Text metadata inference should return expected fields."""
        response = client.post(
            "/text/infer-metadata",
            data={"text": sample_text},
        )
        data = response.json()

        assert "probable_year" in data
        assert "probable_company" in data
        assert "probable_industry" in data
        assert "document_type" in data
        assert "confidence_scores" in data

    def test_infer_metadata_with_filename(self, client: TestClient, sample_text: str):
        """Text metadata inference should accept optional filename."""
        response = client.post(
            "/text/infer-metadata",
            data={
                "text": sample_text,
                "filename": "annual-report-2024.pdf",
            },
        )
        assert response.status_code == 200

    def test_infer_metadata_short_text_returns_400(self, client: TestClient):
        """Short text should return 400 error."""
        response = client.post(
            "/text/infer-metadata",
            data={"text": "Too short"},
        )
        assert response.status_code == 400
        assert "100 characters" in response.json()["detail"]
