"""Tests for semantic analysis endpoints.

Marked `slow` at the file level: these endpoints load real ML models
(sentence-transformers for embeddings + transformers for sentiment), and
combining them in one Python process intermittently segfaults on macOS.
Run with `pytest -m slow` when exercising real model behaviour, e.g.
before a release.
"""

import pytest
from fastapi.testclient import TestClient

from document_analyser.api import app

pytestmark = pytest.mark.slow

client = TestClient(app)

SAMPLE_TEXT = """
INTRODUCTION

This document covers teaching and research activities.

TEACHING ACTIVITIES

I taught three courses this semester focusing on machine learning.
The curriculum development was intensive and rewarding.
Students provided excellent feedback about the learning outcomes.

RESEARCH PROJECTS

My research on climate change modeling produced two publications.
I also secured a grant for sustainable agriculture studies.
The findings were positive and contributed to the field.
"""

DOMAINS = ["Teaching", "Research", "Service", "Administration"]


class TestDomainMapping:
    """Tests for domain mapping endpoint."""

    def test_domain_mapping_success(self) -> None:
        """Test successful domain mapping."""
        response = client.post(
            "/semantic/domain-mapping",
            json={"text": SAMPLE_TEXT, "domains": DOMAINS}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_sections"] > 0
        assert len(data["mappings"]) > 0
        assert "domain_distribution" in data
        assert "average_confidence" in data

    def test_domain_mapping_empty_text(self) -> None:
        """Test domain mapping with empty text."""
        response = client.post(
            "/semantic/domain-mapping",
            json={"text": "", "domains": DOMAINS}
        )

        assert response.status_code == 400

    def test_domain_mapping_insufficient_domains(self) -> None:
        """Test domain mapping with only one domain."""
        response = client.post(
            "/semantic/domain-mapping",
            json={"text": SAMPLE_TEXT, "domains": ["Teaching"]}
        )

        assert response.status_code == 400


class TestStructuralMismatch:
    """Tests for structural mismatch detection endpoint."""

    def test_structural_mismatch_empty_text(self) -> None:
        """Test structural mismatch with empty text."""
        response = client.post(
            "/semantic/structural-mismatch",
            json={
                "text": "",
                "domains": DOMAINS,
                "threshold": 0.3
            }
        )

        assert response.status_code == 400

    def test_structural_mismatch_invalid_threshold(self) -> None:
        """Test structural mismatch with invalid threshold."""
        response = client.post(
            "/semantic/structural-mismatch",
            json={
                "text": SAMPLE_TEXT,
                "domains": DOMAINS,
                "threshold": 1.5
            }
        )

        assert response.status_code == 400


class TestSentimentAnalysis:
    """Tests for sentiment analysis endpoint."""

    def test_sentiment_analysis_success(self) -> None:
        """Test successful sentiment analysis."""
        response = client.post(
            "/semantic/sentiment",
            json={"text": SAMPLE_TEXT}
        )

        assert response.status_code == 200
        data = response.json()

        assert "document_sentiment" in data
        assert "sections" in data
        assert len(data["sections"]) > 0
        assert "total_sentences" in data
        assert "total_paragraphs" in data
        assert "sentiment_distribution" in data

    def test_sentiment_analysis_empty_text(self) -> None:
        """Test sentiment analysis with empty text."""
        response = client.post(
            "/semantic/sentiment",
            json={"text": ""}
        )

        assert response.status_code == 400

    def test_sentiment_analysis_sentiment_scores(self) -> None:
        """Deliberately positive text should produce a positive compound score."""
        positive_text = "This is excellent! I love this. Great work!"

        response = client.post(
            "/semantic/sentiment",
            json={"text": positive_text}
        )

        assert response.status_code == 200
        data = response.json()

        # Check document sentiment structure
        doc_sentiment = data["document_sentiment"]
        assert "positive" in doc_sentiment
        assert "negative" in doc_sentiment
        assert "neutral" in doc_sentiment
        assert "compound" in doc_sentiment
        assert isinstance(doc_sentiment["positive"], (int, float))
        assert isinstance(doc_sentiment["negative"], (int, float))

        # The net polarity for unambiguously positive text must be positive
        # (compound is in the range -1.0 to 1.0; expect > 0 here).
        assert doc_sentiment["compound"] > 0, (
            f"Expected positive compound score for positive text, got {doc_sentiment['compound']}"
        )
