"""Structural mismatch analyzer for detecting thematic dislocation of sentences."""

from typing import Any, Literal

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    SentenceTransformer = None
    np = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError:
    nltk = None
    sent_tokenize = None

from document_analyser.models.schemas import (
    SentenceDislocation,
    StructuralMismatchResponse,
)


class StructuralMismatchAnalyzer:
    """Detect thematic dislocation of sentences within sections."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize with sentence-transformers model."""
        self.model_name = model_name
        self.model = None

        if SentenceTransformer:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None

    def analyze(
        self,
        text: str,
        domains: list[str],
        threshold: float = 0.3,
    ) -> StructuralMismatchResponse:
        """
        Detect sentences thematically dislocated from their parent sections.

        Args:
            text: Document text
            domains: List of domain labels
            threshold: Dislocation score threshold (0.0-1.0)

        Returns:
            StructuralMismatchResponse with detected dislocations
        """
        if not text.strip() or not self.model or not domains:
            return StructuralMismatchResponse(
                total_sentences_analyzed=0, total_sections=0
            )

        # 1. Detect sections
        sections = self._detect_sections(text)

        # 2. Map sections to domains
        section_domain_map = self._map_sections_to_domains(sections, domains)

        # 3. Analyze each sentence within each section
        dislocations: list[SentenceDislocation] = []
        total_sentences = 0

        for section_idx, section in enumerate(sections):
            section_domain = section_domain_map[section_idx]["domain"]
            section_text = section["text"]

            # Tokenize sentences
            sentences = self._get_sentences(section_text)
            total_sentences += len(sentences)

            # Check each sentence against section domain
            for sent_idx, sentence in enumerate(sentences):
                if len(sentence.strip()) < 20:  # Skip very short sentences
                    continue

                # Map sentence to domain
                sent_domain_map = self._map_sentence_to_domain(sentence, domains)
                sent_domain = sent_domain_map["domain"]
                sent_score = sent_domain_map["score"]

                # Calculate dislocation score
                if sent_domain != section_domain:
                    section_score = section_domain_map[section_idx]["score"]
                    dislocation_score = abs(sent_score - section_score)

                    if dislocation_score > threshold:
                        severity = self._calculate_severity(dislocation_score)

                        dislocations.append(SentenceDislocation(
                            sentence_index=total_sentences - len(sentences) + sent_idx,
                            sentence_text=sentence[:150],  # Truncate
                            sentence_domain=sent_domain,
                            parent_section_index=section_idx,
                            parent_section_domain=section_domain,
                            dislocation_score=dislocation_score,
                            severity=severity
                        ))

        # 4. Calculate overall coherence score
        coherence_score = 1.0 - (len(dislocations) / max(total_sentences, 1))

        # 5. Generate recommendations
        recommendations = self._generate_recommendations(dislocations, sections)

        # 6. Count high severity
        high_count = sum(1 for d in dislocations if d.severity == "high")

        return StructuralMismatchResponse(
            total_sentences_analyzed=total_sentences,
            total_sections=len(sections),
            dislocations=dislocations,
            overall_coherence_score=float(coherence_score),
            highly_dislocated_count=high_count,
            recommendations=recommendations
        )

    def _detect_sections(self, text: str) -> list[dict[str, str]]:
        """Detect sections using heuristic patterns."""
        paragraphs = text.split("\n\n")
        sections: list[dict[str, str]] = []

        section_keywords = [
            "introduction", "background", "methodology", "methods", "results",
            "discussion", "conclusion", "abstract", "summary", "findings",
            "recommendations", "analysis", "overview", "scope"
        ]

        current_section: dict[str, str] = {"header": "Introduction", "text": ""}

        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue

            lines = para.split("\n")
            first_line = lines[0].strip() if lines else ""

            is_header = False

            if len(first_line) > 0:
                upper_count = sum(1 for c in first_line if c.isupper())
                upper_ratio = upper_count / len(first_line)
                if upper_ratio > 0.5 and len(first_line) < 100:
                    is_header = True

            if len(first_line) < 60 and any(kw in first_line.lower() for kw in section_keywords):
                is_header = True

            if is_header and i > 0:
                if current_section["text"].strip():
                    sections.append(current_section)
                rest_lines = "\n".join(lines[1:]) if len(lines) > 1 else ""
                current_section = {"header": first_line, "text": rest_lines}
            else:
                current_section["text"] += "\n\n" + para

        if current_section["text"].strip():
            sections.append(current_section)

        if not sections:
            sections = [{"header": "Document", "text": text}]

        return sections

    def _map_sections_to_domains(
        self, sections: list[dict[str, str]], domains: list[str]
    ) -> dict[int, dict[str, Any]]:
        """Map each section to its primary domain."""
        if not self.model or np is None:
            return {}

        section_embeddings = self.model.encode([s["text"] for s in sections])
        domain_embeddings = self.model.encode(domains)

        mapping: dict[int, dict[str, Any]] = {}
        for i, section_emb in enumerate(section_embeddings):
            similarities = self._cosine_similarities(section_emb, domain_embeddings)
            best_idx = int(np.argmax(similarities))
            mapping[i] = {
                "domain": domains[best_idx],
                "score": float(similarities[best_idx])
            }

        return mapping

    def _map_sentence_to_domain(
        self, sentence: str, domains: list[str]
    ) -> dict[str, Any]:
        """Map a sentence to its primary domain."""
        if not self.model or np is None:
            return {"domain": domains[0], "score": 0.5}

        sent_emb = self.model.encode([sentence])[0]
        domain_embeddings = self.model.encode(domains)
        similarities = self._cosine_similarities(sent_emb, domain_embeddings)
        best_idx = int(np.argmax(similarities))

        return {
            "domain": domains[best_idx],
            "score": float(similarities[best_idx])
        }

    def _get_sentences(self, text: str) -> list[str]:
        """Get sentences from text."""
        if sent_tokenize:
            try:
                return sent_tokenize(text)
            except Exception:
                pass

        # Fallback: split by period
        return [s.strip() for s in text.split('.') if s.strip()]

    def _cosine_similarities(
        self, vec1: Any, vec2_matrix: Any
    ) -> Any:
        """Calculate cosine similarity between vectors."""
        if np is None:
            return []

        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norms = vec2_matrix / np.linalg.norm(vec2_matrix, axis=1, keepdims=True)
        return np.dot(vec2_norms, vec1_norm)

    def _calculate_severity(self, score: float) -> Literal["low", "medium", "high"]:
        """Determine severity of dislocation."""
        if score > 0.6:
            return "high"
        elif score > 0.4:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(
        self,
        dislocations: list[SentenceDislocation],
        sections: list[dict[str, str]],
    ) -> list[str]:
        """Generate recommendations based on dislocations."""
        if not dislocations:
            return [
                "Document structure is coherent with no major thematic "
                "dislocations detected."
            ]

        recommendations: list[str] = []
        high_count = sum(1 for d in dislocations if d.severity == "high")

        if high_count > 0:
            recommendations.append(
                f"Found {high_count} highly dislocated sentences. "
                f"Consider reorganizing content to improve thematic coherence."
            )

        # Group by section
        section_issues: dict[int, int] = {}
        for d in dislocations:
            if d.parent_section_index not in section_issues:
                section_issues[d.parent_section_index] = 0
            section_issues[d.parent_section_index] += 1

        for section_idx, count in section_issues.items():
            if count > 3:
                recommendations.append(
                    f"Section {section_idx} has {count} dislocated sentences. "
                    f"Review this section for structural coherence."
                )

        return recommendations
