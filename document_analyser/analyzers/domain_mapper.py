"""Domain mapping analyzer using semantic similarity with sentence-transformers."""

from typing import Any, Literal

try:
    from document_analyser.ml.onnx_models import OnnxEmbedder
    import numpy as np
except ImportError:
    OnnxEmbedder = None
    np = None

from document_analyser.models.schemas import DomainMapping, DomainMappingResponse


class DomainMapper:
    """Map document sections to user-defined domains using semantic similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize domain mapper with sentence-transformers model."""
        self.model_name = model_name
        self.model = None
        self._load_error: str | None = None

        if OnnxEmbedder is None:
            self._load_error = (
                "ONNX embedding backend is unavailable "
                "(onnxruntime / transformers not installed)."
            )
        else:
            try:
                self.model = OnnxEmbedder(model_name)
            except Exception as e:
                self._load_error = (
                    f"Failed to load ONNX embedding model '{model_name}': {e!s}. "
                    "This usually means missing model files, network issues during first load, "
                    "or memory pressure when many ML libraries are loaded in the same process."
                )

    def analyze(self, text: str, domains: list[str]) -> DomainMappingResponse:
        """
        Map sections to domains using cosine similarity.

        Args:
            text: Document text to analyze
            domains: List of domain labels to map sections to

        Returns:
            DomainMappingResponse with section-domain mappings

        Raises:
            RuntimeError: if the embedding model failed to load. Better to fail
                loudly than silently return total_sections=0 (a real prior bug
                that masked test pollution and made debugging painful).
        """
        if not text.strip() or not domains:
            return DomainMappingResponse(total_sections=0)
        if self.model is None:
            raise RuntimeError(
                f"DomainMapper unusable: {self._load_error or 'model not loaded'}"
            )

        # 1. Detect sections using heuristic patterns
        sections = self._detect_sections(text)

        # 2. Generate embeddings for sections and domains
        section_texts = [s["text"] for s in sections]
        section_embeddings = self.model.encode(section_texts)
        domain_embeddings = self.model.encode(domains)

        # 3. Calculate cosine similarities and build mappings
        mappings = []
        for i, (section, section_emb) in enumerate(zip(sections, section_embeddings)):
            similarities = self._cosine_similarities(section_emb, domain_embeddings)

            # Find primary domain (highest similarity)
            best_idx = int(np.argmax(similarities))
            primary_domain = domains[best_idx]
            best_score = float(similarities[best_idx])

            # Determine confidence
            confidence = self._calculate_confidence(best_score)

            mappings.append(DomainMapping(
                section_text=section["text"][:200],  # Truncate for response
                section_index=i,
                primary_domain=primary_domain,
                similarity_score=best_score,
                all_domain_scores={d: float(s) for d, s in zip(domains, similarities)},
                confidence=confidence
            ))

        # 4. Calculate domain distribution
        domain_distribution: dict[str, int] = {}
        for m in mappings:
            domain_distribution[m.primary_domain] = domain_distribution.get(m.primary_domain, 0) + 1

        # 5. Calculate average confidence
        avg_conf = float(np.mean([m.similarity_score for m in mappings])) if mappings else 0.0

        return DomainMappingResponse(
            total_sections=len(sections),
            domains_analyzed=domains,
            mappings=mappings,
            domain_distribution=domain_distribution,
            average_confidence=avg_conf
        )

    def _detect_sections(self, text: str) -> list[dict[str, str]]:
        """
        Detect sections using heuristic patterns.

        Looks for:
        1. ALL CAPS lines (>50% uppercase) as headers
        2. Short lines (<60 chars) containing section keywords
        """
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

            # Check if this is a section header
            is_header = False

            # Pattern 1: ALL CAPS (at least 50% uppercase)
            if len(first_line) > 0:
                upper_count = sum(1 for c in first_line if c.isupper())
                upper_ratio = upper_count / len(first_line)
                if upper_ratio > 0.5 and len(first_line) < 100:
                    is_header = True

            # Pattern 2: Short line with keywords
            if len(first_line) < 60 and any(kw in first_line.lower() for kw in section_keywords):
                is_header = True

            if is_header and i > 0:
                # Save previous section
                if current_section["text"].strip():
                    sections.append(current_section)
                # Start new section
                rest_lines = "\n".join(lines[1:]) if len(lines) > 1 else ""
                current_section = {"header": first_line, "text": rest_lines}
            else:
                # Add to current section
                current_section["text"] += "\n\n" + para

        # Add final section
        if current_section["text"].strip():
            sections.append(current_section)

        # If no sections detected, treat entire text as one section
        if not sections:
            sections = [{"header": "Document", "text": text}]

        return sections

    def _cosine_similarities(
        self, vec1: Any, vec2_matrix: Any
    ) -> Any:
        """Calculate cosine similarity between vec1 and each row in vec2_matrix."""
        if np is None:
            return []

        # Normalize
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norms = vec2_matrix / np.linalg.norm(vec2_matrix, axis=1, keepdims=True)

        # Dot product
        return np.dot(vec2_norms, vec1_norm)

    def _calculate_confidence(
        self, best_score: float
    ) -> Literal["high", "medium", "low"]:
        """Determine confidence level based on similarity score."""
        if best_score > 0.7:
            return "high"
        elif best_score > 0.5:
            return "medium"
        else:
            return "low"
