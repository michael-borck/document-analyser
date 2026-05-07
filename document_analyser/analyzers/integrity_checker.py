"""
Integrity checker for detecting AI patterns and content authenticity issues
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from document_analyser.models.schemas import SuspiciousPatterns


class IntegrityChecker:
    """Detects AI patterns, suspicious content, and integrity issues in text"""

    def __init__(self) -> None:
        """Initialize the integrity checker with AI pattern data"""
        self.patterns_file = Path(__file__).parent.parent / "data" / "ai_patterns.json"
        self.ai_patterns = self._load_patterns()

    def _load_patterns(self) -> dict[str, Any]:
        """Load AI detection patterns from JSON file"""
        try:
            with self.patterns_file.open(encoding='utf-8') as f:
                data: dict[str, Any] = json.load(f)
                return data
        except FileNotFoundError:
            # Fallback to minimal patterns if file not found
            fallback: dict[str, Any] = {
                "patterns": {
                    "ai_phrases": [],
                    "ai_verbs": [],
                    "ai_adjectives": [],
                    "llm_artifacts": []
                },
                "thresholds": {
                    "ai_word_frequency": {"high": 0.15, "medium": 0.08, "low": 0.03}
                }
            }
            return fallback

    def detect_patterns(self, text: str, references: list, documents: list[str] | None = None) -> SuspiciousPatterns:
        """
        Detect suspicious patterns including AI-generated content indicators

        Args:
            text: The text to analyze
            references: List of references/citations
            documents: Optional list of other documents for comparison

        Returns:
            SuspiciousPatterns with detected issues
        """
        # Normalize text for analysis
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        total_words = len(words)

        # Detect AI patterns
        ai_indicators = self._detect_ai_patterns(text, text_lower, words, total_words)

        # Detect self-plagiarism if documents provided
        self_plagiarism = self._detect_self_plagiarism(text, documents) if documents else []

        # Detect citation anomalies
        citation_anomalies = self._detect_citation_anomalies(text, references)

        # Detect style inconsistencies
        style_inconsistencies = self._detect_style_inconsistencies(text)

        # Calculate overall integrity score
        integrity_score = self._calculate_integrity_score(
            ai_indicators,
            self_plagiarism,
            citation_anomalies,
            style_inconsistencies
        )

        # Compile all issues
        all_issues = []

        # Add AI detection results
        if ai_indicators['risk_level'] in ['medium', 'high']:
            all_issues.append(f"AI content detected (confidence: {ai_indicators['confidence']:.1%})")
            if ai_indicators['llm_artifacts']:
                all_issues.append(f"LLM artifacts found: {', '.join(ai_indicators['llm_artifacts'][:3])}")

        all_issues.extend(self_plagiarism)
        all_issues.extend(citation_anomalies)
        all_issues.extend(style_inconsistencies)

        return SuspiciousPatterns(
            self_plagiarism=self_plagiarism,
            citation_anomalies=citation_anomalies,
            style_inconsistencies=style_inconsistencies,
            ai_indicators=ai_indicators,
            integrity_score=integrity_score,
            all_issues=all_issues
        )

    def _detect_ai_patterns(self, text: str, text_lower: str, words: list[str], total_words: int) -> dict[str, Any]:
        """Detect AI-generated content patterns"""
        if total_words == 0:
            return {
                'risk_level': 'low',
                'confidence': 0.0,
                'detected_patterns': {}
            }

        patterns = self.ai_patterns.get('patterns', {})
        thresholds = self.ai_patterns.get('thresholds', {})
        weights = self.ai_patterns.get('weights', {
            'ai_words': 0.25,
            'ai_phrases': 0.3,
            'llm_artifacts': 0.35,
            'structural_markers': 0.1
        })

        results: dict[str, Any] = {
            'ai_word_frequency': 0.0,
            'ai_phrase_count': 0,
            'llm_artifacts': [],
            'em_dash_frequency': 0.0,
            'bullet_ratio': 0.0,
            'detected_ai_words': [],
            'detected_ai_phrases': []
        }

        # Check AI words (verbs and adjectives)
        ai_words = set(patterns.get('ai_verbs', []) + patterns.get('ai_adjectives', []))
        ai_word_count = sum(1 for word in words if word in ai_words)
        results['ai_word_frequency'] = ai_word_count / total_words if total_words > 0 else 0

        # Find which AI words were used
        word_counter = Counter(words)
        for word in ai_words:
            if word in word_counter:
                results['detected_ai_words'].append((word, word_counter[word]))
        results['detected_ai_words'].sort(key=lambda x: x[1], reverse=True)

        # Check AI phrases
        for phrase in patterns.get('ai_phrases', []):
            if phrase in text_lower:
                results['ai_phrase_count'] += text_lower.count(phrase)
                results['detected_ai_phrases'].append(phrase)

        # Check for LLM artifacts
        for artifact in patterns.get('llm_artifacts', []):
            if artifact in text_lower:
                results['llm_artifacts'].append(artifact)

        # Check structural patterns
        structural = patterns.get('structural_patterns', {})

        # Em-dash frequency
        em_dash = structural.get('excessive_em_dash', '—')
        results['em_dash_frequency'] = text.count(em_dash) / total_words if total_words > 0 else 0

        # Bullet point ratio
        lines = text.split('\n')
        bullet_lines = 0
        for line in lines:
            for bullet in structural.get('bullet_indicators', ['•', '-', '*']):
                if line.strip().startswith(bullet):
                    bullet_lines += 1
                    break
        results['bullet_ratio'] = bullet_lines / len(lines) if lines else 0

        # Calculate overall AI confidence score
        score = 0.0

        # Word frequency component
        word_freq_thresholds = thresholds.get('ai_word_frequency', {})
        if results['ai_word_frequency'] >= word_freq_thresholds.get('high', 0.15):
            score += weights['ai_words'] * 1.0
        elif results['ai_word_frequency'] >= word_freq_thresholds.get('medium', 0.08):
            score += weights['ai_words'] * 0.6
        elif results['ai_word_frequency'] >= word_freq_thresholds.get('low', 0.03):
            score += weights['ai_words'] * 0.3

        # Phrase density component
        phrase_thresholds = thresholds.get('ai_phrase_density', {})
        if results['ai_phrase_count'] >= phrase_thresholds.get('high', 5):
            score += weights['ai_phrases'] * 1.0
        elif results['ai_phrase_count'] >= phrase_thresholds.get('medium', 3):
            score += weights['ai_phrases'] * 0.6
        elif results['ai_phrase_count'] >= phrase_thresholds.get('low', 1):
            score += weights['ai_phrases'] * 0.3

        # LLM artifacts component
        if len(results['llm_artifacts']) > 0:
            score += weights['llm_artifacts'] * min(1.0, len(results['llm_artifacts']) / 3)

        # Structural markers component
        structural_score = 0.0
        if results['em_dash_frequency'] >= thresholds.get('em_dash_frequency', {}).get('high', 0.02):
            structural_score += 0.5
        if results['bullet_ratio'] >= thresholds.get('bullet_ratio', {}).get('high', 0.3):
            structural_score += 0.5
        score += weights['structural_markers'] * structural_score

        # Determine risk level
        overall_thresholds = thresholds.get('overall_risk', {})
        if score >= overall_thresholds.get('high', 0.7):
            risk_level = 'high'
        elif score >= overall_thresholds.get('medium', 0.4):
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'risk_level': risk_level,
            'confidence': score,
            'detected_patterns': results,
            'llm_artifacts': results['llm_artifacts'],
            'disclaimer': self.ai_patterns.get('disclaimer', '')
        }

    def _detect_self_plagiarism(self, text: str, documents: list[str] | None) -> list[str]:
        """Detect potential self-plagiarism by comparing with other documents"""
        if not documents or len(documents) < 2:
            return []

        issues = []
        text_sentences = set(re.split(r'[.!?]+', text))
        text_sentences = {s.strip().lower() for s in text_sentences if len(s.strip()) > 20}

        for i, doc in enumerate(documents):
            if doc == text:
                continue

            doc_sentences = set(re.split(r'[.!?]+', doc))
            doc_sentences = {s.strip().lower() for s in doc_sentences if len(s.strip()) > 20}

            overlap = text_sentences.intersection(doc_sentences)
            if len(overlap) > 3:
                overlap_ratio = len(overlap) / len(text_sentences) if text_sentences else 0
                if overlap_ratio > 0.1:
                    issues.append(f"Significant text overlap ({overlap_ratio:.1%}) with document {i+1}")

        return issues

    def _detect_citation_anomalies(self, text: str, references: list) -> list[str]:
        """Detect issues with citations and references"""
        issues = []

        # Check for citation density
        sentences = re.split(r'[.!?]+', text)
        sentences_with_citations = 0

        # Common in-text citation patterns
        citation_patterns = [
            r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s*\d{4}\)',  # (Author, 2024) or (Author et al., 2024)
            r'[A-Z][a-z]+(?:\s+et\s+al\.)?\s+\(\d{4}\)',    # Author (2024) or Author et al. (2024)
            r'\[\d+\]',                                       # [1] style citations
            r'\[[\w\s,]+\d{4}\]'                             # [Author 2024] style
        ]

        for sentence in sentences:
            for pattern in citation_patterns:
                if re.search(pattern, sentence):
                    sentences_with_citations += 1
                    break

        citation_density = sentences_with_citations / len(sentences) if sentences else 0

        # Flag unusual citation patterns
        if len(sentences) > 10:
            if citation_density > 0.8:
                issues.append(f"Excessive citation density ({citation_density:.1%} of sentences)")
            elif citation_density < 0.05 and len(references) > 5:
                issues.append("Many references but few in-text citations")

        # Check for citation clustering
        text_thirds = [text[:len(text)//3], text[len(text)//3:2*len(text)//3], text[2*len(text)//3:]]
        citations_per_third = []

        for third in text_thirds:
            count = 0
            for pattern in citation_patterns:
                count += len(re.findall(pattern, third))
            citations_per_third.append(count)

        total_citations = sum(citations_per_third)
        if total_citations > 10:
            for i, count in enumerate(citations_per_third):
                if count > total_citations * 0.7:
                    position = ['beginning', 'middle', 'end'][i]
                    issues.append(f"Citations heavily clustered in {position} of document")

        return issues

    def _detect_style_inconsistencies(self, text: str) -> list[str]:
        """Detect inconsistencies in writing style"""
        issues: list[str] = []

        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if len(paragraphs) < 3:
            return issues

        # Analyze sentence complexity variation
        paragraph_complexities = []
        for para in paragraphs:
            sentences = re.split(r'[.!?]+', para)
            sentences = [s for s in sentences if s.strip()]
            if sentences:
                avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
                paragraph_complexities.append(avg_length)

        if paragraph_complexities:
            avg_complexity = sum(paragraph_complexities) / len(paragraph_complexities)
            for i, complexity in enumerate(paragraph_complexities):
                deviation = abs(complexity - avg_complexity) / avg_complexity if avg_complexity > 0 else 0
                if deviation > 0.5:  # More than 50% deviation
                    issues.append(f"Paragraph {i+1} has significantly different sentence complexity")

        # Check for spelling variety mixing (US vs UK)
        us_uk_pairs = [
            (r'\bcolor\b', r'\bcolour\b'),
            (r'\banalyze\b', r'\banalyse\b'),
            (r'\borganize\b', r'\borganise\b'),
            (r'\bcenter\b', r'\bcentre\b'),
            (r'\boptimize\b', r'\boptimise\b')
        ]

        mixed_spelling = False
        for us_pattern, uk_pattern in us_uk_pairs:
            has_us = bool(re.search(us_pattern, text, re.IGNORECASE))
            has_uk = bool(re.search(uk_pattern, text, re.IGNORECASE))
            if has_us and has_uk:
                mixed_spelling = True
                break

        if mixed_spelling:
            issues.append("Mixed US/UK spelling detected (possible copy-paste from multiple sources)")

        # Check for sudden tone shifts
        formal_indicators = ['furthermore', 'moreover', 'consequently', 'therefore', 'thus']
        informal_indicators = ["it's", "don't", "won't", "can't", "shouldn't", "you'll"]

        para_formality = []
        for para in paragraphs:
            para_lower = para.lower()
            formal_count = sum(1 for word in formal_indicators if word in para_lower)
            informal_count = sum(1 for word in informal_indicators if word in para_lower)

            if formal_count > informal_count * 2:
                para_formality.append('formal')
            elif informal_count > formal_count * 2:
                para_formality.append('informal')
            else:
                para_formality.append('neutral')

        # Check for abrupt tone changes
        for i in range(1, len(para_formality)):
            if para_formality[i-1] == 'formal' and para_formality[i] == 'informal':
                issues.append(f"Abrupt tone shift from formal to informal at paragraph {i+1}")
            elif para_formality[i-1] == 'informal' and para_formality[i] == 'formal':
                issues.append(f"Abrupt tone shift from informal to formal at paragraph {i+1}")

        return issues

    def _calculate_integrity_score(self, ai_indicators: dict, self_plagiarism: list,
                                  citation_anomalies: list, style_inconsistencies: list) -> float:
        """Calculate overall document integrity score (0-100, higher is better)"""
        base_score = 100.0

        # Deduct for AI indicators
        ai_confidence = ai_indicators.get('confidence', 0)
        base_score -= ai_confidence * 30  # Max 30 point deduction for AI

        # Deduct for self-plagiarism
        base_score -= len(self_plagiarism) * 10  # 10 points per issue

        # Deduct for citation anomalies
        base_score -= len(citation_anomalies) * 5  # 5 points per issue

        # Deduct for style inconsistencies
        base_score -= len(style_inconsistencies) * 5  # 5 points per issue

        # Ensure score stays within 0-100 range
        final_score: float = max(0.0, min(100.0, base_score))
        return final_score
