"""
Readability analysis using various metrics
"""

import re

try:
    import textstat
    from nltk import download
    from nltk.tokenize import sent_tokenize

    # Download required NLTK data
    try:
        sent_tokenize("test")
    except LookupError:
        download('punkt')

except ImportError:
    textstat = None
    sent_tokenize = None

from document_analyser.models.schemas import DocumentAnalysis


class ReadabilityAnalyzer:
    """Analyzes text readability using various metrics"""

    def __init__(self) -> None:
        self.syllable_vowels = "aeiouy"

    def analyze(self, text: str) -> DocumentAnalysis:
        """
        Analyze text readability

        Args:
            text: The text to analyze

        Returns:
            DocumentAnalysis with readability metrics
        """
        if not text.strip():
            return self._empty_analysis()

        # Basic counts
        word_count = self._count_words(text)
        sentence_count = self._count_sentences(text)
        paragraph_count = self._count_paragraphs(text)

        avg_words_per_sentence = word_count / max(sentence_count, 1)

        # Readability scores
        if textstat:
            flesch_score = textstat.flesch_reading_ease(text)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        else:
            # Fallback implementation
            flesch_score = self._calculate_flesch_score(text, word_count, sentence_count)
            flesch_kincaid_grade = self._calculate_flesch_kincaid_grade(text, word_count, sentence_count)

        return DocumentAnalysis(
            word_count=word_count,
            sentence_count=sentence_count,
            avg_words_per_sentence=round(avg_words_per_sentence, 1),
            paragraph_count=paragraph_count,
            flesch_score=round(flesch_score, 1),
            flesch_kincaid_grade=round(flesch_kincaid_grade, 1)
        )

    def _empty_analysis(self) -> DocumentAnalysis:
        """Return empty analysis for empty text"""
        return DocumentAnalysis(
            word_count=0,
            sentence_count=0,
            avg_words_per_sentence=0.0,
            paragraph_count=0,
            flesch_score=0.0,
            flesch_kincaid_grade=0.0
        )

    def _count_words(self, text: str) -> int:
        """Count words in text"""
        words = re.findall(r'\b\w+\b', text.lower())
        return len(words)

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        if textstat and sent_tokenize is not None:
            try:
                sentences = sent_tokenize(text)
                return len(sentences)
            except Exception:
                pass

        # Fallback method
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])

    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs in text"""
        paragraphs = text.split('\n\n')
        return len([p for p in paragraphs if p.strip()])

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        word = word.lower()
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in self.syllable_vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        # Every word has at least one syllable
        return max(syllable_count, 1)

    def _calculate_flesch_score(self, text: str, word_count: int, sentence_count: int) -> float:
        """Calculate Flesch Reading Ease score"""
        if word_count == 0 or sentence_count == 0:
            return 0.0

        # Count syllables
        words = re.findall(r'\b\w+\b', text.lower())
        total_syllables = sum(self._count_syllables(word) for word in words)

        if total_syllables == 0:
            return 0.0

        # Flesch Reading Ease formula
        return 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (total_syllables / word_count)

    def _calculate_flesch_kincaid_grade(self, text: str, word_count: int, sentence_count: int) -> float:
        """Calculate Flesch-Kincaid Grade Level"""
        if word_count == 0 or sentence_count == 0:
            return 0.0

        # Count syllables
        words = re.findall(r'\b\w+\b', text.lower())
        total_syllables = sum(self._count_syllables(word) for word in words)

        if total_syllables == 0:
            return 0.0

        # Flesch-Kincaid Grade Level formula
        return 0.39 * (word_count / sentence_count) + 11.8 * (total_syllables / word_count) - 15.59
