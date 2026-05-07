"""
Word and phrase analysis with frequency statistics and vocabulary metrics
"""

import re
from collections import Counter
from typing import Any

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.util import ngrams

    # NLTK data should be pre-downloaded in Docker build
    # Don't download at import time to avoid permission issues

except ImportError:
    nltk = None
    word_tokenize = None
    sent_tokenize = None
    stopwords = None
    ngrams = None

from document_analyser.models.schemas import PhraseCount, WordAnalysis, WordFrequency


class WordAnalyzer:
    """Analyzes word frequency, vocabulary richness, and phrase patterns"""

    def __init__(self) -> None:
        """Initialize word analyzer with stop words"""
        # Common English stop words as fallback
        self.fallback_stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "would",
            "i",
            "you",
            "we",
            "they",
            "this",
            "these",
            "those",
            "or",
            "but",
            "not",
            "have",
            "had",
            "do",
            "does",
            "did",
            "can",
            "could",
            "should",
            "may",
            "might",
            "must",
        }

    def analyze(self, text: str) -> WordAnalysis:
        """
        Analyze word patterns, frequency, and vocabulary metrics

        Args:
            text: The text to analyze

        Returns:
            WordAnalysis with frequency statistics and vocabulary metrics
        """
        if not text.strip():
            return self._empty_analysis()

        # Clean and tokenize text
        words = self._tokenize_words(text)
        if not words:
            return self._empty_analysis()

        # Get stop words
        stop_words = self._get_stop_words()

        # Filter out stop words and short words
        meaningful_words = [word for word in words if len(word) > 2 and word not in stop_words]

        # Calculate word frequencies
        word_freq = Counter(meaningful_words)
        most_frequent = self._get_most_frequent_words(word_freq)

        # Get unique words (hapax legomena - words that appear only once)
        unique_words = [word for word, count in word_freq.items() if count == 1]

        # Extract meaningful phrases (bigrams and trigrams)
        unique_phrases = self._extract_phrases(text, stop_words)

        # Calculate vocabulary metrics
        all_words = self._tokenize_words(text)
        unique_word_count = len(set(all_words))
        total_word_count = len(all_words)
        vocabulary_richness = (
            (unique_word_count / total_word_count * 100) if total_word_count > 0 else 0.0
        )

        # Split phrases into bigrams and trigrams
        bigrams = [p for p in unique_phrases if len(p.phrase.split()) == 2]
        trigrams = [p for p in unique_phrases if len(p.phrase.split()) == 3]

        return WordAnalysis(
            most_frequent=most_frequent,
            unique_words=unique_words[:50],  # Hapax legomena sample (words appearing once)
            unique_phrases=unique_phrases,
            unique_word_count=unique_word_count,
            total_word_count=total_word_count,
            vocabulary_richness=round(vocabulary_richness, 1),
            top_words=most_frequent,  # Alias for backward compatibility
            bigrams=bigrams,
            trigrams=trigrams,
        )

    def _empty_analysis(self) -> WordAnalysis:
        """Return empty analysis for empty text"""
        return WordAnalysis(most_frequent=[], unique_words=[], unique_phrases=[])

    def _tokenize_words(self, text: str) -> list[str]:
        """Tokenize text into words, handling both NLTK and fallback methods"""
        if nltk and word_tokenize:
            try:
                # Use NLTK tokenization (more sophisticated)
                tokens = word_tokenize(text.lower())
                # Filter out punctuation and non-alphabetic tokens
                return [token for token in tokens if token.isalpha()]
            except Exception:
                # Fall back to regex if NLTK fails
                pass

        # Fallback regex-based tokenization
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        return words

    def _get_stop_words(self) -> set[str]:
        """Get stop words, with fallback if NLTK is not available"""
        if nltk and stopwords:
            try:
                return set(stopwords.words("english"))
            except Exception:
                pass

        return self.fallback_stopwords

    def _get_most_frequent_words(
        self, word_freq: Counter[str], limit: int = 20
    ) -> list[WordFrequency]:
        """Get most frequent words with their counts and relative sizes"""
        if not word_freq:
            return []

        most_common = word_freq.most_common(limit)
        max_count = most_common[0][1] if most_common else 1

        return [
            WordFrequency(
                word=word,
                count=count,
                size=min(
                    100, max(10, int((count / max_count) * 100))
                ),  # Size 10-100 for visualization
            )
            for word, count in most_common
        ]

    def _extract_phrases(
        self, text: str, stop_words: set[str], limit: int = 15
    ) -> list[PhraseCount]:
        """Extract meaningful phrases (n-grams) from text"""
        phrases: list[PhraseCount] = []

        # Extract bigrams (2-word phrases)
        bigrams = self._extract_ngrams(text, 2, stop_words)
        phrases.extend(
            [
                PhraseCount(phrase=phrase, count=count)
                for phrase, count in bigrams.most_common(limit // 2)
            ]
        )

        # Extract trigrams (3-word phrases)
        trigrams = self._extract_ngrams(text, 3, stop_words)
        phrases.extend(
            [
                PhraseCount(phrase=phrase, count=count)
                for phrase, count in trigrams.most_common(limit // 2)
            ]
        )

        # Sort by frequency and return top phrases
        phrases.sort(key=lambda x: x.count, reverse=True)
        return phrases[:limit]

    def _extract_ngrams(self, text: str, n: int, stop_words: set[str]) -> Counter[str]:
        """Extract n-grams from text, filtering out stop words and common patterns"""
        words = self._tokenize_words(text)

        if nltk and ngrams:
            try:
                # Use NLTK n-grams
                phrase_list = list(ngrams(words, n))
            except Exception:
                # Fallback to manual n-gram generation
                phrase_list = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        else:
            # Manual n-gram generation
            phrase_list = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

        # Filter meaningful phrases
        meaningful_phrases = []
        for phrase_tuple in phrase_list:
            # Skip phrases with stop words
            if any(word in stop_words for word in phrase_tuple):
                continue

            # Skip phrases with very short words
            if any(len(word) < 3 for word in phrase_tuple):
                continue

            # Join into string
            phrase = " ".join(phrase_tuple)

            # Skip phrases with repeated words
            if len(set(phrase_tuple)) < len(phrase_tuple):
                continue

            meaningful_phrases.append(phrase)

        return Counter(meaningful_phrases)

    def get_vocabulary_metrics(self, text: str) -> dict[str, Any]:
        """
        Calculate additional vocabulary richness metrics
        (This could be used by other analyzers or future endpoints)
        """
        if not text.strip():
            return {}

        words = self._tokenize_words(text)
        if not words:
            return {}

        unique_word_count = len(set(words))
        total_word_count = len(words)

        # Type-Token Ratio (TTR) - vocabulary richness
        ttr = unique_word_count / total_word_count if total_word_count > 0 else 0

        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Lexical diversity (unique words per 100 words)
        lexical_diversity = (
            (unique_word_count / total_word_count) * 100 if total_word_count > 0 else 0
        )

        return {
            "unique_words": unique_word_count,
            "total_words": total_word_count,
            "type_token_ratio": round(ttr, 3),
            "lexical_diversity": round(lexical_diversity, 1),
            "average_word_length": round(avg_word_length, 1),
        }
