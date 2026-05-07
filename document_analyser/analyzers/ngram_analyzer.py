"""
N-gram analysis utilities (bigrams and trigrams)
"""

import re
from collections import Counter

from document_analyser.models.schemas import Ngram


class NgramAnalyzer:
    def __init__(self) -> None:
        pass

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        return tokens

    def extract_ngrams(
        self, text: str, n: int = 2, top_k: int = 20, filter_terms: list[str] | None = None
    ) -> list[Ngram]:
        """
        Extract n-grams from text.

        Args:
            text: The text to analyze
            n: Size of n-grams (2 for bigrams, 3 for trigrams)
            top_k: Number of top n-grams to return
            filter_terms: Optional list of terms. If provided, only return n-grams
                         that contain at least one of these terms.

        Returns:
            List of Ngram objects with phrase and count
        """
        tokens = self._tokenize(text)
        if len(tokens) < n:
            return []

        ngrams: Counter[str] = Counter()
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            ngrams[phrase] += 1

        # Apply filter if provided
        if filter_terms:
            # Normalize filter terms to lowercase
            normalized_filters = [term.lower().strip() for term in filter_terms]
            filtered_ngrams: Counter[str] = Counter()

            for phrase, count in ngrams.items():
                # Check if any filter term appears in the phrase
                # Support both exact word match and partial phrase match
                phrase_words = set(phrase.split())
                for term in normalized_filters:
                    # Check if term is a single word and matches exactly
                    if (" " not in term and term in phrase_words) or term in phrase:
                        filtered_ngrams[phrase] = count
                        break

            top = filtered_ngrams.most_common(top_k)
        else:
            top = ngrams.most_common(top_k)

        return [Ngram(phrase=p, count=c) for p, c in top]
