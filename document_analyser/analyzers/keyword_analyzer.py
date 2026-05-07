"""
Contextual keyword search analyzer
"""

import re
from typing import Any

from document_analyser.models.schemas import (
    DocumentKeywordResult,
    KeywordMatch,
    KeywordResult,
    KeywordSearchResponse,
    MultiKeywordSearchResponse,
)

# Helper: ensure type hints for older python/mypy
ListOfText = list[tuple[str, str]]


class KeywordAnalyzer:
    def __init__(self, window: int = 5) -> None:
        self.window = window

    def search(self, texts: list[tuple[str, str]], keyword: str) -> KeywordSearchResponse:
        """
        Search for keyword across multiple documents.

        Args:
            texts: List of tuples (document_name, text)
            keyword: Keyword to search

        Returns:
            KeywordSearchResponse with matches
        """
        keyword_lower = keyword.lower()
        matches: list[KeywordMatch] = []

        for doc_name, text in texts:
            for m in re.finditer(re.escape(keyword_lower), text.lower()):
                start = m.start()
                end = m.end()

                # Compute context window in original text
                words = re.findall(r"\b\w+\b", text)
                # Find word indices surrounding the match by character offsets
                # Map char index to word index approximate
                char_to_word = []
                running = 0
                for _idx, w in enumerate(words):
                    start_char = text.lower().find(w, running)
                    end_char = start_char + len(w)
                    char_to_word.append((start_char, end_char))
                    running = end_char

                # Find the word index containing the match start
                word_idx = 0
                for i, (s_char, e_char) in enumerate(char_to_word):
                    if s_char <= start < e_char:
                        word_idx = i
                        break
                    if i == len(char_to_word) - 1:
                        # If not found, approximate
                        word_idx = 0

                left = max(0, word_idx - self.window)
                right = min(len(words), word_idx + self.window + 1)
                context = " ".join(words[left:right])

                matches.append(
                    KeywordMatch(document=doc_name, context=context, start_char=start, end_char=end)
                )

        return KeywordSearchResponse(keyword=keyword, matches=matches)

    def search_multiple(
        self,
        texts: list[tuple[str, str]],
        keywords: list[str],
        context_chars: int = 100,
        max_contexts_per_doc: int = 5,
    ) -> MultiKeywordSearchResponse:
        """
        Search for multiple keywords across multiple documents.

        Args:
            texts: List of tuples (document_name, text)
            keywords: List of keywords to search
            context_chars: Number of characters around each match for context
            max_contexts_per_doc: Maximum context snippets per document per keyword

        Returns:
            MultiKeywordSearchResponse with aggregated results
        """
        results: list[KeywordResult] = []
        total_matches = 0
        documents_with_matches: set[str] = set()

        for keyword in keywords:
            keyword_lower = keyword.lower()
            doc_results: dict[str, dict[str, Any]] = {}

            for doc_name, text in texts:
                text_lower = text.lower()
                matches = list(re.finditer(re.escape(keyword_lower), text_lower))

                if matches:
                    documents_with_matches.add(doc_name)
                    contexts: list[str] = []

                    # Extract context snippets (limit to max_contexts_per_doc)
                    for match in matches[:max_contexts_per_doc]:
                        start = max(0, match.start() - context_chars // 2)
                        end = min(len(text), match.end() + context_chars // 2)

                        # Extend to word boundaries
                        while start > 0 and text[start - 1].isalnum():
                            start -= 1
                        while end < len(text) and text[end].isalnum():
                            end += 1

                        context = text[start:end].strip()
                        if start > 0:
                            context = "..." + context
                        if end < len(text):
                            context = context + "..."

                        contexts.append(context)

                    doc_results[doc_name] = {
                        "count": len(matches),
                        "contexts": contexts,
                    }

            # Build KeywordResult
            by_document = [
                DocumentKeywordResult(
                    document=doc_name,
                    count=data["count"],
                    contexts=data["contexts"],
                )
                for doc_name, data in doc_results.items()
            ]

            keyword_total = sum(d.count for d in by_document)
            total_matches += keyword_total

            results.append(
                KeywordResult(keyword=keyword, total_matches=keyword_total, by_document=by_document)
            )

        return MultiKeywordSearchResponse(
            results=results,
            summary={
                "total_keywords": len(keywords),
                "total_matches": total_matches,
                "documents_searched": len(texts),
                "documents_with_matches": len(documents_with_matches),
            },
        )
