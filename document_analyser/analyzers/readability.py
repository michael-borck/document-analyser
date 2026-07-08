"""
Readability analysis using various metrics
"""

import re

try:
    import textstat
    from nltk import download
    from nltk.tokenize import sent_tokenize

    # Ensure the NLTK resources our readability metrics rely on are present.
    # `punkt` powers sentence tokenisation; `cmudict` is what textstat uses
    # for syllable counting (Flesch/SMOG/etc.). In a packaged, offline build
    # these must be bundled — the download() calls below are only a best-effort
    # fallback for source/dev runs and MUST NOT raise at import time (a clean
    # offline machine can't fetch them, and that's fine: analyze() degrades
    # gracefully rather than failing extraction). See analyze() for the guard.
    def _ensure_nltk(resource: str, probe) -> None:
        try:
            probe()
        except LookupError:
            try:
                download(resource, quiet=True)
            except Exception:  # noqa: BLE001 - offline/clean box: fall back at use time
                pass
        except Exception:  # noqa: BLE001 - never let resource setup abort import
            pass

    from nltk.data import find as _nltk_find

    _ensure_nltk('punkt', lambda: sent_tokenize("test"))
    _ensure_nltk('cmudict', lambda: _nltk_find('corpora/cmudict'))

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

        # Readability scores. textstat's syllable-based metrics need the NLTK
        # `cmudict` corpus; if it isn't available (e.g. a packaged build that
        # didn't bundle it, or an offline first run) textstat raises a
        # LookupError. Readability is a soft signal — never let a missing
        # resource abort extraction. Fall back to the pure-Python estimates.
        flesch_score = flesch_kincaid_grade = 0.0
        gunning_fog = smog_index = automated_readability_index = 0.0
        used_textstat = False
        if textstat:
            try:
                flesch_score = textstat.flesch_reading_ease(text)
                flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
                gunning_fog = textstat.gunning_fog(text)
                smog_index = textstat.smog_index(text)
                automated_readability_index = textstat.automated_readability_index(text)
                used_textstat = True
            except Exception:  # noqa: BLE001 - missing NLTK data / textstat internals
                used_textstat = False

        if not used_textstat:
            # Fallback implementation (advanced indices need textstat; default to 0.0)
            flesch_score = self._calculate_flesch_score(text, word_count, sentence_count)
            flesch_kincaid_grade = self._calculate_flesch_kincaid_grade(text, word_count, sentence_count)
            gunning_fog = 0.0
            smog_index = 0.0
            automated_readability_index = 0.0

        return DocumentAnalysis(
            word_count=word_count,
            sentence_count=sentence_count,
            avg_words_per_sentence=round(avg_words_per_sentence, 1),
            paragraph_count=paragraph_count,
            flesch_score=round(flesch_score, 1),
            flesch_kincaid_grade=round(flesch_kincaid_grade, 1),
            gunning_fog=round(gunning_fog, 1),
            smog_index=round(smog_index, 1),
            automated_readability_index=round(automated_readability_index, 1),
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
