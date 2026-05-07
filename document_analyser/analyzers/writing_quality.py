"""
Writing quality analysis including style, tone, and linguistic patterns
"""

import re
from statistics import variance

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize

    # NLTK data should be pre-downloaded in Docker build
    # Don't download at import time to avoid permission issues

except ImportError:
    nltk = None
    sent_tokenize = None
    word_tokenize = None

from document_analyser.models.schemas import WritingQuality


class WritingQualityAnalyzer:
    """Analyzes writing quality metrics including style, tone, and linguistic patterns"""

    def __init__(self) -> None:
        """Initialize writing quality analyzer with linguistic patterns"""

        # Passive voice indicators
        self.passive_indicators = [
            r'\bis\s+\w+ed\b', r'\bare\s+\w+ed\b', r'\bwas\s+\w+ed\b', r'\bwere\s+\w+ed\b',
            r'\bbeen\s+\w+ed\b', r'\bbeing\s+\w+ed\b', r'\bis\s+\w+en\b', r'\bare\s+\w+en\b',
            r'\bwas\s+\w+en\b', r'\bwere\s+\w+en\b', r'\bbeen\s+\w+en\b', r'\bbeing\s+\w+en\b'
        ]

        # Transition words and phrases
        self.transition_words = {
            'addition': ['furthermore', 'moreover', 'additionally', 'also', 'besides', 'in addition'],
            'contrast': ['however', 'nevertheless', 'nonetheless', 'conversely', 'on the contrary', 'in contrast'],
            'sequence': ['firstly', 'secondly', 'subsequently', 'finally', 'meanwhile', 'then'],
            'causation': ['therefore', 'consequently', 'thus', 'hence', 'as a result', 'because'],
            'example': ['for instance', 'for example', 'such as', 'namely', 'specifically'],
            'summary': ['in conclusion', 'in summary', 'to summarize', 'overall', 'ultimately']
        }

        # Hedging language patterns
        self.hedging_patterns = [
            r'\bmight\b', r'\bmay\b', r'\bcould\b', r'\bwould\b', r'\bshould\b',
            r'\bpossibly\b', r'\bprobably\b', r'\bperhaps\b', r'\blikely\b',
            r'\bappears?\b', r'\bseems?\b', r'\bsuggests?\b', r'\bindicates?\b',
            r'\btends? to\b', r'\bmainly\b', r'\bmostly\b', r'\bgenerally\b',
            r'\boften\b', r'\busually\b', r'\btypically\b', r'\bfrequently\b',
            r'\bsomewhat\b', r'\brather\b', r'\bquite\b', r'\brelatively\b',
            r'\bto some extent\b', r'\bto a certain degree\b', r'\bin some ways\b'
        ]

        # Academic tone indicators
        self.academic_indicators = {
            'formal_verbs': ['demonstrate', 'illustrate', 'indicate', 'suggest', 'propose',
                           'examine', 'investigate', 'analyze', 'evaluate', 'assess'],
            'scholarly_phrases': ['according to', 'research shows', 'studies indicate',
                                'it has been found that', 'evidence suggests', 'data reveal'],
            'formal_connectors': ['furthermore', 'moreover', 'consequently', 'therefore',
                                'nonetheless', 'nevertheless', 'thus', 'hence']
        }

        # Spelling patterns for consistency checking
        self.spelling_variants = [
            (r'\bcolou?r\b', 'color/colour'),
            (r'\banalys[ez]\b', 'analyze/analyse'),
            (r'\borgani[sz]e\b', 'organize/organise'),
            (r'\bcente?r\b', 'center/centre'),
            (r'\boptimi[sz]e\b', 'optimize/optimise'),
            (r'\breali[sz]e\b', 'realize/realise'),
            (r'\bhonour?\b', 'honor/honour'),
            (r'\bflavou?r\b', 'flavor/flavour'),
            (r'\blabou?r\b', 'labor/labour')
        ]

    def analyze(self, text: str) -> WritingQuality:
        """
        Analyze writing quality metrics

        Args:
            text: The text to analyze

        Returns:
            WritingQuality with style and tone metrics
        """
        if not text.strip():
            return self._empty_analysis()

        # Basic text processing
        sentences = self._get_sentences(text)
        words = self._get_words(text)

        if not sentences or not words:
            return self._empty_analysis()

        # Calculate metrics
        passive_voice_percentage = self._calculate_passive_voice(sentences)
        sentence_variety = self._calculate_sentence_variety(sentences)
        transition_words_score = self._calculate_transition_words(text)
        hedging_language = self._calculate_hedging_language(text, len(words))
        academic_tone = self._calculate_academic_tone(text)

        return WritingQuality(
            passive_voice_percentage=passive_voice_percentage,
            sentence_variety=sentence_variety,
            transition_words=transition_words_score,
            hedging_language=hedging_language,
            academic_tone=academic_tone
        )

    def _empty_analysis(self) -> WritingQuality:
        """Return empty analysis for empty text"""
        return WritingQuality(
            passive_voice_percentage=0.0,
            sentence_variety=0.0,
            transition_words=0.0,
            hedging_language=0.0,
            academic_tone=0.0
        )

    def _get_sentences(self, text: str) -> list[str]:
        """Get sentences from text using NLTK or fallback"""
        if nltk and sent_tokenize:
            try:
                result = sent_tokenize(text)
                return [str(s) for s in result]  # Ensure all items are strings
            except Exception:
                pass

        # Fallback sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_words(self, text: str) -> list[str]:
        """Get words from text using NLTK or fallback"""
        if nltk and word_tokenize:
            try:
                tokens = word_tokenize(text.lower())
                return [token for token in tokens if token.isalpha()]
            except Exception:
                pass

        # Fallback word extraction
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    def _calculate_passive_voice(self, sentences: list[str]) -> float:
        """Calculate percentage of sentences containing passive voice"""
        if not sentences:
            return 0.0

        passive_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for pattern in self.passive_indicators:
                if re.search(pattern, sentence_lower):
                    passive_count += 1
                    break

        return round((passive_count / len(sentences)) * 100, 1)

    def _calculate_sentence_variety(self, sentences: list[str]) -> float:
        """Calculate sentence variety based on length variation"""
        if len(sentences) < 2:
            return 0.0

        # Calculate word counts per sentence
        word_counts = []
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence)
            word_counts.append(len(words))

        if len(word_counts) < 2:
            return 0.0

        # Calculate coefficient of variation (normalized variance)
        try:
            mean_length = sum(word_counts) / len(word_counts)
            if mean_length == 0:
                return 0.0

            var = variance(word_counts)
            cv = (var ** 0.5) / mean_length

            # Convert to 0-100 scale (higher = more variety)
            # Typical good writing has CV around 0.3-0.6
            variety_score = min(100, cv * 200)
            return round(float(variety_score), 1)

        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_transition_words(self, text: str) -> float:
        """Calculate transition word usage score"""
        text_lower = text.lower()
        word_count = len(re.findall(r'\b\w+\b', text))

        if word_count == 0:
            return 0.0

        transition_count = 0
        for category in self.transition_words.values():
            for phrase in category:
                # Count occurrences of each transition phrase
                transition_count += len(re.findall(rf'\b{re.escape(phrase)}\b', text_lower))

        # Calculate per 100 words and normalize to 0-100 scale
        transitions_per_100 = (transition_count / word_count) * 100

        # Good writing typically has 2-5 transition phrases per 100 words
        # Score based on optimal range
        if transitions_per_100 < 1:
            score = transitions_per_100 * 30  # Low usage
        elif transitions_per_100 <= 5:
            score = 30 + (transitions_per_100 - 1) * 17.5  # Optimal range
        else:
            score = max(0, 100 - (transitions_per_100 - 5) * 10)  # Overuse penalty

        return round(min(100, score), 1)

    def _calculate_hedging_language(self, text: str, word_count: int) -> float:
        """Calculate hedging language frequency"""
        if word_count == 0:
            return 0.0

        text_lower = text.lower()
        hedging_count = 0

        for pattern in self.hedging_patterns:
            hedging_count += len(re.findall(pattern, text_lower))

        # Calculate per 100 words
        hedging_per_100 = (hedging_count / word_count) * 100

        # Convert to 0-100 scale
        # Academic writing typically has 3-8 hedging phrases per 100 words
        if hedging_per_100 <= 8:
            score = (hedging_per_100 / 8) * 100
        else:
            score = max(0, 100 - (hedging_per_100 - 8) * 5)  # Penalty for overuse

        return round(score, 1)

    def _calculate_academic_tone(self, text: str) -> float:
        """Calculate academic tone strength"""
        text_lower = text.lower()
        word_count = len(re.findall(r'\b\w+\b', text))

        if word_count == 0:
            return 0.0

        academic_score = 0.0

        # Check for formal verbs
        formal_verb_count = 0
        for verb in self.academic_indicators['formal_verbs']:
            formal_verb_count += len(re.findall(rf'\b{verb}', text_lower))

        # Check for scholarly phrases
        scholarly_phrase_count = 0
        for phrase in self.academic_indicators['scholarly_phrases']:
            scholarly_phrase_count += len(re.findall(rf'{re.escape(phrase)}', text_lower))

        # Check for formal connectors
        formal_connector_count = 0
        for connector in self.academic_indicators['formal_connectors']:
            formal_connector_count += len(re.findall(rf'\b{connector}\b', text_lower))

        # Calculate component scores
        verb_score = min(40, (formal_verb_count / word_count) * 1000 * 40)
        phrase_score = min(30, (scholarly_phrase_count / word_count) * 1000 * 30)
        connector_score = min(30, (formal_connector_count / word_count) * 1000 * 30)

        academic_score = verb_score + phrase_score + connector_score

        # Penalty for informal contractions
        contractions = len(re.findall(r"\b\w+'\w+\b", text))
        contraction_penalty = min(20, (contractions / word_count) * 100 * 20)

        final_score = max(0, academic_score - contraction_penalty)
        return round(final_score, 1)

    def detect_spelling_consistency(self, text: str) -> dict[str, list[str]]:
        """
        Detect spelling consistency issues (US vs UK vs AU)

        Returns:
            Dictionary with detected inconsistencies
        """
        text_lower = text.lower()
        inconsistencies = {}

        for pattern, description in self.spelling_variants:
            matches = re.findall(pattern, text_lower)
            if len(set(matches)) > 1:  # Multiple variants found
                inconsistencies[description] = list(set(matches))

        return inconsistencies

    def get_readability_factors(self, text: str) -> dict[str, float]:
        """
        Get additional factors that affect readability
        (This could be used by the readability analyzer or other components)
        """
        sentences = self._get_sentences(text)
        words = self._get_words(text)

        if not sentences or not words:
            return {}

        # Average words per sentence
        avg_words_per_sentence = len(words) / len(sentences)

        # Complex word ratio (words > 6 characters)
        complex_words = [w for w in words if len(w) > 6]
        complex_word_ratio = len(complex_words) / len(words) if words else 0

        # Long sentence ratio (sentences > 20 words)
        long_sentences = 0
        for sentence in sentences:
            sentence_words = re.findall(r'\b\w+\b', sentence)
            if len(sentence_words) > 20:
                long_sentences += 1

        long_sentence_ratio = long_sentences / len(sentences) if sentences else 0

        return {
            'avg_words_per_sentence': round(avg_words_per_sentence, 1),
            'complex_word_ratio': round(complex_word_ratio * 100, 1),
            'long_sentence_ratio': round(long_sentence_ratio * 100, 1)
        }
