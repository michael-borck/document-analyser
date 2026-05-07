"""Granular sentiment analyzer for multi-level sentiment analysis."""

from typing import Any, Literal

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError:
    nltk = None
    sent_tokenize = None

from document_analyser.models.schemas import (
    GranularSentimentResponse,
    ParagraphSentiment,
    SectionSentiment,
    SentenceSentiment,
    SentimentScore,
)


class GranularSentimentAnalyzer:
    """Multi-level sentiment analysis (sentence, paragraph, section)."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    ) -> None:
        """Initialize sentiment analyzer with HuggingFace transformer model."""
        self.model_name = model_name
        self.sentiment_pipeline = None

        if pipeline:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=-1,  # CPU for PyInstaller compatibility
                )
            except Exception:
                self.sentiment_pipeline = None

    def analyze(self, text: str) -> GranularSentimentResponse:
        """
        Analyze sentiment at sentence, paragraph, and section levels.

        Args:
            text: Document text to analyze

        Returns:
            GranularSentimentResponse with multi-level sentiment analysis
        """
        if not text.strip() or not self.sentiment_pipeline:
            return GranularSentimentResponse(total_sentences=0)

        # 1. Detect sections
        sections = self._detect_sections(text)

        # 2. Analyze each section
        section_sentiments: list[SectionSentiment] = []
        total_sentences = 0
        total_paragraphs = 0

        sentiment_counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}

        for section_idx, section in enumerate(sections):
            section_text = section["text"]

            # Split into paragraphs
            paragraphs = [
                p.strip() for p in section_text.split("\n\n") if p.strip()
            ]
            total_paragraphs += len(paragraphs)

            paragraph_sentiments: list[ParagraphSentiment] = []

            for para_idx, paragraph in enumerate(paragraphs):
                # Split into sentences
                sentences = self._get_sentences(paragraph)
                total_sentences += len(sentences)

                sentence_sentiments: list[SentenceSentiment] = []

                for sent_idx, sentence in enumerate(sentences):
                    if len(sentence.strip()) < 10:
                        continue

                    # Analyze sentence sentiment
                    sent_sentiment = self._analyze_sentence(sentence)
                    sentence_sentiments.append(SentenceSentiment(
                        sentence_index=total_sentences - len(sentences) + sent_idx,
                        text=sentence[:200],
                        sentiment=sent_sentiment,
                        dominant_sentiment=self._get_dominant(sent_sentiment)
                    ))

                    # Update counts
                    dominant = self._get_dominant(sent_sentiment)
                    sentiment_counts[dominant] += 1

                # Aggregate paragraph sentiment
                para_sentiment = self._aggregate_sentiments(
                    [s.sentiment for s in sentence_sentiments]
                )
                paragraph_sentiments.append(ParagraphSentiment(
                    paragraph_index=para_idx,
                    sentence_count=len(sentence_sentiments),
                    sentiment=para_sentiment,
                    sentences=sentence_sentiments,
                    dominant_sentiment=self._get_dominant(para_sentiment)
                ))

            # Aggregate section sentiment
            section_sentiment = self._aggregate_sentiments(
                [p.sentiment for p in paragraph_sentiments]
            )
            section_sentiments.append(SectionSentiment(
                section_name=section["header"],
                section_index=section_idx,
                paragraph_count=len(paragraph_sentiments),
                sentiment=section_sentiment,
                paragraphs=paragraph_sentiments,
                dominant_sentiment=self._get_dominant(section_sentiment)
            ))

        # 3. Calculate overall document sentiment
        doc_sentiment = self._aggregate_sentiments(
            [s.sentiment for s in section_sentiments]
        )

        # 4. Calculate sentiment distribution
        total = sum(sentiment_counts.values())
        distribution = {
            k: round(v / total * 100, 1) if total > 0 else 0.0
            for k, v in sentiment_counts.items()
        }

        return GranularSentimentResponse(
            document_sentiment=doc_sentiment,
            sections=section_sentiments,
            total_sentences=total_sentences,
            total_paragraphs=total_paragraphs,
            total_sections=len(sections),
            sentiment_distribution=distribution
        )

    def _analyze_sentence(self, sentence: str) -> SentimentScore:
        """Analyze sentiment of a single sentence."""
        if not self.sentiment_pipeline:
            return SentimentScore()

        # Truncate to model's max length
        truncated = sentence[:512]

        try:
            result = self.sentiment_pipeline(truncated)[0]
            label = result["label"].lower()
            score = result["score"]

            # Convert to SentimentScore
            if label == "positive":
                return SentimentScore(
                    positive=score,
                    negative=1 - score,
                    neutral=0.0,
                    compound=score
                )
            elif label == "negative":
                return SentimentScore(
                    positive=1 - score,
                    negative=score,
                    neutral=0.0,
                    compound=-score
                )
            else:
                return SentimentScore(
                    positive=0.0,
                    negative=0.0,
                    neutral=1.0,
                    compound=0.0
                )
        except Exception:
            return SentimentScore()

    def _aggregate_sentiments(
        self, sentiments: list[SentimentScore]
    ) -> SentimentScore:
        """Aggregate multiple sentiment scores by averaging."""
        if not sentiments:
            return SentimentScore()

        avg_pos = sum(s.positive for s in sentiments) / len(sentiments)
        avg_neg = sum(s.negative for s in sentiments) / len(sentiments)
        avg_neu = sum(s.neutral for s in sentiments) / len(sentiments)
        avg_comp = sum(s.compound for s in sentiments) / len(sentiments)

        return SentimentScore(
            positive=round(avg_pos, 3),
            negative=round(avg_neg, 3),
            neutral=round(avg_neu, 3),
            compound=round(avg_comp, 3)
        )

    def _get_dominant(
        self, sentiment: SentimentScore
    ) -> Literal["positive", "negative", "neutral"]:
        """Get dominant sentiment from scores."""
        if (
            sentiment.positive > sentiment.negative
            and sentiment.positive > sentiment.neutral
        ):
            return "positive"
        elif sentiment.negative > sentiment.neutral:
            return "negative"
        else:
            return "neutral"

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

            if len(first_line) < 60 and any(
                kw in first_line.lower() for kw in section_keywords
            ):
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

    def _get_sentences(self, text: str) -> list[str]:
        """Get sentences from text."""
        if sent_tokenize:
            try:
                return sent_tokenize(text)
            except Exception:
                pass

        return [s.strip() for s in text.split('.') if s.strip()]
