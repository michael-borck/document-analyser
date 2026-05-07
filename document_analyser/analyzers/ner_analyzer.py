"""
Named Entity Recognition analyzer using spaCy
"""

try:
    import spacy
except ImportError:
    spacy = None  # type: ignore

from document_analyser.models.schemas import NEREntity, NERResponse


class NerAnalyzer:
    """Wraps spaCy NER for use in DocumentLens."""

    def __init__(self, model: str = "en_core_web_sm") -> None:
        self.model_name = model
        self.nlp = None
        if spacy:
            try:
                self.nlp = spacy.load(self.model_name)
            except Exception:
                # Model not available; keep nlp as None
                self.nlp = None

    def analyze(self, text: str) -> NERResponse:
        if not text.strip() or not self.nlp:
            return NERResponse(entities=[])

        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append(NEREntity(
                text=str(ent.text),
                label=str(ent.label_),
                start_char=int(ent.start_char),
                end_char=int(ent.end_char)
            ))

        return NERResponse(entities=entities)
