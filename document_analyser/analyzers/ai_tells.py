"""
AI-writing-tell signals — lightweight, advisory markers that a passage may read
as machine-generated or over-polished.

These are *flags to read the document more closely*, not pass/fail judgements:
emoji use, em-dash density (excluding list/table formatting), adverb ratio,
intensifier density, hyperbole, and a list of AI-favoured buzzwords. Severity is
informational. All checks are pure regex/counts — no NLTK/model dependency.
"""

from __future__ import annotations

import re

# --- Emoji / pictographic glyphs (incl. dingbats like ✓ ✗ ★, a common AI tell) ---
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, emoji, supplemental
    "\U0001F1E0-\U0001F1FF"  # regional-indicator flags
    "\U00002600-\U000027BF"  # misc symbols + dingbats (✓ ✗ ★ ✦ …)
    "\U00002B00-\U00002BFF"  # stars / arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000231A-\U0000231B"  # ⌚ ⌛
    "\U000023E9-\U000023FA"  # media controls / ⏰ etc.
    "]",
    flags=re.UNICODE,
)

# Words ending in -ly that are NOT manner adverbs (keeps the ratio honest).
_LY_STOPLIST = {
    "family", "reply", "apply", "supply", "imply", "comply", "rely", "only",
    "early", "italy", "july", "ally", "rally", "bully", "jelly", "belly",
    "folly", "holly", "lily", "ugly", "lonely", "lovely", "silly", "likely",
    "assembly", "anomaly", "monopoly", "panoply", "butterfly", "multiply",
    "daily", "weekly", "monthly", "yearly", "hourly", "nightly", "homely",
}

_INTENSIFIER_RE = re.compile(
    r"\b(?:very|extremely|highly|incredibly|really|truly|absolutely|"
    r"completely|utterly|remarkably|exceptionally)\s+\w+",
    flags=re.IGNORECASE,
)

# Vocabulary disproportionately favoured by LLMs.
_AI_VOCAB = [
    "delve", "intricate", "multifaceted", "nuanced", "paramount", "comprehensive",
    "underscore", "robust", "leverage", "synergy", "holistic", "pivotal",
    "seamless", "streamline", "foster", "endeavor", "facilitate", "navigate",
    "realm", "tapestry", "testament", "landscape", "vibrant", "bustling",
    "crucial", "vital", "intricacies", "underpin", "myriad",
]

_HYPERBOLE = [
    "revolutionary", "groundbreaking", "unprecedented", "game-changing",
    "cutting-edge", "paradigm-shifting", "transformative", "disruptive",
    "world-class", "best-in-class", "state-of-the-art", "trailblazing",
]

# Lines that are list items or table rows — em-dashes here are formatting.
_LIST_RE = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)]|[a-zA-Z][.)])\s")


def _is_structural_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _LIST_RE.match(line):
        return True
    if s.startswith("|") or " | " in s:  # markdown / pipe table row
        return True
    return False


class AiTellsAnalyzer:
    """Computes advisory AI-writing-tell signals for a block of text."""

    def analyze(self, text: str) -> dict:
        words = re.findall(r"[A-Za-z]+", text)
        word_count = len(words)
        flags: list[dict] = []

        # --- Emojis / pictographic glyphs ---
        emojis = _EMOJI_RE.findall(text)
        if emojis:
            uniq = sorted(set(emojis))
            flags.append({
                "type": "emojis",
                "severity": "medium",
                "description": f"{len(emojis)} emoji/pictographic glyph(s) in body text",
                "evidence": " ".join(uniq[:10]),
            })

        # --- Em-dashes (excluding list/table formatting) ---
        em_dashes = 0
        for line in text.splitlines():
            if _is_structural_line(line):
                continue
            em_dashes += len(re.findall(r"—", line))           # —
            em_dashes += len(re.findall(r"(?<!-)--(?!-)", line))    # -- as em-dash
        em_per_1k = round(em_dashes / word_count * 1000, 2) if word_count else 0.0
        if em_dashes >= 5:
            flags.append({
                "type": "em_dashes",
                "severity": "low",
                "description": f"{em_dashes} em-dashes in prose ({em_per_1k} per 1,000 words)",
                "evidence": f"{em_dashes} em-dash occurrences (list/table lines excluded)",
            })

        # --- Adverb ratio (-ly words, minus a non-adverb stop-list) ---
        ly = [w for w in words if len(w) > 3 and w.lower().endswith("ly")
              and w.lower() not in _LY_STOPLIST]
        adverb_ratio = round(len(ly) / word_count, 4) if word_count else 0.0
        if adverb_ratio >= 0.05:
            flags.append({
                "type": "adverb_ratio",
                "severity": "low",
                "description": f"High adverb ratio: {adverb_ratio:.1%} of words end in -ly ({len(ly)}/{word_count})",
                "evidence": ", ".join(sorted({w.lower() for w in ly})[:8]),
            })

        # --- Intensifier phrases ("very X", "highly Y") ---
        intensifiers = _INTENSIFIER_RE.findall(text)
        if len(intensifiers) >= 5:
            flags.append({
                "type": "intensifiers",
                "severity": "low",
                "description": f"Frequent intensifier phrases ({len(intensifiers)} occurrences)",
                "evidence": "; ".join(intensifiers[:5]),
            })

        # --- AI-favoured vocabulary ---
        vocab_hits: dict[str, int] = {}
        for term in _AI_VOCAB:
            n = len(re.findall(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE))
            if n:
                vocab_hits[term] = n
        if len(vocab_hits) >= 3:
            flags.append({
                "type": "ai_vocabulary",
                "severity": "low",
                "description": f"Repeated LLM-favoured vocabulary ({len(vocab_hits)} distinct terms)",
                "evidence": ", ".join(f"{t} (x{c})" for t, c in sorted(vocab_hits.items())),
            })

        # --- Hyperbole ---
        hyperbole = [t for t in _HYPERBOLE
                     if re.search(rf"\b{t.replace('-', '[ -]')}\b", text, flags=re.IGNORECASE)]
        if len(hyperbole) >= 3:
            flags.append({
                "type": "hyperbole",
                "severity": "low",
                "description": f"Multiple hyperbolic terms ({len(hyperbole)})",
                "evidence": ", ".join(hyperbole),
            })

        return {
            "word_count": word_count,
            "emoji_count": len(emojis),
            "em_dash_count": em_dashes,
            "em_dashes_per_1000_words": em_per_1k,
            "adverb_count": len(ly),
            "adverb_ratio": adverb_ratio,
            "intensifier_count": len(intensifiers),
            "ai_vocabulary": vocab_hits,
            "hyperbole_terms": hyperbole,
            "flags": flags,
        }
