"""
Report-structure signals — heading-detected sections, canonical section
coverage, and a curated report-keyword lexicon.

Deterministic and repeatable (pure regex/counts, no NLTK/model dependency).
`extract_text` hands us markdown-ish text (markitdown renders docx/pptx
headings as `#` markers), so headings are detected from those markers first,
with numbered ("2.3 Title") and short ALL-CAPS lines as fallbacks. Each
heading starts a section whose word count rolls up into canonical buckets
(problem, method, evaluation, findings, recommendation, limitations, ...),
and a small consultant/data-mining lexicon counts domain terms.

These are *presence and emphasis* anchors for a marker — "is the evidence
there, and how much of the report is behind it" — never quality judgements.
"""

from __future__ import annotations

import re

# Canonical buckets: heading text containing any term maps to the bucket.
CANONICAL_LEXICONS: dict[str, list[str]] = {
    "introduction": ["introduction", "executive summary", "background"],
    "problem": ["problem", "objective", "business understanding", "framing",
                "brief"],
    "method": ["method", "approach", "data preparation", "workflow",
               "modelling", "modeling", "design", "preparation",
               "data understanding"],
    "evaluation": ["evaluat", "result", "performance", "assess"],
    "findings": ["finding", "insight"],
    "limitations": ["limitation", "caveat", "constraint", "shortcoming"],
    "recommendation": ["recommend", "next step"],
    "conclusion": ["conclusion"],
}

# Curated consultant / data-mining report lexicon (case-insensitive counts).
KEYWORDS: dict[str, str] = {
    "decision": r"\bdecisions?\b",
    "stakeholder": r"\bstakeholders?\b",
    "objective": r"\bobjectives?\b",
    "orange": r"\borange\b",
    "workflow": r"\bworkflows?\b",
    "cross_validation": r"cross[-\s]?validat",
    "train_test": r"\btrain(?:ing)?\b[\s-]*(?:and[\s-]*)?(?:test|holdout)"
                  r"|\btest(?:ing)?\s+set\b|\bholdout\b",
    "model": r"\bmodels?\b",
    "algorithm": r"\balgorithms?\b",
    "baseline": r"\bbaselines?\b",
    "alternative": r"\balternatives?\b",
    "feature": r"\bfeatures?\b",
    "threshold": r"\bthresholds?\b",
    "accuracy": r"\baccuracy\b|\baccurate\b",
    "precision": r"\bprecision\b",
    "recall": r"\brecall\b",
    "f1": r"\bf1\b|f[-\s]?score|f[-\s]?measure",
    "confusion_matrix": r"confusion\s+matrix",
    "auc": r"\bauc\b|area under (?:the )?curve",
    "lift": r"\blift\b",
    "validation": r"\bvalidat(?:e|ed|ing|ion)\b",
    "limitation": r"\blimitations?\b",
    "shortcoming": r"\bshortcomings?\b",
    "caveat": r"\bcaveats?\b",
    "trust": r"\btrust(?:ed|ing|worthy)?\b",
    "bias": r"\bbias(?:ed)?\b",
    "recommend": r"\brecommend(?:s|ed|ation|ations|ing)?\b",
    "risk": r"\brisks?\b",
    "mitigation": r"\bmitigat(?:e|es|ed|ing|ion)s?\b",
    "human_in_the_loop": r"human[\s-]+in[\s-]+the[\s-]+loop|human oversight"
                         r"|human\s+(?:judgement|judgment)",
    "monitoring": r"\bmonitor(?:s|ed|ing)?\b",
    "retrain": r"\bretrain(?:s|ed|ing)?\b",
}

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*)$")
_NUM_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+(\S.{2,98})$")
_WORD_RE = re.compile(r"[a-z']+")


def _heading_level(line: str) -> tuple[int, str] | None:
    """Heading level (1-6) + clean title, or None for body text."""
    m = _MD_HEADING_RE.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    if line.isupper() and 3 < len(line) < 80:
        return 1, line.title()
    m = _NUM_HEADING_RE.match(line)
    if m:
        return min(6, m.group(1).count(".") + 1), m.group(2).strip()
    return None


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text.lower()))


class SectionsAnalyzer:
    """Heading-detected section structure + canonical coverage + keywords."""

    def analyze(self, text: str) -> dict:
        lines = text.splitlines()
        blocks: list[tuple[int | None, str | None, int]] = []
        markdown = numbered = caps = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            level_title = _heading_level(stripped)
            if level_title is None:
                blocks.append((None, None, _word_count(stripped)))
                continue
            level, title = level_title
            if stripped.startswith("#"):
                markdown += 1
            elif _NUM_HEADING_RE.match(stripped):
                numbered += 1
            else:
                caps += 1
            blocks.append((level, title, 0))

        # Roll body words into the section each line belongs to.
        sections: list[dict] = []
        current: dict | None = None
        for level, title, wc in blocks:
            if level is None:
                if current is None:
                    continue  # preamble before any heading
                current["word_count"] += wc
            else:
                current = {"title": title, "level": level, "word_count": wc}
                sections.append(current)

        canonical = {name: False for name in CANONICAL_LEXICONS}
        canonical_wc = {name: 0 for name in CANONICAL_LEXICONS}
        for s in sections:
            title = s["title"].lower()
            for cname, terms in CANONICAL_LEXICONS.items():
                if any(term in title for term in terms):
                    canonical[cname] = True
                    canonical_wc[cname] += s["word_count"]

        if markdown:
            detection = "markdown"
        elif numbered:
            detection = "numbered"
        elif caps:
            detection = "caps"
        else:
            detection = "none"

        return {
            "heading_detection": detection,
            "section_count": len(sections),
            "headings": sections,
            "canonical": canonical,
            "canonical_wc": canonical_wc,
        }

    def keyword_counts(self, text: str) -> dict:
        lower = text.lower()
        return {slug: len(re.findall(rx, lower))
                for slug, rx in KEYWORDS.items()}
