#!/usr/bin/env python3
"""
Deterministic layout pass for PDFs (ADR-0040).

Extracts heading candidates using pure geometry + typography — text spans
whose font size exceeds the page's median body size (bold as a tiebreaker)
— then aligns each candidate into the canonical extracted text (pypdf's
per-page text, joined with "\\n\\n") by order-preserving, whitespace-
normalised substring matching. Aligned headings carry start/end offsets in
full-text coordinates, the same coordinate system every other workflow
joins on (sections, page offsets, suppressed spans).

Design constraints (ADR-0040):
- No ML, no OCR — same PDF in, same headings out, every time (rung 0 of
  the ADR-0035 ladder).
- Best-effort and non-fatal: any failure degrades to an empty layout with
  an error note; text extraction has already committed.
- Misalignment is dropped and counted (fail loudly, ADR-0026), never
  guessed.

Alignment is a pure function of (headings, pages, full_text) and is fully
unit-testable without PyMuPDF; only `extract_heading_candidates` touches
the PDF.
"""

from __future__ import annotations

import re
from typing import Any

# A heading candidate must be at least this many times the page's median
# body font size (bold spans get a lower bar). 1.0 = same size.
SIZE_RATIO_PLAIN = 1.15
SIZE_RATIO_BOLD = 1.05
# Candidate text floor: shorter strings are noise (page furniture).
MIN_HEADING_CHARS = 3
# Whitespace runs collapse to single spaces for alignment.
_WS = re.compile(r"\s+")


def normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to single spaces.

    Returns the normalised string plus a map from normalised index to the
    corresponding original index, so a match found in the normalised text
    can be projected back to exact original offsets.
    """
    norm_chars: list[str] = []
    index_map: list[int] = []
    pending_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            # Emit at most one space, and never leading.
            if norm_chars and not pending_space:
                norm_chars.append(" ")
                index_map.append(i)
            pending_space = True
        else:
            pending_space = False
            norm_chars.append(ch)
            index_map.append(i)
    return "".join(norm_chars), index_map


def align_heading(
    heading_text: str,
    page_norm: str,
    page_map: list[int],
    page_offset: int,
) -> tuple[int, int] | None:
    """Locate one heading inside its page's normalised text.

    Returns (start_offset, end_offset) in full-text coordinates, or None
    when the heading text cannot be located — callers drop and count the
    failure rather than guessing.
    """
    needle, _needle_map = normalise_with_map(heading_text)
    needle = needle.strip()
    if len(needle) < MIN_HEADING_CHARS:
        return None

    span = _find_span(page_norm, needle)
    if span is None:
        # Fallback: match on the first few words — pypdf and PyMuPDF
        # sometimes disagree about intra-heading spacing only.
        words = needle.split(" ")
        if len(words) >= 2:
            span = _find_span(page_norm, " ".join(words[: min(4, len(words))]))
    if span is None:
        return None

    start, end = span
    if start >= len(page_map) or end - 1 >= len(page_map):
        return None
    # `end` is exclusive: the last included char lives at end - 1.
    return (page_offset + page_map[start], page_offset + page_map[end - 1] + 1)


def _find_span(haystack: str, needle: str) -> tuple[int, int] | None:
    """Word-boundary substring search in normalised space."""
    if not needle:
        return None
    idx = haystack.find(needle)
    while idx != -1:
        before_ok = idx == 0 or not haystack[idx - 1].isalnum()
        after = idx + len(needle)
        after_ok = after >= len(haystack) or not haystack[after].isalnum()
        if before_ok and after_ok:
            return (idx, idx + len(needle))
        nxt = haystack.find(needle, idx + 1)
        if nxt == -1:
            return None
        idx = nxt
    return None


def align_headings_to_pages(
    headings: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    full_text: str,
) -> tuple[list[dict[str, Any]], int]:
    """Align per-page heading candidates into full-text offsets.

    `headings` items: {page_number, text, font_size, bold}. `pages` items:
    {page_number, text} — the canonical per-page text the full_text was
    joined from (non-empty pages only, "\\n\\n" separator). Returns
    (aligned_headings, dropped_count), sorted by document order.
    """
    # Page offsets in full_text: walk non-empty pages, accumulate
    # len(text) + 2 per page (the separator; not appended after the last).
    page_offsets: dict[int, int] = {}
    cursor = 0
    non_empty = [p for p in pages if (p.get("text") or "").strip()]
    for i, p in enumerate(non_empty):
        page_offsets[p["page_number"]] = cursor
        cursor += len(p["text"]) + (2 if i < len(non_empty) - 1 else 0)

    # Normalise each page once.
    norm_by_page: dict[int, tuple[str, list[int]]] = {}
    for p in non_empty:
        norm_by_page[p["page_number"]] = normalise_with_map(p["text"])

    aligned: list[dict[str, Any]] = []
    dropped = 0
    for h in headings:
        page_number = h["page_number"]
        norm = norm_by_page.get(page_number)
        offset = page_offsets.get(page_number)
        if norm is None or offset is None:
            dropped += 1
            continue
        page_norm, page_map = norm
        span = align_heading(h["text"], page_norm, page_map, offset)
        if span is None:
            dropped += 1
            continue
        start, end = span
        if end > len(full_text):  # defensive against coordinate drift
            dropped += 1
            continue
        aligned.append(
            {
                "page_number": page_number,
                "text": h["text"],
                "font_size": h["font_size"],
                "bold": h["bold"],
                "start_offset": start,
                "end_offset": end,
            }
        )

    aligned.sort(key=lambda h: (h["start_offset"], h["end_offset"]))
    return aligned, dropped


def extract_heading_candidates(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Heading candidates per page, from PyMuPDF span geometry.

    A line qualifies when its spans' size reaches SIZE_RATIO_PLAIN over the
    page's length-weighted median span size (SIZE_RATIO_BOLD when the spans
    carry PyMuPDF's bold flag). Consecutive qualifying spans on the same
    baseline merge into one heading.
    """
    import fitz  # PyMuPDF — imported here so the module imports without it

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        candidates: list[dict[str, Any]] = []
        for page_number, page in enumerate(doc, start=1):
            data = page.get_text("dict")
            spans: list[dict[str, Any]] = []
            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            spans.append(
                                {
                                    "text": text,
                                    "size": float(span.get("size", 0.0)),
                                    "bold": bool(span.get("flags", 0) & 16),
                                    "y": round(
                                        float(line.get("bbox", [0, 0, 0, 0])[1]), 1
                                    ),
                                    "x": round(
                                        float(line.get("bbox", [0, 0, 0, 0])[0]), 1
                                    ),
                                }
                            )
            if not spans:
                continue

            # Length-weighted median body size for the page.
            sizes: list[float] = []
            for s in spans:
                sizes.extend([s["size"]] * max(1, len(s["text"])))
            sizes.sort()
            body_size = sizes[len(sizes) // 2] if sizes else 0.0
            if body_size <= 0:
                continue

            # Group spans into lines by baseline, reading order.
            lines: dict[float, list[dict[str, Any]]] = {}
            for s in sorted(spans, key=lambda s: (s["y"], s["x"])):
                lines.setdefault(s["y"], []).append(s)

            for _, line_spans in sorted(lines.items()):
                if not line_spans:
                    continue
                size = line_spans[0]["size"]
                bold = all(s["bold"] for s in line_spans)
                ratio = size / body_size
                if ratio < (SIZE_RATIO_BOLD if bold else SIZE_RATIO_PLAIN):
                    continue
                text = " ".join(s["text"] for s in line_spans).strip()
                if len(text) < MIN_HEADING_CHARS or not any(c.isalpha() for c in text):
                    continue
                candidates.append(
                    {
                        "page_number": page_number,
                        "text": text,
                        "font_size": round(size, 1),
                        "bold": bold,
                    }
                )
        return candidates
    finally:
        doc.close()


def run_layout_pass(
    pdf_bytes: bytes,
    pages: list[dict[str, Any]],
    full_text: str,
) -> dict[str, Any]:
    """Full pass: heading candidates -> alignment -> layout result.

    Never raises: any failure returns an empty layout with an error note,
    because text extraction has already committed (ADR-0040: non-fatal).
    """
    try:
        candidates = extract_heading_candidates(pdf_bytes)
    except Exception as e:  # noqa: BLE001 — deliberate: degrade, don't fail
        return {
            "headings": [],
            "pages_scanned": 0,
            "dropped": 0,
            "layout_error": str(e),
        }

    try:
        aligned, dropped = align_headings_to_pages(candidates, pages, full_text)
    except Exception as e:  # noqa: BLE001
        return {
            "headings": [],
            "pages_scanned": len(pages),
            "dropped": len(candidates),
            "layout_error": str(e),
        }

    return {
        "headings": aligned,
        "pages_scanned": len(pages),
        "dropped": dropped,
    }
