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
from collections import defaultdict
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


# ---------------------------------------------------------------------------
# Paragraph blocks: the coding sheet's unit
# ---------------------------------------------------------------------------
# One row per paragraph per SDG. Blank-line splitting cannot find paragraphs in
# pypdf text (PDFs rarely emit blank lines), and it drops short table lines. So
# paragraphs come from page geometry (PyMuPDF text blocks) and are anchored
# into the canonical pypdf text by token sequence, which survives the spacing
# and hyphenation differences the two extractors produce. A block whose tokens
# cannot be found is dropped and counted, never guessed (ADR-0026).

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ANCHOR_SIZES = (12, 8, 5)


def block_text(raw: str) -> str:
    """Join a block's lines into one paragraph, healing line-end hyphenation."""
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    out = ""
    for line in lines:
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = (out + " " + line).strip()
    return out


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def extract_paragraph_candidates(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """One candidate per text block, in page order, with its page number."""
    import fitz  # PyMuPDF — imported here so the module imports without it

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        candidates: list[dict[str, Any]] = []
        for page_number, page in enumerate(doc, start=1):
            for block in page.get_text("blocks"):
                if block[6] != 0:  # 1 = image block
                    continue
                text = block_text(block[4])
                if not _tokens(text):
                    continue
                candidates.append({"page_number": page_number, "text": text})
        return candidates
    finally:
        doc.close()


def _find_token_run(
    tokens: list[tuple[str, int, int]],
    index: dict[tuple[str, str, str], list[int]],
    needle: list[str],
    start: int,
) -> int:
    """First position at or after `start` where `needle` occurs."""
    size = len(needle)
    if size < 3:
        for i in range(start, len(tokens) - size + 1):
            if [t[0] for t in tokens[i : i + size]] == needle:
                return i
        return -1
    for i in index.get((needle[0], needle[1], needle[2]), []):
        if i < start:
            continue
        if [t[0] for t in tokens[i : i + size]] == needle:
            return i
    return -1


def align_paragraph_blocks(
    blocks: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    full_text: str,
) -> tuple[list[dict[str, Any]], int]:
    """Anchor blocks into full-text offsets by order-preserving token search.

    Returns (paragraphs, dropped). Each paragraph carries `page_number`,
    `start_offset` and its own text. Paragraph extents are [start, next start),
    so every keyword span lands in exactly one paragraph: no gaps for the
    export to fall through.
    """
    page_offsets: dict[int, int] = {}
    cursor = 0
    non_empty = [p for p in pages if (p.get("text") or "").strip()]
    for i, p in enumerate(non_empty):
        page_offsets[p["page_number"]] = cursor
        cursor += len(p["text"]) + (2 if i < len(non_empty) - 1 else 0)

    prepared: dict[
        int, tuple[list[tuple[str, int, int]], dict[tuple[str, str, str], list[int]]]
    ] = {}
    for p in non_empty:
        tokens = _tokens(p["text"])
        index: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for i in range(len(tokens) - 2):
            index[(tokens[i][0], tokens[i + 1][0], tokens[i + 2][0])].append(i)
        prepared[p["page_number"]] = (tokens, index)

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for b in blocks:
        by_page[b["page_number"]].append(b)

    aligned: list[dict[str, Any]] = []
    dropped = 0
    for page_number in sorted(by_page):
        page = prepared.get(page_number)
        base = page_offsets.get(page_number)
        if page is None or base is None:
            dropped += len(by_page[page_number])
            continue
        tokens, index = page
        search_from = 0
        for b in by_page[page_number]:
            words = [t[0] for t in _tokens(b["text"])]
            hit = -1
            for size in _ANCHOR_SIZES:
                if len(words) < size:
                    continue
                hit = _find_token_run(tokens, index, words[:size], search_from)
                if hit >= 0:
                    break
            if hit < 0 and len(words) < _ANCHOR_SIZES[-1]:
                hit = _find_token_run(tokens, index, words, search_from)
            if hit < 0:
                dropped += 1
                continue
            search_from = hit
            aligned.append(
                {
                    "page_number": page_number,
                    "start_offset": base + tokens[hit][1],
                    "text": b["text"],
                }
            )

    aligned.sort(key=lambda p: p["start_offset"])
    return aligned, dropped


def run_paragraph_pass(
    pdf_bytes: bytes,
    pages: list[dict[str, Any]],
    full_text: str,
) -> dict[str, Any]:
    """Paragraph pass: candidates -> alignment. Non-fatal, like the layout pass."""
    try:
        candidates = extract_paragraph_candidates(pdf_bytes)
    except Exception as e:  # noqa: BLE001 — degrade, don't fail
        return {"paragraphs": [], "aligned": 0, "dropped": 0, "paragraph_error": str(e)}
    try:
        aligned, dropped = align_paragraph_blocks(candidates, pages, full_text)
    except Exception as e:  # noqa: BLE001
        return {
            "paragraphs": [],
            "aligned": 0,
            "dropped": len(candidates),
            "paragraph_error": str(e),
        }
    return {"paragraphs": aligned, "aligned": len(aligned), "dropped": dropped}


_ASCII_NUMBER = re.compile(r"[0-9]+")


def printed_page_for(label: str | None, has_labels: bool, pdf_page: int) -> int | None:
    """Printed page number for one page, from the PDF's own label.

    No offset is assumed and no university is known: a PDF that declares page
    labels gets them read per page (so roman front matter and restarted
    numbering both work), and a PDF that declares none falls back to its PDF
    page. Labels that are not plain ASCII digits are blank, not guessed.
    """
    if not has_labels:
        return pdf_page
    if label is None:
        return None
    text = label.strip()
    return int(text) if _ASCII_NUMBER.fullmatch(text) else None


def printed_page_map(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """PDF page -> printed page, from the PDF's own page labels.

    A PDF either declares page labels (roman front matter, then arabic) or it
    does not. When it does not, the printed page falls back to the PDF page and
    the fallback is marked, so a landscape report or an unnumbered report shows
    up in the page map rather than quietly shifting citations.
    """
    import io

    import fitz
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    has_labels = False
    try:
        root = reader.trailer.get("/Root") or {}
        has_labels = "/PageLabels" in root
    except Exception:  # noqa: BLE001 — a broken catalog degrades to fallback
        has_labels = False

    labels: list[str] = []
    if has_labels:
        try:
            labels = [str(label) for label in reader.page_labels]
        except Exception:  # noqa: BLE001
            has_labels = False

    rotations: list[int] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        rotations = [int(page.rotation) for page in doc]
    finally:
        doc.close()

    page_map: list[dict[str, Any]] = []
    for index in range(len(reader.pages)):
        label = labels[index] if index < len(labels) else None
        printed = printed_page_for(label, has_labels, index + 1)
        page_map.append(
            {
                "pdf_page": index + 1,
                "printed_page": printed,
                "label": label,
                "rotation": rotations[index] if index < len(rotations) else 0,
                "source": "labels" if has_labels else "fallback",
            }
        )
    return page_map
