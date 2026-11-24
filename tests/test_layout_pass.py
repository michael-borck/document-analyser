#!/usr/bin/env python3
"""
Layout pass tests (ADR-0040).

The fixture PDF is built in-test with PyMuPDF itself — deterministic, no
committed binary. Pure alignment logic is tested without touching a PDF.
"""

import fitz

from document_analyser.services.document_processor import DocumentProcessor
from document_analyser.services.layout_pass import (
    align_heading,
    align_headings_to_pages,
    align_paragraph_blocks,
    block_text,
    normalise_with_map,
    printed_page_for,
    printed_page_map,
    run_layout_pass,
    run_paragraph_pass,
)


def make_fixture_pdf() -> bytes:
    """Two pages: a bold large heading + body on each."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "From the Vice-Chancellor", fontsize=18, fontname="hebo")
    page.insert_text(
        (72, 100),
        "The university advanced renewable energy initiatives this year.",
        fontsize=10,
    )
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Operations", fontsize=16, fontname="hebo")
    page2.insert_text((72, 100), "Water conservation programs cut demand.", fontsize=10)
    return doc.tobytes()


# ---------------------------------------------------------------------------
# Pure alignment logic
# ---------------------------------------------------------------------------


def test_normalise_with_map_projects_matches_back():
    original = "Climate  action\n now"
    norm, index_map = normalise_with_map(original)
    assert norm == "Climate action now"
    # A match found in the normalised text projects to original offsets.
    idx = norm.find("action")
    start = index_map[idx]
    end = index_map[idx + len("action") - 1] + 1
    assert original[start:end] == "action"


def test_align_heading_survives_whitespace_divergence():
    # pypdf and PyMuPDF often disagree about intra-heading spacing: the
    # returned span covers the ORIGINAL text extent, which normalises
    # back to the heading.
    original = "From  the\nVice-Chancellor"
    norm, index_map = normalise_with_map(original)
    span = align_heading("From the Vice-Chancellor", norm, index_map, 40)
    assert span is not None
    start, end = span
    assert start == 40  # heading starts the page text
    sliced = original[start - 40 : end - 40]
    assert normalise_with_map(sliced)[0].strip() == "From the Vice-Chancellor"


def test_align_heading_drops_when_absent():
    norm, index_map = normalise_with_map("Body text only, no heading here.")
    assert align_heading("From the Vice-Chancellor", norm, index_map, 0) is None


def test_align_headings_computes_page_offsets_with_empty_pages():
    # Page 2 is empty — it contributes nothing to full_text and must be
    # skipped in the offset walk (mirrors the "\n\n".join contract).
    pages = [
        {"page_number": 1, "text": "Big Heading"},
        {"page_number": 2, "text": ""},
        {"page_number": 3, "text": "Second Heading"},
    ]
    full_text = "Big Heading\n\nSecond Heading"
    headings = [
        {"page_number": 1, "text": "Big Heading", "font_size": 18.0, "bold": True},
        {"page_number": 3, "text": "Second Heading", "font_size": 16.0, "bold": True},
    ]
    aligned, dropped = align_headings_to_pages(headings, pages, full_text)
    assert dropped == 0
    assert [h["start_offset"] for h in aligned] == [0, full_text.find("Second Heading")]
    for h in aligned:
        assert full_text[h["start_offset"] : h["end_offset"]] == h["text"]


def test_align_headings_counts_dropped_candidates():
    pages = [{"page_number": 1, "text": "Real Heading"}]
    full_text = "Real Heading"
    headings = [
        {"page_number": 1, "text": "Real Heading", "font_size": 18.0, "bold": True},
        {"page_number": 1, "text": "Phantom Heading", "font_size": 17.0, "bold": True},
    ]
    aligned, dropped = align_headings_to_pages(headings, pages, full_text)
    assert len(aligned) == 1
    assert dropped == 1


# ---------------------------------------------------------------------------
# End-to-end over a real (in-test-built) PDF
# ---------------------------------------------------------------------------


def test_extract_text_with_pages_includes_aligned_layout():
    content = make_fixture_pdf()
    processor = DocumentProcessor()
    result = _run(processor, content)

    layout = result["layout"]
    assert layout["layout_error"] is None if "layout_error" in layout else True
    assert layout["dropped"] == 0
    headings = layout["headings"]
    assert len(headings) == 2

    first, second = headings
    assert "Vice-Chancellor" in first["text"]
    assert first["page_number"] == 1
    assert second["page_number"] == 2
    assert "Operations" in second["text"]

    # Offsets must land on the heading inside the canonical full text.
    full = result["full_text"]
    assert "Vice-Chancellor" in full[first["start_offset"] : first["end_offset"]]
    assert "Operations" in full[second["start_offset"] : second["end_offset"]]

    # Heading offsets precede their page's body text.
    assert full[first["end_offset"] :].lstrip().startswith("The university advanced")


def test_layout_pass_is_non_fatal_on_garbage():
    result = run_layout_pass(
        b"not a pdf at all", [{"page_number": 1, "text": "x"}], "x"
    )
    assert result["headings"] == []
    assert "layout_error" in result


def _run(processor: DocumentProcessor, content: bytes) -> dict:
    import asyncio

    return asyncio.run(
        processor.extract_text_with_pages(content, "application/pdf", "fixture.pdf")
    )


# ---------------------------------------------------------------------------
# Paragraph blocks (the coding sheet's unit)
# ---------------------------------------------------------------------------


def test_block_text_heals_line_end_hyphenation():
    assert block_text("cli-\nmate action") == "climate action"
    assert block_text("two\nseparate lines") == "two separate lines"


def test_align_paragraph_blocks_tolerates_punctuation_and_spacing():
    text = (
        "Monash University annual report\n\n"
        "The climate strategy targets net zero by 2035.\n"
        "Water use fell 12 per cent."
    )
    pages = [{"page_number": 1, "text": text}]
    blocks = [
        {"page_number": 1, "text": "Monash University annual report"},
        {"page_number": 1, "text": "The climate strategy targets net zero by 2035."},
        {"page_number": 1, "text": "Water use fell 12 per cent."},
    ]
    aligned, dropped = align_paragraph_blocks(blocks, pages, text)
    assert dropped == 0
    assert [a["text"] for a in aligned] == [b["text"] for b in blocks]
    starts = [a["start_offset"] for a in aligned]
    assert starts == sorted(starts)
    assert text[starts[1] :].startswith("The climate")


def test_align_paragraph_blocks_drops_a_block_absent_from_the_text():
    text = "Only this text is present."
    pages = [{"page_number": 1, "text": text}]
    aligned, dropped = align_paragraph_blocks(
        [{"page_number": 1, "text": "A passage that is not in the text at all"}],
        pages,
        text,
    )
    assert aligned == []
    assert dropped == 1


def test_paragraph_pass_on_a_real_pdf_is_ordered_and_complete():
    content = make_fixture_pdf()
    result = _run(DocumentProcessor(), content)
    paragraph_pass = run_paragraph_pass(content, result["pages"], result["full_text"])
    assert paragraph_pass.get("paragraph_error") is None
    texts = " ".join(p["text"] for p in paragraph_pass["paragraphs"])
    assert "renewable energy" in texts
    assert "Water conservation" in texts
    starts = [p["start_offset"] for p in paragraph_pass["paragraphs"]]
    assert starts == sorted(starts)


def test_printed_page_map_falls_back_without_labels():
    content = make_fixture_pdf()
    page_map = printed_page_map(content)
    assert [p["pdf_page"] for p in page_map] == [1, 2]
    assert [p["printed_page"] for p in page_map] == [1, 2]
    assert all(p["source"] == "fallback" for p in page_map)


def test_printed_page_for_reads_labels_not_a_fixed_offset():
    # ANU's own labels: roman front matter, then arabic two behind the PDF page.
    assert printed_page_for("1", True, 3) == 1
    assert printed_page_for("246", True, 248) == 246
    # Non-numeric labels are blank, never coerced or guessed.
    assert printed_page_for("ii", True, 2) is None
    assert printed_page_for("A-1", True, 9) is None
    assert (
        printed_page_for("²", True, 9) is None
    )  # superscript: isdigit() is true, int() throws
    assert printed_page_for(None, True, 4) is None
    # No declared labels: the PDF page is the fallback.
    assert printed_page_for(None, False, 7) == 7
    assert printed_page_for("x", False, 7) == 7
