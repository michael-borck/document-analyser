"""Coding-sheet endpoints for the v11 researcher workbook.

Two calls, in the order the desktop app makes them:

1. ``POST /layout/paragraphs`` — paragraph blocks anchored to the canonical
   extracted text, plus the PDF-to-printed page map. The paragraph is the
   coding sheet's row unit; the page map keeps citations honest on reports
   whose printed numbering differs from the PDF's.
2. ``POST /coding-sheet/build`` — write the caller's rows into copies of the
   researcher's workbook, one per coder. The app owns matching and row
   assembly; this endpoint owns the workbook so the template's formulas and
   validations survive.
"""

import hashlib
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from document_analyser.manifest import MANIFEST
from document_analyser.services.coding_sheet import CopySpec, Profile, build_workbooks
from document_analyser.services.document_processor import DocumentProcessor
from document_analyser.services.layout_pass import printed_page_map, run_paragraph_pass

router = APIRouter()


class ParagraphRequest(BaseModel):
    """A document on disk (the desktop app already sends paths this way)."""

    file_path: str = Field(..., description="Absolute path to the PDF")


class Paragraph(BaseModel):
    page_number: int
    start_offset: int
    text: str


class PageMapEntry(BaseModel):
    pdf_page: int
    printed_page: int | None
    label: str | None
    rotation: int
    source: str


class ParagraphResponse(BaseModel):
    page_count: int
    full_text_length: int
    full_text_sha256: str
    paragraph_count: int
    dropped: int
    paragraphs: list[Paragraph]
    page_map: list[PageMapEntry]
    rotated_pages: list[int]


class BuildRequest(BaseModel):
    template_path: str = Field(
        ..., description="Absolute path to the workbook template"
    )
    output_dir: str = Field(
        ..., description="Existing directory to write the copies to"
    )
    profile: Profile = Field(..., description="Which columns the engine fills, and how")
    rows: list[dict[str, Any]] = Field(
        ..., description="Payload rows, keyed by profile field"
    )
    copies: list[CopySpec] = Field(..., min_length=1)


class WrittenFile(BaseModel):
    coder: str
    path: str
    rows: int


class BuildResponse(BaseModel):
    files: list[WrittenFile]
    row_count: int


def _pdf_path(file_path: str) -> Path:
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="file_path must be a PDF")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return path


@router.post("/layout/paragraphs", response_model=ParagraphResponse)
async def layout_paragraphs(req: ParagraphRequest) -> ParagraphResponse:
    path = _pdf_path(req.file_path)
    content = path.read_bytes()
    extracted = await DocumentProcessor().extract_text_with_pages(
        content, "application/pdf", path.name
    )
    result = run_paragraph_pass(content, extracted["pages"], extracted["full_text"])
    if result.get("paragraph_error"):
        raise HTTPException(status_code=422, detail=str(result["paragraph_error"]))

    page_map = printed_page_map(content)
    return ParagraphResponse(
        page_count=extracted["total_pages"],
        full_text_length=len(extracted["full_text"]),
        full_text_sha256=hashlib.sha256(
            extracted["full_text"].encode("utf-8")
        ).hexdigest(),
        paragraph_count=result["aligned"],
        dropped=result["dropped"],
        paragraphs=[Paragraph(**p) for p in result["paragraphs"]],
        page_map=[PageMapEntry(**p) for p in page_map],
        rotated_pages=[p["pdf_page"] for p in page_map if p["rotation"] % 360 != 0],
    )


@router.post("/coding-sheet/build", response_model=BuildResponse)
async def build_coding_sheet(req: BuildRequest) -> BuildResponse:
    template = Path(req.template_path)
    if template.suffix.lower() != ".xlsx":
        raise HTTPException(
            status_code=400, detail="template_path must be an .xlsx file"
        )
    try:
        files = build_workbooks(
            template_path=template,
            output_dir=req.output_dir,
            profile=req.profile,
            rows=req.rows,
            copies=req.copies,
            engine_version=MANIFEST["version"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return BuildResponse(
        files=[WrittenFile(**f) for f in files], row_count=len(req.rows)
    )
