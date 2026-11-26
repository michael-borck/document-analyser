"""Write rows into copies of a coding-sheet template, driven by a profile.

The engine has no opinion about SDGs, universities, or any study. A profile
says which sheet to use, where data starts, which Excel column receives which
payload field (or a constant, or the engine version), and which payload field
receives each copy's label. A second domain supplies a different profile; no
code changes.

Safety properties, in order of importance:

- only mapped columns are touched;
- a mapped column must be empty in the template, so a filled workbook is never
  silently loaded over;
- more rows than the template holds is an error, not a truncation;
- a value that begins with "=" is text, not a formula.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string
from pydantic import BaseModel, Field

ENGINE_SOURCES = {"version"}


class CellMapping(BaseModel):
    """One Excel column and where its value comes from."""

    column: str = Field(..., min_length=1)
    field: str | None = None
    constant: Any = None
    engine: str | None = None


class Profile(BaseModel):
    """A coding-sheet layout: which template columns the engine fills, and how."""

    sheet: str = Field(..., min_length=1)
    first_data_row: int = Field(5, ge=2)
    capacity: int | None = Field(None, ge=1)
    cells: list[CellMapping] = Field(..., min_length=1)
    copy_field: str | None = None
    recalculate_on_open: bool = True


class CopySpec(BaseModel):
    """One workbook to write: the label that goes in `copy_field`, and its name."""

    coder: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)


def _column_index(letter: str) -> int:
    try:
        index = column_index_from_string(letter.strip().upper())
    except ValueError as e:
        raise ValueError(f"Invalid column letter: {letter!r}") from e
    if index < 1:
        raise ValueError(f"Invalid column letter: {letter!r}")
    return index


def _validate(profile: Profile) -> list[tuple[int, CellMapping]]:
    mapped: list[tuple[int, CellMapping]] = []
    seen: dict[int, str] = {}
    for cell in profile.cells:
        sources = [s for s in (cell.field, cell.constant, cell.engine) if s is not None]
        if len(sources) != 1:
            raise ValueError(
                f"Column {cell.column}: exactly one of field, constant or engine is required"
            )
        if cell.field is not None and not cell.field.strip():
            raise ValueError(f"Column {cell.column}: field must not be blank")
        if cell.engine is not None and cell.engine not in ENGINE_SOURCES:
            raise ValueError(
                f"Column {cell.column}: unknown engine source {cell.engine!r} "
                f"(known: {sorted(ENGINE_SOURCES)})"
            )
        index = _column_index(cell.column)
        if index in seen:
            raise ValueError(f"Column {cell.column} is mapped more than once")
        seen[index] = cell.column
        mapped.append((index, cell))
    if profile.copy_field is not None:
        if not profile.copy_field.strip():
            raise ValueError("copy_field must not be blank")
        if not any(cell.field == profile.copy_field for _, cell in mapped):
            raise ValueError(
                f"copy_field {profile.copy_field!r} is not mapped to a column; "
                "add a cell with that field"
            )
    return mapped


# XML 1.0 forbids these; openpyxl refuses a cell that contains one. Index pages
# and table rows from PDFs do contain them, so the writer drops them rather than
# failing the whole batch.
_ILLEGAL_XML_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(value: Any) -> Any:
    return _ILLEGAL_XML_CHARS.sub("", value) if isinstance(value, str) else value


def _write(sheet: Any, row: int, column: int, value: Any) -> None:
    value = _clean(value)
    cell = sheet.cell(row=row, column=column)
    cell.value = value
    # A passage or keyword list that begins with "=" is text, not a formula.
    if isinstance(value, str) and value.startswith("="):
        cell.data_type = "s"


def build_workbooks(
    *,
    template_path: str | Path,
    output_dir: str | Path,
    profile: Profile,
    rows: list[dict[str, Any]],
    copies: list[CopySpec],
    engine_version: str,
) -> list[dict[str, Any]]:
    """Write one workbook per copy. Returns {coder, path, rows} per file."""
    template = Path(template_path)
    out_dir = Path(output_dir)
    if not template.is_file():
        raise ValueError(f"Template not found: {template}")
    if not out_dir.is_dir():
        raise ValueError(f"Output directory not found: {out_dir}")
    if not copies:
        raise ValueError("At least one copy is required")

    mapped = _validate(profile)
    for copy in copies:
        if Path(copy.filename).name != copy.filename:
            raise ValueError(f"filename must be a bare file name: {copy.filename!r}")

    written: list[dict[str, Any]] = []
    for copy in copies:
        workbook = load_workbook(template)
        if profile.sheet not in workbook.sheetnames:
            raise ValueError(f"Template has no {profile.sheet!r} sheet")
        sheet = workbook[profile.sheet]
        if profile.recalculate_on_open:
            workbook.calculation.fullCalcOnLoad = True

        available = sheet.max_row - profile.first_data_row + 1
        capacity = profile.capacity if profile.capacity is not None else available
        if capacity > available:
            raise ValueError(
                f"Profile capacity {capacity} exceeds the template's {available} data rows"
            )
        if len(rows) > capacity:
            raise ValueError(
                f"{len(rows)} rows exceeds the {capacity} data rows "
                f"(rows {profile.first_data_row}..{profile.first_data_row + capacity - 1})"
            )

        # A template must have empty mapped columns: mixing exported rows with
        # leftover data would corrupt the sheet silently.
        for index, cell in mapped:
            for row_number in range(
                profile.first_data_row, profile.first_data_row + capacity
            ):
                if sheet.cell(row=row_number, column=index).value is not None:
                    raise ValueError(
                        f"Template column {cell.column} is not empty at row "
                        f"{row_number}; a template must have empty mapped columns"
                    )

        for offset, row in enumerate(rows):
            excel_row = profile.first_data_row + offset
            for index, cell in mapped:
                if cell.field is not None:
                    if cell.field == profile.copy_field:
                        value = copy.coder
                    elif cell.field in row:
                        value = row[cell.field]
                    else:
                        raise ValueError(
                            f"Row {offset + 1} has no {cell.field!r} for column {cell.column}"
                        )
                elif cell.engine is not None:
                    value = engine_version
                else:
                    value = cell.constant
                _write(sheet, excel_row, index, value)

        path = out_dir / copy.filename
        workbook.save(path)
        written.append({"coder": copy.coder, "path": str(path), "rows": len(rows)})
    return written
