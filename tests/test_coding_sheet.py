#!/usr/bin/env python3
"""
Coding-sheet writer tests.

Two contracts matter. The profile decides everything domain-specific (columns,
constants, copies), so the same code writes another study's sheet unchanged.
And the template survives: formulas, data validations and defined names are
still there after the round trip.
"""

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook, load_workbook
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.datavalidation import DataValidation

from document_analyser.api import app
from document_analyser.services.coding_sheet import (
    CellMapping,
    CopySpec,
    Profile,
    build_workbooks,
)


def _template(path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Coding sheet"
    sheet["A4"] = "Claim ID"
    sheet["H4"] = "Keyword(s)"
    sheet["I4"] = "Passage"
    sheet["U4"] = "Row check (auto)"
    sheet["AF4"] = "Coder"
    sheet["U5"] = '=IF($A5="","","Done")'
    sheet["U6"] = '=IF($A6="","","Done")'
    validation = DataValidation(
        type="list", formula1='"Relevant,False positive"', allow_blank=True
    )
    sheet.add_data_validation(validation)
    validation.add("K5:K6")
    workbook.defined_names["PR_Uni"] = DefinedName(
        "PR_Uni", attr_text="'Coding sheet'!$A$5:$A$6"
    )
    workbook.save(path)
    return path


def _profile() -> Profile:
    return Profile(
        sheet="Coding sheet",
        first_data_row=5,
        copy_field="coder",
        cells=[
            CellMapping(column="A", field="claim_id"),
            CellMapping(column="G", field="primary_value"),
            CellMapping(column="H", field="keywords"),
            CellMapping(column="I", field="passage"),
            CellMapping(column="J", field="context"),
            CellMapping(column="AF", field="coder"),
            CellMapping(column="AG", constant="layout-1"),
            CellMapping(column="AH", engine="version"),
            CellMapping(column="AI", constant="words-2026-11"),
        ],
    )


def _rows():
    return [
        {
            "claim_id": "EX-2025-AR-0001",
            "primary_value": 13,
            "keywords": "net zero; emission",
            "passage": "We will reach net zero by 2035.",
            "context": "(previous paragraph)",
        },
        {
            "claim_id": "EX-2025-AR-0002",
            "primary_value": "General",
            "keywords": "SDGs",
            "passage": "=this is text, not a formula",
            "context": "",
        },
    ]


def _copies():
    return [
        CopySpec(coder="Burbridge", filename="sheet_Burbridge.xlsx"),
        CopySpec(coder="Naef", filename="sheet_Naef.xlsx"),
    ]


def test_writes_rows_and_preserves_the_template(tmp_path):
    template = _template(tmp_path / "template.xlsx")
    out = tmp_path / "out"
    out.mkdir()

    files = build_workbooks(
        template_path=template,
        output_dir=out,
        profile=_profile(),
        rows=_rows(),
        copies=_copies(),
        engine_version="9.9.9",
    )

    assert [f["coder"] for f in files] == ["Burbridge", "Naef"]
    assert all(f["rows"] == 2 for f in files)

    workbook = load_workbook(files[0]["path"])
    sheet = workbook["Coding sheet"]
    assert sheet["A5"].value == "EX-2025-AR-0001"
    assert sheet["G5"].value == 13
    assert sheet["G6"].value == "General"
    assert sheet["H5"].value == "net zero; emission"
    assert sheet["J5"].value == "(previous paragraph)"
    assert sheet["AF5"].value == "Burbridge"
    assert sheet["AF6"].value == "Burbridge"
    assert sheet["AG5"].value == "layout-1"
    assert sheet["AH5"].value == "9.9.9"
    assert sheet["AI5"].value == "words-2026-11"

    # The instrument survives: formulas, validations, defined names.
    assert sheet["U5"].value == '=IF($A5="","","Done")'
    assert len(sheet.data_validations.dataValidation) == 1
    assert "PR_Uni" in workbook.defined_names

    # A passage that begins with "=" is text, not a formula.
    assert sheet["I6"].value == "=this is text, not a formula"
    assert sheet["I6"].data_type == "s"

    # The second copy differs only in its copy field.
    other = load_workbook(files[1]["path"])["Coding sheet"]
    assert other["AF5"].value == "Naef"
    assert other["A5"].value == "EX-2025-AR-0001"


def test_empty_mapped_columns_reports_a_filled_template(tmp_path):
    template = _template(tmp_path / "template.xlsx")
    workbook = load_workbook(template)
    workbook["Coding sheet"]["A5"] = "stale row"
    workbook.save(template)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="not empty"):
        build_workbooks(
            template_path=template,
            output_dir=out,
            profile=_profile(),
            rows=_rows()[:1],
            copies=_copies()[:1],
            engine_version="9.9.9",
        )


def test_rejects_more_rows_than_the_template_holds(tmp_path):
    template = _template(tmp_path / "template.xlsx")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="exceeds"):
        build_workbooks(
            template_path=template,
            output_dir=out,
            profile=_profile(),
            rows=_rows() * 2,  # 4 rows against 2 available
            copies=_copies()[:1],
            engine_version="9.9.9",
        )


def test_profile_is_validated(tmp_path):
    template = _template(tmp_path / "template.xlsx")
    out = tmp_path / "out"
    out.mkdir()
    base = dict(
        template_path=template,
        output_dir=out,
        rows=_rows()[:1],
        copies=_copies()[:1],
        engine_version="9.9.9",
    )
    with pytest.raises(ValueError, match="more than once"):
        build_workbooks(
            **base,
            profile=_profile().model_copy(
                update={
                    "cells": [
                        CellMapping(column="A", field="claim_id"),
                        CellMapping(column="a", field="keywords"),
                    ]
                }
            ),
        )
    with pytest.raises(ValueError, match="unknown engine source"):
        build_workbooks(
            **base,
            profile=_profile().model_copy(
                update={
                    "cells": [
                        CellMapping(column="A", field="claim_id"),
                        CellMapping(column="B", engine="nope"),
                    ]
                }
            ),
        )
    with pytest.raises(ValueError, match="not mapped"):
        build_workbooks(
            **base,
            profile=_profile().model_copy(
                update={
                    "copy_field": "reviewer",
                    "cells": [CellMapping(column="A", field="claim_id")],
                }
            ),
        )
    with pytest.raises(ValueError, match="exactly one of"):
        build_workbooks(
            **base,
            profile=_profile().model_copy(
                update={
                    "cells": [
                        CellMapping(column="A", field="claim_id", constant="both")
                    ]
                }
            ),
        )


def test_filename_must_be_a_bare_name(tmp_path):
    template = _template(tmp_path / "template.xlsx")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="bare file name"):
        build_workbooks(
            template_path=template,
            output_dir=out,
            profile=_profile(),
            rows=_rows()[:1],
            copies=[CopySpec(coder="X", filename="../escape.xlsx")],
            engine_version="9.9.9",
        )


def _fixture_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "From the Vice-Chancellor", fontsize=18, fontname="hebo")
    page.insert_text(
        (72, 100), "The university advanced renewable energy.", fontsize=10
    )
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Operations", fontsize=16, fontname="hebo")
    page2.insert_text((72, 100), "Water conservation cut demand.", fontsize=10)
    content = doc.tobytes()
    doc.close()
    return content


def test_paragraphs_endpoint(tmp_path):
    pdf = tmp_path / "fixture.pdf"
    pdf.write_bytes(_fixture_pdf())
    client = TestClient(app)
    response = client.post("/layout/paragraphs", json={"file_path": str(pdf)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["page_count"] == 2
    assert body["paragraph_count"] >= 2
    assert body["dropped"] == 0
    assert len(body["page_map"]) == 2
    assert body["page_map"][0]["source"] == "fallback"
    assert len(body["full_text_sha256"]) == 64


def test_build_endpoint(tmp_path):
    template = _template(tmp_path / "template.xlsx")
    out = tmp_path / "out"
    out.mkdir()
    client = TestClient(app)
    response = client.post(
        "/coding-sheet/build",
        json={
            "template_path": str(template),
            "output_dir": str(out),
            "profile": _profile().model_dump(),
            "rows": _rows(),
            "copies": [{"coder": "Burbridge", "filename": "sheet_Burbridge.xlsx"}],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["row_count"] == 2
    assert len(body["files"]) == 1
    assert (out / "sheet_Burbridge.xlsx").is_file()


def test_build_endpoint_rejects_a_bad_profile(tmp_path):
    template = _template(tmp_path / "template.xlsx")
    out = tmp_path / "out"
    out.mkdir()
    client = TestClient(app)
    response = client.post(
        "/coding-sheet/build",
        json={
            "template_path": str(template),
            "output_dir": str(out),
            "profile": {
                "sheet": "Coding sheet",
                "cells": [{"column": "A"}, {"column": "B", "constant": "x"}],
            },
            "rows": [{"claim_id": "X"}],
            "copies": [{"coder": "X", "filename": "x.xlsx"}],
        },
    )
    assert response.status_code == 400


def test_strips_control_characters_from_text(tmp_path):
    template = _template(tmp_path / "template.xlsx")
    out = tmp_path / "out"
    out.mkdir()
    rows = _rows()
    rows[0]["passage"] = "Audit\x08 and assurance\x0b 113, 130"
    build_workbooks(
        template_path=template,
        output_dir=out,
        profile=_profile(),
        rows=rows,
        copies=[CopySpec(coder="X", filename="x.xlsx")],
        engine_version="9.9.9",
    )
    sheet = load_workbook(out / "x.xlsx")["Coding sheet"]
    assert sheet["I5"].value == "Audit and assurance 113, 130"
