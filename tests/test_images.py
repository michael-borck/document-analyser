"""
Tests for embedded-image extraction (service + /files/extract-images route).

Fixture documents are generated with Pillow (its PDF writer embeds each
page's image as an XObject) and zipfile (DOCX media archive) — no binary
fixtures needed.
"""

import base64
import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from document_analyser.api import app
from document_analyser.services.image_extractor import extract_images

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _solid_image(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color)


def _pdf_bytes(pages: list[Image.Image]) -> bytes:
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()


@pytest.fixture
def pdf_with_images() -> bytes:
    """3-page PDF: page 1 and 2 carry the SAME image (dup), page 3 a different one."""
    image_a = _solid_image((600, 400), (200, 30, 30))
    image_b = _solid_image((300, 500), (30, 30, 200))
    return _pdf_bytes([image_a, image_a.copy(), image_b])


@pytest.fixture
def docx_with_images() -> bytes:
    """Minimal DOCX-shaped zip with two media images (one tiny)."""
    big = io.BytesIO()
    _solid_image((400, 300), (10, 120, 10)).save(big, format="PNG")
    tiny = io.BytesIO()
    _solid_image((10, 10), (0, 0, 0)).save(tiny, format="PNG")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", "<document/>")
        z.writestr("word/media/image1.png", big.getvalue())
        z.writestr("word/media/image2.png", tiny.getvalue())
    return buf.getvalue()


class TestImageExtractorService:
    def test_pdf_images_extracted_with_page_numbers(self, pdf_with_images: bytes):
        result = extract_images(pdf_with_images, "application/pdf")

        assert result["total_images"] == 2
        pages = [img["page_number"] for img in result["images"]]
        assert pages == [1, 3]  # page 2's image deduplicated against page 1's

    def test_pdf_duplicate_images_deduplicated(self, pdf_with_images: bytes):
        result = extract_images(pdf_with_images, "application/pdf")

        assert result["skipped"]["duplicate"] == 1
        hashes = {img["hash_sha256"] for img in result["images"]}
        assert len(hashes) == 2

    def test_renditions_are_decodable_base64_images(self, pdf_with_images: bytes):
        result = extract_images(pdf_with_images, "application/pdf")

        for img in result["images"]:
            for key in ("thumbnail_base64", "image_base64"):
                decoded = Image.open(io.BytesIO(base64.b64decode(img[key])))
                assert decoded.width > 0
            assert max(
                Image.open(io.BytesIO(base64.b64decode(img["thumbnail_base64"]))).size
            ) <= 320

    def test_dimensions_reported_from_source_image(self, pdf_with_images: bytes):
        result = extract_images(pdf_with_images, "application/pdf")

        first = result["images"][0]
        assert (first["width"], first["height"]) == (600, 400)

    def test_min_dimension_filters_tiny_images(self, docx_with_images: bytes):
        result = extract_images(docx_with_images, DOCX_MIME)

        assert result["total_images"] == 1
        assert result["skipped"]["tiny"] == 1

    def test_docx_images_have_no_page_number(self, docx_with_images: bytes):
        result = extract_images(docx_with_images, DOCX_MIME)

        assert result["images"][0]["page_number"] is None
        assert result["images"][0]["name"] == "image1.png"

    def test_max_images_cap(self, pdf_with_images: bytes):
        result = extract_images(pdf_with_images, "application/pdf", max_images=1)

        assert result["total_images"] == 1
        assert result["skipped"]["over_limit"] == 1

    def test_unsupported_type_returns_empty(self):
        result = extract_images(b"just text", "text/plain")

        assert result == {
            "total_images": 0,
            "images": [],
            "skipped": {"tiny": 0, "duplicate": 0, "undecodable": 0, "over_limit": 0},
        }


@pytest.fixture
def local_client():
    """TestClient whose ASGI scope reports a localhost peer (the route is
    localhost-only, and the default TestClient peer is 'testclient')."""
    with TestClient(app, client=("127.0.0.1", 9999)) as c:
        yield c


class TestExtractImagesRoute:
    def test_extracts_images_from_pdf_on_disk(
        self, local_client: TestClient, pdf_with_images: bytes, tmp_path: Path
    ):
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(pdf_with_images)

        response = local_client.post(
            "/files/extract-images", json={"file_path": str(pdf_path)}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "report.pdf"
        assert data["content_type"] == "application/pdf"
        assert data["total_images"] == 2
        assert data["images"][0]["page_number"] == 1
        assert data["images"][0]["thumbnail_mime"] in ("image/jpeg", "image/png")

    def test_rejects_non_local_client(self, pdf_with_images: bytes, tmp_path: Path):
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(pdf_with_images)
        with TestClient(app, client=("203.0.113.7", 9999)) as remote:
            response = remote.post(
                "/files/extract-images", json={"file_path": str(pdf_path)}
            )

        assert response.status_code == 403

    def test_missing_file_returns_400(self, local_client: TestClient):
        response = local_client.post(
            "/files/extract-images", json={"file_path": "/nonexistent/nope.pdf"}
        )

        assert response.status_code == 400

    def test_unsupported_format_returns_400(
        self, local_client: TestClient, tmp_path: Path
    ):
        txt_path = tmp_path / "notes.txt"
        txt_path.write_text("no images in here")

        response = local_client.post(
            "/files/extract-images", json={"file_path": str(txt_path)}
        )

        assert response.status_code == 400
        assert "PDF and DOCX" in response.json()["detail"]
