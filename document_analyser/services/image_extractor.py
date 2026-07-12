"""
Embedded-image extraction from documents (PDF, DOCX).

Extraction only — this module finds images inside container documents and
returns their bytes plus locating metadata (page number, index, dimensions,
content hash). It deliberately does NOT analyse image content; in the lens
family that is image-analyser's job, and callers compose the two.

Design notes:
- PDFs: pypdf's `page.images` (already a core dependency). Page numbers are
  1-based so callers can deep-link viewers ("#page=N").
- DOCX: images live in the archive's `word/media/`; a flow format has no
  pages, so `page_number` is None for them.
- Dedup by SHA-256 of the embedded bytes: annual reports repeat the same
  logo on every page — without dedup a gallery is 90% logo.
- Tiny images (icons, bullets, rules) are skipped via `min_dimension`.
- Two renditions are returned per image: a thumbnail for grids and a
  display rendition capped at `display_max` px (open the original document
  for the true full-resolution image).
- Per-image failures (exotic filters pypdf can't decode, e.g. JBIG2) are
  counted in `skipped.undecodable`, never fatal.
"""

import base64
import hashlib
import io
import zipfile
from typing import Any

from PIL import Image

# Formats browsers can render natively; anything else is transcoded.
_BROWSER_SAFE = {"JPEG", "PNG", "GIF", "WEBP"}

_DOCX_MEDIA_PREFIX = "word/media/"


def extract_images(
    content: bytes,
    content_type: str,
    *,
    min_dimension: int = 32,
    max_images: int = 200,
    thumbnail_max: int = 320,
    display_max: int = 1600,
) -> dict[str, Any]:
    """
    Extract embedded images from a document.

    Args:
        content: Raw document bytes.
        content_type: MIME type of the document.
        min_dimension: Skip images whose width or height is below this.
        max_images: Stop after this many kept images (counted in skipped.over_limit).
        thumbnail_max: Longest side of the thumbnail rendition.
        display_max: Longest side of the display rendition.

    Returns:
        {"total_images": int, "images": [...], "skipped": {...}}
        Each image dict carries page_number (1-based, None for DOCX),
        image_index, width, height, hash_sha256, and base64 thumbnail +
        display renditions with their MIME types.
    """
    extractor = _Extractor(
        min_dimension=min_dimension,
        max_images=max_images,
        thumbnail_max=thumbnail_max,
        display_max=display_max,
    )

    if content_type == "application/pdf":
        extractor.walk_pdf(content)
    elif content_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        extractor.walk_docx(content)
    # Other formats (txt, md, pptx for now) simply have no extractable
    # images here — return an empty, well-formed result.

    return {
        "total_images": len(extractor.images),
        "images": extractor.images,
        "skipped": extractor.skipped,
    }


class _Extractor:
    def __init__(
        self, *, min_dimension: int, max_images: int, thumbnail_max: int, display_max: int
    ) -> None:
        self.min_dimension = min_dimension
        self.max_images = max_images
        self.thumbnail_max = thumbnail_max
        self.display_max = display_max
        self.images: list[dict[str, Any]] = []
        self.skipped = {"tiny": 0, "duplicate": 0, "undecodable": 0, "over_limit": 0}
        self._seen_hashes: set[str] = set()

    def walk_pdf(self, content: bytes) -> None:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        for page_index, page in enumerate(reader.pages):
            try:
                page_images = page.images
            except Exception:
                # A page whose resources are malformed shouldn't sink the rest.
                self.skipped["undecodable"] += 1
                continue
            for image_file in page_images:
                try:
                    raw = image_file.data
                    pil = image_file.image
                    if pil is None:
                        raise ValueError("no decodable image")
                except Exception:
                    self.skipped["undecodable"] += 1
                    continue
                self._keep(raw, pil, page_number=page_index + 1, name=image_file.name)

    def walk_docx(self, content: bytes) -> None:
        try:
            archive = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile:
            self.skipped["undecodable"] += 1
            return
        with archive:
            media_names = sorted(
                n for n in archive.namelist() if n.startswith(_DOCX_MEDIA_PREFIX)
            )
            for name in media_names:
                raw = archive.read(name)
                try:
                    pil = Image.open(io.BytesIO(raw))
                    pil.load()
                except Exception:
                    self.skipped["undecodable"] += 1
                    continue
                # DOCX is a flow format: there is no page number to anchor to.
                self._keep(raw, pil, page_number=None, name=name.removeprefix(_DOCX_MEDIA_PREFIX))

    def _keep(
        self, raw: bytes, pil: Image.Image, *, page_number: int | None, name: str
    ) -> None:
        if min(pil.width, pil.height) < self.min_dimension:
            self.skipped["tiny"] += 1
            return

        digest = hashlib.sha256(raw).hexdigest()
        if digest in self._seen_hashes:
            self.skipped["duplicate"] += 1
            return

        if len(self.images) >= self.max_images:
            self.skipped["over_limit"] += 1
            return

        try:
            display_b64, display_mime = _rendition(pil, self.display_max)
            thumb_b64, thumb_mime = _rendition(pil, self.thumbnail_max)
        except Exception:
            self.skipped["undecodable"] += 1
            return

        self._seen_hashes.add(digest)
        self.images.append(
            {
                "page_number": page_number,
                "image_index": len(self.images),
                "name": name,
                "width": pil.width,
                "height": pil.height,
                "format": (pil.format or "unknown").lower(),
                "hash_sha256": digest,
                "thumbnail_base64": thumb_b64,
                "thumbnail_mime": thumb_mime,
                "image_base64": display_b64,
                "image_mime": display_mime,
            }
        )


def _rendition(pil: Image.Image, max_side: int) -> tuple[str, str]:
    """Downscale to fit max_side and encode as base64 PNG (alpha) or JPEG."""
    img = pil
    if max(img.width, img.height) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)

    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    buf = io.BytesIO()
    if has_alpha:
        img.convert("RGBA").save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        mime = "image/jpeg"
    return base64.b64encode(buf.getvalue()).decode("ascii"), mime
