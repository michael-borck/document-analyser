"""Capability manifest for the lens family (see auto-analyser manifest discovery).

document-analyser is auto-routable: its file extensions imply the analysis.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _version() -> str:
    try:
        return version("document-analyser")
    except PackageNotFoundError:
        return "0.0.0"


MANIFEST: dict = {
    "name": "document-analyser",
    "version": _version(),
    "role": "analyser",
    "accepts": ["document", "prose"],
    "extensions": [".pdf", ".docx", ".pptx", ".txt", ".md", ".qmd", ".rst"],
    "auto_routable": True,
    "produces": "DocumentAnalysis",
}
