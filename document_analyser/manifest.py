"""Capability manifest for the lens family (see auto-analyser manifest discovery).

document-analyser is auto-routable: its file extensions imply the analysis.
"""
from __future__ import annotations

from lens_contract import make_manifest

MANIFEST = make_manifest(
    name="document-analyser",
    accepts=["document", "prose"],
    extensions=[".pdf", ".docx", ".pptx", ".txt", ".md", ".qmd", ".rst"],
    auto_routable=True,
    produces="DocumentAnalysis",
)
