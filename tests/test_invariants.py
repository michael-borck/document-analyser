"""Invariant tests — fast, no real ML models, run by default.

These tests guard against the failure modes that motivated the recent
audit and clean-up: packaging bugs (records-analyser style), silent
graceful-degradation in analysers, and version-string drift across
hardcoded literals. They're cheap; they should always run.
"""

from importlib.metadata import version
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def test_package_imports_cleanly() -> None:
    """The package must import without optional ML deps installed.

    A real bug we've already paid for: records-analyser's package was
    un-importable for weeks because no test ever exercised the bare
    import statement. This test is the smoke alarm — if it ever fails,
    something at module-load time is broken (missing required dep, syntax
    error, circular import, etc.).
    """
    import document_analyser  # noqa: F401
    from document_analyser.main import document_analyser as app  # noqa: F401


def test_health_version_matches_installed_package() -> None:
    """/health must report the actual installed package version.

    Drift trap: route handlers used to hardcode "1.0.0" while the package
    was at 0.2.0 — tests "passed" because they hardcoded the same wrong
    string. Pin the route to importlib.metadata and verify it stays
    pinned.
    """
    from document_analyser.main import document_analyser as app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["version"] == version("document-analyser")


def test_root_version_matches_installed_package() -> None:
    """/root must also report the installed package version (same drift trap)."""
    from document_analyser.main import document_analyser as app

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["version"] == version("document-analyser")


def test_domain_mapper_raises_when_model_unavailable() -> None:
    """DomainMapper must fail loudly, not silently return total_sections=0.

    This is the regression guard for the prior bug where the analyser
    returned a valid-looking response with zero sections when its
    sentence-transformers model failed to load. Tests "passed" by
    accident because the model loaded fine in dev. Now if the model is
    unavailable, analyze() raises — and this test pins that behaviour.
    """
    from document_analyser.analyzers.domain_mapper import DomainMapper

    # Construct an instance, then null out the model to simulate a load
    # failure. This is more direct than patching the SentenceTransformer
    # symbol because it tests the analyse() path's contract regardless
    # of how loading was attempted.
    mapper = DomainMapper.__new__(DomainMapper)
    mapper.model_name = "test"
    mapper.model = None
    mapper._load_error = "simulated load failure for test"

    with pytest.raises(RuntimeError, match="simulated load failure"):
        mapper.analyze("Some document text.", ["DomainA", "DomainB"])


def test_domain_mapper_load_error_captured_when_sentence_transformers_missing() -> None:
    """When sentence-transformers isn't importable, the failure reason is captured.

    Guards the "silent None" pattern — we want self._load_error populated
    so analyse() can include it in the RuntimeError message.
    """
    # Patch the symbol the module imported at top level. The DomainMapper
    # __init__ checks `if SentenceTransformer is None` (after the optional
    # import at module top), so we patch that bound name.
    with patch("document_analyser.analyzers.domain_mapper.SentenceTransformer", None):
        from document_analyser.analyzers.domain_mapper import DomainMapper
        mapper = DomainMapper()
        assert mapper.model is None
        assert mapper._load_error is not None
        assert "sentence-transformers" in mapper._load_error
