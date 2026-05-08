"""
Tests for health and root endpoints.

These are smoke tests to ensure the API is running correctly.
"""

from importlib.metadata import version as _pkg_version

from fastapi.testclient import TestClient

EXPECTED_VERSION = _pkg_version("document-analyser")


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client: TestClient):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client: TestClient):
        """Health endpoint should return ok status."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "ok"

    def test_health_returns_version(self, client: TestClient):
        """Health endpoint should return version information."""
        response = client.get("/health")
        data = response.json()

        assert "version" in data
        assert data["version"] == EXPECTED_VERSION


class TestRootEndpoint:
    """Tests for the root (/) endpoint."""

    def test_root_returns_200(self, client: TestClient):
        """Root endpoint should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_service_info(self, client: TestClient):
        """Root endpoint should return service information."""
        response = client.get("/")
        data = response.json()

        assert data["service"] == "DocumentAnalyser"
        assert data["status"] == "running"


class TestDocsEndpoints:
    """Tests for API documentation endpoints."""

    def test_swagger_docs_available(self, client: TestClient):
        """Swagger UI should be available at /docs."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client: TestClient):
        """ReDoc should be available at /redoc."""
        response = client.get("/redoc")
        assert response.status_code == 200
