"""
Tests for health and root endpoints.

These are smoke tests to ensure the API is running correctly.
"""

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client: TestClient):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client: TestClient):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"

    def test_health_returns_version(self, client: TestClient):
        """Health endpoint should return version information."""
        response = client.get("/health")
        data = response.json()

        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_health_returns_uptime(self, client: TestClient):
        """Health endpoint should return uptime as a positive number."""
        response = client.get("/health")
        data = response.json()

        assert "uptime" in data
        assert isinstance(data["uptime"], (int, float))
        assert data["uptime"] >= 0


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

        assert data["service"] == "DocumentLens"
        assert data["status"] == "running"

    def test_root_lists_available_endpoints(self, client: TestClient):
        """Root endpoint should list available API endpoints."""
        response = client.get("/")
        data = response.json()

        assert "endpoints" in data
        assert "available" in data["endpoints"]

        available = data["endpoints"]["available"]
        assert "health" in available
        assert "text_analysis" in available
        assert "file_processing" in available


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

    def test_openapi_schema_available(self, client: TestClient):
        """OpenAPI schema should be available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "paths" in data
