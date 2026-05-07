"""
Pytest configuration and shared fixtures for DocumentLens API tests.
"""

import os
from collections.abc import Generator
from pathlib import Path

# Set a high rate limit for testing BEFORE importing the document_analyser
os.environ["RATE_LIMIT"] = "1000/minute"

import pytest
from fastapi.testclient import TestClient

from document_analyser.main import document_analyser


# === Test Client Fixtures ===


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """
    Provides a FastAPI TestClient for the entire test session.
    Uses session scope for efficiency - the document_analyser is only initialized once.
    """
    with TestClient(document_analyser) as test_client:
        yield test_client


@pytest.fixture
async def async_client():
    """
    Provides an async httpx client for async test functions.
    Use this when testing async behavior specifically.
    """
    import httpx

    transport = httpx.ASGITransport(document_analyser=document_analyser)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# === Path Fixtures ===


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Returns the path to the test-data directory."""
    return Path(__file__).parent.parent / "test-data"


@pytest.fixture(scope="session")
def sample_pdf_paths(test_data_dir: Path) -> list[Path]:
    """Returns list of all PDF files in test-data directory."""
    return list(test_data_dir.glob("*.pdf"))


@pytest.fixture(scope="session")
def sample_docx_paths(test_data_dir: Path) -> list[Path]:
    """Returns list of all DOCX files in test-data directory."""
    return list(test_data_dir.glob("*.docx"))


# === Sample Data Fixtures ===


@pytest.fixture
def sample_text() -> str:
    """Provides sample text for text analysis tests."""
    return """
    The Australian Securities and Investments Commission (ASIC) released its 
    annual report for the 2024-2025 financial year. The report highlights 
    significant regulatory achievements and outlines future strategic priorities.
    
    Key findings include improved market integrity and enhanced consumer protection 
    measures. The Commission processed over 50,000 complaints and conducted 
    numerous enforcement actions throughout the reporting period.
    
    According to Smith et al. (2024), regulatory frameworks continue to evolve 
    in response to emerging technologies. The implementation of new guidelines 
    has been praised by industry stakeholders (Jones, 2023).
    
    For more information, visit https://asic.gov.au or contact the Commission 
    directly. The full report is available at DOI: 10.1234/asic.2024.report
    """


@pytest.fixture
def sample_academic_text() -> str:
    """Provides sample academic text with citations for academic analysis tests."""
    return """
    Introduction
    
    This study examines the relationship between corporate governance and 
    financial performance in Australian listed companies. Previous research 
    has established significant correlations between board composition and 
    firm value (Brown & Taylor, 2022; Williams, 2021).
    
    The methodology follows established protocols as described by Johnson (2020). 
    Data was collected from the ASX 200 companies over a five-year period 
    (2019-2024). Statistical analysis was performed using regression models 
    as recommended by Anderson et al. (2019).
    
    Results indicate a positive correlation (r=0.72, p<0.001) between board 
    independence and return on equity. These findings align with theoretical 
    predictions (Porter, 2018) and international studies (Chen & Liu, 2023).
    
    References
    
    Anderson, R., Smith, J., & Lee, M. (2019). Statistical methods for 
    corporate research. Journal of Finance, 45(2), 112-134.
    
    Brown, A., & Taylor, S. (2022). Board composition and firm performance. 
    Corporate Governance Review, 18(4), 45-67. https://doi.org/10.1234/cgr.2022.045
    
    Chen, W., & Liu, H. (2023). International perspectives on governance. 
    Asian Business Studies, 12(1), 23-41.
    """


@pytest.fixture
def minimal_text() -> str:
    """Provides minimal text for edge case testing."""
    return "This is a short sentence."


@pytest.fixture
def empty_text() -> str:
    """Provides empty text for validation testing."""
    return ""


# === File Content Fixtures ===


@pytest.fixture
def pdf_content(sample_pdf_paths: list[Path]) -> bytes | None:
    """Returns content of the first available PDF file."""
    if sample_pdf_paths:
        return sample_pdf_paths[0].read_bytes()
    return None


@pytest.fixture
def pdf_file_info(sample_pdf_paths: list[Path]) -> dict | None:
    """Returns info about the first available PDF file for upload tests."""
    if sample_pdf_paths:
        pdf_path = sample_pdf_paths[0]
        return {
            "path": pdf_path,
            "name": pdf_path.name,
            "content": pdf_path.read_bytes(),
            "content_type": "application/pdf",
        }
    return None


# === Parameterized Test Data ===


def get_test_files() -> list[tuple[str, str]]:
    """
    Returns list of (file_path, content_type) tuples for parameterized tests.
    This allows easy extension when adding DOCX, TXT, etc.
    """
    test_data_dir = Path(__file__).parent.parent / "test-data"
    files = []

    # Add PDFs
    for pdf in test_data_dir.glob("*.pdf"):
        files.append((str(pdf), "application/pdf"))

    # Add DOCX (when available)
    for docx in test_data_dir.glob("*.docx"):
        files.append(
            (str(docx), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        )

    # Add TXT (when available)
    for txt in test_data_dir.glob("*.txt"):
        files.append((str(txt), "text/plain"))

    return files


# Pytest parametrize helper - use in tests like:
# @pytest.mark.parametrize("file_path,content_type", TEST_FILES)
TEST_FILES = get_test_files()


# === Markers ===


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "pdf: marks tests that require PDF files")
    config.addinivalue_line("markers", "docx: marks tests that require DOCX files")
