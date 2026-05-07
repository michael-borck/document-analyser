"""
DOI resolution service using CrossRef API for academic reference validation
"""

import asyncio
import re
from typing import Any
from urllib.parse import quote

import httpx


class DOIResolver:
    """Resolves DOIs using CrossRef API with caching and error handling"""

    def __init__(self) -> None:
        """Initialize DOI resolver with CrossRef API configuration"""
        self.crossref_base_url = "https://api.crossref.org"
        self.doi_base_url = "https://doi.org"

        # DOI pattern validation
        self.doi_pattern = re.compile(r'^10\.\d{4,}/[-._;()/:\w\[\]]+$')

        # Request headers for CrossRef API
        self.headers = {
            'User-Agent': 'DocumentLens/1.0.0 (https://github.com/michael-borck/document-lens; mailto:contact@example.com)',
            'Accept': 'application/json'
        }

        # Cache for resolved DOIs (simple in-memory cache)
        self._cache: dict[str, dict[str, Any]] = {}

    async def resolve_dois(self, dois: list[str]) -> dict[str, Any]:
        """
        Resolve a list of DOIs using CrossRef API

        Args:
            dois: List of DOI strings to resolve

        Returns:
            Dictionary with resolved, unresolved, and metadata
        """
        if not dois:
            return {
                "resolved": [],
                "unresolved": [],
                "total": 0,
                "errors": []
            }

        # Clean and validate DOIs
        clean_dois = []
        invalid_dois = []

        for doi in dois:
            cleaned = self._clean_doi(doi)
            if cleaned and self._is_valid_doi(cleaned):
                clean_dois.append(cleaned)
            else:
                invalid_dois.append(doi)

        # Resolve DOIs concurrently
        resolved = []
        unresolved = []
        errors = []

        if clean_dois:
            # Use semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
            tasks = [self._resolve_single_doi(doi, semaphore) for doi in clean_dois]

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for doi, result in zip(clean_dois, results, strict=False):
                    if isinstance(result, Exception):
                        errors.append(f"Error resolving {doi}: {result!s}")
                        unresolved.append(doi)
                    elif result is None:
                        unresolved.append(doi)
                    else:
                        resolved.append(result)

            except Exception as e:
                errors.append(f"Batch resolution error: {e!s}")
                unresolved.extend(clean_dois)

        # Add invalid DOIs to unresolved
        unresolved.extend(invalid_dois)

        return {
            "resolved": resolved,
            "unresolved": unresolved,
            "total": len(dois),
            "errors": errors,
            "statistics": {
                "valid_format": len(clean_dois),
                "invalid_format": len(invalid_dois),
                "successfully_resolved": len(resolved),
                "resolution_rate": round((len(resolved) / len(dois)) * 100, 1) if dois else 0
            }
        }

    async def _resolve_single_doi(self, doi: str, semaphore: asyncio.Semaphore) -> dict[str, Any] | None:
        """
        Resolve a single DOI with rate limiting

        Args:
            doi: Clean DOI string
            semaphore: Asyncio semaphore for rate limiting

        Returns:
            Resolved DOI metadata or None if resolution failed
        """
        # Check cache first
        if doi in self._cache:
            return self._cache[doi]

        async with semaphore:
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(10.0),
                    headers=self.headers,
                    follow_redirects=True
                ) as client:

                    # Try CrossRef API first
                    crossref_result = await self._query_crossref(client, doi)
                    if crossref_result:
                        self._cache[doi] = crossref_result
                        return crossref_result

                    # Fallback to direct DOI resolution
                    direct_result = await self._query_doi_direct(client, doi)
                    if direct_result:
                        self._cache[doi] = direct_result
                        return direct_result

                    return None

            except Exception as e:
                # Log error but don't raise to prevent stopping other resolutions
                print(f"Error resolving DOI {doi}: {e!s}")
                return None

    async def _query_crossref(self, client: httpx.AsyncClient, doi: str) -> dict[str, Any] | None:
        """Query CrossRef API for DOI metadata"""
        try:
            url = f"{self.crossref_base_url}/works/{quote(doi, safe='')}"
            response = await client.get(url)

            if response.status_code == 200:
                data = response.json()
                work = data.get('message', {})

                return {
                    "doi": doi,
                    "title": self._extract_title(work),
                    "authors": self._extract_authors(work),
                    "journal": work.get('container-title', ['Unknown'])[0] if work.get('container-title') else 'Unknown',
                    "year": self._extract_year(work),
                    "type": work.get('type', 'unknown'),
                    "url": work.get('URL', f"{self.doi_base_url}/{doi}"),
                    "is_oa": work.get('is-referenced-by-count', 0) > 0,  # Rough OA indicator
                    "citation_count": work.get('is-referenced-by-count', 0),
                    "publisher": work.get('publisher', 'Unknown'),
                    "source": "crossref"
                }
            elif response.status_code == 404:
                return None  # DOI not found
            else:
                # Other errors, try fallback
                return None

        except Exception:
            return None

    async def _query_doi_direct(self, client: httpx.AsyncClient, doi: str) -> dict[str, Any] | None:
        """Query DOI directly as fallback"""
        try:
            url = f"{self.doi_base_url}/{doi}"
            response = await client.head(url)

            if response.status_code in [200, 301, 302]:
                return {
                    "doi": doi,
                    "title": "Unknown (Direct resolution)",
                    "authors": [],
                    "journal": "Unknown",
                    "year": "Unknown",
                    "type": "unknown",
                    "url": url,
                    "is_oa": False,
                    "citation_count": 0,
                    "publisher": "Unknown",
                    "source": "direct"
                }
            else:
                return None

        except Exception:
            return None

    def _clean_doi(self, doi: str) -> str:
        """Clean and normalize DOI string"""
        if not doi:
            return ""

        # Remove common prefixes and whitespace
        cleaned = doi.strip()

        # Remove URL prefixes
        prefixes = [
            "https://doi.org/",
            "http://doi.org/",
            "https://dx.doi.org/",
            "http://dx.doi.org/",
            "doi:",
            "DOI:"
        ]

        for prefix in prefixes:
            if cleaned.upper().startswith(prefix.upper()):
                cleaned = cleaned[len(prefix):]
                break

        return cleaned.strip()

    def _is_valid_doi(self, doi: str) -> bool:
        """Validate DOI format"""
        return bool(self.doi_pattern.match(doi))

    def _extract_title(self, work: dict[str, Any]) -> str:
        """Extract title from CrossRef work data"""
        titles = work.get('title', [])
        if titles and isinstance(titles, list) and titles[0]:
            return str(titles[0])
        return "Unknown Title"

    def _extract_authors(self, work: dict[str, Any]) -> list[str]:
        """Extract authors from CrossRef work data"""
        authors = work.get('author', [])
        author_names = []

        for author in authors:
            if isinstance(author, dict):
                given = author.get('given', '')
                family = author.get('family', '')
                if family:
                    if given:
                        author_names.append(f"{given} {family}")
                    else:
                        author_names.append(family)

        return author_names

    def _extract_year(self, work: dict[str, Any]) -> str:
        """Extract publication year from CrossRef work data"""
        # Try different date fields
        date_fields = ['published-print', 'published-online', 'published', 'created']

        for field in date_fields:
            date_info = work.get(field, {})
            if isinstance(date_info, dict):
                date_parts = date_info.get('date-parts', [])
                if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
                    return str(date_parts[0][0])

        return "Unknown"

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_dois": len(self._cache),
            "cache_size_bytes": sum(len(str(v)) for v in self._cache.values())
        }

    def clear_cache(self) -> None:
        """Clear the DOI resolution cache"""
        self._cache.clear()

    async def validate_doi_accessibility(self, doi: str) -> dict[str, Any]:
        """
        Check if a DOI is accessible and get basic information

        Args:
            doi: DOI string to validate

        Returns:
            Dictionary with accessibility information
        """
        cleaned_doi = self._clean_doi(doi)

        if not self._is_valid_doi(cleaned_doi):
            return {
                "doi": doi,
                "is_valid_format": False,
                "is_accessible": False,
                "error": "Invalid DOI format"
            }

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
                headers=self.headers,
                follow_redirects=True
            ) as client:

                url = f"{self.doi_base_url}/{cleaned_doi}"
                response = await client.head(url)

                return {
                    "doi": cleaned_doi,
                    "is_valid_format": True,
                    "is_accessible": response.status_code in [200, 301, 302],
                    "status_code": response.status_code,
                    "final_url": str(response.url) if hasattr(response, 'url') else url,
                    "error": None if response.status_code in [200, 301, 302] else f"HTTP {response.status_code}"
                }

        except Exception as e:
            return {
                "doi": cleaned_doi,
                "is_valid_format": True,
                "is_accessible": False,
                "error": str(e)
            }
