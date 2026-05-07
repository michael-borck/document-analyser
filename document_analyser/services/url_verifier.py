"""
URL verification service with async checking, caching, and SSL validation
"""

import asyncio
import re
import ssl
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import httpx


class URLVerifier:
    """Verifies URLs in references with comprehensive checking and caching"""

    def __init__(self) -> None:
        """Initialize URL verifier with configuration and caching"""
        # URL pattern validation
        self.url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )

        # Request headers
        self.headers = {
            'User-Agent': 'DocumentLens/1.0.0 URL Verifier (https://github.com/michael-borck/document-lens)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0'
        }

        # Cache for verified URLs (URL -> result dict)
        self._cache: dict[str, dict[str, Any]] = {}

        # Cache expiration time (1 hour)
        self._cache_duration = timedelta(hours=1)

    async def verify_urls(self, urls: list[str]) -> dict[str, Any]:
        """
        Verify a list of URLs with comprehensive checking

        Args:
            urls: List of URLs to verify

        Returns:
            Dictionary with verification results and statistics
        """
        if not urls:
            return {
                "verified": [],
                "broken": [],
                "total": 0,
                "errors": [],
                "statistics": {
                    "valid_format": 0,
                    "invalid_format": 0,
                    "accessible": 0,
                    "redirects": 0,
                    "ssl_issues": 0,
                    "timeouts": 0
                }
            }

        # Clean and validate URLs
        clean_urls = []
        invalid_urls = []

        for url in urls:
            cleaned = self._clean_url(url)
            if cleaned and self._is_valid_url(cleaned):
                clean_urls.append(cleaned)
            else:
                invalid_urls.append(url)

        # Verify URLs concurrently
        verified = []
        broken = []
        errors = []
        stats = {
            "valid_format": len(clean_urls),
            "invalid_format": len(invalid_urls),
            "accessible": 0,
            "redirects": 0,
            "ssl_issues": 0,
            "timeouts": 0
        }

        if clean_urls:
            # Use semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
            tasks = [self._verify_single_url(url, semaphore) for url in clean_urls]

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for url, result in zip(clean_urls, results, strict=False):
                    if isinstance(result, Exception):
                        errors.append(f"Error verifying {url}: {result!s}")
                        broken.append({
                            "url": url,
                            "error": str(result),
                            "accessible": False
                        })
                    elif isinstance(result, dict) and result.get("accessible"):
                        verified.append(result)
                        stats["accessible"] += 1
                        if result.get("redirected"):
                            stats["redirects"] += 1
                        if result.get("ssl_issue"):
                            stats["ssl_issues"] += 1
                    else:
                        broken_entry = {
                            "url": url,
                            "error": "Unknown verification failure",
                            "accessible": False
                        }
                        if isinstance(result, dict):
                            broken_entry = result
                            if result.get("timeout"):
                                stats["timeouts"] += 1
                            if result.get("ssl_issue"):
                                stats["ssl_issues"] += 1
                        broken.append(broken_entry)

            except Exception as e:
                errors.append(f"Batch verification error: {e!s}")
                # Add all URLs as broken if batch fails
                for url in clean_urls:
                    broken.append({
                        "url": url,
                        "error": "Batch verification failed",
                        "accessible": False
                    })

        # Add invalid URLs to broken list
        for url in invalid_urls:
            broken.append({
                "url": url,
                "error": "Invalid URL format",
                "accessible": False
            })

        return {
            "verified": verified,
            "broken": broken,
            "total": len(urls),
            "errors": errors,
            "statistics": stats
        }

    async def _verify_single_url(self, url: str, semaphore: asyncio.Semaphore) -> dict[str, Any]:
        """
        Verify a single URL with comprehensive checking

        Args:
            url: Clean URL string
            semaphore: Asyncio semaphore for rate limiting

        Returns:
            URL verification result
        """
        # Check cache first
        cache_key = url
        if cache_key in self._cache:
            cached_result = self._cache[cache_key]
            # Check if cache is still valid
            if datetime.now() - cached_result.get("cached_at", datetime.min) < self._cache_duration:
                return cached_result

        async with semaphore:
            try:
                result = await self._check_url_accessibility(url)

                # Cache the result
                result["cached_at"] = datetime.now()
                self._cache[cache_key] = result

                return result

            except Exception as e:
                error_result = {
                    "url": url,
                    "accessible": False,
                    "error": str(e),
                    "timeout": "timeout" in str(e).lower(),
                    "ssl_issue": "ssl" in str(e).lower() or "certificate" in str(e).lower(),
                    "cached_at": datetime.now()
                }
                self._cache[cache_key] = error_result
                return error_result

    async def _check_url_accessibility(self, url: str) -> dict[str, Any]:
        """Check if URL is accessible and get metadata"""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False  # Allow checking sites with cert issues
        ssl_context.verify_mode = ssl.CERT_NONE

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                headers=self.headers,
                follow_redirects=True,
                verify=False  # Don't fail on SSL issues, but detect them
            ) as client:

                # Make HEAD request first (faster)
                try:
                    response = await client.head(url)
                except httpx.HTTPStatusError:
                    # If HEAD fails, try GET (some servers don't support HEAD)
                    response = await client.get(url)

                # Check if URL was redirected
                redirected = str(response.url) != url
                redirect_chain = []

                if redirected and hasattr(response, 'history'):
                    redirect_chain = [str(r.url) for r in response.history]
                    redirect_chain.append(str(response.url))

                # Check SSL certificate validity
                ssl_issue = False
                ssl_error = None

                if url.startswith('https://'):
                    try:
                        # Try to make a request with SSL verification
                        async with httpx.AsyncClient(
                            timeout=httpx.Timeout(5.0),
                            verify=True
                        ) as ssl_client:
                            await ssl_client.head(url)
                    except (httpx.RequestError, ssl.SSLError) as e:
                        ssl_issue = True
                        ssl_error = str(e)

                # Determine accessibility
                accessible = response.status_code < 400

                return {
                    "url": url,
                    "accessible": accessible,
                    "status_code": response.status_code,
                    "final_url": str(response.url),
                    "redirected": redirected,
                    "redirect_chain": redirect_chain,
                    "ssl_issue": ssl_issue,
                    "ssl_error": ssl_error,
                    "content_type": response.headers.get("content-type", ""),
                    "server": response.headers.get("server", ""),
                    "response_time_ms": None,  # Could add timing if needed
                    "error": None
                }

        except httpx.TimeoutException:
            return {
                "url": url,
                "accessible": False,
                "error": "Request timeout",
                "timeout": True,
                "ssl_issue": False
            }
        except Exception as e:
            return {
                "url": url,
                "accessible": False,
                "error": str(e),
                "timeout": False,
                "ssl_issue": "ssl" in str(e).lower() or "certificate" in str(e).lower()
            }

    def _clean_url(self, url: str) -> str:
        """Clean and normalize URL string"""
        if not url:
            return ""

        # Remove whitespace and common prefixes
        cleaned = url.strip()

        # Add protocol if missing
        if not cleaned.startswith(('http://', 'https://')):
            # Default to https for security
            cleaned = f"https://{cleaned}"

        return cleaned

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        if not url:
            return False

        # Check regex pattern
        if not self.url_pattern.match(url):
            return False

        # Parse URL to check components
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc and parsed.scheme in ['http', 'https'])
        except Exception:
            return False

    async def check_url_status(self, url: str) -> dict[str, Any]:
        """
        Quick check of a single URL status

        Args:
            url: URL to check

        Returns:
            Status information for the URL
        """
        cleaned_url = self._clean_url(url)

        if not self._is_valid_url(cleaned_url):
            return {
                "url": url,
                "valid_format": False,
                "accessible": False,
                "error": "Invalid URL format"
            }

        semaphore = asyncio.Semaphore(1)
        result = await self._verify_single_url(cleaned_url, semaphore)

        return {
            "url": url,
            "valid_format": True,
            "accessible": result.get("accessible", False),
            "status_code": result.get("status_code"),
            "final_url": result.get("final_url"),
            "redirected": result.get("redirected", False),
            "ssl_issue": result.get("ssl_issue", False),
            "error": result.get("error")
        }

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_urls": len(self._cache),
            "cache_size_bytes": sum(len(str(v)) for v in self._cache.values())
        }

    def clear_cache(self) -> None:
        """Clear the URL verification cache"""
        self._cache.clear()

    def clear_expired_cache(self) -> int:
        """Clear expired cache entries and return count of cleared entries"""
        now = datetime.now()
        expired_keys = []

        for key, value in self._cache.items():
            cached_at = value.get("cached_at", datetime.min)
            if now - cached_at > self._cache_duration:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)
