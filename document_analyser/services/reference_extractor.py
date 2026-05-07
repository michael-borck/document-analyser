"""
Reference extraction service for citations and bibliography
"""

import re
from dataclasses import dataclass


@dataclass
class Reference:
    """Represents a citation reference"""
    text: str
    type: str  # 'apa', 'mla', 'chicago', 'url', 'doi'
    author: str = ""
    year: str = ""
    title: str = ""
    url: str = ""
    doi: str = ""

class ReferenceExtractor:
    """Extracts and analyzes citations from academic text"""

    def __init__(self) -> None:
        self.patterns = {
            'doi': re.compile(r'10\.\d{4,}/[-._;()/:\w\[\]]+', re.IGNORECASE),
            'url': re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE),
            'apa': re.compile(r'[A-Z][a-z]+,\s+[A-Z]\.(?:\s+[A-Z]\.)*\s+\(\d{4}\)', re.IGNORECASE),
            'mla': re.compile(r'[A-Z][a-z]+,\s+[A-Z][a-z]+\.\s+"[^"]+"\s+[A-Z][a-z\s]+,', re.IGNORECASE),
            'chicago': re.compile(r'[A-Z][a-z]+,\s+[A-Z][a-z]+\.\s+"[^"]+"\.\s+[A-Z][a-z\s]+:', re.IGNORECASE),
        }

    def extract_references(self, text: str, style: str = "auto") -> list[Reference]:
        """
        Extract references from text

        Args:
            text: The text to analyze
            style: Citation style ('auto', 'apa', 'mla', 'chicago')

        Returns:
            List of extracted references
        """
        references = []

        # Extract DOIs
        doi_matches = self.patterns['doi'].findall(text)
        for doi in doi_matches:
            references.append(Reference(
                text=doi,
                type='doi',
                doi=doi
            ))

        # Extract URLs
        url_matches = self.patterns['url'].findall(text)
        for url in url_matches:
            # Skip DOI URLs to avoid duplicates
            if not url.startswith(('http://dx.doi.org', 'https://doi.org')):
                references.append(Reference(
                    text=url,
                    type='url',
                    url=url
                ))

        # Extract citation style references
        if style == "auto":
            # Try all styles
            for style_name, pattern in self.patterns.items():
                if style_name not in ['doi', 'url']:
                    matches = pattern.findall(text)
                    for match in matches:
                        references.append(Reference(
                            text=match,
                            type=style_name
                        ))
        else:
            # Extract specific style
            if style in self.patterns:
                matches = self.patterns[style].findall(text)
                for match in matches:
                    references.append(Reference(
                        text=match,
                        type=style
                    ))

        return references

    def detect_citation_style(self, text: str) -> str:
        """Detect the primary citation style used in the text"""
        style_counts = {}

        for style_name, pattern in self.patterns.items():
            if style_name not in ['doi', 'url']:
                matches = pattern.findall(text)
                style_counts[style_name] = len(matches)

        if not style_counts:
            return 'unknown'

        return max(style_counts, key=lambda x: style_counts[x])

    def extract_in_text_citations(self, text: str) -> list[str]:
        """Extract in-text citations from the document"""
        # Common in-text citation patterns
        patterns = [
            r'\([A-Z][a-z]+(?:\s*,\s*\d{4})\)',  # (Author, 2024)
            r'\([A-Z][a-z]+(?:\s+et\s+al\.)?(?:\s*,\s*\d{4})\)',  # (Author et al., 2024)
            r'[A-Z][a-z]+\s+\(\d{4}\)',  # Author (2024)
            r'[A-Z][a-z]+(?:\s+et\s+al\.)?\s+\(\d{4}\)',  # Author et al. (2024)
        ]

        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)

        return list(set(citations))  # Remove duplicates
