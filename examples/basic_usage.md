# basic_usage

Minimal ways to run document-analyser.

## Install

```bash
pip install document-analyser
```

## CLI

```bash
document-analyser report.pdf --json
```

Accepts `.pdf`, `.docx`, `.pptx`, `.txt`, `.md`, `.qmd`, `.rst`. Without `--json` it prints a human-readable summary.

## Python

The package re-exports the canonical text extractor:

```python
from document_analyser import extract_text

text = extract_text("report.pdf")
print(text)
```

## HTTP

```bash
document-analyser serve            # http://localhost:8000
curl -F file=@report.pdf http://localhost:8000/analyse
```
