# document-analyser

Extracts text from documents and returns readability metrics, word counts, and structural information. Accepts PDF, DOCX, PPTX, and plain text formats.

Part of the [analyser family](#the-analyser-family).

## Install

```bash
pip install document-analyser
```

Requires Python 3.11+.

## Usage

### Python

```python
from app.analyser import DocumentAnalyser

result = DocumentAnalyser().analyse("report.pdf")

print(f"Words:       {result['word_count']}")
print(f"Sentences:   {result['sentence_count']}")
print(f"Readability: {result['readability']['flesch_reading_ease']:.1f} (Flesch)")
print(result["text"][:500])
```

### CLI

```bash
# Human-readable summary
document-analyser report.pdf

# Machine-readable JSON
document-analyser thesis.docx --json

# Start the HTTP server
document-analyser serve --port 8000
```

### HTTP API

```bash
curl -X POST http://localhost:8000/analyse \
  -F "file=@report.pdf"
```

## Supported formats

| Format | Extensions |
|---|---|
| PDF | `.pdf` |
| Word | `.docx` |
| PowerPoint | `.pptx` |
| Plain text | `.txt` `.md` |

## Output

```json
{
  "format": "pdf",
  "file_path": "/path/to/report.pdf",
  "file_size": 204800,
  "page_count": 12,
  "word_count": 4823,
  "sentence_count": 312,
  "paragraph_count": 89,
  "text": "Executive summary...",
  "readability": {
    "flesch_reading_ease": 52.3,
    "flesch_kincaid_grade": 11.2,
    "gunning_fog": 13.8,
    "smog_index": 12.1,
    "automated_readability_index": 11.9
  }
}
```

## The analyser family

Low-level analysis tools. Each accepts files directly and returns structured JSON. Build your own UI or pipeline on top.

| Package | Handles |
|---|---|
| [speech-analyser](https://github.com/michael-borck/speech-analyser) | audio and video files — transcript and speech metrics |
| [video-analyser](https://github.com/michael-borck/video-analyser) | video files — frames, scenes, and visual quality |
| [document-analyser](https://github.com/michael-borck/document-analyser) | PDF, DOCX, PPTX, TXT — text and readability |
| [code-analyser](https://github.com/michael-borck/code-analyser) | source code — style, complexity, and quality metrics |
| [records-analyser](https://github.com/michael-borck/records-analyser) | CSV, Excel, SQLite, Parquet, JSON — data profiling |
| [multi-analyser](https://github.com/michael-borck/multi-analyser) | any file — detects format and routes to the right tool |

## Licence

MIT
