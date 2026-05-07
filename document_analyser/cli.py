"""CLI entry point for document-analyser.

Usage:
  document-analyser report.pdf
  document-analyser thesis.docx --json
  document-analyser slides.pptx
  document-analyser serve
  document-analyser serve --port 8000 --host 0.0.0.0
"""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    import argparse

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        _main_serve(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        prog="document-analyser",
        description="Extract text and readability metrics from documents",
    )
    parser.add_argument("file", type=Path, help="Document to analyse (PDF, DOCX, PPTX, TXT, MD)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")
    _cmd_analyse(parser.parse_args())


def _main_serve(argv: list[str]) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="document-analyser serve", description="Start the HTTP server")
    parser.add_argument("--port", type=int, default=int(os.getenv("DOCUMENT_LENS_PORT", "8000")))
    parser.add_argument("--host", default=os.getenv("DOCUMENT_LENS_HOST", "127.0.0.1"))
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    _cmd_serve(parser.parse_args(argv))


def _cmd_analyse(args) -> None:
    path = args.file
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    suffix = path.suffix.lower()

    try:
        text = _extract_text(path, suffix)
    except Exception as e:
        print(f"Error: could not extract text: {e}", file=sys.stderr)
        sys.exit(1)

    from document_analyser.analyzers.readability import ReadabilityAnalyzer
    analysis = ReadabilityAnalyzer().analyze(text)

    result = {
        "format": suffix.lstrip("."),
        "file_path": str(path.resolve()),
        "file_size": path.stat().st_size,
        "word_count": analysis.word_count,
        "sentence_count": analysis.sentence_count,
        "paragraph_count": analysis.paragraph_count,
        "text": text,
        "readability": {
            "flesch_reading_ease": analysis.flesch_reading_ease,
            "flesch_kincaid_grade": analysis.flesch_kincaid_grade,
            "gunning_fog": analysis.gunning_fog,
            "smog_index": analysis.smog_index,
            "automated_readability_index": analysis.automated_readability_index,
        },
    }

    if args.as_json:
        print(json.dumps(result, indent=2, default=str))
        return

    print(f"Format:      {result['format']}")
    print(f"File size:   {result['file_size']:,} bytes")
    print(f"Words:       {result['word_count']}")
    print(f"Sentences:   {result['sentence_count']}")
    print(f"Paragraphs:  {result['paragraph_count']}")
    r = result["readability"]
    print(f"Flesch:      {r['flesch_reading_ease']:.1f} (grade {r['flesch_kincaid_grade']:.1f})")
    print(f"Gunning Fog: {r['gunning_fog']:.1f}")


def _extract_text(path: Path, suffix: str) -> str:
    if suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()
    else:
        from markitdown import MarkItDown
        result = MarkItDown().convert(str(path))
        return result.text_content.strip()


def _cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run(
        "document_analyser.main:document_analyser",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
