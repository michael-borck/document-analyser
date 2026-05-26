"""CLI entry point for document-analyser.

Usage:
  document-analyser report.pdf
  document-analyser thesis.docx --json
  document-analyser slides.pptx
  document-analyser serve
  document-analyser serve --port 8000 --host 0.0.0.0
"""

import json
import sys
from pathlib import Path


def main() -> None:
    import argparse

    from lens_contract import run_contract_subcommands

    from document_analyser.manifest import MANIFEST

    # `serve` and `manifest` are the family's shared subcommands (lens-contract).
    if run_contract_subcommands(
        MANIFEST,
        app_path="document_analyser.api:app",
        default_port=8000,
        env_prefix="DOCUMENT_ANALYSER",
    ):
        return

    parser = argparse.ArgumentParser(
        prog="document-analyser",
        description="Extract text and readability metrics from documents",
    )
    parser.add_argument("file", type=Path, help="Document to analyse (PDF, DOCX, PPTX, TXT, MD)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")
    _cmd_analyse(parser.parse_args())


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
            "flesch_reading_ease": analysis.flesch_score,
            "flesch_kincaid_grade": analysis.flesch_kincaid_grade,
            "gunning_fog": analysis.gunning_fog,
            "smog_index": analysis.smog_index,
            "automated_readability_index": analysis.automated_readability_index,
        },
    }

    # Additive: .pptx gets a slide-design block on top of prose/readability.
    # The text-only extraction can't see slide structure / images / layouts.
    if suffix == ".pptx":
        try:
            from document_analyser.analyzers.slide_design import analyse_pptx
            result["slide_design"] = analyse_pptx(path).to_dict()
        except Exception as e:
            result["slide_design_error"] = str(e)

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
    sd = result.get("slide_design")
    if sd:
        print()
        print("Slide design:")
        print(f"  slides:           {sd['slide_count']}")
        titled = int(round(sd['title_coverage'] * sd['slide_count']))
        print(f"  titled:           {titled}/{sd['slide_count']}  ({sd['title_coverage']:.0%} coverage)")
        print(f"  avg words/slide:  {sd['avg_words_per_slide']:.1f}  (max {sd['max_words_per_slide']})")
        print(f"  avg images/slide: {sd['avg_images_per_slide']:.2f}  ({sd['total_images']} total)")
        print(f"  max bullet depth: {sd['max_bullet_depth']}")
        print(f"  layouts used:     {sd['distinct_layouts']}  ({', '.join(sd['layout_names'])})")
        if sd['empty_slides']:
            print(f"  empty slides:     {sd['empty_slides']}")
        if sd['text_overloaded_slides']:
            print(f"  overloaded (>80 words): {sd['text_overloaded_slides']} slide(s)")
    elif result.get("slide_design_error"):
        print()
        print(f"Slide design: error — {result['slide_design_error']}")


def _extract_text(path: Path, suffix: str) -> str:
    # Canonical extractor: the single home for binary -> text in the family.
    from document_analyser.extraction import extract_text
    return extract_text(path)


if __name__ == "__main__":
    main()
