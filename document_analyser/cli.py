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

    from document_analyser.analysis import analyse_document

    try:
        result = analyse_document(path)
    except Exception as e:
        print(f"Error: could not analyse document: {e}", file=sys.stderr)
        sys.exit(1)

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


if __name__ == "__main__":
    main()
