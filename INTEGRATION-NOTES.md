# document-analyser integration notes (from ISYS6020 marking pipeline)

Date: 2026-05-21

## Problem

The README advertises a clean library API:

    from app.analyser import DocumentAnalyser
    result = DocumentAnalyser().analyse("report.pdf")
    result["text"], result["word_count"], result["page_count"], result["readability"]

This API does NOT exist in either the published 0.2.1 wheel or the local 0.3.1
source. The package is structured as a FastAPI/CiteSight backend.

## What actually exists (0.3.1)

- Text extraction: only via the private `document_analyser.cli._extract_text(path, suffix)`
  (sync, path-based, not part of the public API) or
  `document_analyser.services.document_processor.DocumentProcessor.extract_text_with_pages`
  (async, bytes-based).
- Readability: CLEAN and usable.
  `from document_analyser.analyzers.readability import ReadabilityAnalyzer`
  `ReadabilityAnalyzer().analyze(text) -> DocumentAnalysis` with fields
  `word_count, sentence_count, avg_words_per_sentence, paragraph_count,
  flesch_score, flesch_kincaid_grade`.

## What the marking pipeline does as a result

- Uses `ReadabilityAnalyzer` directly for Flesch metrics (mapping `flesch_score`
  to `flesch_reading_ease`).
- Uses `pypdf` (primary) + `pymupdf` (fallback) for text extraction, because the
  documented extraction facade is missing.

## Recommendation

Add a public top-level facade matching the README, e.g. a `DocumentAnalyser`
class (or module function) with sync `analyse(path) -> dict` returning
`text`, `word_count`, `page_count`, and a `readability` sub-dict, so downstream
tools can use the documented API instead of reaching into private internals.
Until then, downstream code should not rely on `app.analyser`.

---

## ~~Known issue~~ FIXED (2026-05-24): `/semantic/sentiment` crashed on macOS

**Symptom:** with the `[nlp]`/ML extras installed, calling `/semantic/sentiment`
**hard-crashes the uvicorn server** — the process exits 138 (SIGBUS). The
`resource_tracker: ... leaked semaphore {'/loky-...'}` line is a **red herring**:
it's just sklearn's import-time joblib tracker dying alongside the crash, not the
cause. `/health`, `/manifest`, `/text` (readability, writing quality, vocabulary,
spaCy NER), and — importantly — `/semantic/domain-mapping` + `/semantic/structural-mismatch`
(sentence-transformers) are **unaffected**. Only the transformers-`pipeline`
sentiment path crashes.

**Reproduce:**

    .venv/bin/document-analyser serve --port 8000 --host 127.0.0.1
    curl -X POST http://127.0.0.1:8000/semantic/sentiment \
      -H 'Content-Type: application/json' -d '{"text":"I love this. It is terrible."}'

**Root cause (confirmed 2026-05-24):** none of the original hypotheses. It is **not**
fork/threading/OpenMP and **not** dual-libomp — `OMP_NUM_THREADS=1`,
`KMP_DUPLICATE_LIB_OK=TRUE`, `torch.set_num_threads(1)`, and `torch.inference_mode()`
each make no difference, and sentiment crashes on its own with sentence-transformers
(sklearn) never imported. The faulthandler native stack lands in
`torch.nn.Linear.forward` (a GEMM) inside the distilbert forward pass.

`transformers.pipeline` / `from_pretrained` hands back **mmap-backed safetensors
weight views**. On macOS/arm64, torch's GEMM faults (SIGBUS) while paging those
mmap'd weight pages in during the first forward. Two controls prove it: reading
every weight byte via `safetensors.load_file` is fine, and a raw torch GEMM of the
same shape is fine — only the mmap-backed *model forward* faults. `SentenceTransformer`
(domain-mapping / structural-mismatch) materialises its weights differently, which is
why those endpoints never crashed.

**Fix (applied):** clone the pipeline's parameters onto the heap right after
construction, in `GranularSentimentAnalyzer.__init__`:

    for param in self.sentiment_pipeline.model.parameters():
        param.data = param.data.clone()

`.clone()` is the operative call — it forces a real copy off the mmap. The params are
already contiguous, so `.contiguous()` alone is a no-op and does **not** help.
Verified: unpatched forward exits 138; patched forward returns correct logits
(`POSITIVE 0.9995`) and exits 0. The full repro (`domain-mapping` then `sentiment`)
now runs clean end to end.

**Context:** found while integrating FeedForward (`../../feed-forward`) with this
analyser over HTTP. FeedForward uses `/text` only and defers sentiment, so this is not
blocking that work — but a hard cross-platform crash should be fixed here.

> Note: this checkout also has **uncommitted edits** from the same session — a fix to
> the `/analyse` route + CLI (they referenced `flesch_reading_ease`/`gunning_fog`/
> `smog_index`/`automated_readability_index`, which now exist on `DocumentAnalysis`).
> Those are unrelated to this crash; keep or commit them separately.
