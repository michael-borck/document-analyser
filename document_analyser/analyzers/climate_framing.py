#!/usr/bin/env python3
"""ClimateBERT framing analyzer (ADR-0039, interpretable-ML rung).

Third-party models from the climate-disclosure NLP community. If this
analyzer contributes to published work, cite the originals:

- ``climatebert/distilroberta-base-climate-detector`` — Bingler, Kraus,
  Leippold & Senni (2022), *"How climate change is discussed in financial
  disclosures"* (ClimateBERT project, University of Zurich).
- ``climatebert/distilroberta-base-climate-commitment`` — Bingler, Kraus,
  Leippold & Senni (2022), same programme.
- ``climatebert/netzero-reduction`` — Schimanski, Reding, Reding & Schär
  (EMNLP 2023).

The per-passage pipeline shape (reader -> annotators over a shared
document structure) is inspired by ReportParse (Morio, In, Yoon,
Rowlands & Manning, IJCAI 2024 demo), which validated these model
choices on this exact corpus. All models and ReportParse are Apache-2.0.

Role in the app's method ladder (ADR-0035/0039): these models never
record framing codes. They suggest — the app renders their output as
flagged chips; the recorded value is always the human's accepted one.
The only framing this analyzer proposes is Aspirational (legend value
1): a target direction detected without the number + date that the
deterministic rung-0 rule already claims as Quantified.
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from transformers import pipeline as hf_pipeline
except ImportError:  # optional [nlp] extra — degrade, don't crash
    hf_pipeline = None

try:
    from huggingface_hub import model_info as hf_model_info
except ImportError:
    hf_model_info = None

# Pinned model ids. Revisions resolve at first load: the env override
# wins, else the resolved commit sha is fetched best-effort and cached —
# suggestion provenance is always "model@something" so reruns are
# attributable (ADR-0039).
MODEL_SPECS: dict[str, str] = {
    "detector": "climatebert/distilroberta-base-climate-detector",
    "commitment": "climatebert/distilroberta-base-climate-commitment",
    "target": "climatebert/netzero-reduction",
}

# Minimum confidences. Tuned conservative — suggestions are flagged for
# human review, so recall matters less than never wasting the
# researcher's attention (ADR-0039: precision over noise).
DETECTOR_MIN_CONFIDENCE = 0.5
TARGET_MIN_CONFIDENCE = 0.5
# Truncation: the distilroberta models take 512 tokens; passages arrive
# sentence-grain from the app, so this is generous.
MAX_CHARS = 2000

logger = logging.getLogger(__name__)


class ClimateFramingAnalyzer:
    """Suggests Aspirational framing from ClimateBERT target detection.

    Never raises from model absence: ``available`` reports the state and
    ``suggest_batch`` returns ``available=False`` results — the app
    degrades to the deterministic rung-0 rules (ADR-0039 non-fatal
    contract).
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, Any] | None = None
        self._load_error: str | None = None
        self._revisions: dict[str, str] = {}

    @property
    def available(self) -> bool:
        """True when the three pipelines are loaded (or load on demand)."""
        if self._pipelines is not None:
            return True
        return self._load() is not None

    @property
    def load_error(self) -> str | None:
        if self._pipelines is None and self._load_error is None:
            self._load()
        return self._load_error

    def _load(self) -> dict[str, Any] | None:
        if self._pipelines is not None:
            return self._pipelines
        if self._load_error is not None and not self._pipelines and hf_pipeline is None:
            # A previous attempt failed with transformers missing — don't
            # retry import churn on every call; a restart re-imports.
            return None
        if hf_pipeline is None:
            self._load_error = "transformers not installed (optional [nlp] extra)"
            return None
        try:
            pipes: dict[str, Any] = {}
            for key, name in MODEL_SPECS.items():
                pipe = hf_pipeline(
                    "text-classification",
                    model=name,
                    device=-1,  # CPU for PyInstaller compatibility (ADR-0022)
                    truncation=True,
                )
                # macOS/arm64 SIGBUS workaround: clone mmap-backed weights
                # onto the heap before the first forward pass — same fix as
                # sentiment_analyzer (see INTEGRATION-NOTES.md).
                for param in pipe.model.parameters():
                    param.data = param.data.clone()
                pipes[key] = pipe
            self._pipelines = pipes
            self._load_error = None
            return pipes
        except Exception as e:  # noqa: BLE001 — degrade, don't fail
            self._load_error = str(e)
            return None

    def _revision(self, key: str) -> str:
        """Best-effort resolved revision for provenance (model@sha)."""
        if key in self._revisions:
            return self._revisions[key]
        env = os.environ.get("DOCUMENT_ANALYSER_CLIMATEBERT_REVISIONS", "")
        # Env form: "detector=<sha>,target=<sha>" — pins without code changes.
        for part in env.split(","):
            if "=" in part and part.split("=", 1)[0].strip() == key:
                rev = part.split("=", 1)[1].strip()
                self._revisions[key] = rev
                return rev
        rev = "main"
        if hf_model_info is not None:
            try:
                sha = hf_model_info(MODEL_SPECS[key]).sha
                rev = sha[:10] if sha else "main"
            except Exception as e:  # noqa: BLE001 — offline: fall back to 'main'
                logger.debug("Revision lookup failed for %s: %s", MODEL_SPECS[key], e)
        self._revisions[key] = rev
        return rev

    def _classify(self, key: str, text: str) -> dict[str, float] | None:
        pipes = self._load()
        if pipes is None:
            return None
        pipe = pipes[key]
        raw = pipe(text[:MAX_CHARS], truncation=True)
        # Normalize pipeline output shapes (single dict or list of dicts).
        items = raw if isinstance(raw, list) else [raw]
        if items and isinstance(items[0], list):
            items = items[0]
        labels = getattr(pipe.model.config, "id2label", {})
        out: dict[str, float] = {}
        for item in items:
            label_id = str(item.get("label", ""))
            try:
                readable = labels.get(int(label_id.replace("LABEL_", "")), label_id)
            except ValueError:
                readable = label_id
            out[str(readable).lower()] = float(item.get("score", 0.0))
        return out

    @staticmethod
    def _top_label(labels: dict[str, float]) -> tuple[str, float]:
        """The argmax label — how a text-classification decision is read."""
        return max(labels.items(), key=lambda kv: kv[1])

    def suggest_batch(self, passages: list[str]) -> dict[str, Any]:
        """Framing suggestions per passage.

        Returns ``{"available": bool, "error": str | None, "models": {...},
        "results": [...]}`` where each result is::

            {"climate": bool, "commitment": bool, "target": str,
             "target_score": float, "framing_value": "1" | None,
             "model_revision": str}

        ``framing_value`` is ``"1"`` (Aspirational) when a net-zero or
        reduction target is detected above threshold on a climate-related
        passage — the direction-of-value signal. Quantified (2) and Limit
        (3) belong to the deterministic rung; this analyzer never claims
        them.
        """
        pipes = self._load()
        if pipes is None:
            return {
                "available": False,
                "error": self._load_error or "ClimateBERT models not loaded",
                "models": {},
                "results": [None] * len(passages),
            }

        models = {
            key: f"{MODEL_SPECS[key]}@{self._revision(key)}" for key in MODEL_SPECS
        }
        results: list[dict[str, Any]] = []
        for passage in passages:
            if not passage or not passage.strip():
                results.append(None)
                continue

            detector_labels = self._classify("detector", passage)
            climate = False
            if detector_labels:
                top_label, top_score = self._top_label(detector_labels)
                climate = (
                    "climate" in top_label
                    and "not" not in top_label
                    and top_score >= DETECTOR_MIN_CONFIDENCE
                )

            if not climate:
                results.append(
                    {
                        "climate": False,
                        "commitment": False,
                        "target": "none",
                        "target_score": 0.0,
                        "framing_value": None,
                        "model_revision": models["target"],
                    }
                )
                continue

            target_labels = self._classify("target", passage)
            target, target_score = "none", 0.0
            framing_value: str | None = None
            if target_labels:
                top_label, top_score = self._top_label(target_labels)
                if top_score >= TARGET_MIN_CONFIDENCE:
                    if "net zero" in top_label:
                        target = "net-zero"
                    elif "reduction" in top_label:
                        target = "reduction"
                    if target != "none":
                        target_score = top_score
                        framing_value = "1"  # Aspirational — a direction of value

            commitment = False
            commitment_labels = self._classify("commitment", passage)
            if commitment_labels:
                top_label, top_score = self._top_label(commitment_labels)
                commitment = (
                    "no" not in top_label.split(" ")[:1]
                    and top_score >= TARGET_MIN_CONFIDENCE
                )

            results.append(
                {
                    "climate": climate,
                    "commitment": commitment,
                    "target": target,
                    "target_score": round(target_score, 4),
                    "framing_value": framing_value,
                    "model_revision": models["target"],
                }
            )

        return {"available": True, "error": None, "models": models, "results": results}
