"""Torch-free ONNX inference for the semantic analyzers.

Replaces sentence-transformers / transformers-pipeline (both drag in torch,
~200-400 MB, and transformers eagerly imports torch in generation/) with
onnxruntime + the Rust `tokenizers` library directly — no torch, no transformers.
Outputs are verified identical to the torch models: MiniLM embeddings match at
cosine 1.0; distilbert-sst2 labels + scores match to 4 decimals.

The models are pre-exported ONNX variants, prefetched into the HF cache at build
time (see the desktop app's tauri.yml) and loaded offline in the frozen bundle.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

# Short/torch model name -> the repo that carries an `onnx/model.onnx`.
_EMBED_REPOS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}
_SENTIMENT_REPOS = {
    # Xenova ships the same weights as distilbert-...-sst-2-english, ONNX-exported.
    "distilbert-base-uncased-finetuned-sst-2-english": "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
}

_EMBED_DIM = 384
# Match each model's training truncation so embeddings are identical for long
# inputs (MiniLM truncates at 256; distilbert-sst2 at 512).
_EMBED_MAX_LEN = 256
_SENTIMENT_MAX_LEN = 512


@lru_cache(maxsize=4)
def _session(repo: str, max_length: int) -> tuple[Any, Any, frozenset[str]]:
    """Load (and cache) an ONNX session + Rust tokenizer for a repo."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    onnx_path = hf_hub_download(repo, filename="onnx/model.onnx")
    tokenizer = Tokenizer.from_file(hf_hub_download(repo, filename="tokenizer.json"))
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=max_length)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = frozenset(i.name for i in sess.get_inputs())
    return sess, tokenizer, input_names


def _feed(tokenizer: Any, texts: list[str], input_names: frozenset[str]) -> tuple[dict, np.ndarray]:
    encs = tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encs], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    feed: dict[str, np.ndarray] = {}
    if "input_ids" in input_names:
        feed["input_ids"] = input_ids
    if "attention_mask" in input_names:
        feed["attention_mask"] = attention_mask
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
    return feed, attention_mask


class OnnxEmbedder:
    """Drop-in for SentenceTransformer: ``encode(list[str]) -> (n, dim)`` float32,
    mean-pooled and L2-normalized (matches all-MiniLM-L6-v2's default)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.repo = _EMBED_REPOS.get(model_name, model_name)

    def encode(self, texts: Any, normalize_embeddings: bool = True, **_: Any) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        sess, tokenizer, input_names = _session(self.repo, _EMBED_MAX_LEN)
        feed, attention_mask = _feed(tokenizer, texts, input_names)
        token_embeddings = sess.run(None, feed)[0]  # (n, seq, dim)
        mask = attention_mask[:, :, None].astype(np.float32)
        pooled = (token_embeddings * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        if normalize_embeddings:
            pooled = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return pooled.astype(np.float32)


class OnnxSentiment:
    """Drop-in for the transformers sentiment pipeline: callable returning a list
    of ``{'label': 'POSITIVE'|'NEGATIVE', 'score': float}`` (one per input, like
    the pipeline — so ``pipe(text)[0]`` still works)."""

    _LABELS = {0: "NEGATIVE", 1: "POSITIVE"}

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> None:
        self.repo = _SENTIMENT_REPOS.get(model_name, model_name)

    def __call__(self, texts: Any) -> list[dict[str, Any]]:
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            return []
        sess, tokenizer, input_names = _session(self.repo, _SENTIMENT_MAX_LEN)
        feed, _ = _feed(tokenizer, batch, input_names)
        logits = sess.run(None, feed)[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return [
            {"label": self._LABELS[int(row.argmax())], "score": float(row.max())}
            for row in probs
        ]
