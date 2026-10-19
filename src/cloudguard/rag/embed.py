"""Embedding backends for the RAG index.

FastEmbedder is the live implementation; FakeEmbedder is deterministic and network/model-free
for tests. Nothing here imports fastembed at module level, so mock-mode tests and CI never
need the `agent` extra - same lazy-import convention as GeminiClient in llm.py.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import numpy as np

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    """Live mode: calls fastembed's ONNX runtime locally (no network once the model is
    cached). query_embed()/passage_embed() apply each model's own required prefix
    internally - bge-small-en-v1.5 prefixes queries, not passages - so we never hardcode
    a prefix string ourselves; fastembed's model registry already knows it per model.
    """

    name: str

    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding

        self.name = model_name or os.environ.get("CLOUDGUARD_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self._model = TextEmbedding(model_name=self.name)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.array(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(list(self._model.query_embed([text]))[0], dtype=np.float32)


class FakeEmbedder:
    """Deterministic hashed bag-of-words - what tests use. hashlib.md5, not Python's
    builtin hash(), because str hashing is randomized per-process (PYTHONHASHSEED) and
    this must be the same vector every run, every process, for a stale-index check that
    compares model *names* to still mean something in tests."""

    name = "fake-bow-v1"
    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dim
            v[idx] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)
