"""A small in-repo BM25 over chunk text. Parameters and tokenization are fixed and recorded in
evals/RESULTS.md (pre-registration + addendum); nothing here is tuned on the golden set.

  * tokens: lowercase alphanumeric runs, minus STOPWORDS. No stemming, no camelCase splitting.
  * document text: title + heading path + text, one bag of words, no per-field boosts.
  * idf = ln(1 + (N - n + 0.5) / (n + 0.5)) (never negative).
  * score(D, Q) = sum over UNIQUE query terms q of
        idf(q) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |D| / avgdl))
  * only chunks with a score > 0 are ranked; ties break by chunk_id.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

K1 = 1.5
B = 0.75

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it my of on or so that the this to was what when which with".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def chunk_document_text(chunk: dict[str, Any]) -> str:
    return f"{chunk['title']}\n{chunk['heading_path']}\n{chunk['text']}"


class BM25Index:
    def __init__(self, doc_ids: list[str], docs_tokens: list[list[str]], k1: float = K1, b: float = B):
        if len(doc_ids) != len(docs_tokens):
            raise ValueError("doc_ids and docs_tokens must be the same length")
        self.doc_ids = doc_ids
        self.k1, self.b = k1, b
        self._tf = [Counter(tokens) for tokens in docs_tokens]
        self._len = [len(tokens) for tokens in docs_tokens]
        n_docs = len(doc_ids)
        self._avgdl = (sum(self._len) / n_docs) if n_docs else 0.0
        doc_freq: Counter[str] = Counter()
        for tf in self._tf:
            doc_freq.update(tf.keys())
        self._idf = {t: math.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for t, n in doc_freq.items()}

    @classmethod
    def from_chunks(cls, chunks: Iterable[dict[str, Any]], k1: float = K1, b: float = B) -> "BM25Index":
        chunks = list(chunks)
        return cls([c["chunk_id"] for c in chunks], [tokenize(chunk_document_text(c)) for c in chunks], k1, b)

    def scores(self, query: str) -> list[float]:
        """One score per document, in doc order (0.0 for documents with no matching term)."""
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]  # unique, in order
        out = []
        for tf, dl in zip(self._tf, self._len):
            norm = self.k1 * (1 - self.b + self.b * dl / self._avgdl) if self._avgdl else self.k1
            out.append(sum(
                self._idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if tf[t]
            ))
        return out

    def rank(self, query: str) -> list[tuple[str, float]]:
        """(doc_id, score) for matching documents only (score > 0), best first, ties by doc_id."""
        scored = [(d, s) for d, s in zip(self.doc_ids, self.scores(query)) if s > 0]
        return sorted(scored, key=lambda x: (-x[1], x[0]))
