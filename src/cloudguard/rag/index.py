"""Build and search the RAG index: corpus/index/{vectors.npy, chunks.jsonl, meta.json}.

DEFAULT_MANIFEST_PATH / DEFAULT_INDEX_DIR / get_default_embedder are module-level and read
at call time (inside search_default), not captured as default argument values - so tests
can monkeypatch them directly on this module and search_docs picks up the change with no
code change on its side. That's what keeps the real FastEmbedder out of unit tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from cloudguard.models import CorpusManifest
from cloudguard.rag.bm25 import BM25Index
from cloudguard.rag.chunk import WARN_WORDS, Chunk, word_count, chunk_article
from cloudguard.rag.embed import Embedder
from cloudguard.rag.fusion import RRF_K, fuse_rrf

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = ROOT / "docs" / "corpus_manifest.json"
DEFAULT_RAW_DIR = ROOT / "corpus" / "raw"
DEFAULT_INDEX_DIR = ROOT / "corpus" / "index"

MODES = ("vector", "bm25", "hybrid")
# Read at call time by search_default (so it can be monkeypatched). Stays "vector" until the
# decision rule in evals/RESULTS.md is met.
DEFAULT_MODE = "vector"


class StaleIndexError(RuntimeError):
    """The on-disk index doesn't match the current manifest or embedding model."""


def get_default_embedder() -> Embedder:
    from cloudguard.rag.embed import FastEmbedder

    return FastEmbedder()


def manifest_hash(manifest_path: Path) -> str:
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


@dataclass
class SearchResult:
    chunk_id: str
    article_id: str
    title: str
    heading: str
    topic: str
    url: str
    text: str
    score: float


def build_index(manifest_path: Path, raw_dir: Path, embedder: Embedder, out_dir: Path) -> dict:
    manifest = CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    chunks: list[Chunk] = []
    for article in manifest.articles:
        chunks.extend(chunk_article(article, raw_dir))

    seen_ids: set[str] = set()
    for c in chunks:
        if c.chunk_id in seen_ids:
            raise ValueError(f"duplicate chunk_id: {c.chunk_id!r}")
        seen_ids.add(c.chunk_id)

    # Reported, not silenced: anything still over WARN_WORDS after chunk.py's own
    # table/code-preserving, prose-splitting remediation is a table or a code block that
    # genuinely can't be split further.
    over_warn_chunks = [
        {"chunk_id": c.chunk_id, "word_count": word_count(c.text)}
        for c in chunks if word_count(c.text) > WARN_WORDS
    ]

    texts = [c.embedding_text() for c in chunks]
    vectors = embedder.embed_passages(texts)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vectors.npy", vectors.astype(np.float32))
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")

    meta = {
        "manifest_hash": manifest_hash(manifest_path),
        "model_name": embedder.name,
        "chunk_count": len(chunks),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return {**meta, "over_warn_chunks": over_warn_chunks, "warn_limit_words": WARN_WORDS}


def _load_index(index_dir: Path) -> tuple[dict, list[dict], np.ndarray]:
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    vectors = np.load(index_dir / "vectors.npy")
    return meta, chunks, vectors


def _take_with_cap(
    ranked: Iterable[tuple[int, float]], chunks: list[dict], k: int, per_article_cap: int,
) -> list[SearchResult]:
    """Walk (chunk index, score) pairs best-first; keep at most per_article_cap chunks per
    article and stop at k. The same cap-then-truncate step for every mode."""
    results: list[SearchResult] = []
    per_article: dict[str, int] = {}
    for i, score in ranked:
        c = chunks[i]
        if per_article.get(c["article_id"], 0) >= per_article_cap:
            continue
        per_article[c["article_id"]] = per_article.get(c["article_id"], 0) + 1
        results.append(SearchResult(
            chunk_id=c["chunk_id"], article_id=c["article_id"], title=c["title"],
            heading=c["heading_path"], topic=c["topic"], url=c["url"], text=c["text"],
            score=score,
        ))
        if len(results) >= k:
            break
    return results


def search(
    query: str,
    k: int,
    embedder: Embedder,
    manifest_path: Path,
    index_dir: Path,
    per_article_cap: int = 2,
    mode: str = "vector",
) -> list[SearchResult]:
    """mode: "vector" (cosine; score is the cosine), "bm25" (score is the BM25 score) or
    "hybrid" (RRF over the full vector and BM25 rankings; score is the RRF score). Scores from
    different modes are on different scales and are not comparable."""
    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {MODES}")

    meta, chunks, vectors = _load_index(index_dir)

    current_hash = manifest_hash(manifest_path)
    if meta["manifest_hash"] != current_hash or meta["model_name"] != embedder.name:
        raise StaleIndexError(
            f"index is stale (manifest or model changed since it was built), "
            f"run `cloudguard build-index`"
        )

    if not chunks:
        return []

    if mode == "bm25":
        # Built at load time from chunks.jsonl; the corpus is small, so no cache to go stale.
        by_id = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        ranking = BM25Index.from_chunks(chunks).rank(query)
        return _take_with_cap(((by_id[d], s) for d, s in ranking), chunks, k, per_article_cap)

    qvec = embedder.embed_query(query)
    qnorm = np.linalg.norm(qvec)
    if qnorm > 0:
        qvec = qvec / qnorm

    scores = vectors @ qvec

    if mode == "vector":
        order = np.argsort(-scores)
        return _take_with_cap(((i, float(scores[i])) for i in order), chunks, k, per_article_cap)

    # hybrid: fuse the complete vector ranking (ties by chunk_id, so the input is deterministic)
    # with the BM25 ranking of matching chunks.
    by_id = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    vector_ranking = [chunks[i]["chunk_id"] for i in sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i]["chunk_id"]))]
    bm25_ranking = [d for d, _ in BM25Index.from_chunks(chunks).rank(query)]
    fused = fuse_rrf([vector_ranking, bm25_ranking], k=RRF_K)
    return _take_with_cap(((by_id[d], s) for d, s in fused), chunks, k, per_article_cap)


def search_default(query: str, k: int = 3) -> list[SearchResult]:
    return search(
        query, k,
        embedder=get_default_embedder(),
        manifest_path=DEFAULT_MANIFEST_PATH,
        index_dir=DEFAULT_INDEX_DIR,
        mode=DEFAULT_MODE,
    )
