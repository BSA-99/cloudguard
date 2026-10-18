"""Day-1 placeholder for search_docs: keyword scoring over a tiny curated corpus.

Day 3 replaces this with embeddings over ~30 public Microsoft Learn pages. The function
signature and return shape stay the same, so the tool contract (and its tests) don't change.
Summaries in docs/corpus.json are written in our own words; the URL is the citation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "how", "what", "do", "i"}


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


def search(corpus_path: Path, query: str, k: int = 3) -> list[dict[str, Any]]:
    docs = json.loads(corpus_path.read_text(encoding="utf-8"))
    q = set(_tokens(query))
    if not q:
        return []
    scored = []
    for d in docs:
        body = _tokens(f"{d['title']} {d['title']} {' '.join(d.get('keywords', []))} {d['summary']}")
        score = sum(1 for t in body if t in q)
        if score:
            scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    return [{"title": d["title"], "url": d["url"], "summary": d["summary"], "score": s} for s, d in scored[:k]]
