"""Reciprocal rank fusion. Pure and deterministic."""

from __future__ import annotations

import math

RRF_K = 60


def fuse_rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """rankings: each a list of ids, best first (1-based ranks). Returns (id, score) best first,
    score = sum over the rankings containing the id of 1 / (k + rank). Ties break by id.
    math.fsum makes the score independent of the order of `rankings`."""
    contributions: dict[str, list[float]] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:  # a repeated id would otherwise count twice
                continue
            seen.add(item)
            contributions.setdefault(item, []).append(1.0 / (k + rank))
    fused = [(item, math.fsum(parts)) for item, parts in contributions.items()]
    return sorted(fused, key=lambda x: (-x[1], x[0]))
