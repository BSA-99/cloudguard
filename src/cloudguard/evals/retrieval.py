"""Retrieval-only eval: how well rag.index.search() finds the right article/section for a
fixed golden query set (evals/retrieval_golden.jsonl). This measures retrieval in
isolation, not the full agent loop - Day 4 builds a broader eval harness on top of this.

score_ranking() and aggregate() are pure (no search, no embedder) so the metric math is
unit-testable on a hand-made ranking, independent of anything real retrieval returns.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cloudguard.rag.embed import Embedder
from cloudguard.rag.index import search

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN_PATH = ROOT / "evals" / "retrieval_golden.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "evals" / "results"

OUT_OF_SCOPE = "out_of_scope"


@dataclass
class GoldenQuery:
    query: str
    expected_articles: list[str]
    source: str
    expected_sections: list[str] | None = None
    category: str = "in_scope"


def load_golden(path: Path) -> list[GoldenQuery]:
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        queries.append(GoldenQuery(
            query=d["query"], expected_articles=d["expected_articles"], source=d["source"],
            expected_sections=d.get("expected_sections"), category=d.get("category", "in_scope"),
        ))
    return queries


def slug_of(url: str) -> str | None:
    return url.rsplit("#", 1)[1] if "#" in url else None


def _dedupe_by_article(
    ranked: list[tuple[str, str | None, float]],
) -> list[tuple[str, str | None, float]]:
    """Keep only the first (best-ranked) chunk per article_id, so a page that took 2 of
    the per_article_cap slots doesn't inflate its own rank for article-level hit@k."""
    seen: set[str] = set()
    deduped = []
    for item in ranked:
        aid = item[0]
        if aid in seen:
            continue
        seen.add(aid)
        deduped.append(item)
    return deduped


@dataclass
class QueryScore:
    query: str
    source: str
    category: str
    hit_at_1: bool | None  # None when category is out_of_scope (excluded, not scored as a miss)
    hit_at_3: bool | None
    hit_at_5: bool | None
    reciprocal_rank: float | None
    top3_article_ids: list[str]
    section_hit_at_3: bool | None  # None when expected_sections wasn't given, or category is out_of_scope
    top1_score: float | None  # the top-ranked (post-dedup) result's similarity score; None if no results


def score_ranking(
    query: str,
    source: str,
    ranked: list[tuple[str, str | None, float]],  # [(article_id, heading_slug_or_None, score), ...] in rank order
    expected_articles: list[str],
    expected_sections: list[str] | None,
    category: str = "in_scope",
) -> QueryScore:
    """Pure: takes an already-ranked list, nothing calls search() here."""
    deduped = _dedupe_by_article(ranked)
    top1_score = deduped[0][2] if deduped else None

    if category == OUT_OF_SCOPE:
        return QueryScore(
            query=query, source=source, category=category,
            hit_at_1=None, hit_at_3=None, hit_at_5=None, reciprocal_rank=None,
            top3_article_ids=[aid for aid, _, _ in deduped[:3]],
            section_hit_at_3=None, top1_score=top1_score,
        )

    rank_of_hit = next((i + 1 for i, (aid, _, _) in enumerate(deduped) if aid in expected_articles), None)
    reciprocal_rank = 1.0 / rank_of_hit if rank_of_hit else 0.0

    section_hit_at_3 = None
    if expected_sections is not None:
        section_hit_at_3 = any(
            aid in expected_articles and slug in expected_sections
            for aid, slug, _ in deduped[:3]
        )

    return QueryScore(
        query=query,
        source=source,
        category=category,
        hit_at_1=rank_of_hit is not None and rank_of_hit <= 1,
        hit_at_3=rank_of_hit is not None and rank_of_hit <= 3,
        hit_at_5=rank_of_hit is not None and rank_of_hit <= 5,
        reciprocal_rank=reciprocal_rank,
        top3_article_ids=[aid for aid, _, _ in deduped[:3]],
        section_hit_at_3=section_hit_at_3,
        top1_score=top1_score,
    )


@dataclass
class MetricBlock:
    num_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    section_hit_at_3: float | None
    section_eval_count: int


@dataclass
class EvalReport:
    overall: MetricBlock
    by_source: dict[str, MetricBlock] = field(default_factory=dict)
    misses_top3: list[dict[str, Any]] = field(default_factory=list)
    per_query: list[dict[str, Any]] = field(default_factory=list)
    top1_score_stats: dict[str, float | None] = field(
        default_factory=lambda: {"min": None, "median": None, "max": None}
    )
    out_of_scope: list[dict[str, Any]] = field(default_factory=list)


def _metric_block(scores: list[QueryScore]) -> MetricBlock:
    """scores must already be in-scope only - callers filter out out_of_scope first."""
    n = len(scores)
    section_scores = [s.section_hit_at_3 for s in scores if s.section_hit_at_3 is not None]
    return MetricBlock(
        num_queries=n,
        hit_at_1=sum(s.hit_at_1 for s in scores) / n if n else 0.0,
        hit_at_3=sum(s.hit_at_3 for s in scores) / n if n else 0.0,
        hit_at_5=sum(s.hit_at_5 for s in scores) / n if n else 0.0,
        mrr=sum(s.reciprocal_rank for s in scores) / n if n else 0.0,
        section_hit_at_3=(sum(section_scores) / len(section_scores)) if section_scores else None,
        section_eval_count=len(section_scores),
    )


def aggregate(golden: list[GoldenQuery], scores: list[QueryScore]) -> EvalReport:
    """Pure: no search, no embedder - rolls up overall + per-source metrics and the
    missed-top-3 list from an already-scored list of queries. out_of_scope queries are
    excluded from hit@k/MRR (they have no expected article to hit) and reported separately
    with just their top1_score, for picking a score threshold that separates them."""
    pairs = list(zip(golden, scores))
    in_scope_pairs = [(gq, s) for gq, s in pairs if s.category != OUT_OF_SCOPE]
    in_scope_scores = [s for _, s in in_scope_pairs]

    by_source: dict[str, list[QueryScore]] = {}
    for s in in_scope_scores:
        by_source.setdefault(s.source, []).append(s)

    misses = [
        {"query": s.query, "source": s.source, "expected_articles": gq.expected_articles,
         "top3_article_ids": s.top3_article_ids}
        for gq, s in in_scope_pairs if not s.hit_at_3
    ]
    per_query = [
        {"query": s.query, "source": s.source, "category": s.category,
         "expected_articles": gq.expected_articles,
         "hit_at_1": s.hit_at_1, "hit_at_3": s.hit_at_3, "hit_at_5": s.hit_at_5,
         "reciprocal_rank": s.reciprocal_rank, "top3_article_ids": s.top3_article_ids,
         "section_hit_at_3": s.section_hit_at_3, "top1_score": s.top1_score}
        for gq, s in pairs
    ]

    top1_scores = [s.top1_score for s in in_scope_scores if s.top1_score is not None]
    top1_score_stats = {
        "min": min(top1_scores) if top1_scores else None,
        "median": statistics.median(top1_scores) if top1_scores else None,
        "max": max(top1_scores) if top1_scores else None,
    }
    out_of_scope = [
        {"query": s.query, "source": s.source, "top1_score": s.top1_score}
        for _, s in pairs if s.category == OUT_OF_SCOPE
    ]

    return EvalReport(
        overall=_metric_block(in_scope_scores),
        by_source={source: _metric_block(group) for source, group in by_source.items()},
        misses_top3=misses,
        per_query=per_query,
        top1_score_stats=top1_score_stats,
        out_of_scope=out_of_scope,
    )


def run_eval(
    golden: list[GoldenQuery],
    embedder: Embedder,
    manifest_path: Path,
    index_dir: Path,
    k: int = 5,
) -> EvalReport:
    scores = []
    for gq in golden:
        results = search(gq.query, k=k, embedder=embedder, manifest_path=manifest_path, index_dir=index_dir)
        ranked = [(r.article_id, slug_of(r.url), r.score) for r in results]
        scores.append(score_ranking(
            gq.query, gq.source, ranked, gq.expected_articles, gq.expected_sections, category=gq.category,
        ))
    return aggregate(golden, scores)
