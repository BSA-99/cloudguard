"""Metric math (score_ranking / aggregate) is pure and tested on hand-made rankings, no
search() or embedder involved. One integration test wires run_eval end to end against a
tiny FakeEmbedder-built index, following tests/test_rag.py's fixture conventions.

ranked tuples are (article_id, heading_slug_or_None, score) throughout - score matters now
for top1_score / the out_of_scope threshold report, not just for ranking order.
"""

from __future__ import annotations

import json
from pathlib import Path

from cloudguard.evals.retrieval import (
    GoldenQuery,
    QueryScore,
    aggregate,
    load_golden,
    run_eval,
    score_ranking,
    slug_of,
)
from cloudguard.rag.embed import FakeEmbedder
from cloudguard.rag.index import build_index

REPO = "MicrosoftDocs/azure-docs"


# ------------------------------------------------------------------------- slug_of

def test_slug_of_extracts_fragment():
    assert slug_of("https://learn.microsoft.com/azure/locks#lock-inheritance") == "lock-inheritance"


def test_slug_of_none_when_no_fragment():
    assert slug_of("https://learn.microsoft.com/azure/locks") is None


# ------------------------------------------------------------------- score_ranking

def test_hit_at_1_sets_all_higher_hits_true():
    s = score_ranking("q", "human", [("a", None, 0.9), ("b", None, 0.5)], ["a"], None)
    assert s.hit_at_1 and s.hit_at_3 and s.hit_at_5
    assert s.reciprocal_rank == 1.0
    assert s.top1_score == 0.9


def test_hit_at_3_but_not_at_1():
    ranked = [("x", None, 0.9), ("y", None, 0.8), ("a", None, 0.7), ("z", None, 0.6)]
    s = score_ranking("q", "human", ranked, ["a"], None)
    assert not s.hit_at_1
    assert s.hit_at_3 and s.hit_at_5
    assert s.reciprocal_rank == 1.0 / 3


def test_complete_miss():
    ranked = [("x", None, 0.9), ("y", None, 0.8), ("z", None, 0.7), ("w", None, 0.6), ("v", None, 0.5)]
    s = score_ranking("q", "human", ranked, ["a"], None)
    assert not s.hit_at_1 and not s.hit_at_3 and not s.hit_at_5
    assert s.reciprocal_rank == 0.0
    assert s.top3_article_ids == ["x", "y", "z"]


def test_two_expected_articles_better_ranked_one_wins():
    ranked = [("x", None, 0.9), ("b", None, 0.8), ("a", None, 0.7)]
    s = score_ranking("q", "human", ranked, ["a", "b"], None)
    assert not s.hit_at_1  # rank of first expected hit is 2
    assert s.hit_at_3
    assert s.reciprocal_rank == 1.0 / 2


def test_section_hit_true_when_top3_slug_matches():
    ranked = [("a", "lock-inheritance", 0.9), ("b", None, 0.5)]
    s = score_ranking("q", "human", ranked, ["a"], ["lock-inheritance"])
    assert s.section_hit_at_3 is True


def test_section_hit_false_when_article_matches_but_slug_does_not():
    ranked = [("a", "some-other-section", 0.9), ("b", None, 0.5)]
    s = score_ranking("q", "human", ranked, ["a"], ["lock-inheritance"])
    assert s.section_hit_at_3 is False


def test_section_hit_none_when_expected_sections_not_given():
    ranked = [("a", "whatever", 0.9), ("b", None, 0.5)]
    s = score_ranking("q", "human", ranked, ["a"], None)
    assert s.section_hit_at_3 is None


# ------------------------------------------------------------- score_ranking: dedup

def test_duplicate_article_chunks_do_not_inflate_rank():
    # "a" takes the first two slots (two chunks of the same page); the real 2nd-ranked
    # *article* is "b", which should score as rank 2, not rank 3.
    ranked = [("a", None, 0.9), ("a", "other-section", 0.85), ("b", None, 0.7)]
    s = score_ranking("q", "human", ranked, ["b"], None)
    assert not s.hit_at_1
    assert s.hit_at_3
    assert s.reciprocal_rank == 1.0 / 2


def test_top3_article_ids_has_no_duplicates_after_dedup():
    ranked = [("a", None, 0.9), ("a", None, 0.85), ("b", None, 0.7), ("c", None, 0.6)]
    s = score_ranking("q", "human", ranked, ["z"], None)
    assert s.top3_article_ids == ["a", "b", "c"]


def test_dedup_never_changes_top1_score():
    ranked = [("a", None, 0.42), ("a", None, 0.99)]
    s = score_ranking("q", "human", ranked, ["a"], None)
    assert s.top1_score == 0.42  # the literal top-ranked chunk's score, not the max


# ------------------------------------------------------------- score_ranking: out_of_scope

def test_out_of_scope_skips_hit_at_k_and_mrr():
    ranked = [("x", None, 0.6), ("y", None, 0.5)]
    s = score_ranking("q", "human", ranked, [], None, category="out_of_scope")
    assert s.hit_at_1 is None
    assert s.hit_at_3 is None
    assert s.hit_at_5 is None
    assert s.reciprocal_rank is None
    assert s.section_hit_at_3 is None


def test_out_of_scope_still_reports_top1_score_and_top3():
    ranked = [("x", None, 0.6), ("y", None, 0.5), ("z", None, 0.4)]
    s = score_ranking("q", "human", ranked, [], None, category="out_of_scope")
    assert s.top1_score == 0.6
    assert s.top3_article_ids == ["x", "y", "z"]


def test_out_of_scope_with_no_results_has_none_top1_score():
    s = score_ranking("q", "human", [], [], None, category="out_of_scope")
    assert s.top1_score is None
    assert s.top3_article_ids == []


# ----------------------------------------------------------------------- aggregate

def _golden(
    query: str, expected: list[str], source: str,
    sections: list[str] | None = None, category: str = "in_scope",
) -> GoldenQuery:
    return GoldenQuery(query=query, expected_articles=expected, source=source,
                        expected_sections=sections, category=category)


def test_aggregate_overall_means_and_per_source_breakdown():
    golden = [
        _golden("q1", ["a"], "claude-drafted"),
        _golden("q2", ["a"], "claude-drafted"),
        _golden("q3", ["a"], "human"),
    ]
    scores = [
        score_ranking("q1", "claude-drafted", [("a", None, 0.9)], ["a"], None),  # hit@1
        score_ranking("q2", "claude-drafted", [("x", None, 0.9), ("x", None, 0.8), ("x", None, 0.7)],
                       ["a"], None),  # miss
        score_ranking("q3", "human", [("x", None, 0.9), ("a", None, 0.8)], ["a"], None),  # hit@3, not @1
    ]

    report = aggregate(golden, scores)

    assert report.overall.num_queries == 3
    assert report.overall.hit_at_1 == 1 / 3
    assert report.overall.hit_at_3 == 2 / 3
    assert report.overall.mrr == (1.0 + 0.0 + 0.5) / 3

    assert report.by_source["claude-drafted"].num_queries == 2
    assert report.by_source["claude-drafted"].hit_at_1 == 0.5
    assert report.by_source["human"].num_queries == 1
    assert report.by_source["human"].hit_at_1 == 0.0
    assert report.by_source["human"].hit_at_3 == 1.0


def test_aggregate_groups_on_whatever_source_strings_appear_in_first_seen_order():
    """No source name is special: a third (or fourth, or made-up) label gets its own row."""
    sources = ["claude-drafted", "human-edited", "human", "human-edited", "some-new-label"]
    golden = [_golden(f"q{i}", ["a"], src) for i, src in enumerate(sources)]
    scores = [score_ranking(f"q{i}", src, [("a", None, 0.9)], ["a"], None) for i, src in enumerate(sources)]

    report = aggregate(golden, scores)

    assert list(report.by_source) == ["claude-drafted", "human-edited", "human", "some-new-label"]
    assert report.by_source["human-edited"].num_queries == 2
    assert report.by_source["some-new-label"].num_queries == 1
    assert sum(b.num_queries for b in report.by_source.values()) == report.overall.num_queries == 5


def test_aggregate_source_present_only_on_out_of_scope_queries_has_no_by_source_row():
    golden = [_golden("q1", ["a"], "claude-drafted"), _golden("q2", [], "human-edited", category="out_of_scope")]
    scores = [
        score_ranking("q1", "claude-drafted", [("a", None, 0.9)], ["a"], None),
        score_ranking("q2", "human-edited", [("x", None, 0.5)], [], None, category="out_of_scope"),
    ]

    report = aggregate(golden, scores)

    assert list(report.by_source) == ["claude-drafted"]
    assert report.out_of_scope[0]["source"] == "human-edited"


def test_load_golden_accepts_arbitrary_source_strings(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(
        json.dumps({"query": "a", "expected_articles": ["x"], "source": "human-edited"}) + "\n"
        + json.dumps({"query": "b", "expected_articles": ["x"], "source": "anything at all"}) + "\n",
        encoding="utf-8",
    )
    assert [g.source for g in load_golden(path)] == ["human-edited", "anything at all"]


def test_aggregate_misses_list_only_contains_non_hit_at_3_queries():
    golden = [_golden("hit", ["a"], "human"), _golden("miss", ["a"], "human")]
    scores = [
        score_ranking("hit", "human", [("a", None, 0.9)], ["a"], None),
        score_ranking("miss", "human", [("x", None, 0.9), ("y", None, 0.8), ("z", None, 0.7)], ["a"], None),
    ]

    report = aggregate(golden, scores)

    assert len(report.misses_top3) == 1
    assert report.misses_top3[0]["query"] == "miss"
    assert report.misses_top3[0]["expected_articles"] == ["a"]
    assert report.misses_top3[0]["top3_article_ids"] == ["x", "y", "z"]


def test_aggregate_section_hit_at_3_excludes_queries_without_expected_sections():
    golden = [_golden("with-section", ["a"], "human", ["s1"]), _golden("no-section", ["a"], "human")]
    scores = [
        score_ranking("with-section", "human", [("a", "s1", 0.9)], ["a"], ["s1"]),
        score_ranking("no-section", "human", [("a", None, 0.9)], ["a"], None),
    ]

    report = aggregate(golden, scores)

    assert report.overall.section_eval_count == 1
    assert report.overall.section_hit_at_3 == 1.0


# ------------------------------------------------------- aggregate: out_of_scope handling

def test_aggregate_excludes_out_of_scope_from_overall_and_by_source():
    golden = [
        _golden("in1", ["a"], "human"),
        _golden("in2", ["a"], "human"),
        _golden("oos", [], "human", category="out_of_scope"),
    ]
    scores = [
        score_ranking("in1", "human", [("a", None, 0.9)], ["a"], None),  # hit@1
        score_ranking("in2", "human", [("x", None, 0.9)], ["a"], None),  # miss
        score_ranking("oos", "human", [("x", None, 0.95)], [], None, category="out_of_scope"),
    ]

    report = aggregate(golden, scores)

    assert report.overall.num_queries == 2  # out_of_scope not counted
    assert report.overall.hit_at_1 == 0.5
    assert report.by_source["human"].num_queries == 2
    assert report.misses_top3 == [m for m in report.misses_top3 if m["query"] != "oos"]
    assert all(m["query"] != "oos" for m in report.misses_top3)


def test_aggregate_out_of_scope_report_has_query_source_and_top1_score():
    golden = [_golden("oos1", [], "human", category="out_of_scope"),
              _golden("oos2", [], "claude-drafted", category="out_of_scope")]
    scores = [
        score_ranking("oos1", "human", [("x", None, 0.95)], [], None, category="out_of_scope"),
        score_ranking("oos2", "claude-drafted", [], [], None, category="out_of_scope"),
    ]

    report = aggregate(golden, scores)

    assert report.out_of_scope == [
        {"query": "oos1", "source": "human", "top1_score": 0.95, "vector_top1_cosine": None},
        {"query": "oos2", "source": "claude-drafted", "top1_score": None, "vector_top1_cosine": None},
    ]


def test_out_of_scope_report_carries_vector_cosine_next_to_the_modes_own_score():
    golden = [_golden("oos", [], "human", category="out_of_scope")]
    scores = [score_ranking("oos", "human", [("x", None, 0.0328)], [], None, category="out_of_scope",
                            vector_top1_cosine=0.61)]
    report = aggregate(golden, scores)
    assert report.out_of_scope == [{"query": "oos", "source": "human", "top1_score": 0.0328, "vector_top1_cosine": 0.61}]


def test_vector_top1_stats_are_over_in_scope_queries_only():
    golden = [_golden("a", ["a"], "human"), _golden("b", ["a"], "human"), _golden("o", [], "human", category="out_of_scope")]
    scores = [
        score_ranking("a", "human", [("a", None, 0.03)], ["a"], None, vector_top1_cosine=0.8),
        score_ranking("b", "human", [("a", None, 0.02)], ["a"], None, vector_top1_cosine=0.6),
        score_ranking("o", "human", [("x", None, 0.01)], [], None, category="out_of_scope", vector_top1_cosine=0.1),
    ]
    stats = aggregate(golden, scores).vector_top1_score_stats
    assert (stats["min"], stats["max"]) == (0.6, 0.8)


def test_aggregate_top1_score_stats_computed_over_in_scope_only():
    golden = [
        _golden("in1", ["a"], "human"),
        _golden("in2", ["a"], "human"),
        _golden("in3", ["a"], "human"),
        _golden("oos", [], "human", category="out_of_scope"),
    ]
    scores = [
        score_ranking("in1", "human", [("a", None, 0.9)], ["a"], None),
        score_ranking("in2", "human", [("a", None, 0.5)], ["a"], None),
        score_ranking("in3", "human", [("a", None, 0.1)], ["a"], None),
        score_ranking("oos", "human", [("x", None, 0.99)], [], None, category="out_of_scope"),
    ]

    report = aggregate(golden, scores)

    assert report.top1_score_stats == {"min": 0.1, "median": 0.5, "max": 0.9}


def test_aggregate_top1_score_stats_all_none_when_no_results_ever():
    golden = [_golden("q", ["a"], "human")]
    scores = [score_ranking("q", "human", [], ["a"], None)]

    report = aggregate(golden, scores)

    assert report.top1_score_stats == {"min": None, "median": None, "max": None}


def test_aggregate_out_of_scope_empty_list_when_none_present():
    golden = [_golden("q", ["a"], "human")]
    scores = [score_ranking("q", "human", [("a", None, 0.9)], ["a"], None)]

    report = aggregate(golden, scores)

    assert report.out_of_scope == []


# -------------------------------------------------------------------- load_golden

def test_load_golden_parses_optional_sections_and_source(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"query": "q1", "expected_articles": ["a"], "source": "human"}\n'
        '{"query": "q2", "expected_articles": ["b"], "expected_sections": ["s1"], "source": "claude-drafted"}\n',
        encoding="utf-8",
    )

    queries = load_golden(path)

    assert len(queries) == 2
    assert queries[0].expected_sections is None
    assert queries[0].source == "human"
    assert queries[1].expected_sections == ["s1"]


def test_load_golden_category_defaults_and_parses(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"query": "q1", "expected_articles": ["a"], "source": "human"}\n'
        '{"query": "q2", "expected_articles": [], "source": "human", "category": "out_of_scope"}\n',
        encoding="utf-8",
    )

    queries = load_golden(path)

    assert queries[0].category == "in_scope"
    assert queries[1].category == "out_of_scope"
    assert queries[1].expected_articles == []


# ---------------------------------------------------------------- run_eval (integration)

def _manifest(tmp_path: Path, articles: list[dict]) -> Path:
    manifest = {
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "repos": [{"repo": REPO, "commit_sha": "0" * 40, "license": "CC-BY-4.0", "attribution": "test fixture"}],
        "known_issues": [],
        "articles": articles,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _article(id_: str, title: str, *, topic: str = "locks") -> dict:
    return {
        "id": id_, "title": title,
        "learn_url": f"https://learn.microsoft.com/azure/fixture/{id_}",
        "repo": REPO, "source_path": f"articles/fixture/{id_}.md", "topic": topic,
    }


def test_run_eval_wires_search_scoring_and_aggregation_end_to_end(tmp_path):
    manifest_path = _manifest(tmp_path, [
        _article("locks", "Lock your resources"),
        _article("other", "Something unrelated", topic="policy"),
    ])
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "locks.md").write_text(
        "---\ntitle: Lock your resources\n---\n"
        "## Overview\n\nwidget widget widget lock content here.\n",
        encoding="utf-8",
    )
    (raw_dir / "other.md").write_text(
        "---\ntitle: Something unrelated\n---\n## Overview\n\nnothing to do with widgets.\n",
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    embedder = FakeEmbedder()
    build_index(manifest_path, raw_dir, embedder, index_dir)

    golden = [GoldenQuery(query="widget", expected_articles=["locks"], source="claude-drafted")]
    report = run_eval(golden, embedder, manifest_path, index_dir, k=5)

    assert report.overall.num_queries == 1
    assert report.overall.hit_at_1 == 1.0
    assert report.by_source["claude-drafted"].hit_at_1 == 1.0
    assert report.misses_top3 == []


def test_run_eval_out_of_scope_query_excluded_from_metrics_but_scored(tmp_path):
    manifest_path = _manifest(tmp_path, [_article("locks", "Lock your resources")])
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "locks.md").write_text(
        "---\ntitle: Lock your resources\n---\n## Overview\n\nwidget widget widget lock content here.\n",
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    embedder = FakeEmbedder()
    build_index(manifest_path, raw_dir, embedder, index_dir)

    golden = [
        GoldenQuery(query="widget", expected_articles=["locks"], source="claude-drafted"),
        GoldenQuery(query="what's the weather today", expected_articles=[], source="human",
                    category="out_of_scope"),
    ]
    report = run_eval(golden, embedder, manifest_path, index_dir, k=5)

    assert report.overall.num_queries == 1  # only the in-scope query counted
    assert len(report.out_of_scope) == 1
    assert report.out_of_scope[0]["query"] == "what's the weather today"
    assert report.out_of_scope[0]["top1_score"] is not None
