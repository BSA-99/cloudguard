"""BM25, RRF fusion, and the retrieval `mode` flag. Hand-made corpora and FakeEmbedder, no real model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cloudguard import cli
from cloudguard.evals.retrieval import GoldenQuery, run_eval
from cloudguard.rag import index as rag_index
from cloudguard.rag.bm25 import B, K1, STOPWORDS, BM25Index, tokenize
from cloudguard.rag.embed import FakeEmbedder
from cloudguard.rag.fusion import RRF_K, fuse_rrf
from cloudguard.rag.index import MODES, StaleIndexError, build_index, search

REPO = "MicrosoftDocs/azure-docs"


# --- tokenizer ------------------------------------------------------------------------------

def test_tokenize_lowercases_splits_on_punctuation_and_drops_stopwords():
    assert tokenize("How do I lock-resources in the Azure CLI?") == ["lock", "resources", "azure", "cli"]


def test_tokenize_does_not_stem_or_split_camel_case():
    assert tokenize("CanNotDelete locks locking") == ["cannotdelete", "locks", "locking"]


def test_stopword_list_is_the_pre_registered_one():
    assert STOPWORDS == frozenset(
        "a an and are as at be by can do does for from how i in is it my of on or so that the this to was what when which with".split()
    )
    assert (K1, B) == (1.5, 0.75)


# --- BM25 -----------------------------------------------------------------------------------

def _idf(n_docs: int, n: int) -> float:
    return math.log(1 + (n_docs - n + 0.5) / (n + 0.5))


def _tf_part(tf: int, dl: int, avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    return tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


def test_bm25_scores_match_hand_computation():
    docs = [["lock", "lock", "group"], ["group", "delete"], ["unrelated", "words", "here", "now"]]
    index = BM25Index(["d1", "d2", "d3"], docs)
    avgdl = (3 + 2 + 4) / 3

    scores = index.scores("lock group")

    expected_d1 = _idf(3, 1) * _tf_part(2, 3, avgdl) + _idf(3, 2) * _tf_part(1, 3, avgdl)
    expected_d2 = _idf(3, 2) * _tf_part(1, 2, avgdl)
    assert scores[0] == pytest.approx(expected_d1)
    assert scores[1] == pytest.approx(expected_d2)
    assert scores[2] == 0.0


def test_bm25_rare_term_outweighs_common_term():
    docs = [["common", "rare"], ["common"], ["common"], ["common"]]
    index = BM25Index(["a", "b", "c", "d"], docs)
    assert index.scores("rare")[0] > index.scores("common")[0]


def test_bm25_longer_document_scores_lower_for_same_term_frequency():
    index = BM25Index(["short", "long"], [["term"], ["term"] + ["filler"] * 9])
    short, long_ = index.scores("term")
    assert short > long_


def test_bm25_term_frequency_saturates():
    index = BM25Index(["a", "b"], [["t"] * 2 + ["x"] * 8, ["t"] * 20 + ["x"] * 8])
    low, high = index.scores("t")
    assert low < high < low * 3  # 10x the tf is nowhere near 10x the score


def test_bm25_repeated_query_terms_count_once():
    index = BM25Index(["a", "b"], [["lock", "x"], ["y", "z"]])
    assert index.scores("lock lock lock") == index.scores("lock")


def test_bm25_unmatched_query_gives_empty_ranking():
    index = BM25Index(["a"], [["lock"]])
    assert index.rank("zebra") == []
    assert index.rank("the and of") == []  # all stopwords


def test_bm25_rank_excludes_zero_scores_and_breaks_ties_by_id():
    index = BM25Index(["b", "a", "c"], [["t"], ["t"], ["other"]])
    assert [d for d, _ in index.rank("t")] == ["a", "b"]  # identical scores -> id order; "c" unranked


def test_bm25_idf_is_never_negative_even_for_a_term_in_every_document():
    index = BM25Index(["a", "b"], [["t"], ["t"]])
    assert all(s > 0 for s in index.scores("t"))


def test_bm25_from_chunks_uses_title_heading_and_text():
    chunks = [
        {"chunk_id": "c1", "title": "Zebra guide", "heading_path": "Stripes", "text": "plain"},
        {"chunk_id": "c2", "title": "Other", "heading_path": "Else", "text": "plain"},
    ]
    index = BM25Index.from_chunks(chunks)
    assert [d for d, _ in index.rank("zebra")] == ["c1"]
    assert [d for d, _ in index.rank("stripes")] == ["c1"]
    assert {d for d, _ in index.rank("plain")} == {"c1", "c2"}


# --- RRF ------------------------------------------------------------------------------------

def test_rrf_scores_on_hand_made_rankings():
    fused = dict(fuse_rrf([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_default_k_is_60():
    assert RRF_K == 60
    assert dict(fuse_rrf([["a"]]))["a"] == pytest.approx(1 / 61)


def test_rrf_item_in_both_rankings_beats_item_in_one_at_the_same_rank():
    fused = fuse_rrf([["x", "shared"], ["shared", "y"]])
    assert fused[0][0] == "shared"


def test_rrf_ties_break_by_id_and_are_order_independent():
    r1, r2 = ["b", "a"], ["a", "b"]  # a and b tie exactly (ranks 2+1 and 1+2)
    forward = fuse_rrf([r1, r2])
    backward = fuse_rrf([r2, r1])
    assert [i for i, _ in forward] == ["a", "b"] == [i for i, _ in backward]
    assert forward[0][1] == forward[1][1]
    assert forward == backward


def test_rrf_tie_between_single_ranking_items_breaks_by_id():
    # "z" is rank 1 of the first ranking only, "a" rank 1 of the second only: same score
    assert [i for i, _ in fuse_rrf([["z"], ["a"]])] == ["a", "z"]


def test_rrf_k_changes_the_result():
    rankings = [["a", "b"], ["b", "c"], ["c", "a"], ["a"]]
    assert fuse_rrf(rankings, k=1) != fuse_rrf(rankings, k=60)
    assert dict(fuse_rrf([["a", "b"]], k=1))["b"] == pytest.approx(1 / 3)


def test_rrf_empty_and_duplicate_inputs():
    assert fuse_rrf([]) == [] and fuse_rrf([[], []]) == []
    assert dict(fuse_rrf([["a", "a"]]))["a"] == pytest.approx(1 / 61)  # a repeat doesn't count twice


# --- search(mode=...) -----------------------------------------------------------------------

def _article(id_: str, title: str, topic: str = "locks") -> dict:
    return {"id": id_, "title": title, "learn_url": f"https://learn.microsoft.com/azure/fixture/{id_}",
            "repo": REPO, "source_path": f"articles/fixture/{id_}.md", "topic": topic}


@pytest.fixture
def fixture_index(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "repos": [{"repo": REPO, "commit_sha": "0" * 40, "license": "CC-BY-4.0", "attribution": "fixture"}],
        "known_issues": [],
        "articles": [_article("locks", "Lock your resources"), _article("policy", "Azure Policy", "policy"),
                     _article("disks", "Managed disks", "disks")],
    }), encoding="utf-8")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "locks.md").write_text("---\ntitle: Lock\nms.date: 01/01/2026\n---\n## Overview\n\nA zebra lock prevents deletion of a resource group.\n", encoding="utf-8")
    (raw / "policy.md").write_text("---\ntitle: Policy\nms.date: 01/01/2026\n---\n## Overview\n\nPolicy definitions assign rules to subscriptions and groups.\n", encoding="utf-8")
    (raw / "disks.md").write_text("---\ntitle: Disks\nms.date: 01/01/2026\n---\n## Overview\n\nAn unattached managed disk still costs money.\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    embedder = FakeEmbedder()
    build_index(manifest, raw, embedder, index_dir)
    return manifest, index_dir, embedder


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_results_for_a_matching_query(fixture_index, mode):
    manifest, index_dir, embedder = fixture_index
    results = search("zebra lock", 3, embedder, manifest, index_dir, mode=mode)
    assert results and len(results) <= 3
    if mode != "vector":  # a FakeEmbedder cosine can legitimately be <= 0
        assert all(r.score > 0 for r in results)


def test_bm25_mode_ranks_the_exact_term_match_first_and_scores_are_bm25(fixture_index):
    manifest, index_dir, embedder = fixture_index
    results = search("zebra", 3, embedder, manifest, index_dir, mode="bm25")
    assert [r.article_id for r in results] == ["locks"]  # only matching chunks are ranked
    # N=3 chunks, "zebra" in exactly 1; a BM25 score, not a cosine or an RRF score
    chunk_len = len(tokenize("Lock\nOverview\nA zebra lock prevents deletion of a resource group."))
    other_lens = [len(tokenize(f"{t}\nOverview\n{x}")) for t, x in [
        ("Policy", "Policy definitions assign rules to subscriptions and groups."),
        ("Disks", "An unattached managed disk still costs money.")]]
    avgdl = (chunk_len + sum(other_lens)) / 3
    assert results[0].score == pytest.approx(_idf(3, 1) * _tf_part(1, chunk_len, avgdl))


def test_hybrid_scores_are_rrf_scores(fixture_index):
    manifest, index_dir, embedder = fixture_index
    results = search("zebra", 3, embedder, manifest, index_dir, mode="hybrid")
    assert all(0 < r.score <= 2 / 61 for r in results)
    assert results[0].article_id == "locks"  # in both rankings


def test_vector_mode_is_the_default_and_unchanged(fixture_index):
    manifest, index_dir, embedder = fixture_index
    default = search("zebra lock", 3, embedder, manifest, index_dir)
    explicit = search("zebra lock", 3, embedder, manifest, index_dir, mode="vector")
    assert default == explicit
    assert all(-1.0 <= r.score <= 1.0 for r in default)


def test_unknown_mode_is_rejected(fixture_index):
    manifest, index_dir, embedder = fixture_index
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        search("x", 3, embedder, manifest, index_dir, mode="semantic")


def test_search_default_uses_the_module_default_mode_which_is_vector():
    assert rag_index.DEFAULT_MODE == "vector"


def test_per_article_cap_applies_after_fusion(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "repos": [{"repo": REPO, "commit_sha": "0" * 40, "license": "CC-BY-4.0", "attribution": "f"}],
        "known_issues": [], "articles": [_article("big", "Big")],
    }), encoding="utf-8")
    raw = tmp_path / "raw"
    raw.mkdir()
    sections = "\n".join(f"## Part {i}\n\n" + ("zebra " * 200) for i in range(5))
    (raw / "big.md").write_text("---\ntitle: Big\nms.date: 01/01/2026\n---\n" + sections, encoding="utf-8")
    embedder = FakeEmbedder()
    build_index(manifest, raw, embedder, tmp_path / "idx")
    for mode in MODES:
        results = search("zebra", 5, embedder, manifest, tmp_path / "idx", mode=mode)
        assert len(results) <= 2, mode


# --- stale index fires in every mode --------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_stale_index_error_still_fires_when_the_manifest_changes(fixture_index, mode):
    manifest, index_dir, embedder = fixture_index
    manifest.write_text(manifest.read_text() + " ", encoding="utf-8")
    with pytest.raises(StaleIndexError):
        search("zebra", 3, embedder, manifest, index_dir, mode=mode)


@pytest.mark.parametrize("mode", MODES)
def test_stale_index_error_still_fires_when_the_embedding_model_changes(fixture_index, mode):
    manifest, index_dir, embedder = fixture_index

    class OtherEmbedder(FakeEmbedder):
        name = "some-other-model"

    with pytest.raises(StaleIndexError):
        search("zebra", 3, OtherEmbedder(), manifest, index_dir, mode=mode)


# --- eval + CLI mode flag -------------------------------------------------------------------

def test_run_eval_vector_mode_cosine_equals_top1_score_and_other_modes_report_both(fixture_index):
    manifest, index_dir, embedder = fixture_index
    golden = [
        GoldenQuery(query="zebra lock", expected_articles=["locks"], source="human"),
        GoldenQuery(query="what is the weather", expected_articles=[], source="human", category="out_of_scope"),
    ]
    vector = run_eval(golden, embedder, manifest, index_dir, k=5, mode="vector")
    assert all(q["vector_top1_cosine"] == q["top1_score"] for q in vector.per_query)

    for mode in ("bm25", "hybrid"):
        report = run_eval(golden, embedder, manifest, index_dir, k=5, mode=mode)
        by_query = {q["query"]: q for q in report.per_query}
        for q in vector.per_query:  # the comparable cosine is mode-independent
            assert by_query[q["query"]]["vector_top1_cosine"] == q["vector_top1_cosine"]
        oos = report.out_of_scope[0]
        assert oos["vector_top1_cosine"] == vector.out_of_scope[0]["top1_score"]
        assert set(oos) == {"query", "source", "top1_score", "vector_top1_cosine"}


def test_cli_mode_flag_is_parsed_and_recorded_in_the_results_json(tmp_path, fixture_index, monkeypatch, capsys):
    manifest, index_dir, _ = fixture_index
    monkeypatch.setattr(rag_index, "DEFAULT_MANIFEST_PATH", manifest)
    monkeypatch.setattr(rag_index, "DEFAULT_INDEX_DIR", index_dir)
    monkeypatch.setattr("cloudguard.rag.embed.FastEmbedder", FakeEmbedder)
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        json.dumps({"query": "zebra lock", "expected_articles": ["locks"], "source": "human"}) + "\n"
        + json.dumps({"query": "weather", "expected_articles": [], "source": "human", "category": "out_of_scope"}) + "\n",
        encoding="utf-8")

    for mode in MODES:
        results_dir = tmp_path / f"res_{mode}"
        assert cli.main(["eval-retrieval", "--mode", mode, "--golden", str(golden), "--results-dir", str(results_dir)]) == 0
        out = capsys.readouterr().out
        assert f"mode: {mode}" in out and "vector top1 cosine=" in out
        (path,) = Path(results_dir).glob("retrieval_*.json")
        data = json.loads(path.read_text())
        assert data["mode"] == mode
        assert data["retrieval_params"]["bm25_k1"] == 1.5 and data["retrieval_params"]["rrf_k"] == 60
        assert "vector_top1_cosine" in data["out_of_scope"][0]


def test_cli_defaults_to_vector_and_rejects_unknown_modes(tmp_path, fixture_index, monkeypatch, capsys):
    manifest, index_dir, _ = fixture_index
    monkeypatch.setattr(rag_index, "DEFAULT_MANIFEST_PATH", manifest)
    monkeypatch.setattr(rag_index, "DEFAULT_INDEX_DIR", index_dir)
    monkeypatch.setattr("cloudguard.rag.embed.FastEmbedder", FakeEmbedder)
    golden = tmp_path / "g.jsonl"
    golden.write_text(json.dumps({"query": "zebra", "expected_articles": ["locks"], "source": "human"}) + "\n", encoding="utf-8")

    assert cli.main(["eval-retrieval", "--golden", str(golden), "--results-dir", str(tmp_path / "r")]) == 0
    assert json.loads(next((tmp_path / "r").glob("*.json")).read_text())["mode"] == "vector"
    with pytest.raises(SystemExit):
        cli.main(["eval-retrieval", "--mode", "semantic"])
