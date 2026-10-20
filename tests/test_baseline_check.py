"""The vector-baseline gate (evals/RESULTS.md). The comparison logic is tested on hand-edited copies
of the real baseline file; running the real embedder is a manual step (`python -m
cloudguard.evals.baseline_check`), because tests never load the real model.
"""

from __future__ import annotations

import copy
import json

import pytest

from cloudguard.evals.baseline_check import BASELINE_PATH, compare_results, main, max_score_diff


@pytest.fixture
def baseline():
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def test_baseline_file_exists_and_has_the_compared_fields(baseline):
    assert BASELINE_PATH.name == "retrieval_20261020T011040Z.json"
    # per_query lists every golden query; overall counts only the in-scope ones.
    assert len(baseline["per_query"]) == baseline["overall"]["num_queries"] + len(baseline["out_of_scope"]) == 28


def test_identical_results_reproduce(baseline):
    assert compare_results(baseline, copy.deepcopy(baseline)) == []


def test_provenance_fields_and_extra_keys_are_ignored(baseline):
    cand = copy.deepcopy(baseline)
    cand["timestamp"] = "2099-01-01T00:00:00+00:00"
    cand["git_commit"] = "deadbeef"
    cand["index_meta"]["built_at"] = "2099"
    cand["mode"] = "vector"
    cand["per_query"][0]["vector_top1_cosine"] = 0.5
    assert compare_results(baseline, cand) == []


@pytest.mark.parametrize("path, value", [
    (("overall", "hit@1"), 0.0),
    (("overall", "hit@3"), 0.5),
    (("overall", "hit@5"), 0.5),
    (("overall", "mrr"), 0.1),
    (("overall", "section_hit@3"), 0.0),
])
def test_each_overall_metric_change_is_reported(baseline, path, value):
    cand = copy.deepcopy(baseline)
    cand[path[0]][path[1]] = value
    diffs = compare_results(baseline, cand)
    assert len(diffs) == 1 and path[1] in diffs[0]


def test_a_tiny_top1_score_change_fails_by_default_and_passes_only_with_an_explicit_tolerance(baseline):
    cand = copy.deepcopy(baseline)
    cand["per_query"][0]["top1_score"] += 1e-9
    assert compare_results(baseline, cand)
    assert compare_results(baseline, cand, score_tol=1e-6) == []
    assert 0 < max_score_diff(baseline, cand) < 1e-8


def test_a_changed_ranking_or_missing_query_is_reported(baseline):
    cand = copy.deepcopy(baseline)
    cand["per_query"][0]["top3_article_ids"] = list(reversed(cand["per_query"][0]["top3_article_ids"]))
    assert any("top3_article_ids" in d for d in compare_results(baseline, cand))

    cand = copy.deepcopy(baseline)
    cand["per_query"].pop()
    assert any("per_query count" in d for d in compare_results(baseline, cand))


def test_out_of_scope_score_change_is_reported(baseline):
    cand = copy.deepcopy(baseline)
    cand["out_of_scope"][0]["top1_score"] += 0.01
    assert any("out_of_scope" in d for d in compare_results(baseline, cand))


def test_stale_index_or_different_model_is_reported(baseline):
    cand = copy.deepcopy(baseline)
    cand["index_meta"]["manifest_hash"] = "other"
    cand["model_name"] = "other-model"
    diffs = compare_results(baseline, cand)
    assert any("manifest_hash" in d for d in diffs) and any("model_name" in d for d in diffs)


def test_a_non_vector_candidate_is_rejected(baseline):
    cand = copy.deepcopy(baseline)
    cand["mode"] = "hybrid"
    assert any("mode" in d for d in compare_results(baseline, cand))


def test_main_exit_codes_and_message(tmp_path, baseline, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(baseline))
    assert main(["--candidate", str(good)]) == 0
    assert "reproduces the baseline exactly" in capsys.readouterr().out

    baseline["overall"]["mrr"] = 0.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(baseline))
    assert main(["--candidate", str(bad)]) == 1
    assert "Stop and investigate before running hybrid" in capsys.readouterr().out
