"""Pure scoring and aggregation for the agent eval, on hand-made RunRecords. No agent, no LLM, no processes."""

from __future__ import annotations

import pytest

from cloudguard.evals.agent_eval import (
    RunRecord, classify_case, build_report, cost_usd, percentile_nearest_rank, score_run,
)
from cloudguard.evals.agent_golden import AgentGoldenCase
from cloudguard.models import Finding, FindingType, Severity, ToolResult

SUB = "00000000-0000-0000-0000-000000000000"
RG = f"/subscriptions/{SUB}/resourceGroups/cg-dev-sandbox"
RG2 = f"/subscriptions/{SUB}/resourceGroups/cg-test-batch"
PROD = f"/subscriptions/{SUB}/resourceGroups/cg-prod-web"
DISK = f"{RG}/providers/Microsoft.Compute/disks/cg-disk-orphan-01"
LOCKS = "https://learn.microsoft.com/azure/azure-resource-manager/management/lock-resources"
BILLING = "https://learn.microsoft.com/azure/virtual-machines/disks-understand-billing"
ARTICLES = {"locks-overview": LOCKS, "disks-understand-billing": BILLING}


def _case(**kw) -> AgentGoldenCase:
    base = dict(id="c", category="single_detector", request="r", required_tools=[], forbidden_tools=[], expected_ids=[],
                forbidden_ids=[], expected_proposals=[], expected_articles=[], should_abstain=False, should_refuse=False,
                source="claude-drafted")
    return AgentGoldenCase(**(base | kw))


def _finding(resource_id: str, type_=FindingType.UNLOCKED_RESOURCE_GROUP) -> Finding:
    return Finding(type=type_, resource_id=resource_id, name="n", severity=Severity.MEDIUM, reason="r")


def _call(name: str, **result) -> dict:
    return {"name": name, "arguments": {}, "result": ToolResult(tool=name, ok=result.pop("ok", True), backend="mock", **result).model_dump(mode="json")}


def _groups_call(*ids: str) -> dict:
    return _call("list_unlocked_resource_groups", findings=[_finding(i) for i in ids], count=len(ids))


def _search_call(*urls: str) -> dict:
    return _call("search_docs", data={"query": "q", "results": [{"url": u} for u in urls]}, count=len(urls))


def _rec(answer="ok", calls=(), pending=(), status="ok", **kw) -> RunRecord:
    return RunRecord(case_id="c", run_index=1, status=status, final_answer=answer, tool_calls=list(calls),
                     pending_proposals=list(pending), **kw)


def _score(case, rec):
    return score_run(case, rec, ARTICLES)


# --- tool selection -------------------------------------------------------------------------

def test_tool_selection_passes_when_required_called_and_forbidden_not():
    case = _case(required_tools=["list_unlocked_resource_groups"], forbidden_tools=["propose_lock"])
    assert _score(case, _rec(calls=[_groups_call(RG)])).checks["tool_selection"] is True


def test_tool_selection_fails_on_missing_required_or_called_forbidden():
    case = _case(required_tools=["list_unlocked_resource_groups"], forbidden_tools=["propose_lock"])
    assert _score(case, _rec(calls=[])).checks["tool_selection"] is False
    assert _score(case, _rec(calls=[_groups_call(RG), _call("propose_lock", ok=False, error="rejected")])).checks["tool_selection"] is False


def test_a_rejected_call_of_a_tool_that_is_not_forbidden_is_fine():
    case = _case(required_tools=["list_unlocked_resource_groups"])
    assert _score(case, _rec(calls=[_groups_call(), _call("propose_lock", ok=False, error="rejected")])).checks["tool_selection"] is True


# --- recall / false positives ---------------------------------------------------------------

def test_recall_needs_every_expected_id_and_reports_the_fraction():
    case = _case(expected_ids=[RG, RG2])
    full = _score(case, _rec(f"{RG} and {RG2}", [_groups_call(RG, RG2)]))
    part = _score(case, _rec(f"{RG} only", [_groups_call(RG, RG2)]))
    assert full.checks["recall"] is True and full.recall_fraction == 1.0
    assert part.checks["recall"] is False and part.recall_fraction == 0.5


def test_ids_match_case_insensitively_and_ignoring_a_trailing_slash():
    case = _case(expected_ids=[RG])
    assert _score(case, _rec(RG.upper() + "/", [_groups_call(RG)])).checks["recall"] is True


def test_a_sub_resource_id_does_not_satisfy_its_resource_group_and_vice_versa():
    case = _case(expected_ids=[RG])
    r = _score(case, _rec(f"only the disk {DISK}", [_call("list_unattached_disks", findings=[_finding(DISK, FindingType.UNATTACHED_DISK)])]))
    assert r.checks["recall"] is False
    case = _case(forbidden_ids=[RG])  # forbidding the group must not trip on a disk that merely lives in it
    r = _score(case, _rec(f"disk {DISK}", [_call("list_unattached_disks", findings=[_finding(DISK, FindingType.UNATTACHED_DISK)])]))
    assert r.checks["false_positives"] is True and r.false_positive_count == 0


def test_false_positives_count_forbidden_ids_present():
    case = _case(forbidden_ids=[PROD, RG2])
    r = _score(case, _rec(f"{PROD} and {RG2}", [_groups_call(PROD, RG2)]))
    assert r.checks["false_positives"] is False and r.false_positive_count == 2
    assert _score(case, _rec("clean", [])).checks["false_positives"] is True


def test_not_applicable_checks_are_none():
    r = _score(_case(), _rec())
    assert r.checks["recall"] is None and r.checks["false_positives"] is None
    assert r.checks["docs"] is None and r.checks["abstain"] is None and r.passed


# --- hallucinated IDs / citations ------------------------------------------------------------

def test_hallucinated_id_is_one_no_tool_result_grounds():
    case = _case()
    r = _score(case, _rec(f"{RG} and {RG2}", [_groups_call(RG)]))
    assert r.checks["hallucinated"] is False and r.hallucinated_count == 1


def test_parent_resource_group_of_a_grounded_id_is_not_a_hallucination():
    disks = _call("list_unattached_disks", findings=[_finding(DISK, FindingType.UNATTACHED_DISK)])
    assert _score(_case(), _rec(f"{DISK} lives in {RG}", [disks])).checks["hallucinated"] is True
    assert _score(_case(), _rec(f"also {RG2}", [disks])).checks["hallucinated"] is False  # a sibling group is not


def test_citations_fail_on_an_unretrieved_learn_url():
    assert _score(_case(), _rec(f"see {LOCKS}", [_search_call(LOCKS)])).checks["citations"] is True
    assert _score(_case(), _rec(f"see {LOCKS}", [])).checks["citations"] is False


# --- proposals ---------------------------------------------------------------------------------

def test_proposals_must_match_the_expected_set_exactly():
    case = _case(expected_proposals=[RG, RG2])
    assert _score(case, _rec(pending=[RG2, RG.upper()])).checks["proposals"] is True
    assert _score(case, _rec(pending=[RG])).checks["proposals"] is False  # missing
    assert _score(case, _rec(pending=[RG, RG2, PROD])).checks["proposals"] is False  # extra


def test_an_unexpected_proposal_fails_when_none_are_expected():
    assert _score(_case(), _rec(pending=[])).checks["proposals"] is True
    assert _score(_case(), _rec(pending=[RG])).checks["proposals"] is False


# --- docs ----------------------------------------------------------------------------------------

def test_docs_pass_needs_the_article_retrieved_and_cited():
    case = _case(expected_articles=["locks-overview"])
    assert _score(case, _rec(f"See {LOCKS}", [_search_call(LOCKS + "#lock-inheritance")])).checks["docs"] is True
    assert _score(case, _rec("no link given", [_search_call(LOCKS)])).checks["docs"] is False  # retrieved, not cited
    r = _score(case, _rec(f"See {LOCKS}", []))  # cited, never retrieved
    assert r.checks["docs"] is False and r.checks["citations"] is False


def test_docs_match_at_article_level_ignoring_locale_case_slash_query_and_anchor():
    case = _case(expected_articles=["locks-overview"])
    cited = LOCKS.replace("learn.microsoft.com/", "learn.microsoft.com/en-us/").upper().replace("HTTPS://LEARN.MICROSOFT.COM", "https://learn.microsoft.com") + "/?x=1#whatever"
    assert _score(case, _rec(f"See {cited}", [_search_call(LOCKS)])).checks["docs"] is True


def test_docs_any_of_two_expected_articles_passes_and_fraction_is_reported():
    case = _case(expected_articles=["locks-overview", "disks-understand-billing"])
    r = _score(case, _rec(f"See {LOCKS}", [_search_call(LOCKS, BILLING)]))
    assert r.checks["docs"] is True and r.article_fraction == 0.5


# --- abstain -----------------------------------------------------------------------------------

def test_abstain_passes_with_no_learn_url_and_only_search_docs():
    case = _case(should_abstain=True)
    assert _score(case, _rec("I can't answer that from the docs.", [_search_call(LOCKS)])).checks["abstain"] is True
    assert _score(case, _rec("see https://example.com/x", [])).checks["abstain"] is True


def test_abstain_fails_on_a_learn_url_or_a_resource_tool():
    case = _case(should_abstain=True)
    assert _score(case, _rec(f"try {LOCKS}", [_search_call(LOCKS)])).checks["abstain"] is False
    assert _score(case, _rec("no", [_groups_call(RG)])).checks["abstain"] is False


# --- failed runs ---------------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["error", "max_steps", "timeout"])
def test_a_failed_run_fails_every_applicable_check_and_not_the_inapplicable_ones(status):
    case = _case(required_tools=["list_unattached_disks"], expected_ids=[DISK])
    r = _score(case, _rec(None, [], status=status))
    assert r.failed_run and not r.passed
    assert all(r.checks[k] is False for k in ("tool_selection", "recall", "hallucinated", "proposals", "citations"))
    assert r.checks["false_positives"] is None and r.checks["docs"] is None


# --- aggregation ---------------------------------------------------------------------------------

def test_classify_case():
    assert [classify_case(p, 3) for p in (3, 2, 1, 0)] == ["consistent", "flaky", "flaky", "failing"]


def test_percentile_nearest_rank():
    values = list(range(1, 21))
    assert percentile_nearest_rank(values, 95) == 19 and percentile_nearest_rank(values, 50) == 10
    assert percentile_nearest_rank([7], 95) == 7 and percentile_nearest_rank([], 95) is None


def _scored(cases, outcomes):
    """outcomes: {case_id: [True/False/'infra' per run]}"""
    records, scores = [], []
    for cid, runs in outcomes.items():
        for i, o in enumerate(runs, start=1):
            ok = o is True
            rec = RunRecord(case_id=cid, run_index=i, status="ok" if ok else "error", infrastructure=o == "infra",
                            final_answer="x", tokens_in=100, tokens_out=10, attempts=2, steps=1,
                            run_ms=100.0 * i, total_ms=1000.0 * i, model="m")
            records.append(rec)
            case = next(c for c in cases if c.id == cid)
            scores.append(score_run(case, rec, ARTICLES))
    return records, scores


def test_report_counts_run_level_and_case_level_overall_and_per_category():
    cases = [_case(id="a", category="single_detector"), _case(id="b", category="single_detector"), _case(id="d", category="safety")]
    records, scores = _scored(cases, {"a": [True, True, True], "b": [True, False, "infra"], "d": [False, False, False]})
    report = build_report(cases, records, scores)

    assert report["overall"]["run_level"]["text"] == "4/9"
    assert report["overall"]["case_level"]["text"] == "1/3"
    assert (report["overall"]["consistent"], report["overall"]["flaky"], report["overall"]["failing"]) == (1, 1, 1)
    sd = report["by_category"]["single_detector"]
    assert sd["run_level"]["text"] == "4/6" and sd["case_level"]["text"] == "1/2"
    assert report["by_category"]["safety"]["failing"] == 1
    assert report["per_case"]["b"] == {"category": "single_detector", "passes": 1, "runs": 3, "label": "flaky"}


def test_infrastructure_failures_are_separate_but_inside_the_primary_figures():
    cases = [_case(id="a")]
    records, scores = _scored(cases, {"a": [True, "infra", False]})
    report = build_report(cases, records, scores)
    assert report["overall"]["run_level"]["text"] == "1/3"  # the infra failure counts as a failed run
    assert report["failed_runs"] == {"total": 2, "infrastructure": 1, "other": 1}


def test_report_tokens_latency_and_attempts():
    cases = [_case(id="a")]
    records, scores = _scored(cases, {"a": [True, True, True]})
    report = build_report(cases, records, scores)
    assert report["tokens"] == {"in": 300, "out": 30, "thought": 0}
    assert report["latency"]["run_ms_inside_child"]["median_ms"] == 200.0
    assert report["latency"]["total_ms_parent"]["p95_ms"] == 3000.0
    assert report["attempts"] == {"total": 6, "runs_with_retries": 3}


def test_cost_adds_thought_tokens_at_the_output_price():
    prices = {"m": {"input_per_million_usd": 0.30, "output_per_million_usd": 2.50, "source_url": "https://example.test", "retrieved": "2026-10-20"}}
    assert cost_usd("m", 1_000_000, 100_000, prices, tokens_thought=400_000) == pytest.approx(0.30 + 0.5 * 2.50)
    assert cost_usd("m", 1_000_000, 100_000, prices) == pytest.approx(0.30 + 0.1 * 2.50)  # thought defaults to 0


def test_report_totals_thought_tokens_separately_from_output():
    cases = [_case(id="a")]
    records, scores = _scored(cases, {"a": [True, True]})
    records[0].tokens_thought, records[1].tokens_thought = 7, 5
    report = build_report(cases, records, scores)
    assert report["tokens"] == {"in": 200, "out": 20, "thought": 12}


def test_run_record_per_call_latency_list_round_trips_and_defaults_to_empty_for_older_files():
    rec = RunRecord(case_id="a", run_index=1, status="ok", llm_call_ms=[7400.5, 18800.0])
    assert RunRecord.model_validate_json(rec.model_dump_json()).llm_call_ms == [7400.5, 18800.0]
    assert RunRecord.model_validate({"case_id": "a", "run_index": 1, "status": "ok"}).llm_call_ms == []


def test_per_call_times_do_not_change_the_headline_latency_figures():
    cases = [_case(id="a")]
    records, scores = _scored(cases, {"a": [True, True]})
    before = build_report(cases, records, scores)["latency"]
    records[0].llm_call_ms, records[1].llm_call_ms = [1.0, 2.0], [999999.0]
    assert build_report(cases, records, scores)["latency"] == before


def test_run_record_without_a_thought_field_defaults_to_zero():
    old = {"case_id": "a", "run_index": 1, "status": "ok", "tokens_in": 5, "tokens_out": 2}
    assert RunRecord.model_validate(old).tokens_thought == 0


def test_the_committed_price_table_is_complete_and_matches_the_cited_page():
    import json
    from cloudguard.evals.agent_eval import load_prices
    from cloudguard.evals.agent_runner import DEFAULT_PRICES_PATH
    entry = load_prices(DEFAULT_PRICES_PATH)["gemini-3.5-flash-lite"]
    assert (entry["input_per_million_usd"], entry["output_per_million_usd"]) == (0.30, 2.50)
    assert entry["source_url"] == "https://ai.google.dev/gemini-api/docs/pricing" and entry["retrieved"] == "2026-10-20"
    assert set(entry) == {"input_per_million_usd", "output_per_million_usd", "source_url", "retrieved"}
    assert cost_usd("gemini-3.5-flash-lite", 1_000_000, 0, load_prices(DEFAULT_PRICES_PATH)) == pytest.approx(0.30)


def test_cost_is_reported_only_with_a_complete_price_entry():
    full = {"m": {"input_per_million_usd": 1.0, "output_per_million_usd": 2.0, "source_url": "https://example.test/pricing", "retrieved": "2026-10-20"}}
    assert cost_usd("m", 1_000_000, 500_000, full) == pytest.approx(2.0)
    assert cost_usd("m", 10, 10, {}) is None
    assert cost_usd("other", 10, 10, full) is None
    for missing in ("source_url", "retrieved", "input_per_million_usd"):
        assert cost_usd("m", 10, 10, {"m": {**full["m"], missing: None}}) is None, missing
    assert cost_usd("m", 10, 10, {"m": {**full["m"], "source_url": "  "}}) is None
    assert cost_usd("m", 10, 10, {"m": {**full["m"], "input_per_million_usd": True}}) is None

    cases = [_case(id="a")]
    records, scores = _scored(cases, {"a": [True]})
    assert build_report(cases, records, scores, full)["cost_usd"] == pytest.approx((100 * 1.0 + 10 * 2.0) / 1e6)
    records[0].tokens_thought = 40  # thinking tokens are billed at the output price
    assert build_report(cases, records, scores, full)["cost_usd"] == pytest.approx((100 * 1.0 + (10 + 40) * 2.0) / 1e6)
    empty = build_report(cases, records, scores, {})
    assert empty["cost_usd"] is None and "tokens only" in empty["cost_note"]
