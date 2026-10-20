"""app/view.py: pure data-shaping tests, no Streamlit import. Uses FakeLLM-produced
AgentRuns where that's natural, and directly-constructed Finding/proposal dicts where
precision matters more (grouping, the raw-string test, the pending/decided split).
"""

from __future__ import annotations

from app.view import citations_banner, group_findings, grounding_banner, run_stats, split_proposals
from cloudguard.agent import run_agent
from cloudguard.llm import FakeLLM, LLMResponse, ToolCall, Usage
from cloudguard.models import Finding, FindingType, Severity

SUB = "00000000-0000-0000-0000-000000000000"


def _finding(type_: FindingType, resource_id: str, name: str = "x", flags: list[str] | None = None,
             tags: dict | None = None) -> Finding:
    return Finding(
        type=type_, resource_id=resource_id, name=name, severity=Severity.MEDIUM, reason="test",
        evidence={"tags": tags or {}}, untrusted_text_flags=flags or [],
    )


def test_group_findings_groups_by_type_in_first_seen_order():
    disk = _finding(FindingType.UNATTACHED_DISK, f"/subscriptions/{SUB}/resourceGroups/a/disks/d1")
    rg1 = _finding(FindingType.UNLOCKED_RESOURCE_GROUP, f"/subscriptions/{SUB}/resourceGroups/a")
    rg2 = _finding(FindingType.UNLOCKED_RESOURCE_GROUP, f"/subscriptions/{SUB}/resourceGroups/b")

    grouped = group_findings([disk, rg1, rg2])

    assert list(grouped.keys()) == ["unattached_disk", "unlocked_resource_group"]
    assert grouped["unattached_disk"] == [disk]
    assert grouped["unlocked_resource_group"] == [rg1, rg2]


def test_group_findings_with_no_findings_is_empty():
    assert group_findings([]) == {}


def test_group_findings_preserves_untrusted_text_flags_exactly():
    flagged = _finding(FindingType.UNLOCKED_RESOURCE_GROUP, f"/subscriptions/{SUB}/resourceGroups/a",
                        flags=["override_instructions", "tool_directive"])
    clean = _finding(FindingType.UNLOCKED_RESOURCE_GROUP, f"/subscriptions/{SUB}/resourceGroups/b")

    grouped = group_findings([flagged, clean])

    assert grouped["unlocked_resource_group"][0].untrusted_text_flags == ["override_instructions", "tool_directive"]
    assert grouped["unlocked_resource_group"][1].untrusted_text_flags == []


def test_group_findings_passes_through_a_malicious_tag_value_unchanged():
    """The exact scenario the view layer must never touch: markdown/HTML-looking text in a
    tag must come out byte-for-byte identical, so streamlit_app.py can safely st.text() it
    without stripping or escaping doing something unexpected either way."""
    malicious = "[click me](https://evil.example) and **bold**"
    finding = _finding(FindingType.UNLOCKED_RESOURCE_GROUP, f"/subscriptions/{SUB}/resourceGroups/a",
                        tags={"note": malicious})

    grouped = group_findings([finding])

    assert grouped["unlocked_resource_group"][0].evidence["tags"]["note"] == malicious


def test_grounding_banner_ok(backend):
    llm = FakeLLM([LLMResponse(text="all clear")])
    run = run_agent(llm, backend, "audit")

    banner = grounding_banner(run)
    assert banner.status == "ok"
    assert banner.ungrounded_ids == []


def test_grounding_banner_failed(backend):
    invented = f"/subscriptions/{SUB}/resourceGroups/x/providers/Microsoft.Compute/disks/does-not-exist"
    llm = FakeLLM([
        LLMResponse(text=None, tool_calls=[ToolCall(id="c1", name="list_unattached_disks", arguments={})]),
        LLMResponse(text=f"Found a problem disk at {invented}."),
    ])
    run = run_agent(llm, backend, "audit")

    banner = grounding_banner(run)
    assert banner.status == "failed"
    assert banner.ungrounded_ids == [invented]


def test_grounding_banner_not_checked_when_run_hit_max_steps(backend):
    script = [
        LLMResponse(text=None, tool_calls=[ToolCall(id="c1", name="list_unattached_disks", arguments={})])
        for _ in range(8)
    ]
    run = run_agent(FakeLLM(script), backend, "audit", max_steps=8)

    banner = grounding_banner(run)
    assert banner.status == "not_checked"
    assert banner.ungrounded_ids == []


def test_split_proposals_separates_pending_from_decided_preserving_order():
    proposals = [
        {"proposal_id": "p1", "status": "pending"},
        {"proposal_id": "p2", "status": "approved"},
        {"proposal_id": "p3", "status": "pending"},
        {"proposal_id": "p4", "status": "rejected"},
    ]

    pending, decided = split_proposals(proposals)

    assert [p["proposal_id"] for p in pending] == ["p1", "p3"]
    assert [p["proposal_id"] for p in decided] == ["p2", "p4"]


def test_split_proposals_with_no_proposals():
    assert split_proposals([]) == ([], [])


def test_run_stats_reports_every_field(backend):
    llm = FakeLLM([
        LLMResponse(text=None, tool_calls=[ToolCall(id="c1", name="list_unattached_disks", arguments={})],
                    usage=Usage(10, 5), attempts=2),
        LLMResponse(text="done", usage=Usage(20, 8), attempts=1),
    ])
    run = run_agent(llm, backend, "audit")

    stats = run_stats(run, model="gemini-3.5-flash-lite")

    assert stats == {
        "run_id": run.run_id,
        "model": "gemini-3.5-flash-lite",
        "steps": 2,
        "stopped_reason": "final_answer",
        "tokens": "30 in / 13 out",
        "attempts": 3,
        "latency_ms": round(run.total_latency_ms),
    }


LEARN = "https://learn.microsoft.com/azure/azure-resource-manager/management/lock-resources"


def test_citations_banner_ok_and_external_listed(backend):
    run = run_agent(FakeLLM([LLMResponse(text="no links, see https://example.com")]), backend, "audit")
    banner = citations_banner(run)
    assert banner.status == "ok"
    assert banner.external_urls == ["https://example.com"]
    assert banner.ungrounded_urls == []


def test_citations_banner_failed_lists_ungrounded_urls(backend):
    run = run_agent(FakeLLM([LLMResponse(text=f"See {LEARN}")]), backend, "audit")
    banner = citations_banner(run)
    assert banner.status == "failed"
    assert banner.ungrounded_urls == [LEARN]
    assert run.final_answer == f"See {LEARN}"


def test_citations_banner_none_is_not_checked_never_ok(backend):
    script = [
        LLMResponse(text=None, tool_calls=[ToolCall(id="c1", name="list_unattached_disks", arguments={})])
        for _ in range(3)
    ]
    run = run_agent(FakeLLM(script), backend, "audit", max_steps=3)
    assert run.citations_ok is None
    assert citations_banner(run).status == "not_checked"
