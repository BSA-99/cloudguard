"""Citation grounding: learn.microsoft.com URLs in an answer must have come from a
search_docs result in the same run. Pure-function tests plus run_agent tests driven by
FakeLLM with search_docs's retrieval stubbed (no network, no embedding model).
"""

from __future__ import annotations

import json

import pytest

from cloudguard import cli, tools
from cloudguard.agent import run_agent
from cloudguard.grounding import check_citations, check_grounding, extract_urls, normalize_doc_url
from cloudguard.llm import FakeLLM, LLMResponse, ToolCall
from cloudguard.models import ToolResult
from cloudguard.rag.index import SearchResult
from tests.conftest import read_log

LOCKS = "https://learn.microsoft.com/azure/azure-resource-manager/management/lock-resources"
LOCKS_ANCHOR = LOCKS + "#lock-inheritance"
POLICY = "https://learn.microsoft.com/azure/governance/policy/overview"


def _search_result(*urls: str, ok: bool = True) -> ToolResult:
    return ToolResult(
        tool="search_docs", ok=ok, backend="mock", count=len(urls),
        data={"query": "q", "results": [{"url": u, "title": "t", "heading": "h"} for u in urls]} if ok else None,
        error=None if ok else "boom",
    )


# --- normalize_doc_url ---------------------------------------------------------------

def test_normalize_lowercases_scheme_and_host_and_path():
    n = normalize_doc_url("HTTPS://Learn.Microsoft.com/Azure/Foo")
    assert n.host == "learn.microsoft.com"
    assert n.article == "learn.microsoft.com/azure/foo"


def test_normalize_strips_query_trailing_slash_and_splits_anchor():
    n = normalize_doc_url("https://learn.microsoft.com/azure/foo/?view=azure-cli-latest#Some-Anchor")
    assert n.article == "learn.microsoft.com/azure/foo"
    assert n.anchor == "some-anchor"


@pytest.mark.parametrize("locale", ["en-us", "EN-US", "pt-br", "zh-hans-cn", "en"])
def test_normalize_strips_locale_segment(locale):
    assert normalize_doc_url(f"https://learn.microsoft.com/{locale}/azure/foo").article == "learn.microsoft.com/azure/foo"


def test_normalize_does_not_strip_non_locale_first_segment():
    assert normalize_doc_url("https://learn.microsoft.com/entra/identity").article == "learn.microsoft.com/entra/identity"
    assert normalize_doc_url("https://learn.microsoft.com/azure/foo").article == "learn.microsoft.com/azure/foo"


def test_normalize_no_or_empty_anchor_is_none():
    assert normalize_doc_url(LOCKS).anchor is None
    assert normalize_doc_url(LOCKS + "#").anchor is None


def test_normalize_percent_decodes_anchor():
    assert normalize_doc_url(LOCKS + "#a%2Db").anchor == "a-b"


# --- extract_urls --------------------------------------------------------------------

def test_extract_bare_url_strips_trailing_prose_punctuation():
    assert extract_urls(f"See {LOCKS}.") == [LOCKS]
    assert extract_urls(f"(see {LOCKS}), and more") == [LOCKS]


def test_extract_markdown_link_does_not_leak_closing_paren():
    assert extract_urls(f"Read [the docs]({LOCKS_ANCHOR}) now") == [LOCKS_ANCHOR]


def test_extract_markdown_link_with_title_and_dedup():
    text = f'[a]({LOCKS} "Lock docs") and again {LOCKS} and <{POLICY}>'
    assert extract_urls(text) == [LOCKS, POLICY]


def test_extract_no_urls():
    assert extract_urls("no links here") == []


# --- check_citations -----------------------------------------------------------------

def test_retrieved_and_cited_url_passes():
    r = check_citations(f"Use locks: {LOCKS}", [_search_result(LOCKS)])
    assert r.citations_ok is True
    assert r.ungrounded_urls == []
    assert r.cited_urls[0]["grounded"] is True


def test_invented_learn_url_fails_and_is_listed():
    invented = "https://learn.microsoft.com/azure/made-up/article"
    r = check_citations(f"{LOCKS} and {invented}", [_search_result(LOCKS)])
    assert r.citations_ok is False
    assert r.ungrounded_urls == [invented]


def test_unretrieved_anchor_on_retrieved_article_passes_with_anchor_matched_false():
    r = check_citations(f"See {LOCKS}#never-retrieved", [_search_result(LOCKS_ANCHOR)])
    assert r.citations_ok is True
    assert r.cited_urls[0]["anchor_matched"] is False


def test_retrieved_anchor_gives_anchor_matched_true_and_no_anchor_gives_none():
    r = check_citations(f"{LOCKS_ANCHOR.upper().replace('HTTPS', 'https')} {POLICY}", [_search_result(LOCKS_ANCHOR, POLICY)])
    by_article = {c["article"]: c for c in r.cited_urls}
    assert by_article["learn.microsoft.com/azure/azure-resource-manager/management/lock-resources"]["anchor_matched"] is True
    assert by_article["learn.microsoft.com/azure/governance/policy/overview"]["anchor_matched"] is None


@pytest.mark.parametrize("variant", [
    LOCKS.replace("learn.microsoft.com/", "learn.microsoft.com/en-us/"),
    LOCKS.replace("https://learn.microsoft.com", "HTTPS://LEARN.MICROSOFT.COM"),
    LOCKS.upper().replace("HTTPS://LEARN.MICROSOFT.COM", "https://learn.microsoft.com"),
    LOCKS + "/",
    LOCKS + "?view=azure-cli-latest",
    LOCKS.replace("learn.microsoft.com/", "learn.microsoft.com/de-de/") + "/?x=1#lock-inheritance",
])
def test_locale_case_slash_and_query_variants_of_a_retrieved_url_pass(variant):
    r = check_citations(f"See {variant}", [_search_result(LOCKS)])
    assert r.citations_ok is True, variant


def test_answer_without_citations_is_ok_with_empty_lists():
    r = check_citations("No links.", [_search_result(LOCKS)])
    assert (r.citations_ok, r.cited_urls, r.ungrounded_urls, r.external_urls) == (True, [], [], [])


def test_non_microsoft_url_goes_to_external_not_ungrounded():
    r = check_citations("See https://example.com/x and https://learn.microsoft.com.evil.io/azure/a", [])
    assert r.citations_ok is True
    assert r.ungrounded_urls == []
    assert r.external_urls == ["https://example.com/x", "https://learn.microsoft.com.evil.io/azure/a"]


def test_failed_search_docs_result_grounds_nothing():
    r = check_citations(f"See {LOCKS}", [_search_result(LOCKS, ok=False)])
    assert r.ungrounded_urls == [LOCKS]


def test_url_with_no_search_at_all_fails():
    assert check_citations(f"See {LOCKS}", []).citations_ok is False


def test_check_grounding_carries_citation_fields_and_keeps_grounding_ok_id_only():
    g = check_grounding(f"See {POLICY}", [_search_result(LOCKS)])
    assert g.grounding_ok is True
    assert g.citations_ok is False
    assert g.ungrounded_urls == [POLICY]


# --- run_agent: per-run scope, audit log, CLI warning --------------------------------

def _stub_search(monkeypatch, urls):
    def fake(query, k=3):
        return [SearchResult(chunk_id=f"c{i}", article_id=f"a{i}", title="t", heading="h", topic="locks",
                             url=u, text="text", score=0.9) for i, u in enumerate(urls)]
    monkeypatch.setattr(tools.rag_index, "search_default", fake)


def _search_then_answer(answer: str) -> FakeLLM:
    return FakeLLM([
        LLMResponse(text=None, tool_calls=[ToolCall(id="c1", name="search_docs", arguments={"query": "locks"})]),
        LLMResponse(text=answer),
    ])


def test_run_agent_cited_retrieved_url_passes_and_audit_entry_has_citation_fields(backend, monkeypatch, isolated_audit_log):
    _stub_search(monkeypatch, [LOCKS_ANCHOR])
    run = run_agent(_search_then_answer(f"Use a lock ({LOCKS}) - see https://example.com"), backend, "q")
    assert run.citations_ok is True
    assert run.external_urls == ["https://example.com"]
    entry = [e for e in read_log(isolated_audit_log) if e["tool"] == "grounding_check"][0]
    assert entry["citations_ok"] is True
    assert entry["cited_urls"][0]["anchor_matched"] is None
    assert entry["ungrounded_urls"] == []
    assert entry["external_urls"] == ["https://example.com"]


def test_url_retrieved_in_a_different_run_fails(backend, monkeypatch):
    _stub_search(monkeypatch, [LOCKS])
    first = run_agent(_search_then_answer(f"See {LOCKS}"), backend, "q")
    assert first.citations_ok is True

    # Second run never calls search_docs, so run 1's retrieval must not count.
    second = run_agent(FakeLLM([LLMResponse(text=f"See {LOCKS}")]), backend, "q")
    assert second.citations_ok is False
    assert second.ungrounded_urls == [LOCKS]


def test_cli_prints_visible_warning_and_leaves_answer_unchanged(backend, monkeypatch, capsys):
    _stub_search(monkeypatch, [LOCKS])
    invented = "https://learn.microsoft.com/azure/invented/page"
    answer = f"Per {invented}, do it."
    run = run_agent(_search_then_answer(answer), backend, "q")

    cli._print_agent_run(run, FakeLLM([]))
    out = capsys.readouterr().out

    assert run.final_answer == answer
    assert answer in out
    assert "WARNING" in out and f"!!   {invented}" in out
    assert "== Citations ==" in out and "FAILED" in out


def test_cli_no_warning_when_citations_ok(backend, monkeypatch, capsys):
    _stub_search(monkeypatch, [LOCKS])
    run = run_agent(_search_then_answer(f"See {LOCKS}"), backend, "q")
    cli._print_agent_run(run, FakeLLM([]))
    assert "WARNING" not in capsys.readouterr().out


def test_never_checked_run_is_not_reported_as_passing_by_cli(backend, capsys):
    script = [LLMResponse(text=None, tool_calls=[ToolCall(id="c1", name="list_unattached_disks", arguments={})])
              for _ in range(2)]
    run = run_agent(FakeLLM(script), backend, "q", max_steps=2)
    assert run.citations_ok is None

    cli._print_agent_run(run, FakeLLM([]))
    out = capsys.readouterr().out

    citations_section = out.split("== Citations ==")[1].split("==")[0]
    assert "not checked" in citations_section
    assert "ok" not in citations_section
    assert "WARNING" not in out
