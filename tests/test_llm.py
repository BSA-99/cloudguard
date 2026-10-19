"""Unit tests for FakeLLM and the TOOL_SPECS -> Gemini conversion only. GeminiClient itself
needs a real key and network, same convention as `pytest -m live` for the Azure backend."""

import pytest

from cloudguard.llm import FakeLLM, LLMResponse, ToolCall, Usage, tool_specs_to_gemini
from cloudguard.tools import TOOL_SPECS


def test_fake_llm_replays_script_in_order():
    r1 = LLMResponse(text="first")
    r2 = LLMResponse(text=None, tool_calls=[ToolCall(id="c1", name="get_resource", arguments={"resource_id": "x"})])
    llm = FakeLLM([r1, r2])

    assert llm.generate([{"role": "user", "content": "hi"}], []) is r1
    assert llm.generate([{"role": "user", "content": "again"}], TOOL_SPECS) is r2


def test_fake_llm_records_calls():
    llm = FakeLLM([LLMResponse(text="ok")])
    messages = [{"role": "user", "content": "hi"}]
    llm.generate(messages, TOOL_SPECS)

    assert llm.calls == [{"messages": messages, "tool_specs": TOOL_SPECS}]


def test_fake_llm_raises_when_script_is_exhausted():
    llm = FakeLLM([LLMResponse(text="only one")])
    llm.generate([], [])
    with pytest.raises(AssertionError, match="script exhausted"):
        llm.generate([], [])


def test_fake_llm_response_defaults():
    r = LLMResponse(text="hi")
    assert r.tool_calls == []
    assert r.usage == Usage(0, 0)


def test_tool_specs_to_gemini_converts_every_real_tool():
    declarations = tool_specs_to_gemini(TOOL_SPECS)
    assert [d["name"] for d in declarations] == [s["name"] for s in TOOL_SPECS]
    for d in declarations:
        assert set(d.keys()) == {"name", "description", "parameters"}


def test_tool_specs_to_gemini_uppercases_types_and_keeps_empty_object_schema():
    find_weak_tls = next(d for d in tool_specs_to_gemini(TOOL_SPECS) if d["name"] == "find_weak_tls")
    assert find_weak_tls["parameters"] == {"type": "OBJECT", "properties": {}, "required": []}


def test_tool_specs_to_gemini_recurses_into_properties_and_keeps_required():
    get_resource = next(d for d in tool_specs_to_gemini(TOOL_SPECS) if d["name"] == "get_resource")
    params = get_resource["parameters"]
    assert params["type"] == "OBJECT"
    assert params["required"] == ["resource_id"]
    assert params["properties"]["resource_id"]["type"] == "STRING"
    assert "description" in params["properties"]["resource_id"]


def test_tool_specs_to_gemini_keeps_numeric_constraints_and_defaults():
    search_docs = next(d for d in tool_specs_to_gemini(TOOL_SPECS) if d["name"] == "search_docs")
    k_schema = search_docs["parameters"]["properties"]["k"]
    assert k_schema == {"type": "INTEGER", "minimum": 1, "maximum": 10, "default": 3}
