"""Tests that exercise the real google-genai/python-dotenv types (error classes, response
shape) rather than pure Python. Skipped, not failed, when the `agent` extra isn't installed,
so `pip install -e ".[dev]"` + `pytest -q` stays green without it.

The autouse `no_live_llm_calls` fixture in conftest.py is what actually makes these safe to
run: it strips GEMINI_API_KEY, neutralizes load_dotenv(), and blocks the real network call
class-wide. These tests prove that guard actually holds.
"""

import pytest

pytest.importorskip("google.genai")
pytest.importorskip("dotenv")

from google.genai import errors, types  # noqa: E402

from cloudguard.llm import GeminiClient, LLMError  # noqa: E402


def test_real_env_file_key_is_never_loaded_during_tests(tmp_path, monkeypatch):
    """Proves a test run can't reach Gemini even when a real-looking .env exists on disk."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=leaked-should-never-load\n")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        GeminiClient()


def test_network_guard_blocks_a_real_call_even_with_an_explicit_key():
    client = GeminiClient(api_key="definitely-not-a-real-key")
    with pytest.raises(RuntimeError, match="must not call the real Gemini API"):
        client.generate([{"role": "user", "content": "hi"}], [])


def _fake_success() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text="ok")]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=1, candidates_token_count=1),
    )


@pytest.mark.parametrize("code,status", [(503, "UNAVAILABLE"), (429, "RESOURCE_EXHAUSTED")])
def test_retries_transient_errors_then_succeeds(monkeypatch, code, status):
    client = GeminiClient(api_key="fake", max_retries=3)
    monkeypatch.setattr("cloudguard.llm.time.sleep", lambda s: None)

    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise errors.APIError(code, {"error": {"message": "transient", "status": status}})
        return _fake_success()

    # Instance-level override takes precedence over the class-level block from conftest.
    monkeypatch.setattr(client._client.models, "generate_content", flaky)

    response = client.generate([{"role": "user", "content": "hi"}], [])

    assert response.text == "ok"
    assert calls["n"] == 3
    assert response.attempts == 3


def test_raises_clear_error_after_exhausting_retries(monkeypatch):
    client = GeminiClient(api_key="fake", max_retries=3)
    monkeypatch.setattr("cloudguard.llm.time.sleep", lambda s: None)

    def always_503(**kwargs):
        raise errors.APIError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}})

    monkeypatch.setattr(client._client.models, "generate_content", always_503)

    with pytest.raises(LLMError) as exc_info:
        client.generate([{"role": "user", "content": "hi"}], [])
    assert exc_info.value.status == "UNAVAILABLE"
    assert exc_info.value.message == "overloaded"
    assert exc_info.value.model == client.model


def test_mixed_text_and_function_call_parts_extracted_without_sdk_warning(monkeypatch, caplog):
    """response.text / response.function_calls each log a warning whenever a function_call
    part is present. generate() must read candidates[0].content.parts directly instead, so
    tool-calling turns (which always mix a function_call part in) never trigger that log."""
    client = GeminiClient(api_key="fake", max_retries=3)

    mixed = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[
            types.Part(text="looking into it"),
            types.Part(function_call=types.FunctionCall(id="c1", name="list_unattached_disks", args={})),
        ]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=1, candidates_token_count=1),
    )
    monkeypatch.setattr(client._client.models, "generate_content", lambda **kwargs: mixed)

    with caplog.at_level("WARNING"):
        response = client.generate([{"role": "user", "content": "hi"}], [])

    assert response.text == "looking into it"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "list_unattached_disks"
    assert caplog.records == []


def test_system_role_message_becomes_system_instruction_not_a_content_turn(monkeypatch):
    """Gemini has no "system" Content role; agent.py's SYSTEM_PROMPT must end up as
    config.system_instruction, and must not appear in `contents` at all."""
    client = GeminiClient(api_key="fake")
    captured = {}

    def fake_generate_content(**kwargs):
        captured.update(kwargs)
        return _fake_success()

    monkeypatch.setattr(client._client.models, "generate_content", fake_generate_content)

    client.generate(
        [{"role": "system", "content": "be a good auditor"}, {"role": "user", "content": "hi"}], [],
    )

    assert captured["config"].system_instruction == "be a good auditor"
    assert len(captured["contents"]) == 1
    assert captured["contents"][0].role == "user"


def test_tool_result_becomes_a_function_response_part_matched_by_id(monkeypatch):
    """Tool results must reach Gemini as a function_response Part correlated to the
    function_call that produced it via a shared id, never as plain user-role text - that's
    what lets Gemini (and the model) tell "data returned by a tool I called" apart from
    "a new instruction from the user", independent of what the data block's text says."""
    client = GeminiClient(api_key="fake")
    captured = {}
    monkeypatch.setattr(client._client.models, "generate_content",
                         lambda **kwargs: captured.update(kwargs) or _fake_success())

    messages = [
        {"role": "user", "content": "check disks"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "name": "list_unattached_disks", "arguments": {}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "list_unattached_disks",
         "content": "<tool_result tool=\"list_unattached_disks\">...</tool_result>"},
    ]
    client.generate(messages, [])

    contents = captured["contents"]
    assistant_content, tool_content = contents[1], contents[2]

    call_part = assistant_content.parts[0].function_call
    assert assistant_content.role == "model"
    assert call_part.id == "call_1"

    # role="user", not "function": Gemini's generateContent rejects role="function" outright
    # (400 INVALID_ARGUMENT - this is the exact bug a live contract test caught).
    assert tool_content.role == "user"
    response_part = tool_content.parts[0].function_response
    assert response_part is not None
    assert tool_content.parts[0].text is None  # not plain text
    assert response_part.id == call_part.id == "call_1"  # matched to the call that produced it
    assert response_part.name == "list_unattached_disks"


def test_assistant_turn_echoed_back_preserves_the_original_content_including_thought_signature(monkeypatch):
    """agent.py rebuilds the assistant's tool-call turn from our neutral ToolCall list only
    as a fallback. When the conversation was produced by this same client, the exact
    Content Gemini returned - thought_signature included - must be echoed back unchanged,
    not reconstructed."""
    client = GeminiClient(api_key="fake")
    captured_calls = []
    monkeypatch.setattr(client._client.models, "generate_content",
                         lambda **kwargs: captured_calls.append(kwargs) or _fake_success())

    original_call_part = types.Part(
        function_call=types.FunctionCall(id="call_1", name="list_unattached_disks", args={}),
        thought_signature=b"opaque-signature-bytes",
    )
    original_content = types.Content(role="model", parts=[original_call_part])

    # First turn: the model's own response (as agent.py would have received it).
    first_response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=original_content)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=1, candidates_token_count=1),
    )
    monkeypatch.setattr(client._client.models, "generate_content", lambda **kwargs: first_response)
    from cloudguard.llm import _from_gemini_response
    response = _from_gemini_response(first_response, attempts=1)
    assert response.provider_turn is original_content

    # Second turn: agent.py echoes it back via the generic message dict.
    monkeypatch.setattr(client._client.models, "generate_content",
                         lambda **kwargs: captured_calls.append(kwargs) or _fake_success())
    messages = [
        {"role": "user", "content": "check disks"},
        {"role": "assistant", "content": response.text, "tool_calls": [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls
        ], "provider_turn": response.provider_turn},
        {"role": "tool", "tool_call_id": "call_1", "name": "list_unattached_disks", "content": "{}"},
    ]
    client.generate(messages, [])

    echoed_content = captured_calls[0]["contents"][1]
    assert echoed_content is original_content  # the exact object, not a rebuilt proxy
    assert echoed_content.parts[0].thought_signature == b"opaque-signature-bytes"


def test_non_retryable_error_is_not_retried(monkeypatch):
    client = GeminiClient(api_key="fake", max_retries=3)
    monkeypatch.setattr("cloudguard.llm.time.sleep", lambda s: None)

    calls = {"n": 0}

    def bad_request(**kwargs):
        calls["n"] += 1
        raise errors.APIError(400, {"error": {"message": "bad request", "status": "INVALID_ARGUMENT"}})

    monkeypatch.setattr(client._client.models, "generate_content", bad_request)

    with pytest.raises(LLMError) as exc_info:
        client.generate([{"role": "user", "content": "hi"}], [])
    assert calls["n"] == 1
    assert exc_info.value.status == "INVALID_ARGUMENT"
    assert exc_info.value.message == "bad request"
