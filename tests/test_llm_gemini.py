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

from cloudguard.llm import GeminiClient  # noqa: E402


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


def test_raises_clear_error_after_exhausting_retries(monkeypatch):
    client = GeminiClient(api_key="fake", max_retries=3)
    monkeypatch.setattr("cloudguard.llm.time.sleep", lambda s: None)

    def always_503(**kwargs):
        raise errors.APIError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}})

    monkeypatch.setattr(client._client.models, "generate_content", always_503)

    with pytest.raises(RuntimeError, match=r"failed after 3 attempts.*503 UNAVAILABLE"):
        client.generate([{"role": "user", "content": "hi"}], [])


def test_non_retryable_error_is_not_retried(monkeypatch):
    client = GeminiClient(api_key="fake", max_retries=3)
    monkeypatch.setattr("cloudguard.llm.time.sleep", lambda s: None)

    calls = {"n": 0}

    def bad_request(**kwargs):
        calls["n"] += 1
        raise errors.APIError(400, {"error": {"message": "bad request", "status": "INVALID_ARGUMENT"}})

    monkeypatch.setattr(client._client.models, "generate_content", bad_request)

    with pytest.raises(errors.APIError):
        client.generate([{"role": "user", "content": "hi"}], [])
    assert calls["n"] == 1
