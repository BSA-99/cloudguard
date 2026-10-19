"""Provider-neutral LLM client for the Day-2 agent loop.

Messages are a plain list of dicts so no provider's SDK types leak into agent.py:

    {"role": "system", "content": "..."}   # at most one, first in the list by convention
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [{"id": ..., "name": ..., "arguments": {...}}]}
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}   # JSON-stringified result

Each LLMClient translates this shape to and from its provider's native format.
GeminiClient is the only live implementation today; an AnthropicClient would implement the
same LLMClient protocol and the same message shape, translating to Claude's content blocks
instead of Gemini's Content/Part objects.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(0, 0))
    attempts: int = 1  # how many calls generate_content() took, including retries
    raw: Any = None  # the provider's native response, for logging/debugging only
    # The provider's own representation of this turn (for Gemini: the candidate's Content,
    # parts/thought_signature/etc. intact), so it can be echoed back on the next call
    # unchanged instead of rebuilt (lossily) from our neutral ToolCall list. None for
    # FakeLLM and any message history not produced by this same client.
    provider_turn: Any = None


class LLMClient(Protocol):
    name: str


class LLMError(Exception):
    """A clean, provider-neutral LLM failure: status + message + which model, so a caller
    like cmd_agent can print one line instead of a traceback. Anything that is NOT an
    LLMError (a real CloudGuard bug) is left to propagate normally."""

    def __init__(self, status: str | None, message: str | None, model: str):
        self.status = status
        self.message = message
        self.model = model
        super().__init__(f"{status}: {message} (model: {model})")

    def generate(self, messages: list[dict[str, Any]], tool_specs: list[dict[str, Any]]) -> LLMResponse: ...


# Our TOOL_SPECS (tools.py) use the Anthropic-style shape: name/description/input_schema,
# with lowercase JSON-Schema types. Gemini's FunctionDeclaration wants name/description/
# parameters, with uppercase Schema types. This mapping (and the recursion below) is the
# part worth unit testing without needing the google-genai SDK installed.
_JSON_SCHEMA_TYPE_TO_GEMINI = {
    "object": "OBJECT",
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
}

# Keys that mean the same thing in both schema dialects and can be copied verbatim.
_SCHEMA_PASSTHROUGH_KEYS = ("description", "enum", "required", "minimum", "maximum", "default")


def _json_schema_to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "type" in schema:
        out["type"] = _JSON_SCHEMA_TYPE_TO_GEMINI[schema["type"]]
    if "properties" in schema:
        out["properties"] = {k: _json_schema_to_gemini_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema:
        out["items"] = _json_schema_to_gemini_schema(schema["items"])
    for key in _SCHEMA_PASSTHROUGH_KEYS:
        if key in schema:
            out[key] = schema[key]
    return out


def tool_specs_to_gemini(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our TOOL_SPECS into Gemini functionDeclaration dicts. Pure function, no
    google-genai import, so it's unit-testable without the SDK installed."""
    return [
        {
            "name": spec["name"],
            "description": spec["description"],
            "parameters": _json_schema_to_gemini_schema(spec["input_schema"]),
        }
        for spec in tool_specs
    ]


# Transient errors worth retrying: 429 (rate limited) and 503 (model temporarily overloaded).
# Anything else (bad request, auth failure, ...) is never retried.
_RETRYABLE_CODES = (429, 503)


class GeminiClient:
    """Live mode: calls Gemini through the google-genai SDK.

    GEMINI_API_KEY and CLOUDGUARD_MODEL are read from the environment (optionally loaded
    from a local .env via python-dotenv). Both google-genai and python-dotenv are imported
    lazily, inside __init__, so mock-mode tests and CI never need the `agent` extra.

    Retries 429 and 503 with exponential backoff, up to max_retries attempts, then raises a
    RuntimeError naming the attempt count and the last error instead of retrying forever.
    """

    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None, max_retries: int = 3):
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()  # no-op if there's no .env; never overrides an already-set env var
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (env var or .env file)")

        self._genai = genai
        self.model = model or os.environ.get("CLOUDGUARD_MODEL", "gemini-3.5-flash-lite")
        self.max_retries = max_retries
        self._client = genai.Client(api_key=key)

    def generate(self, messages: list[dict[str, Any]], tool_specs: list[dict[str, Any]]) -> LLMResponse:
        types = self._genai.types
        # Gemini has no "system" turn in contents; a system prompt is a separate config
        # field instead, so system-role messages are pulled out rather than converted.
        system_texts = [m["content"] for m in messages if m["role"] == "system"]
        contents = [_to_gemini_content(types, m) for m in messages if m["role"] != "system"]

        config_kwargs: dict[str, Any] = {}
        if tool_specs:
            declarations = tool_specs_to_gemini(tool_specs)
            config_kwargs["tools"] = [types.Tool(function_declarations=declarations)]
        if system_texts:
            config_kwargs["system_instruction"] = "\n\n".join(system_texts)
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response, attempts = self._call_with_retry(contents, config)
        return _from_gemini_response(response, attempts)

    def _call_with_retry(self, contents: list[Any], config: Any) -> tuple[Any, int]:
        from google.genai import errors

        delay = 1.0
        last_error: Any = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(model=self.model, contents=contents, config=config)
                return response, attempt + 1
            except errors.APIError as e:
                if e.code not in _RETRYABLE_CODES:
                    raise LLMError(e.status, e.message, self.model) from e
                last_error = e
                if attempt == self.max_retries - 1:
                    break
                time.sleep(delay)
                delay *= 2
        raise LLMError(last_error.status, last_error.message, self.model) from last_error


def _to_gemini_content(types: Any, message: dict[str, Any]) -> Any:
    role = message["role"]
    if role == "user":
        return types.Content(role="user", parts=[types.Part(text=message.get("content") or "")])
    if role == "assistant":
        # Prefer echoing back exactly what Gemini sent us (parts, thought_signature, any
        # field we don't model) rather than rebuilding a lossy proxy from our own ToolCall
        # list. Only a conversation this same GeminiClient produced has this; FakeLLM-driven
        # runs and hand-built message lists fall back to rebuilding from id/name/arguments.
        provider_turn = message.get("provider_turn")
        if provider_turn is not None:
            return provider_turn
        parts = []
        if message.get("content"):
            parts.append(types.Part(text=message["content"]))
        for call in message.get("tool_calls") or []:
            parts.append(types.Part(function_call=types.FunctionCall(
                id=call["id"], name=call["name"], args=call["arguments"],
            )))
        return types.Content(role="model", parts=parts)
    if role == "tool":
        # role="user", not "function": Gemini's generateContent API rejects role="function"
        # outright (400 INVALID_ARGUMENT). The function_response Part is what identifies
        # this as a tool result, not the Content-level role.
        return types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            id=message.get("tool_call_id"), name=message["name"], response={"output": message["content"]},
        ))])
    raise ValueError(f"unknown message role: {role!r}")


def _from_gemini_response(response: Any, attempts: int) -> LLMResponse:
    # Read candidates[0].content.parts directly rather than response.text /
    # response.function_calls: both of those log a warning whenever a function_call part is
    # present (which is every tool-calling turn), even though that's expected, not an error.
    provider_turn = response.candidates[0].content if response.candidates else None
    parts = provider_turn.parts or [] if provider_turn else []

    text_chunks = []
    tool_calls = []
    for i, part in enumerate(parts):
        if part.function_call is not None:
            fc = part.function_call
            tool_calls.append(ToolCall(id=fc.id or f"call_{i}", name=fc.name, arguments=fc.args or {}))
        elif part.text is not None:
            text_chunks.append(part.text)

    usage = response.usage_metadata
    return LLMResponse(
        text="".join(text_chunks) or None,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        ),
        attempts=attempts,
        raw=response,
        provider_turn=provider_turn,
    )


class FakeLLM:
    """Replays scripted LLMResponses in order. No network, no key — what tests and CI use."""

    name = "fake"

    def __init__(self, script: list[LLMResponse]):
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages: list[dict[str, Any]], tool_specs: list[dict[str, Any]]) -> LLMResponse:
        self.calls.append({"messages": messages, "tool_specs": tool_specs})
        if not self._script:
            raise AssertionError("FakeLLM script exhausted: the agent loop called generate() more times than scripted")
        return self._script.pop(0)
