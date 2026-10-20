"""LLM factories for the agent-eval runner tests. Imported by path ("tests.agent_fakes:name") inside a
spawned child process, so they must be plain module-level functions."""

from __future__ import annotations

import os
import time

from cloudguard.llm import FakeLLM, LLMError, LLMResponse, ToolCall, Usage


def _call(name: str, args: dict | None = None, cid: str = "c1") -> LLMResponse:
    return LLMResponse(text=None, tool_calls=[ToolCall(id=cid, name=name, arguments=args or {})],
                       usage=Usage(10, 2), attempts=1)


def scripted(case_id: str, run_index: int):
    """sd-disks-1 -> lists disks and answers; pl-all -> proposes a lock; others -> a one-line answer."""
    if case_id == "pl-named":
        rg = f"/subscriptions/{os.environ.get('SUB', '00000000-0000-0000-0000-000000000000')}/resourceGroups/cg-dev-sandbox"
        return FakeLLM([_call("list_unlocked_resource_groups"), _call("propose_lock", {"resource_group_id": rg, "reason": "no lock"}, "c2"),
                        LLMResponse(text=f"Proposed a lock for {rg}.", usage=Usage(10, 5), attempts=1)])
    if case_id == "sd-disks-1":
        return FakeLLM([_call("list_unattached_disks"), LLMResponse(text="Two disks found.", usage=Usage(10, 5, thought_tokens=40), attempts=2)])
    return FakeLLM([LLMResponse(text="Nothing to report.", usage=Usage(5, 3), attempts=1)])


def hangs(case_id: str, run_index: int):
    time.sleep(60)


def crashes(case_id: str, run_index: int):
    os._exit(3)


def raises_unavailable(case_id: str, run_index: int):
    raise LLMError("UNAVAILABLE", "model overloaded", "fake-model")


def raises_bad_request(case_id: str, run_index: int):
    raise LLMError("INVALID_ARGUMENT", "bad request", "fake-model")


def leaks_key(case_id: str, run_index: int):
    raise RuntimeError(f"auth failed for key {os.environ.get('GEMINI_API_KEY')}")
