"""Opt-in contract test against the real Gemini API.

FakeLLM-scripted tests can't catch a provider-shape mismatch (a role the real API rejects,
a field it requires) - only a real call can. This is the test that would have caught the
role="function" bug: it does one real 2-turn exchange - the model calls a tool, we send
back a tool result the way agent.py builds it, and the second call must not raise.

  CLOUDGUARD_LIVE_LLM=1 pytest -m live_llm -q

Needs `pip install -e ".[agent]"` and a real GEMINI_API_KEY (env var or .env). The autouse
no_live_llm_calls guard in conftest.py only steps aside for a test marked live_llm when
CLOUDGUARD_LIVE_LLM=1 is actually set - every other test stays blocked regardless.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(os.environ.get("CLOUDGUARD_LIVE_LLM") != "1", reason="live LLM tests disabled"),
]


def test_two_turn_tool_call_round_trip_against_the_real_api():
    from cloudguard.llm import GeminiClient
    from cloudguard.tools import TOOL_SPECS

    client = GeminiClient()
    tool_spec = next(s for s in TOOL_SPECS if s["name"] == "list_unattached_disks")

    messages = [{"role": "user", "content": "Check this subscription for unattached disks."}]
    first = client.generate(messages, [tool_spec])
    assert first.tool_calls, "model should have called list_unattached_disks"
    call = first.tool_calls[0]
    assert call.name == "list_unattached_disks"

    messages.append({
        "role": "assistant", "content": first.text,
        "tool_calls": [{"id": call.id, "name": call.name, "arguments": call.arguments}],
        "provider_turn": first.provider_turn,
    })
    messages.append({
        "role": "tool", "tool_call_id": call.id, "name": call.name,
        "content": '<tool_result tool="list_unattached_disks">'
                   '{"tool": "list_unattached_disks", "ok": true, "count": 0, "findings": []}'
                   '</tool_result>',
    })

    # The real assertion: this must not raise (it used to, with role="function").
    second = client.generate(messages, [tool_spec])
    assert second.text or second.tool_calls
