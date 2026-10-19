"""CLI tests for `cloudguard agent`. FakeLLM injected via cli.main(..., llm=...), no network."""

from __future__ import annotations

import pytest

from cloudguard import cli
from cloudguard.llm import FakeLLM, LLMError, LLMResponse, ToolCall


def _tool_call(name: str, arguments: dict | None = None, call_id: str = "c1") -> LLMResponse:
    return LLMResponse(text=None, tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments or {})])


class _RaisingLLM:
    """A minimal LLMClient test double that raises whatever it's given, instead of
    replaying a script. No google-genai needed - just exercises cmd_agent's except clause."""

    name = "fake"

    def __init__(self, exc: Exception):
        self._exc = exc

    def generate(self, messages, tool_specs):
        raise self._exc


def test_agent_command_reports_findings_answer_grounding_proposals_and_stats(capsys):
    llm = FakeLLM([
        _tool_call("list_unattached_disks"),
        LLMResponse(text="Found 2 unattached disks."),
    ])

    exit_code = cli.main(["--mock", "agent", "audit this subscription"], llm=llm)
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "== Findings ==" in out
    assert "unattached_disk" in out
    assert "cg-disk-orphan-01" in out and "cg-disk-orphan-02" in out
    assert "== Answer ==" in out
    assert "Found 2 unattached disks." in out
    assert "== Grounding ==" in out
    assert "ok" in out
    assert "== Proposals ==" in out
    assert "(none)" in out
    assert "== Run stats ==" in out
    assert "stopped_reason: final_answer" in out
    assert "model: fake" in out  # FakeLLM has no .model attribute; falls back to .name


def test_agent_command_reports_a_created_proposal(capsys):
    SUB = "00000000-0000-0000-0000-000000000000"
    rg_id = f"/subscriptions/{SUB}/resourceGroups/cg-dev-sandbox"
    llm = FakeLLM([
        _tool_call("list_unlocked_resource_groups", call_id="c1"),
        _tool_call("propose_lock", {"resource_group_id": rg_id, "reason": "no lock found"}, call_id="c2"),
        LLMResponse(text=f"Proposed a lock on {rg_id}."),
    ])

    exit_code = cli.main(["--mock", "agent", "audit"], llm=llm)
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "== Proposals ==" in out
    assert "[pending  ]" in out
    assert rg_id in out
    assert "az lock create" in out and "not executed" in out


def test_agent_command_exits_nonzero_when_grounding_fails(capsys):
    invented = "/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/x" \
               "/providers/Microsoft.Compute/disks/does-not-exist"
    llm = FakeLLM([
        _tool_call("list_unattached_disks"),
        LLMResponse(text=f"Found a problem disk at {invented}."),
    ])

    exit_code = cli.main(["--mock", "agent", "audit"], llm=llm)
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "FAILED" in out
    assert invented in out


def test_agent_command_exits_nonzero_on_max_steps(capsys):
    script = [_tool_call("list_unattached_disks") for _ in range(2)]
    llm = FakeLLM(script)

    exit_code = cli.main(["--mock", "agent", "audit", "--max-steps", "2"], llm=llm)
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "stopped_reason: max_steps" in out
    assert "(no final answer — stopped at max_steps)" in out


def test_agent_command_prints_one_clean_line_and_exits_2_on_llm_error(capsys):
    llm = _RaisingLLM(LLMError("INVALID_ARGUMENT", "Role 'function' is not supported", "gemini-3.5-flash-lite"))

    exit_code = cli.main(["--mock", "agent", "audit"], llm=llm)
    out = capsys.readouterr().out

    assert exit_code == 2
    assert out.strip() == "LLM call failed: INVALID_ARGUMENT Role 'function' is not supported (model: gemini-3.5-flash-lite)"


def test_agent_command_lets_non_llm_errors_propagate_as_a_real_traceback():
    """Only LLMError gets the clean one-line treatment; an actual CloudGuard bug must still
    surface as a normal exception, not get silently swallowed or mislabeled."""
    llm = _RaisingLLM(ValueError("some unrelated bug"))

    with pytest.raises(ValueError, match="some unrelated bug"):
        cli.main(["--mock", "agent", "audit"], llm=llm)
