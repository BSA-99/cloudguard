"""Proves the live_llm relaxation in conftest.py is scoped correctly: both the `live_llm`
marker AND CLOUDGUARD_LIVE_LLM=1 are required, not either alone. Needs neither google-genai
nor python-dotenv, so it always runs (no importorskip), unlike the other Gemini-shape tests.
"""

from __future__ import annotations

import os

from tests.conftest import _apply_live_llm_guard


class _UnmarkedNode:
    def get_closest_marker(self, name: str):
        return None  # simulates a test with no `live_llm` marker


class _FakeRequest:
    node = _UnmarkedNode()


def test_unmarked_test_is_still_blocked_even_with_the_env_var_set(monkeypatch):
    monkeypatch.setenv("CLOUDGUARD_LIVE_LLM", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "should-be-stripped-regardless")

    _apply_live_llm_guard(_FakeRequest(), monkeypatch)

    assert os.environ.get("GEMINI_API_KEY") is None


class _MarkedNode:
    def get_closest_marker(self, name: str):
        return object() if name == "live_llm" else None


class _MarkedRequest:
    node = _MarkedNode()


def test_marked_test_is_relaxed_only_when_the_env_var_is_also_set(monkeypatch):
    monkeypatch.delenv("CLOUDGUARD_LIVE_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "should-still-be-stripped-without-the-env-var")

    _apply_live_llm_guard(_MarkedRequest(), monkeypatch)

    assert os.environ.get("GEMINI_API_KEY") is None


def test_marked_test_is_relaxed_when_both_conditions_hold(monkeypatch):
    monkeypatch.setenv("CLOUDGUARD_LIVE_LLM", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "a-real-looking-key")

    _apply_live_llm_guard(_MarkedRequest(), monkeypatch)

    assert os.environ.get("GEMINI_API_KEY") == "a-real-looking-key"
