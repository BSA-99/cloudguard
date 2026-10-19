import json
from pathlib import Path

import pytest

from cloudguard.backends import MockBackend
from cloudguard.models import GroundTruth

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_audit_log(tmp_path, monkeypatch):
    """Every test writes to its own audit log, never the repo's."""
    path = tmp_path / "audit_log.jsonl"
    monkeypatch.setenv("CLOUDGUARD_AUDIT_LOG", str(path))
    return path


@pytest.fixture(autouse=True)
def isolated_proposals(tmp_path, monkeypatch):
    """Every test writes to its own proposals store, never the repo's."""
    path = tmp_path / "proposals.jsonl"
    monkeypatch.setenv("CLOUDGUARD_PROPOSALS", str(path))
    return path


def _block_real_gemini_calls(*args, **kwargs):
    raise RuntimeError("Tests must not call the real Gemini API; use FakeLLM.")


@pytest.fixture(autouse=True)
def no_live_llm_calls(monkeypatch):
    """Tests must never reach a real LLM: strip GEMINI_API_KEY, neutralize load_dotenv() so
    a real .env on disk can't repopulate it, and make the one real network call Gemini-side
    fail loudly if anything still reaches it. Each guard is a no-op if the `agent` extra
    (google-genai / python-dotenv) isn't installed, so CI without it stays green."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    try:
        import dotenv
    except ImportError:
        pass
    else:
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    try:
        from google.genai.models import Models
    except ImportError:
        pass
    else:
        monkeypatch.setattr(Models, "generate_content", _block_real_gemini_calls)


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend(ROOT / "mock" / "subscription_snapshot.json")


@pytest.fixture
def truth() -> GroundTruth:
    return GroundTruth.model_validate_json((ROOT / "ground_truth.json").read_text())


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
