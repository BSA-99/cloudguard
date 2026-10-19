import json
import os
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


def _apply_live_llm_guard(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard logic, factored out of the fixture below so it can be called directly from
    a test with a stand-in `request`, without fighting pytest's own fixture-ordering timing.

    Relaxed only for a test marked `live_llm`, and only when CLOUDGUARD_LIVE_LLM=1 - both
    conditions, not either: an unmarked test must stay blocked even if someone has that env
    var set for an unrelated reason (e.g. running the whole suite with it exported)."""
    if request.node.get_closest_marker("live_llm") and os.environ.get("CLOUDGUARD_LIVE_LLM") == "1":
        return
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


@pytest.fixture(autouse=True)
def no_live_llm_calls(request, monkeypatch):
    """Tests must never reach a real LLM: strip GEMINI_API_KEY, neutralize load_dotenv() so
    a real .env on disk can't repopulate it, and make the one real network call Gemini-side
    fail loudly if anything still reaches it. Each guard is a no-op if the `agent` extra
    (google-genai / python-dotenv) isn't installed, so CI without it stays green. The one
    exception: a test marked `live_llm`, when CLOUDGUARD_LIVE_LLM=1 - see
    tests/test_gemini_contract.py."""
    _apply_live_llm_guard(request, monkeypatch)


@pytest.fixture(scope="session")
def _default_rag_fixture_index(tmp_path_factory) -> tuple[Path, Path]:
    """A tiny 1-article index, built once with FakeEmbedder, that stands in for the real
    corpus/index/ during every test. Built here (not committed) so no test depends on it
    having any particular content - only that search_docs works structurally."""
    from cloudguard.rag.embed import FakeEmbedder
    from cloudguard.rag.index import build_index

    base = tmp_path_factory.mktemp("default_rag_fixture")
    manifest_path = base / "manifest.json"
    raw_dir = base / "raw"
    index_dir = base / "index"
    raw_dir.mkdir()

    manifest_path.write_text(json.dumps({
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "repos": [{"repo": "MicrosoftDocs/azure-docs", "commit_sha": "0" * 40,
                   "license": "CC-BY-4.0", "attribution": "test fixture"}],
        "known_issues": [],
        "articles": [{
            "id": "fixture-lock-resources", "title": "Lock your Azure resources",
            "learn_url": "https://learn.microsoft.com/azure/fixture/lock-resources",
            "repo": "MicrosoftDocs/azure-docs",
            "source_path": "articles/fixture/lock-resources.md", "topic": "locks",
        }],
    }), encoding="utf-8")
    (raw_dir / "fixture-lock-resources.md").write_text(
        "---\ntitle: Lock your Azure resources\nms.date: 01/01/2026\n---\n"
        "## Overview\n\nA resource lock prevents accidental deletion of a resource group.\n",
        encoding="utf-8",
    )

    build_index(manifest_path, raw_dir, FakeEmbedder(), index_dir)
    return manifest_path, index_dir


@pytest.fixture(autouse=True)
def no_real_embedder_loads(_default_rag_fixture_index, monkeypatch):
    """search_docs is called via TOOLS[name](backend, **kwargs), with no per-call way to
    inject an embedder - so the injection seam lives here, on rag.index's module-level
    defaults, exactly like no_live_llm_calls does for GeminiClient. Every test gets this
    tiny FakeEmbedder-built index automatically; the real FastEmbedder (fastembed, a real
    model download) is never constructed during the test suite."""
    from cloudguard.rag import index as rag_index
    from cloudguard.rag.embed import FakeEmbedder

    manifest_path, index_dir = _default_rag_fixture_index
    monkeypatch.setattr(rag_index, "DEFAULT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(rag_index, "DEFAULT_INDEX_DIR", index_dir)
    monkeypatch.setattr(rag_index, "get_default_embedder", FakeEmbedder)


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend(ROOT / "mock" / "subscription_snapshot.json")


@pytest.fixture
def truth() -> GroundTruth:
    return GroundTruth.model_validate_json((ROOT / "ground_truth.json").read_text())


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
