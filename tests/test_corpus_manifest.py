"""docs/corpus_manifest.json: schema, uniqueness, and that every manifest entry has a
real fetched file in corpus/raw/. No network - this only reads what's already committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloudguard.models import CorpusManifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs" / "corpus_manifest.json"

ALLOWED_TOPICS = {
    "locks", "policy", "resource_graph", "disks", "storage_tls",
    "storage_security", "rbac", "identity", "tagging_governance", "cost",
}


@pytest.fixture(scope="module")
def manifest() -> CorpusManifest:
    return CorpusManifest.model_validate_json(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_validates_against_the_schema(manifest):
    assert len(manifest.articles) >= 1
    assert len(manifest.repos) >= 1


def test_article_ids_are_unique(manifest):
    ids = [a.id for a in manifest.articles]
    assert len(ids) == len(set(ids))


def test_every_topic_is_from_the_allowed_set(manifest):
    assert ALLOWED_TOPICS == {
        "locks", "policy", "resource_graph", "disks", "storage_tls",
        "storage_security", "rbac", "identity", "tagging_governance", "cost",
    }  # documents the exact allowed set this test enforces
    for a in manifest.articles:
        assert a.topic in ALLOWED_TOPICS, f"{a.id} has topic {a.topic!r}, not in the allowed set"


def test_every_learn_url_starts_with_the_learn_microsoft_domain(manifest):
    for a in manifest.articles:
        assert a.learn_url.startswith("https://learn.microsoft.com/"), a.id


def test_every_article_repo_is_declared_in_the_top_level_repos_list(manifest):
    declared = {r.repo for r in manifest.repos}
    for a in manifest.articles:
        assert a.repo in declared, f"{a.id} references undeclared repo {a.repo!r}"


def test_every_manifest_entry_has_a_matching_file_in_corpus_raw(manifest):
    raw_dir = ROOT / "corpus" / "raw"
    missing = [a.id for a in manifest.articles if not (raw_dir / f"{a.id}.md").exists()]
    assert missing == [], f"missing fetched files for: {missing}"


def test_known_issues_reference_a_declared_repo(manifest):
    declared = {r.repo for r in manifest.repos}
    for issue in manifest.known_issues:
        assert issue.repo in declared


def test_manifest_is_still_valid_raw_json():
    """Belt-and-suspenders: the file itself must be plain, well-formed JSON (not just
    something pydantic happens to coerce)."""
    json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
