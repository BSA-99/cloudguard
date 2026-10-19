"""scripts/fetch_corpus.py's known_issues classification. No network: fetch_fn is a fake,
injected into cmd_fetch directly, following the same importlib pattern as test_seed.py.
"""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_fetch_corpus():
    spec = importlib.util.spec_from_file_location("fetch_corpus", ROOT / "scripts" / "fetch_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_corpus = _load_fetch_corpus()

REPO = "MicrosoftDocs/azure-compute-docs"
SHA = "deadbeef"


def _manifest(known_issues: list[dict]) -> dict:
    return {
        "repos": [{"repo": REPO, "commit_sha": SHA, "license": "CC-BY-4.0", "attribution": "x"}],
        "known_issues": known_issues,
        "articles": [{
            "id": "disks-types", "title": "x", "learn_url": "https://learn.microsoft.com/x",
            "repo": REPO, "source_path": "articles/virtual-machines/disks-types.md", "topic": "disks",
        }],
    }


ARTICLE_BODY = (
    b"---\ntitle: x\n---\n"
    b"[!INCLUDE [broken](./includes/managed-disks-ultra-disks-GA-scope-and-limitations.md)]\n"
)
INCLUDE_PATH = "articles/virtual-machines/includes/managed-disks-ultra-disks-GA-scope-and-limitations.md"


def _http_404(url: str) -> bytes:
    raise urllib.error.HTTPError(url, 404, "Not Found", None, None)


def test_only_known_failures_exit_zero(tmp_path, capsys):
    manifest = _manifest([{"repo": REPO, "source_path": INCLUDE_PATH, "reason": "upstream casing bug"}])

    def fake_fetch(url: str) -> bytes:
        if url.endswith(INCLUDE_PATH):
            return _http_404(url)
        return ARTICLE_BODY

    exit_code = fetch_corpus.cmd_fetch(manifest, tmp_path, fetch_fn=fake_fetch)
    out, err = capsys.readouterr()

    assert exit_code == 0
    assert "KNOWN" in out and INCLUDE_PATH in out
    assert "FAILED" not in out
    assert err == ""  # never printed to stderr
    assert "known issues: 1" in out
    assert "unexpected failures: 0" in out


def test_one_unexpected_failure_exits_one(tmp_path, capsys):
    manifest = _manifest([])  # nothing declared as a known issue

    def fake_fetch(url: str) -> bytes:
        if url.endswith(INCLUDE_PATH):
            return _http_404(url)
        return ARTICLE_BODY

    exit_code = fetch_corpus.cmd_fetch(manifest, tmp_path, fetch_fn=fake_fetch)
    out, err = capsys.readouterr()

    assert exit_code == 1
    assert "FAILED" in err and INCLUDE_PATH in err
    assert "KNOWN" not in out
    assert "known issues: 0" in out
    assert "unexpected failures: 1" in out


def test_known_issue_that_now_succeeds_exits_zero_with_notice(tmp_path, capsys):
    manifest = _manifest([{"repo": REPO, "source_path": INCLUDE_PATH, "reason": "upstream casing bug"}])

    def fake_fetch(url: str) -> bytes:
        return ARTICLE_BODY if not url.endswith(INCLUDE_PATH) else b"include content, no longer 404ing"

    exit_code = fetch_corpus.cmd_fetch(manifest, tmp_path, fetch_fn=fake_fetch)
    out, _ = capsys.readouterr()

    assert exit_code == 0
    assert "NOTICE" in out and "consider removing" in out and INCLUDE_PATH in out
    assert "known issues: 0" in out  # it didn't fail this run, so it's not a "known hit"
    assert "unexpected failures: 0" in out
