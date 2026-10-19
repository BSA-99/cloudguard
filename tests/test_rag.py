"""chunk.py / embed.py / index.py, FakeEmbedder only. Small fixture corpora are built
inline per test (tmp_path), following this project's existing convention (FakeLLM scripts
built inline in test_agent.py, etc.) rather than a separate static fixtures directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloudguard import tools
from cloudguard.models import CorpusManifest
from cloudguard.rag import index as rag_index
from cloudguard.rag.chunk import Chunk, chunk_article
from cloudguard.rag.embed import FakeEmbedder
from cloudguard.rag.index import StaleIndexError, build_index, search

REPO = "MicrosoftDocs/azure-docs"


def _manifest(tmp_path: Path, articles: list[dict]) -> Path:
    manifest = {
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "repos": [{"repo": REPO, "commit_sha": "0" * 40, "license": "CC-BY-4.0", "attribution": "test fixture"}],
        "known_issues": [],
        "articles": articles,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _article(id_: str, title: str, *, topic: str = "locks", learn_url: str | None = None,
             sections: list[str] | None = None) -> dict:
    a = {
        "id": id_, "title": title,
        "learn_url": learn_url or f"https://learn.microsoft.com/azure/fixture/{id_}",
        "repo": REPO, "source_path": f"articles/fixture/{id_}.md", "topic": topic,
    }
    if sections is not None:
        a["sections"] = sections
    return a


def _write_raw(raw_dir: Path, id_: str, content: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{id_}.md").write_text(content, encoding="utf-8")


def _write_include(raw_dir: Path, path: str, content: str) -> None:
    p = raw_dir / "includes" / "azure-docs" / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _chunks_for(tmp_path: Path, id_: str, title: str, body: str, **kwargs) -> list[Chunk]:
    manifest_path = _manifest(tmp_path, [_article(id_, title, **kwargs)])
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, id_, body)
    manifest = CorpusManifest.model_validate_json(manifest_path.read_text())
    return chunk_article(manifest.articles[0], raw_dir)


# ---------------------------------------------------------------------- include splicing

def test_include_is_spliced_inline(tmp_path):
    manifest_path = _manifest(tmp_path, [_article("locks", "Lock your Azure resources")])
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, "locks", (
        "---\ntitle: Lock your Azure resources\nms.date: 01/15/2026\n---\n"
        "## Overview\n\n[!INCLUDE [intro](./includes/lock-intro.md)]\n"
    ))
    _write_include(raw_dir, "articles/fixture/includes/lock-intro.md",
                    "A lock prevents accidental deletion of a resource group.")

    manifest = CorpusManifest.model_validate_json(manifest_path.read_text())
    chunks = chunk_article(manifest.articles[0], raw_dir)

    assert len(chunks) == 1
    assert "A lock prevents accidental deletion of a resource group." in chunks[0].text
    assert "[!INCLUDE" not in chunks[0].text
    assert chunks[0].ms_date == "01/15/2026"


def test_unresolved_include_is_removed_not_left_as_syntax(tmp_path):
    chunks = _chunks_for(tmp_path, "nofile", "No file", (
        "---\ntitle: No file\n---\n"
        "## Overview\n\n[!INCLUDE [missing](./includes/does-not-exist.md)]\n\nReal text.\n"
    ))
    assert "[!INCLUDE" not in chunks[0].text
    assert "Real text." in chunks[0].text


def test_cross_repo_include_is_removed(tmp_path):
    chunks = _chunks_for(tmp_path, "crossrepo", "Cross repo", (
        "---\ntitle: Cross repo\n---\n"
        "## Overview\n\n"
        "[!INCLUDE [boilerplate](~/reusable-content/ce-skilling/azure/includes/gdpr-intro-sentence.md)]\n\n"
        "Real text.\n"
    ))
    assert "[!INCLUDE" not in chunks[0].text
    assert "Real text." in chunks[0].text


# ------------------------------------------------------------------------ DocFX stripping

def test_docfx_note_zone_and_image_are_stripped(tmp_path):
    chunks = _chunks_for(tmp_path, "docfx", "DocFX syntax test", (
        "---\ntitle: DocFX syntax test\n---\n"
        "## Overview\n\n"
        "> [!NOTE]\n> This note should become plain text.\n\n"
        ':::zone pivot="azure-portal"\n'
        "Portal-pivot content stays (only the marker lines are dropped).\n"
        ":::zone-end\n\n"
        ':::image source="./media/lock.png" alt-text="Lock icon":::\n\n'
        "Ordinary paragraph after.\n"
    ))
    text = chunks[0].text

    assert "[!NOTE]" not in text
    assert "This note should become plain text." in text
    assert "> This note" not in text  # blockquote prefix stripped off the alert body
    assert ":::zone" not in text and ":::image" not in text
    assert "Portal-pivot content stays" in text  # content between zone markers is kept
    assert "Ordinary paragraph after." in text


def test_docfx_tab_markers_are_stripped(tmp_path):
    chunks = _chunks_for(tmp_path, "tabs", "Tabs test", (
        "---\ntitle: Tabs test\n---\n"
        "## Apply a lock\n\n"
        "# [Azure CLI](#tab/azure-cli)\n\nUse az lock create.\n\n"
        "# [Azure portal](#tab/azure-portal)\n\nUse the Locks blade.\n"
    ))
    text = chunks[0].text

    assert "#tab/" not in text
    assert "[Azure CLI]" not in text and "[Azure portal]" not in text
    assert "Use az lock create." in text
    assert "Use the Locks blade." in text


def test_boilerplate_trailer_sections_are_dropped(tmp_path):
    chunks = _chunks_for(tmp_path, "trailer", "Trailer test", (
        "---\ntitle: Trailer test\n---\n"
        "## Overview\n\nReal content here.\n"
        "## Next steps\n\nLearn more elsewhere.\n"
        "## Related content\n\nSee also these links.\n"
        "## See also\n\nMore links.\n"
    ))
    headings = {c.heading_path for c in chunks}

    assert headings == {"Overview"}
    assert all("Learn more elsewhere" not in c.text for c in chunks)


# -------------------------------------------------------------- code blocks never split

def test_fenced_code_block_is_never_split(tmp_path):
    code_lines = "\n".join(f"line_{i} = {i}" for i in range(140))
    body = (
        "---\ntitle: Code heavy\n---\n"
        "## Example\n\n" + ("intro word " * 60) + "\n\n"
        "```python\n" + code_lines + "\n```\n\n" + ("outro word " * 60) + "\n"
    )
    chunks = _chunks_for(tmp_path, "code", "Code heavy", body, topic="policy")

    for c in chunks:
        assert c.text.count("```") % 2 == 0, f"chunk has an unbalanced code fence: {c.chunk_id}"
    holder = next(c for c in chunks if "line_0 = 0" in c.text)
    assert "line_139 = 139" in holder.text  # both ends of the block in the same chunk


# ----------------------------------------------------------------------------- anchors

def test_heading_anchor_slugs_follow_learn_rules(tmp_path):
    chunks = _chunks_for(tmp_path, "slug", "Slug test", (
        "---\ntitle: Slug test\n---\n"
        "## Enforce TLS 1.2 (recommended)!\n\nSome text.\n"
        "### Azure CLI & PowerShell\n\nMore text.\n"
    ))

    h2 = next(c for c in chunks if c.heading_path == "Enforce TLS 1.2 (recommended)!")
    assert h2.url.endswith("#enforce-tls-12-recommended")
    h3 = next(c for c in chunks if "Azure CLI & PowerShell" in c.heading_path)
    assert h3.url.endswith("#azure-cli-powershell")


def test_chunk_ids_use_full_heading_path_no_collision_across_h2s(tmp_path):
    """Two H2s sharing an H3 name (e.g. both have an "Azure CLI" subsection) must not
    collide: chunk_id is built from the full heading path, not just the innermost slug."""
    chunks = _chunks_for(tmp_path, "collide", "Collision test", (
        "---\ntitle: Collision test\n---\n"
        "## Policy\n\n### Azure CLI\n\nPolicy CLI steps.\n"
        "## Locks\n\n### Azure CLI\n\nLock CLI steps.\n"
    ))

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert any("policy" in i and "azure-cli" in i for i in ids)
    assert any("locks" in i and "azure-cli" in i for i in ids)


def test_build_index_raises_on_a_genuine_chunk_id_duplicate(tmp_path):
    manifest_path = _manifest(tmp_path, [
        _article("dup", "Dup A"),
        _article("dup", "Dup B"),  # same id - illegal, but exercises the safety net
    ])
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, "dup", "---\ntitle: Dup\n---\n## Overview\n\nSame content either way.\n")

    with pytest.raises(ValueError, match="duplicate chunk_id"):
        build_index(manifest_path, raw_dir, FakeEmbedder(), tmp_path / "index")


# ------------------------------------------------------------------- sections allowlist

def test_sections_allowlist_keeps_only_named_headings_plus_intro(tmp_path):
    chunks = _chunks_for(tmp_path, "allow", "Allowlist test", (
        "---\ntitle: Allowlist test\n---\n"
        "Intro text before any heading.\n\n"
        "## Keep me\n\nKept content.\n"
        "## Drop me\n\nDropped content.\n"
    ), sections=["Keep me"])
    headings = {c.heading_path for c in chunks}

    assert headings == {"", "Keep me"}
    assert all("Dropped content" not in c.text for c in chunks)


# ------------------------------------------------------------------ embedding vs. storage

def test_embedding_text_prefixes_title_and_heading_but_stored_text_stays_bare():
    c = Chunk(chunk_id="x", article_id="a", title="My Title", heading_path="My Heading",
              topic="locks", url="https://x", text="bare body text")
    assert c.text == "bare body text"
    assert c.embedding_text() == "My Title > My Heading\n\nbare body text"


# ------------------------------------------------------------------------------ search

def test_search_caps_results_per_article(tmp_path):
    manifest_path = _manifest(tmp_path, [
        _article("dominant", "Dominant article"),
        _article("other", "Other article", topic="policy"),
    ])
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, "dominant", (
        "---\ntitle: Dominant article\n---\n"
        "## Widget alpha\n\nwidget widget widget alpha content here.\n"
        "## Widget beta\n\nwidget widget widget beta content here.\n"
        "## Widget gamma\n\nwidget widget widget gamma content here.\n"
    ))
    _write_raw(raw_dir, "other", (
        "---\ntitle: Other article\n---\n## Something else\n\nwidget appears once here too.\n"
    ))

    index_dir = tmp_path / "index"
    embedder = FakeEmbedder()
    build_index(manifest_path, raw_dir, embedder, index_dir)

    results = search("widget", k=4, embedder=embedder, manifest_path=manifest_path, index_dir=index_dir)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.article_id] = counts.get(r.article_id, 0) + 1
    assert len(results) == 3  # 3 "dominant" chunks exist but only 2 are allowed through
    assert counts["dominant"] == 2
    assert counts["other"] == 1


def test_stale_index_raises_on_manifest_change(tmp_path):
    manifest_path = _manifest(tmp_path, [_article("a", "A")])
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, "a", "---\ntitle: A\n---\n## Overview\n\nSome content.\n")
    index_dir = tmp_path / "index"
    embedder = FakeEmbedder()
    build_index(manifest_path, raw_dir, embedder, index_dir)

    manifest_path.write_text(manifest_path.read_text() + "\n")  # changes the file's hash

    with pytest.raises(StaleIndexError, match="stale"):
        search("content", k=3, embedder=embedder, manifest_path=manifest_path, index_dir=index_dir)


def test_stale_index_raises_on_model_change(tmp_path):
    manifest_path = _manifest(tmp_path, [_article("a", "A")])
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, "a", "---\ntitle: A\n---\n## Overview\n\nSome content.\n")
    index_dir = tmp_path / "index"
    build_index(manifest_path, raw_dir, FakeEmbedder(), index_dir)

    class _DifferentModel(FakeEmbedder):
        name = "different-model"

    with pytest.raises(StaleIndexError, match="stale"):
        search("content", k=3, embedder=_DifferentModel(), manifest_path=manifest_path, index_dir=index_dir)


# --------------------------------------------------------------------- search_docs shape

def test_search_docs_output_shape(tmp_path, backend, monkeypatch):
    manifest_path = _manifest(tmp_path, [_article("a", "A")])
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, "a", "---\ntitle: A\n---\n## Overview\n\n" + ("content word " * 50) + "\n")
    index_dir = tmp_path / "index"
    build_index(manifest_path, raw_dir, FakeEmbedder(), index_dir)

    monkeypatch.setattr(rag_index, "DEFAULT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(rag_index, "DEFAULT_INDEX_DIR", index_dir)
    monkeypatch.setattr(rag_index, "get_default_embedder", FakeEmbedder)

    result = tools.search_docs(backend, query="content", k=3)

    assert result.ok
    assert result.data["results"]
    hit = result.data["results"][0]
    assert set(hit.keys()) == {"chunk_id", "title", "heading", "url", "snippet", "score"}
    assert hit["url"].startswith("https://learn.microsoft.com/")
    assert len(hit["snippet"]) <= 600


# ------------------------------------------------------------------------ link rewriting

def test_relative_md_link_with_anchor_is_resolved_against_learn_url(tmp_path):
    # learn_url: https://learn.microsoft.com/azure/fixture/links -> dirname .../azure/fixture
    chunks = _chunks_for(tmp_path, "links", "Links test", (
        "---\ntitle: Links test\n---\n"
        "## Overview\n\nSee [Authorize access](../common/authorize-data-access.md#roles) for details.\n"
    ))
    text = chunks[0].text
    assert "[Authorize access](https://learn.microsoft.com/azure/common/authorize-data-access#roles)" in text


def test_relative_md_link_same_directory_bare_filename_is_resolved(tmp_path):
    chunks = _chunks_for(tmp_path, "links", "Links test", (
        "---\ntitle: Links test\n---\n"
        "## Overview\n\nSee [sibling page](sibling-page.md) for details.\n"
    ))
    text = chunks[0].text
    assert "[sibling page](https://learn.microsoft.com/azure/fixture/sibling-page)" in text


def test_root_relative_link_gets_the_learn_host_prefixed(tmp_path):
    chunks = _chunks_for(tmp_path, "links", "Links test", (
        "---\ntitle: Links test\n---\n"
        "## Overview\n\nSee the [workload identity overview](/azure/aks/workload-identity-overview).\n"
    ))
    text = chunks[0].text
    assert "[workload identity overview](https://learn.microsoft.com/azure/aks/workload-identity-overview)" in text


def test_image_link_becomes_just_its_alt_text(tmp_path):
    chunks = _chunks_for(tmp_path, "links", "Links test", (
        "---\ntitle: Links test\n---\n"
        "## Overview\n\nAn image: ![Lock icon](./media/lock.png) in the middle of a sentence.\n"
    ))
    text = chunks[0].text
    assert "Lock icon" in text
    assert "](./media/lock.png)" not in text
    assert "![" not in text


def test_absolute_http_link_is_left_unchanged(tmp_path):
    chunks = _chunks_for(tmp_path, "links", "Links test", (
        "---\ntitle: Links test\n---\n"
        "## Overview\n\nSee [external docs](https://example.com/docs) for more.\n"
    ))
    assert "[external docs](https://example.com/docs)" in chunks[0].text


def test_unresolvable_relative_non_markdown_link_becomes_plain_text(tmp_path):
    chunks = _chunks_for(tmp_path, "links", "Links test", (
        "---\ntitle: Links test\n---\n"
        "## Overview\n\nSee [the diagram](./media/diagram.png) for more.\n"
    ))
    text = chunks[0].text
    assert "the diagram" in text
    assert "](./media/diagram.png)" not in text


def test_link_inside_a_fenced_code_block_is_never_rewritten(tmp_path):
    chunks = _chunks_for(tmp_path, "links", "Links test", (
        "---\ntitle: Links test\n---\n"
        "## Overview\n\n```text\n[fake link](../should/not/be/touched.md)\n```\n"
    ))
    assert "[fake link](../should/not/be/touched.md)" in chunks[0].text


# ------------------------------------------------------------ oversized-chunk remediation

def test_oversized_table_chunk_is_left_whole(tmp_path):
    rows = "\n".join(f"| Recommendation {i} | {'word ' * 20} | - |" for i in range(15))
    body = (
        "---\ntitle: Table test\n---\n"
        "## Recommendations\n\n"
        "| Recommendation | Comments | Defender |\n|-|----|--|\n" + rows + "\n"
    )
    chunks = _chunks_for(tmp_path, "table", "Table test", body)

    assert len(chunks) == 1
    assert len(chunks[0].text.split()) > 380
    assert chunks[0].text.count("|-|----|--|") == 1  # the table survived as one block


def test_oversized_code_chunk_is_left_whole(tmp_path):
    code = "\n".join(f"line_{i} = {i}  # padding to push this well past 380 words" for i in range(60))
    body = "---\ntitle: Code test\n---\n## Example\n\n```python\n" + code + "\n```\n"
    chunks = _chunks_for(tmp_path, "code", "Code test", body)

    assert len(chunks) == 1
    assert len(chunks[0].text.split()) > 380
    assert chunks[0].text.count("```") == 2


def test_oversized_bullet_list_prose_is_split_at_item_boundaries(tmp_path):
    items = "\n".join(f"- `property{i}` (optional): {'word ' * 25} more detail text here." for i in range(20))
    body = "---\ntitle: List test\n---\n## Properties\n\n" + items + "\n"
    chunks = _chunks_for(tmp_path, "listy", "List test", body)

    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 380 for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))  # still unique after the extra split
    assert all(c.heading_path == "Properties" for c in chunks)
    # Every property survived somewhere, and none got cut mid-item.
    combined = "\n".join(c.text for c in chunks)
    for i in range(20):
        assert f"`property{i}`" in combined


def test_build_index_reports_only_the_chunks_still_over_the_limit_after_remediation(tmp_path):
    items = "\n".join(f"- `property{i}` (optional): {'word ' * 25} more detail." for i in range(20))
    manifest_path = _manifest(tmp_path, [_article("listy", "List test")])
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, "listy", "---\ntitle: List test\n---\n## Properties\n\n" + items + "\n")

    result = build_index(manifest_path, raw_dir, FakeEmbedder(), tmp_path / "index")

    assert result["over_warn_chunks"] == []  # the list was splittable, so nothing remains oversized
