"""Grounding check: does the agent's final answer only cite resource IDs that a tool result
actually returned this run? Pure function, no LLM, no I/O - agent.py calls it after a final
answer, and the Day-4 eval harness can call it the same way over recorded runs.

Citation grounding (check_citations) applies the same idea to learn.microsoft.com URLs:
a cited URL must have come from a successful search_docs result in this same run.

The rule is ARTICLE-LEVEL. A URL is grounded if its normalized article path (lowercased
host, no locale, no query, no trailing slash, no #anchor) matches the article path of some
URL search_docs returned this run. The #anchor never affects grounding; `anchor_matched`
records separately whether that exact anchor was retrieved, as a precision signal. Why not
anchor-level: retrieval returns section chunks, and the model legitimately cites the
article or a neighbouring heading of an article it did retrieve. Anchors are slugs of
headings that Learn can rename and that we slugify ourselves, so demanding an exact anchor
match would raise false alarms on real, retrieved articles. What the check is for is catching
invented or recalled-from-memory articles, and that is an article-level question.
Grounding is scoped per run because it only ever sees that run's tool results.

Failure never removes anything: the answer is shown unchanged with a visible warning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import unquote, urlsplit

from cloudguard.models import ToolResult

# Same GUID shape as backends.ARM_ID_PATTERN. The charset after resourceGroups/ deliberately
# excludes backticks, quotes, commas, colons and whitespace - none of those are ever valid in
# an ARM ID, so they terminate extraction on their own, with no lookahead needed. `.` and `)`
# ARE valid in real IDs (provider namespaces like Microsoft.Compute; resource names can
# contain parens), so a trailing one is stripped only when it's clearly prose punctuation,
# not part of the name - see _strip_trailing_punctuation.
_CANDIDATE_RE = re.compile(
    r"/subscriptions/[0-9a-fA-F-]{36}/resourceGroups/[A-Za-z0-9_.\-/()]+",
    re.IGNORECASE,
)

# The /subscriptions/<sub>/resourceGroups/<rg> prefix of any ID - used to derive a grounded
# resource's parent resource group, which is implied by the ID, not invented.
_RG_PREFIX_RE = re.compile(r"^(/subscriptions/[^/]+/resourceGroups/[^/]+)", re.IGNORECASE)


def _strip_trailing_punctuation(candidate: str) -> str:
    """Azure resource names never end in a literal '.'; a trailing ')' with no matching '('
    earlier in the candidate is markdown/prose wrapping, not part of the name."""
    while candidate:
        if candidate[-1] == ".":
            candidate = candidate[:-1]
        elif candidate[-1] == ")" and candidate.count("(") < candidate.count(")"):
            candidate = candidate[:-1]
        else:
            break
    return candidate


def _normalize(resource_id: str) -> str:
    """ARM IDs are case-insensitive; a trailing slash never changes identity."""
    return resource_id.rstrip("/").lower()


def extract_resource_ids(text: str) -> list[str]:
    """Every ARM resource ID mentioned in free text, in first-seen order, deduplicated."""
    seen: dict[str, str] = {}
    for m in _CANDIDATE_RE.finditer(text or ""):
        candidate = _strip_trailing_punctuation(m.group(0))
        key = _normalize(candidate)
        if key and key not in seen:
            seen[key] = candidate
    return list(seen.values())


def _grounded_ids(tool_results: Iterable[ToolResult]) -> set[str]:
    """Every resource ID a *successful* tool result actually returned this run: findings,
    plus get_resource's own `data["id"]`. A not-found get_resource result has ok=False, so
    the ID the model asked about never enters this set."""
    ids: set[str] = set()
    for result in tool_results:
        if not result.ok:
            continue
        for f in result.findings:
            ids.add(f.resource_id)
        if result.data and result.data.get("id"):
            ids.add(result.data["id"])
    return ids


def _with_parent_resource_groups(ids: set[str]) -> set[str]:
    """A grounded resource's own resource group is implied by its ID, not invented, so it
    counts as grounded too - but nothing else: not sibling resources, not other groups."""
    derived = set(ids)
    for resource_id in ids:
        m = _RG_PREFIX_RE.match(resource_id)
        if m:
            derived.add(m.group(1))
    return derived


LEARN_HOST = "learn.microsoft.com"

# `en-us`, `pt-br`, `zh-hans-cn`... or a bare `en`. Deliberately not "any two letters": a
# real first path segment must never be eaten as a locale.
_LOCALE_SEGMENT_RE = re.compile(r"^/(?:en|[a-z]{2}-[a-z]{2,4}(?:-[a-z]{2})?)(?=/|$)", re.IGNORECASE)

# [text](url) first so the closing paren is consumed by the link, then bare URLs. Bare
# URLs stop at whitespace, quotes, backticks, angle brackets and brackets; trailing prose
# punctuation is stripped afterwards.
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*<?(https?://[^\s)>]+(?:\([^\s)]*\)[^\s)>]*)*)>?[^)]*\)", re.IGNORECASE)
_BARE_URL_RE = re.compile(r"https?://[^\s<>\"'`\[\]]+", re.IGNORECASE)


@dataclass(frozen=True)
class NormalizedUrl:
    host: str
    article: str  # "<host>/<path>", lowercased, no locale/query/anchor/trailing slash
    anchor: str | None  # lowercased, percent-decoded, no '#'; None if absent/empty


def normalize_doc_url(url: str) -> NormalizedUrl:
    """Reduce a docs URL to its article identity plus its anchor."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    path = _LOCALE_SEGMENT_RE.sub("", parts.path, count=1).rstrip("/").lower()
    anchor = unquote(parts.fragment).lower() or None
    return NormalizedUrl(host=host, article=f"{host}{path}", anchor=anchor)


def _strip_url_trailing_punctuation(url: str) -> str:
    while url:
        if url[-1] in ".,;:!?*_":
            url = url[:-1]
        elif url[-1] == ")" and url.count("(") < url.count(")"):
            url = url[:-1]
        else:
            break
    return url


def extract_urls(text: str) -> list[str]:
    """Every http(s) URL in free text (Markdown links and bare), first-seen order, deduplicated."""
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for m in _MD_LINK_RE.finditer(text or ""):
        found.append((m.start(1), m.group(1)))
        taken.append((m.start(), m.end()))
    for m in _BARE_URL_RE.finditer(text or ""):
        if not any(a <= m.start() < b for a, b in taken):
            found.append((m.start(), m.group(0)))
    seen: dict[str, str] = {}
    for _, raw in sorted(found):
        url = _strip_url_trailing_punctuation(raw)
        if url and url not in seen:
            seen[url] = url
    return list(seen)


@dataclass
class CitationReport:
    # One dict per cited learn.microsoft.com URL: url, article, anchor, grounded, anchor_matched
    # (anchor_matched is None when the cited URL has no anchor).
    cited_urls: list[dict] = field(default_factory=list)
    ungrounded_urls: list[str] = field(default_factory=list)
    external_urls: list[str] = field(default_factory=list)
    citations_ok: bool = True


def _retrieved_doc_urls(tool_results: Iterable[ToolResult]) -> list[str]:
    urls: list[str] = []
    for result in tool_results:
        if result.tool != "search_docs" or not result.ok or not result.data:
            continue
        urls.extend(r["url"] for r in result.data.get("results", []) if r.get("url"))
    return urls


def check_citations(answer_text: str, tool_results: Iterable[ToolResult]) -> CitationReport:
    retrieved = [normalize_doc_url(u) for u in _retrieved_doc_urls(tool_results)]
    articles = {n.article for n in retrieved}
    anchors = {(n.article, n.anchor) for n in retrieved if n.anchor}

    report = CitationReport()
    for url in extract_urls(answer_text):
        n = normalize_doc_url(url)
        if n.host != LEARN_HOST:
            report.external_urls.append(url)
            continue
        grounded = n.article in articles
        report.cited_urls.append({
            "url": url, "article": n.article, "anchor": n.anchor, "grounded": grounded,
            "anchor_matched": None if n.anchor is None else (n.article, n.anchor) in anchors,
        })
        if not grounded:
            report.ungrounded_urls.append(url)
    report.citations_ok = not report.ungrounded_urls
    return report


@dataclass
class GroundingReport:
    grounded_ids: list[str] = field(default_factory=list)
    ungrounded_ids: list[str] = field(default_factory=list)
    grounding_ok: bool = True  # resource IDs only; citations have their own citations_ok
    cited_urls: list[dict] = field(default_factory=list)
    ungrounded_urls: list[str] = field(default_factory=list)
    external_urls: list[str] = field(default_factory=list)
    citations_ok: bool = True


def check_grounding(answer_text: str, tool_results: Iterable[ToolResult]) -> GroundingReport:
    tool_results = list(tool_results)
    grounded_raw = _with_parent_resource_groups(_grounded_ids(tool_results))
    grounded_normalized = {_normalize(i) for i in grounded_raw}

    grounded_ids, ungrounded_ids = [], []
    for candidate in extract_resource_ids(answer_text):
        bucket = grounded_ids if _normalize(candidate) in grounded_normalized else ungrounded_ids
        bucket.append(candidate)

    citations = check_citations(answer_text, tool_results)
    return GroundingReport(
        grounded_ids=grounded_ids, ungrounded_ids=ungrounded_ids, grounding_ok=not ungrounded_ids,
        cited_urls=citations.cited_urls, ungrounded_urls=citations.ungrounded_urls,
        external_urls=citations.external_urls, citations_ok=citations.citations_ok,
    )
