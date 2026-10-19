"""Turn one fetched article (corpus/raw/<id>.md + its spliced includes) into retrieval
chunks: strip DocFX/front-matter syntax that has no retrieval value, split by heading,
and keep each chunk under the embedding model's context window.

bge-small-en-v1.5 truncates silently at 512 tokens (~0.75 words/token => ~384 words), so
chunks target 180-320 words, not the full window - leaving headroom for the "{title} >
{heading}" context line that gets prefixed onto the chunk text before embedding (the
embedded string differs from the stored/returned chunk text, which stays bare for
snippets).
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from cloudguard.models import CorpusArticle

TARGET_MIN_WORDS = 180
TARGET_MAX_WORDS = 320
OVERLAP_WORDS = 40
WARN_WORDS = 380  # close to the ~384-word truncation point; build_index warns, doesn't fail

# Pure boilerplate trailer sections with no retrieval value - always dropped, regardless of
# whether the manifest entry also sets a `sections` allowlist.
_ALWAYS_DROPPED_HEADINGS = {"next steps", "related content", "see also"}

_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_INCLUDE_RE = re.compile(r"\[!INCLUDE\s*\[[^\]]*\]\(([^)]+)\)\]")
_ZONE_OR_IMAGE_RE = re.compile(r"^:::\s*(zone|zone-end|image)\b.*$", re.IGNORECASE)
_ALERT_START_RE = re.compile(r"^>\s*\[!(NOTE|WARNING|TIP|IMPORTANT|CAUTION)\]\s*$", re.IGNORECASE)
_BLOCKQUOTE_PREFIX_RE = re.compile(r"^>\s?")
_TAB_MARKER_RE = re.compile(r"^#\s*\[[^\]]*\]\(#tab/[^)]*\)\s*$")
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.*?)\s*$")
_FENCE_RE = re.compile(r"^\s*```")
_MD_LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:-]*-{2,}[\s:|-]*\|?\s*$")
_TOP_LEVEL_LIST_RE = re.compile(r"^(-|\*|\d+\.)\s")


@dataclass
class Chunk:
    chunk_id: str
    article_id: str
    title: str
    heading_path: str  # e.g. "Enforce TLS > Azure portal", "" for the intro section
    topic: str
    url: str
    text: str  # bare chunk text, used for snippets - never the embedded (contextualized) string
    ms_date: str | None = None

    def embedding_text(self) -> str:
        """What actually gets embedded: title + heading context prefixed onto the bare
        text, so a short chunk isn't embedded with no idea what page/section it's from."""
        prefix = self.title if not self.heading_path else f"{self.title} > {self.heading_path}"
        return f"{prefix}\n\n{self.text}"


def _slugify(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)  # punctuation removed
    s = re.sub(r"\s+", "-", s)  # spaces -> hyphens
    return s


def _strip_front_matter(text: str) -> tuple[str, str | None, str | None]:
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return text, None, None
    front = m.group(1)
    title_m = re.search(r"^title:\s*(.+)$", front, re.MULTILINE)
    date_m = re.search(r"^ms\.date:\s*(.+)$", front, re.MULTILINE)
    title = title_m.group(1).strip().strip("\"'") if title_m else None
    ms_date = date_m.group(1).strip().strip("\"'") if date_m else None
    return text[m.end():], title, ms_date


def _splice_includes(text: str, article: CorpusArticle, raw_dir) -> str:
    base_dir = posixpath.dirname(article.source_path)
    repo_slug = article.repo.split("/")[-1]

    def _replace(m: re.Match) -> str:
        inc = m.group(1).strip()
        if inc.startswith("~/"):
            return ""  # cross-repo include, never fetched - not topic content anyway
        resolved = posixpath.normpath(posixpath.join(base_dir, inc))
        inc_path = raw_dir / "includes" / repo_slug / resolved
        if inc_path.exists():
            return inc_path.read_text(encoding="utf-8")
        return ""  # unresolved (e.g. a documented known_issues 404) - drop, never guess

    return _INCLUDE_RE.sub(_replace, text)


def _resolve_relative_link(url: str, learn_url: str) -> str | None:
    """Resolve a relative markdown-page link against the article's own learn_url (Learn
    URLs mirror the docs repo's folder structure after stripping articles|docs/ and .md,
    so resolving against the URL itself - not the repo source_path - is correct and repo-
    agnostic). Returns None for anything that isn't a plain relative .md-or-root-relative
    page link - the caller falls back to the link text alone for those."""
    if url.startswith(("http://", "https://", "#", "mailto:")):
        return url  # already absolute (or a same-page anchor) - nothing to resolve

    path, _, anchor = url.partition("#")
    anchor = f"#{anchor}" if anchor else ""

    if path.endswith(".md"):
        path = path[: -len(".md")]
    elif not path or not path.startswith("/"):
        return None  # relative, but not a markdown page (e.g. an image/asset link), or empty

    if path.startswith("/"):
        return f"https://learn.microsoft.com{path}{anchor}"

    parsed = urlsplit(learn_url)
    base_dir = posixpath.dirname(parsed.path)
    resolved_path = posixpath.normpath(posixpath.join(base_dir, path))
    return f"{parsed.scheme}://{parsed.netloc}{resolved_path}{anchor}"


def _rewrite_links(text: str, learn_url: str) -> str:
    def _replace(m: re.Match) -> str:
        is_image, link_text, url = m.group(1), m.group(2), m.group(3)
        if is_image:
            return link_text
        resolved = _resolve_relative_link(url, learn_url)
        return link_text if resolved is None else f"[{link_text}]({resolved})"

    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        out.append(line if in_fence else _MD_LINK_RE.sub(_replace, line))
    return "\n".join(out)


def _strip_docfx(text: str) -> str:
    out: list[str] = []
    in_alert = False
    for line in text.splitlines():
        if _ZONE_OR_IMAGE_RE.match(line) or _TAB_MARKER_RE.match(line):
            continue
        if _ALERT_START_RE.match(line):
            in_alert = True
            continue
        if in_alert:
            if line.startswith(">"):
                out.append(_BLOCKQUOTE_PREFIX_RE.sub("", line))
                continue
            in_alert = False
        out.append(line)
    return "\n".join(out)


@dataclass
class _Section:
    heading_path: str  # "" for the intro section
    anchor_heading: str | None  # the innermost heading text, for the URL anchor; None for intro
    lines: list[str]


def _split_into_sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    current = _Section(heading_path="", anchor_heading=None, lines=[])
    h2_text: str | None = None
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            current.lines.append(line)
            continue
        heading_m = _HEADING_RE.match(line) if not in_fence else None
        if heading_m:
            sections.append(current)
            level, heading_text = len(heading_m.group(1)), heading_m.group(2)
            if level == 2:
                h2_text = heading_text
                path = heading_text
            else:
                path = f"{h2_text} > {heading_text}" if h2_text else heading_text
            current = _Section(heading_path=path, anchor_heading=heading_text, lines=[])
        else:
            current.lines.append(line)
    sections.append(current)
    return [s for s in sections if any(l.strip() for l in s.lines)]


def _atomic_units(lines: list[str]) -> list[str]:
    units: list[str] = []
    buf: list[str] = []
    in_fence = False
    fence_buf: list[str] = []
    for line in lines:
        if _FENCE_RE.match(line):
            if not in_fence:
                if buf:
                    units.append("\n".join(buf))
                    buf = []
                in_fence = True
                fence_buf = [line]
            else:
                fence_buf.append(line)
                units.append("\n".join(fence_buf))
                fence_buf = []
                in_fence = False
            continue
        if in_fence:
            fence_buf.append(line)
            continue
        if line.strip() == "":
            if buf:
                units.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)
    if fence_buf:  # unterminated fence - keep it rather than lose content
        units.append("\n".join(fence_buf))
    if buf:
        units.append("\n".join(buf))
    return units


def word_count(s: str) -> int:
    return len(s.split())


def _is_code_unit(unit: str) -> bool:
    return unit.lstrip().startswith("```")


def _pack_units(units: list[str]) -> list[str]:
    if not units:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        w = word_count(unit)
        if current and current_words >= TARGET_MIN_WORDS and current_words + w > TARGET_MAX_WORDS:
            chunks.append("\n\n".join(current))
            # Overlap is prose continuity, not a chance to duplicate a whole code block:
            # a fenced-code unit is never pulled into the next chunk's overlap, however
            # large it is - that would both bloat the next chunk and defeat the point of
            # "small" overlap.
            overlap_units: list[str] = []
            ow = 0
            for u in reversed(current):
                if _is_code_unit(u):
                    continue
                overlap_units.insert(0, u)
                ow += word_count(u)
                if ow >= OVERLAP_WORDS:
                    break
            current, current_words = overlap_units, ow
        current.append(unit)
        current_words += w
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _is_code(body: str) -> bool:
    return "```" in body


def _is_table(body: str) -> bool:
    """A GFM table's header-separator row (e.g. "|-|----|--|" or "| --- | --- |") is a
    reliable, simple signal - tables often have long cells so they read as "oversized"
    by word count alone, but splitting a table row by row would wreck it."""
    return any(_TABLE_SEPARATOR_RE.match(line) for line in body.splitlines())


def _reflow_prose_units(body: str) -> list[str]:
    """Finer-grained units for re-splitting an oversized prose body: blank-line
    paragraphs, and any paragraph that is itself one unbroken block (e.g. a long bullet
    list with no blank lines between items) further split at top-level list-item
    boundaries, so there's still a "paragraph boundary" to split at."""
    paragraphs = _atomic_units(body.splitlines())
    units: list[str] = []
    for p in paragraphs:
        if word_count(p) <= TARGET_MAX_WORDS:
            units.append(p)
            continue
        items: list[str] = []
        current: list[str] = []
        for line in p.splitlines():
            if _TOP_LEVEL_LIST_RE.match(line) and current:
                items.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            items.append("\n".join(current))
        units.extend(items if len(items) > 1 else [p])
    return units


def _split_oversized_prose(bodies: list[str]) -> list[str]:
    """Run after normal packing: a chunk that's still over WARN_WORDS and isn't a table or
    a fenced code block (both of which must stay whole) gets re-split at paragraph/list-
    item boundaries and re-packed to the normal target size."""
    result: list[str] = []
    for body in bodies:
        if word_count(body) <= WARN_WORDS or _is_code(body) or _is_table(body):
            result.append(body)
            continue
        finer_units = _reflow_prose_units(body)
        if len(finer_units) <= 1:
            result.append(body)  # no paragraph/list boundary found - nothing to split on
            continue
        result.extend(_pack_units(finer_units))
    return result


def chunk_article(article: CorpusArticle, raw_dir) -> list[Chunk]:
    raw_path = raw_dir / f"{article.id}.md"
    text = raw_path.read_text(encoding="utf-8")

    text, front_matter_title, ms_date = _strip_front_matter(text)
    text = _splice_includes(text, article, raw_dir)
    text = _rewrite_links(text, article.learn_url)
    text = _strip_docfx(text)

    title = front_matter_title or article.title
    allowlist = {h.lower() for h in article.sections} if article.sections else None

    chunks: list[Chunk] = []
    for section in _split_into_sections(text):
        heading_lower = (section.anchor_heading or "").lower()
        if heading_lower in _ALWAYS_DROPPED_HEADINGS:
            continue
        if allowlist is not None and section.anchor_heading is not None and heading_lower not in allowlist:
            continue

        units = _atomic_units(section.lines)
        section_chunks = _pack_units(units)
        section_chunks = _split_oversized_prose(section_chunks)
        path_slug = "__".join(_slugify(p) for p in section.heading_path.split(" > ")) if section.heading_path else "intro"
        url = article.learn_url if section.anchor_heading is None else f"{article.learn_url}#{_slugify(section.anchor_heading)}"

        for i, body in enumerate(section_chunks):
            chunks.append(Chunk(
                chunk_id=f"{article.id}__{path_slug}__{i}",
                article_id=article.id,
                title=title,
                heading_path=section.heading_path,
                topic=article.topic,
                url=url,
                text=body,
                ms_date=ms_date,
            ))
    return chunks
