"""Fetch the pinned RAG corpus described by docs/corpus_manifest.json.

Never clones azure-docs (it's huge): fetches individual files straight from
raw.githubusercontent.com at each repo's pinned commit_sha.

  python scripts/fetch_corpus.py              # normal run: fetch articles + includes at the pinned SHAs
  python scripts/fetch_corpus.py --resolve     # re-resolve each repo's current commit SHA into the manifest,
                                                # then stop - does not fetch any content. Run this, then re-run
                                                # without --resolve to pin and fetch at the new SHA.

Saves each article to corpus/raw/<id>.md. Also fetches every [!INCLUDE [...](path)] this
docset references (one level deep) into corpus/raw/includes/<repo>/<resolved path>, because
much of the real content in these docs lives in includes, not the article body itself.
A `~/`-prefixed include points at a different pinned repo (e.g. MicrosoftDocs/reusable-content)
declared in that repo's .openpublishing.publish.config.json; those are generic reusable
boilerplate (free-trial banners, GDPR notices), not topic content, so this script logs and
skips them rather than pinning a fifth repo for them.

Fails loudly: every 404 (article or include) is collected and printed, never swallowed. A
failure whose (repo, path) is listed in the manifest's known_issues prints as KNOWN on
stdout and does not affect the exit code - that list documents *why* it's expected, not an
excuse to go silent. Any other failure prints FAILED on stderr and makes the run exit 1. If
a known_issues path fetches successfully (Microsoft fixed it), that's reported too, as a
notice to remove the now-stale entry - still exit 0.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "corpus_manifest.json"
DEFAULT_OUT_DIR = ROOT / "corpus" / "raw"

INCLUDE_RE = re.compile(r"\[!INCLUDE\s*\[[^\]]*\]\(([^)]+)\)\]")
USER_AGENT = "cloudguard-fetch-corpus (https://github.com/MicrosoftDocs - read-only docs fetch)"


def raw_url(repo: str, commit_sha: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit_sha}/{path}"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def resolve_commit_sha(repo: str) -> str:
    """git ls-remote needs no clone and isn't subject to any GitHub API rate limit."""
    out = subprocess.run(
        ["git", "ls-remote", f"https://github.com/{repo}.git", "main"],
        capture_output=True, text=True, check=True,
    )
    line = out.stdout.strip().splitlines()[0]
    return line.split()[0]


def cmd_resolve(manifest: dict[str, Any], manifest_path: Path) -> int:
    for r in manifest["repos"]:
        new_sha = resolve_commit_sha(r["repo"])
        if new_sha != r["commit_sha"]:
            print(f"{r['repo']}: {r['commit_sha']} -> {new_sha}")
        r["commit_sha"] = new_sha
    manifest["fetched_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


def find_includes(text: str) -> tuple[list[str], list[str]]:
    """(same_repo_relative_includes, skipped_cross_repo_includes)."""
    same_repo, cross_repo = [], []
    for inc in INCLUDE_RE.findall(text):
        inc = inc.strip()
        (cross_repo if inc.startswith("~/") else same_repo).append(inc)
    return same_repo, cross_repo


class _FetchOutcome:
    """Collects what happened across the whole run, so article and include fetches share
    one classification: known-failure / unexpected-failure / known-issue-now-resolved."""

    def __init__(self, known_issue_paths: set[tuple[str, str]]):
        self.known_issue_paths = known_issue_paths
        self.failures: list[tuple[str, str, str]] = []       # unexpected: (repo, path, reason)
        self.known_hits: list[tuple[str, str, str]] = []      # expected: (repo, path, reason)
        self.resolved_known: set[tuple[str, str]] = set()     # known_issues that fetched fine this run

    def try_fetch(self, fetch_fn, repo: str, path: str, sha: str) -> bytes | None:
        url = raw_url(repo, sha, path)
        is_known = (repo, path) in self.known_issue_paths
        try:
            body = fetch_fn(url)
        except urllib.error.HTTPError as e:
            reason = f"HTTP {e.code}"
            if is_known:
                self.known_hits.append((repo, path, reason))
                print(f"KNOWN   {repo}  {path}  ({reason}) - see manifest known_issues")
            else:
                self.failures.append((repo, path, reason))
                print(f"FAILED  {repo}  {path}  ({reason})", file=sys.stderr)
            return None
        if is_known:
            self.resolved_known.add((repo, path))
        return body


def cmd_fetch(manifest: dict[str, Any], out_dir: Path, fetch_fn=fetch) -> int:
    shas = {r["repo"]: r["commit_sha"] for r in manifest["repos"]}
    known_issue_paths = {
        (i["repo"], i["source_path"]) for i in manifest.get("known_issues", [])
    }
    outcome = _FetchOutcome(known_issue_paths)

    articles_ok = 0
    includes_ok = 0
    total_bytes = 0
    skipped_cross_repo: set[str] = set()

    for article in manifest["articles"]:
        repo, path, sha = article["repo"], article["source_path"], shas[article["repo"]]
        body = outcome.try_fetch(fetch_fn, repo, path, sha)
        if body is None:
            continue

        out_path = out_dir / f"{article['id']}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(body)
        articles_ok += 1
        total_bytes += len(body)
        print(f"fetched {repo}  {path}  ({len(body):,} bytes) -> {out_path.relative_to(out_dir)}")

        text = body.decode("utf-8", errors="replace")
        same_repo_includes, cross_repo_includes = find_includes(text)
        skipped_cross_repo.update(cross_repo_includes)

        base_dir = posixpath.dirname(path)
        for inc in same_repo_includes:
            resolved = posixpath.normpath(posixpath.join(base_dir, inc))
            inc_body = outcome.try_fetch(fetch_fn, repo, resolved, sha)
            if inc_body is None:
                continue

            repo_slug = repo.split("/")[-1]
            inc_out_path = out_dir / "includes" / repo_slug / resolved
            inc_out_path.parent.mkdir(parents=True, exist_ok=True)
            inc_out_path.write_bytes(inc_body)
            includes_ok += 1
            total_bytes += len(inc_body)

    print()
    print("== Summary ==")
    print(f"articles fetched: {articles_ok}/{len(manifest['articles'])}")
    print(f"includes fetched: {includes_ok}")
    print(f"cross-repo includes skipped (not topic content; see module docstring): {len(skipped_cross_repo)}")
    print(f"total size: {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")
    print(f"known issues: {len(outcome.known_hits)}")
    print(f"unexpected failures: {len(outcome.failures)}")

    if outcome.known_hits:
        print(f"\nKNOWN ({len(outcome.known_hits)}):")
        for repo, path, reason in outcome.known_hits:
            print(f"  {repo}  {path}  ({reason})")

    if outcome.resolved_known:
        print("\nNOTICE: these known_issues entries fetched successfully this run - "
              "consider removing them from docs/corpus_manifest.json:")
        for repo, path in sorted(outcome.resolved_known):
            print(f"  {repo}  {path}")

    if outcome.failures:
        print(f"\nFAILED ({len(outcome.failures)}):", file=sys.stderr)
        for repo, path, reason in outcome.failures:
            print(f"  {repo}  {path}  ({reason})", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fetch_corpus")
    p.add_argument("--resolve", action="store_true", help="re-resolve each repo's current commit SHA, then stop")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    if args.resolve:
        return cmd_resolve(manifest, args.manifest)
    return cmd_fetch(manifest, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
