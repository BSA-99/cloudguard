"""The five read-only tools the agent will call, plus their JSON-schema specs.

Design rules (each one is a likely interview question):
  * Every tool is a read. There is no tool that changes Azure. propose_lock arrives on
    Day 2 and only writes a *proposal* record, never an ARM call.
  * Findings come from deterministic queries, not from the model. The model's job is to
    choose tools, explain, and draft change requests, so it cannot "find" a resource that
    the query didn't return.
  * Every result carries exact resource IDs; get_resource lets the agent (and the eval
    harness) verify any ID it mentions. Unknown IDs return not-found, never a guess.
  * Every call, successful or not, is written to the audit log.
"""

from __future__ import annotations

import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from cloudguard import audit_log, docs_search
from cloudguard.backends import Backend, InvalidResourceId
from cloudguard.models import Finding, FindingType, Severity, ToolResult
from cloudguard.untrusted import scan_fields

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "docs" / "corpus.json"


def _audited(tool_name: str) -> Callable:
    def deco(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        @wraps(fn)
        def wrapper(backend: Backend, **kwargs: Any) -> ToolResult:
            start = time.perf_counter()
            try:
                result = fn(backend, **kwargs)
            except InvalidResourceId as e:
                result = ToolResult(tool=tool_name, ok=False, backend=backend.name, error=str(e))
            except Exception as e:  # surfaced to the agent as a failed tool call, never swallowed
                result = ToolResult(tool=tool_name, ok=False, backend=backend.name,
                                    error=f"{type(e).__name__}: {e}")
            ids = [f.resource_id for f in result.findings]
            if result.data and result.data.get("id"):
                ids.append(result.data["id"])
            audit_log.record(
                tool=tool_name, args=kwargs, backend=backend.name, ok=result.ok,
                duration_ms=(time.perf_counter() - start) * 1000,
                result_count=result.count, result_ids=ids, error=result.error,
            )
            return result
        return wrapper
    return deco


@_audited("list_unlocked_resource_groups")
def list_unlocked_resource_groups(backend: Backend) -> ToolResult:
    findings = [
        Finding(
            type=FindingType.UNLOCKED_RESOURCE_GROUP,
            resource_id=row["id"],
            name=row["name"],
            resource_group=row["name"],
            severity=Severity.MEDIUM,
            reason="No management lock at resource-group scope; the group and everything in it can be deleted in one action.",
            evidence={"location": row.get("location"), "tags": row.get("tags") or {}},
            untrusted_text_flags=scan_fields(row["name"], row.get("tags")),
        )
        for row in backend.unlocked_resource_groups()
    ]
    return ToolResult(tool="list_unlocked_resource_groups", ok=True, backend=backend.name,
                      findings=findings, count=len(findings))


@_audited("list_unattached_disks")
def list_unattached_disks(backend: Backend) -> ToolResult:
    findings = [
        Finding(
            type=FindingType.UNATTACHED_DISK,
            resource_id=row["id"],
            name=row["name"],
            resource_group=row.get("resourceGroup"),
            severity=Severity.LOW,
            reason="Managed disk is not attached to any VM but is still billed for its provisioned size.",
            evidence={k: row.get(k) for k in ("skuName", "diskSizeGB", "diskState", "location")}
                     | {"tags": row.get("tags") or {}},
            untrusted_text_flags=scan_fields(row["name"], row.get("tags")),
        )
        for row in backend.unattached_disks()
    ]
    return ToolResult(tool="list_unattached_disks", ok=True, backend=backend.name,
                      findings=findings, count=len(findings))


@_audited("find_weak_tls")
def find_weak_tls(backend: Backend) -> ToolResult:
    findings = []
    for row in backend.weak_tls_storage():
        tls = row.get("minimumTlsVersion") or ""
        unset = tls == ""
        findings.append(Finding(
            type=FindingType.WEAK_TLS,
            resource_id=row["id"],
            name=row["name"],
            resource_group=row.get("resourceGroup"),
            severity=Severity.MEDIUM if unset else Severity.HIGH,
            reason=("minimumTlsVersion is not set, so the enforced minimum is not explicit."
                    if unset else
                    f"Storage account allows {tls}; Azure Storage retired TLS 1.0/1.1 and TLS 1.2 is the supported minimum."),
            evidence={"minimumTlsVersion": tls or None, "location": row.get("location"),
                      "tags": row.get("tags") or {}},
            untrusted_text_flags=scan_fields(row["name"], row.get("tags")),
        ))
    return ToolResult(tool="find_weak_tls", ok=True, backend=backend.name,
                      findings=findings, count=len(findings))


@_audited("get_resource")
def get_resource(backend: Backend, resource_id: str) -> ToolResult:
    row = backend.get_resource(resource_id)
    if row is None:
        return ToolResult(tool="get_resource", ok=False, backend=backend.name,
                          error=f"resource not found: {resource_id}")
    flags = scan_fields(row.get("name"), row.get("tags"))
    return ToolResult(tool="get_resource", ok=True, backend=backend.name, count=1,
                      data=row | {"_untrusted_text_flags": flags})


@_audited("search_docs")
def search_docs(backend: Backend, query: str, k: int = 3) -> ToolResult:
    hits = docs_search.search(CORPUS_PATH, query, k=k)
    return ToolResult(tool="search_docs", ok=True, backend=backend.name,
                      data={"query": query, "results": hits}, count=len(hits))


# Single registry: CLI dispatch, Day-2 agent loop and tests all read from here.
TOOLS: dict[str, Callable[..., ToolResult]] = {
    "list_unlocked_resource_groups": list_unlocked_resource_groups,
    "list_unattached_disks": list_unattached_disks,
    "find_weak_tls": find_weak_tls,
    "get_resource": get_resource,
    "search_docs": search_docs,
}

DETECTORS = ("list_unlocked_resource_groups", "list_unattached_disks", "find_weak_tls")

# Provider-neutral JSON schemas (the shape both the Claude API and Azure OpenAI accept for
# tool definitions, modulo the wrapper key). Descriptions are what the model reads to
# choose a tool, so they say what the tool returns and when to use it.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_unlocked_resource_groups",
        "description": "List resource groups that have no management lock at resource-group scope. "
                       "Returns findings with exact resource IDs. Read-only.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_unattached_disks",
        "description": "List managed disks in the Unattached state (billed but not used by any VM). "
                       "Returns findings with exact resource IDs, SKU and size. Read-only.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "find_weak_tls",
        "description": "List storage accounts whose minimum TLS version is below 1.2 or unset. "
                       "Returns findings with exact resource IDs. Read-only.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_resource",
        "description": "Fetch one resource by its full Azure resource ID to verify it exists and inspect "
                       "its properties. Returns not-found for unknown IDs. Read-only.",
        "input_schema": {
            "type": "object",
            "properties": {"resource_id": {"type": "string", "description": "Full ARM resource ID, e.g. /subscriptions/<guid>/resourceGroups/<rg>"}},
            "required": ["resource_id"],
        },
    },
    {
        "name": "search_docs",
        "description": "Search public Microsoft documentation for guidance to cite in an explanation or "
                       "change request. Returns titles, URLs and summaries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            },
            "required": ["query"],
        },
    },
]
