"""Append-only JSONL record of every tool call: what was asked, what came back, how long.

One line per call so it greps, tails and loads into pandas without ceremony. Day 6 mirrors
this into Postgres; the JSONL stays as the local source of truth.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PATH = "audit_log.jsonl"

# ContextVars, not env vars: an env var is process-global, so two concurrent agent runs
# (threads, or future asyncio tasks) would clobber each other's state. A ContextVar gives
# each thread (and each asyncio task) its own value with no extra plumbing.
_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("cloudguard_run_id", default=None)
# None means "no active run context at all" (propose_lock must reject outright). An empty
# set means a run is active but list_unlocked_resource_groups hasn't succeeded in it yet.
_current_unlocked_rg_ids: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "cloudguard_unlocked_rg_ids", default=None
)


def current_run_id() -> str | None:
    return _current_run_id.get()


def current_unlocked_resource_group_ids() -> set[str] | None:
    return _current_unlocked_rg_ids.get()


def record_unlocked_resource_group_ids(ids: Iterable[str]) -> None:
    """Called by list_unlocked_resource_groups on success. A no-op outside an active run -
    there's nothing to authorize against, and propose_lock rejects that case separately."""
    current = _current_unlocked_rg_ids.get()
    if current is not None:
        current.update(ids)


@contextlib.contextmanager
def run_id_scope(run_id: str):
    """Every audit_log.record() call made (directly or via tools._audited) while this
    context is active, in this thread/task, is tagged with run_id. Also opens a fresh,
    empty "unlocked resource groups seen this run" set, scoped and isolated the same way -
    a run never sees another run's (or no run's) state."""
    run_token = _current_run_id.set(run_id)
    rg_token = _current_unlocked_rg_ids.set(set())
    try:
        yield
    finally:
        _current_run_id.reset(run_token)
        _current_unlocked_rg_ids.reset(rg_token)


def log_path() -> Path:
    return Path(os.environ.get("CLOUDGUARD_AUDIT_LOG", DEFAULT_PATH))


def record(
    *,
    tool: str,
    args: dict[str, Any],
    backend: str,
    ok: bool,
    duration_ms: float,
    result_count: int,
    result_ids: list[str],
    error: str | None = None,
    run_id: str | None = None,
    grounded_ids: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "call_id": str(uuid.uuid4()),
        "run_id": run_id or _current_run_id.get(),
        "tool": tool,
        "args": args,
        "backend": backend,
        "ok": ok,
        "duration_ms": round(duration_ms, 2),
        "result_count": result_count,
        "result_ids": result_ids,
        "grounded_ids": grounded_ids,
        "error": error,
        **(extra or {}),
    }
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry
