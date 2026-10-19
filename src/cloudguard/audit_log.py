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
from typing import Any

DEFAULT_PATH = "audit_log.jsonl"

# A ContextVar, not an env var: an env var is process-global, so two concurrent agent runs
# (threads, or future asyncio tasks) would clobber each other's run_id. A ContextVar gives
# each thread (and each asyncio task) its own value with no extra plumbing.
_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("cloudguard_run_id", default=None)


def current_run_id() -> str | None:
    return _current_run_id.get()


@contextlib.contextmanager
def run_id_scope(run_id: str):
    """Every audit_log.record() call made (directly or via tools._audited) while this
    context is active, in this thread/task, is tagged with run_id."""
    token = _current_run_id.set(run_id)
    try:
        yield
    finally:
        _current_run_id.reset(token)


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
    }
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry
