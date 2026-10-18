"""Append-only JSONL record of every tool call: what was asked, what came back, how long.

One line per call so it greps, tails and loads into pandas without ceremony. Day 6 mirrors
this into Postgres; the JSONL stays as the local source of truth.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PATH = "audit_log.jsonl"


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
) -> dict[str, Any]:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "call_id": str(uuid.uuid4()),
        "run_id": run_id or os.environ.get("CLOUDGUARD_RUN_ID"),
        "tool": tool,
        "args": args,
        "backend": backend,
        "ok": ok,
        "duration_ms": round(duration_ms, 2),
        "result_count": result_count,
        "result_ids": result_ids,
        "error": error,
    }
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry
