"""Remediation proposals: propose_lock drafts a lock request for a human to approve.

Nothing here ever calls Azure. propose_lock writes a proposal record; decide() records a
human's approval or rejection. The two are deliberately separate: decide() is not a tool,
is never registered in TOOLS/TOOL_SPECS, and nothing in this module executes anything
against a subscription - approving a proposal only changes what's in proposals.jsonl.

Storage is event-based and append-only: a `created` event, then at most one `decided`
event. Current status is derived by replaying a proposal's events, so history is never
overwritten (mirrors audit_log.py's JSONL convention).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cloudguard import audit_log
from cloudguard.backends import Backend, validate_resource_id
from cloudguard.models import ToolResult
from cloudguard.untrusted import scan_fields

DEFAULT_PATH = "proposals.jsonl"
LOCK_LEVEL = "CanNotDelete"  # fixed; ReadOnly can break running workloads and is never offered
MAX_REASON_LENGTH = 1000

# A bare resource group ID only - no /providers/... suffix. propose_lock is specifically
# about resource groups, so a sub-resource ID (syntactically valid ARM ID, wrong shape for
# this tool) is rejected here, distinct from validate_resource_id's general syntax check.
_RG_ONLY_PATTERN = re.compile(r"^/subscriptions/[0-9a-fA-F-]{36}/resourceGroups/[-\w.()]{1,90}$", re.IGNORECASE)


class ProposalRejected(ValueError):
    """propose_lock's own business-rule rejections - not an ARM-ID-syntax problem."""


def log_path() -> Path:
    return Path(os.environ.get("CLOUDGUARD_PROPOSALS", DEFAULT_PATH))


def _append_event(event: dict[str, Any]) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


def _read_events() -> list[dict[str, Any]]:
    path = log_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_proposals() -> list[dict[str, Any]]:
    """Every proposal's current state, replayed from its events, in first-created order.
    The UI reads this fresh on every render, so decisions persist across restarts."""
    order: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for e in _read_events():
        pid = e["proposal_id"]
        if pid not in by_id:
            order.append(pid)
            by_id[pid] = {}
        by_id[pid].update({k: v for k, v in e.items() if k != "event"})
    return [by_id[pid] for pid in order]


def get_proposal(proposal_id: str) -> dict[str, Any] | None:
    """Replays every event for this proposal_id, in order. The `created` event supplies the
    base record; a later `decided` event overlays status/decided_by/decided_at/note."""
    return next((p for p in list_proposals() if p["proposal_id"] == proposal_id), None)


def _find_existing_proposal_this_run(run_id: str | None, resource_group_id: str) -> dict[str, Any] | None:
    normalized = resource_group_id.lower()
    created_ids = [
        e["proposal_id"] for e in _read_events()
        if e.get("event") == "created" and e.get("run_id") == run_id
        and e.get("resource_group_id", "").lower() == normalized
    ]
    return get_proposal(created_ids[-1]) if created_ids else None


def _suggested_command(resource_group_id: str) -> str:
    rg_name = resource_group_id.rsplit("/", 1)[-1]
    quoted_name = shlex.quote(rg_name)
    return f"az lock create -n cg-proposed-lock -t {LOCK_LEVEL} -g {quoted_name}  # proposal only — not executed"


def _create_proposal(run_id: str | None, resource_group_id: str, reason: str, backend_name: str) -> dict[str, Any]:
    event = {
        "event": "created",
        "proposal_id": str(uuid.uuid4()),
        "run_id": run_id,
        "backend": backend_name,
        "resource_group_id": resource_group_id,
        "lock_level": LOCK_LEVEL,
        # The reason is model-authored free text - untrusted, stored verbatim, flagged.
        "reason": reason,
        "reason_untrusted_text_flags": scan_fields(reason, None),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "suggested_command": _suggested_command(resource_group_id),
    }
    _append_event(event)
    return {k: v for k, v in event.items() if k != "event"}


def propose_lock(backend: Backend, resource_group_id: str, reason: str) -> ToolResult:
    rg_id = validate_resource_id(resource_group_id)

    if not _RG_ONLY_PATTERN.match(rg_id):
        raise ProposalRejected(f"propose_lock only accepts a resource group ID, not a sub-resource: {resource_group_id!r}")
    if len(reason) > MAX_REASON_LENGTH:
        raise ProposalRejected(f"reason must be at most {MAX_REASON_LENGTH} characters (got {len(reason)})")

    unlocked_ids = audit_log.current_unlocked_resource_group_ids()
    if unlocked_ids is None:
        raise ProposalRejected("propose_lock requires an active agent run")
    if rg_id.lower() not in {i.lower() for i in unlocked_ids}:
        raise ProposalRejected(
            f"{resource_group_id} was not returned by list_unlocked_resource_groups this run; "
            "propose_lock only accepts resource groups already confirmed unlocked this run."
        )

    run_id = audit_log.current_run_id()
    existing = _find_existing_proposal_this_run(run_id, rg_id)
    proposal = existing if existing is not None else _create_proposal(run_id, rg_id, reason, backend.name)
    return ToolResult(tool="propose_lock", ok=True, backend=backend.name, count=1, data=proposal)


def decide(proposal_id: str, decision: str, decided_by: str, note: str = "") -> dict[str, Any]:
    if decision not in ("approved", "rejected"):
        raise ValueError(f"decision must be 'approved' or 'rejected', got {decision!r}")
    if not decided_by or not decided_by.strip():
        raise ValueError("decided_by is required")

    current = get_proposal(proposal_id)
    if current is None:
        raise ValueError(f"no such proposal: {proposal_id!r}")
    if current["status"] != "pending":
        raise ValueError(f"proposal {proposal_id} is already {current['status']}; cannot decide again")

    _append_event({
        "event": "decided",
        "proposal_id": proposal_id,
        "status": decision,
        "decided_by": decided_by,
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
    })
    return get_proposal(proposal_id)
