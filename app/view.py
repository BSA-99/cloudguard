"""All data shaping for streamlit_app.py, as plain functions with no Streamlit import -
so they're unit-testable without a running app. Every function here must pass untrusted
text (tag values, the model's answer, a proposal's reason) through unchanged: it's
streamlit_app.py's job to render it as plain text (st.text/st.code), never markdown/HTML,
but the view layer's job is to never mangle it in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from cloudguard.agent import AgentRun
from cloudguard.models import Finding


def group_findings(findings: list[Finding]) -> dict[str, list[Finding]]:
    """Groups by finding type, in first-seen order. Findings themselves (including
    untrusted_text_flags and evidence) pass through untouched."""
    grouped: dict[str, list[Finding]] = {}
    for f in findings:
        grouped.setdefault(f.type.value, []).append(f)
    return grouped


@dataclass
class GroundingBanner:
    status: Literal["ok", "failed", "not_checked"]
    ungrounded_ids: list[str]


def grounding_banner(run: AgentRun) -> GroundingBanner:
    if run.grounding_ok is None:
        return GroundingBanner(status="not_checked", ungrounded_ids=[])
    if run.grounding_ok:
        return GroundingBanner(status="ok", ungrounded_ids=[])
    return GroundingBanner(status="failed", ungrounded_ids=list(run.ungrounded_ids))


@dataclass
class CitationsBanner:
    status: Literal["ok", "failed", "not_checked"]
    cited_count: int
    ungrounded_urls: list[str]
    external_urls: list[str]


def citations_banner(run: AgentRun) -> CitationsBanner:
    """None means the check never ran (no final answer) and must never read as passing."""
    if run.citations_ok is None:
        return CitationsBanner("not_checked", 0, [], [])
    status: Literal["ok", "failed"] = "ok" if run.citations_ok else "failed"
    return CitationsBanner(status, len(run.cited_urls), list(run.ungrounded_urls), list(run.external_urls))


def split_proposals(proposals: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(pending, decided), each in the order list_proposals() returned them."""
    pending = [p for p in proposals if p["status"] == "pending"]
    decided = [p for p in proposals if p["status"] != "pending"]
    return pending, decided


def run_stats(run: AgentRun, model: str) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "model": model,
        "steps": run.steps,
        "stopped_reason": run.stopped_reason,
        "tokens": f"{run.total_input_tokens} in / {run.total_output_tokens} out",
        "attempts": run.total_attempts,
        "latency_ms": round(run.total_latency_ms),
    }
