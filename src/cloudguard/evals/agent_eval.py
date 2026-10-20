"""Pure scoring for the agent eval (protocol: "Agent eval protocol (pre-registered)" in
evals/RESULTS.md). Nothing here runs an agent, calls a model or touches the filesystem except
`load_prices`. Every function takes recorded data (`RunRecord`s, the golden cases), so a results
file can be re-scored later from the file alone.

Checks per run, all rule-based; a check is applicable only for the cases the protocol names, and a
failed run (error, max_steps, timeout) fails every applicable check:
  tool_selection  every case          required_tools subset of called, forbidden_tools disjoint from called
  recall          expected_ids        every expected ID is among the IDs extracted from the answer
  false_positives forbidden_ids       no forbidden ID among the extracted IDs
  hallucinated    every case         no extracted ID that production grounding doesn't ground
  proposals       every case         pending proposals at the end == expected_proposals exactly
  citations       every case         grounding.check_citations is ok for the answer
  docs            expected_articles   at least one expected article retrieved by search_docs AND cited (article level)
  abstain         should_abstain      no learn.microsoft.com URL in the answer, no tool other than search_docs called
IDs are compared as exact extracted IDs (grounding.extract_resource_ids), never as substrings.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from cloudguard.evals.agent_golden import EXPECTED_COUNTS, AgentGoldenCase
from cloudguard.grounding import LEARN_HOST, check_grounding, extract_resource_ids, extract_urls, normalize_doc_url
from cloudguard.models import ToolResult

RUN_TIMEOUT_S = 300  # wall-clock, the parent's view of the child; a runaway guard (120 s until the 2026-10-20 addendum in evals/RESULTS.md)
FROZEN_GOLDEN_SHA256 = "d656d074224bb27de58231a0d6dcecee49376cabeeb29d6cd9678874d346c156"
# The two statuses GeminiClient retries (429, 503). One that survives the retries is an
# infrastructure failure; every other LLM or runtime error is a plain "error".
INFRA_STATUSES = frozenset({"RESOURCE_EXHAUSTED", "UNAVAILABLE"})
CHECKS = ("tool_selection", "recall", "false_positives", "hallucinated", "proposals", "citations", "docs", "abstain")
SEARCH = "search_docs"

Status = Literal["ok", "error", "max_steps", "timeout"]


class RunRecord(BaseModel):
    """What one run stores in the results file: enough to recompute every metric."""

    case_id: str
    run_index: int
    status: Status
    infrastructure: bool = False
    error: str | None = None
    model: str | None = None
    final_answer: str | None = None
    stopped_reason: str | None = None
    steps: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)  # {"name", "arguments", "result": ToolResult dump}
    pending_proposals: list[str] = Field(default_factory=list)  # resource group IDs with a pending proposal at the end
    tokens_in: int = 0
    tokens_out: int = 0  # visible output tokens only
    tokens_thought: int = 0  # thinking tokens, reported separately by the API; billed at the output rate
    attempts: int = 0  # LLM calls made including retries (AgentRun.total_attempts)
    llm_ms: float = 0.0  # LLM-call time only
    llm_call_ms: list[float] = Field(default_factory=list)  # per-call wall time, in call order (extra detail)
    run_ms: float = 0.0  # measured in the child around run_agent only
    total_ms: float = 0.0  # measured by the parent: process start to exit, startup included


@dataclass
class RunScore:
    case_id: str
    run_index: int
    failed_run: bool
    checks: dict[str, bool | None] = field(default_factory=dict)  # None = not applicable
    recall_fraction: float | None = None
    false_positive_count: int | None = None
    hallucinated_count: int = 0
    article_fraction: float | None = None

    @property
    def passed(self) -> bool:
        return all(v for v in self.checks.values() if v is not None)

    @property
    def failed_checks(self) -> list[str]:
        return [k for k, v in self.checks.items() if v is False]


def _norm_id(resource_id: str) -> str:
    return resource_id.rstrip("/").lower()


def _article_paths(urls: list[str]) -> set[str]:
    return {normalize_doc_url(u).article for u in urls}


def score_run(case: AgentGoldenCase, rec: RunRecord, article_urls: dict[str, str]) -> RunScore:
    """article_urls maps corpus article id -> its learn_url (from docs/corpus_manifest.json)."""
    failed = rec.status != "ok"
    answer = rec.final_answer or ""
    called = {tc["name"] for tc in rec.tool_calls}
    results = [ToolResult.model_validate(tc["result"]) for tc in rec.tool_calls]
    extracted = {_norm_id(i) for i in extract_resource_ids(answer)}
    grounding = check_grounding(answer, results)

    score = RunScore(case_id=rec.case_id, run_index=rec.run_index, failed_run=failed)
    raw: dict[str, bool | None] = {k: None for k in CHECKS}

    raw["tool_selection"] = set(case.required_tools) <= called and not (set(case.forbidden_tools) & called)

    if case.expected_ids:
        wanted = {_norm_id(i) for i in case.expected_ids}
        score.recall_fraction = len(wanted & extracted) / len(wanted)
        raw["recall"] = wanted <= extracted
    if case.forbidden_ids:
        forbidden = {_norm_id(i) for i in case.forbidden_ids}
        score.false_positive_count = len(forbidden & extracted)
        raw["false_positives"] = score.false_positive_count == 0

    score.hallucinated_count = len(grounding.ungrounded_ids)
    raw["hallucinated"] = score.hallucinated_count == 0
    raw["proposals"] = {_norm_id(i) for i in rec.pending_proposals} == {_norm_id(i) for i in case.expected_proposals}
    raw["citations"] = grounding.citations_ok

    if case.expected_articles:
        retrieved = _article_paths([
            r["url"] for tr in results if tr.tool == SEARCH and tr.ok and tr.data for r in tr.data.get("results", [])
        ])
        cited = {normalize_doc_url(u).article for u in extract_urls(answer) if normalize_doc_url(u).host == LEARN_HOST}
        satisfied = [
            a for a in case.expected_articles
            if a in article_urls and normalize_doc_url(article_urls[a]).article in retrieved
            and normalize_doc_url(article_urls[a]).article in cited
        ]
        score.article_fraction = len(satisfied) / len(case.expected_articles)
        raw["docs"] = bool(satisfied)

    if case.should_abstain:
        learn_cited = any(normalize_doc_url(u).host == LEARN_HOST for u in extract_urls(answer))
        raw["abstain"] = not learn_cited and called <= {SEARCH}

    score.checks = {k: (False if failed and v is not None else v) for k, v in raw.items()}
    return score


# --- aggregation ---------------------------------------------------------------------------

def classify_case(passes: int, runs: int) -> str:
    """consistent = every run passed (3/3 with 3 runs), failing = none did, flaky = in between."""
    return "consistent" if passes == runs else "failing" if passes == 0 else "flaky"


def percentile_nearest_rank(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(p / 100 * len(ordered)), 1) - 1]


def load_prices(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")).get("models", {}) if path.exists() else {}


def cost_usd(
    model: str | None, tokens_in: int, tokens_out: int, prices: dict[str, dict[str, Any]], tokens_thought: int = 0,
) -> float | None:
    """input x input_price + (output + thought) x output_price: thinking tokens are billed as output.
    Only if the model's entry has numeric prices AND a source URL AND a retrieval date; else None."""
    entry = prices.get(model or "")
    if not entry:
        return None
    nums = (entry.get("input_per_million_usd"), entry.get("output_per_million_usd"))
    if not all(isinstance(n, (int, float)) and not isinstance(n, bool) and n >= 0 for n in nums):
        return None
    if not str(entry.get("source_url") or "").strip() or not str(entry.get("retrieved") or "").strip():
        return None
    return (tokens_in * nums[0] + (tokens_out + tokens_thought) * nums[1]) / 1_000_000


def _kn(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "text": f"{k}/{n}"}


def build_report(
    cases: list[AgentGoldenCase], records: list[RunRecord], scores: list[RunScore],
    prices: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prices = prices or {}
    category_of = {c.id: c.category for c in cases}
    by_case: dict[str, list[RunScore]] = defaultdict(list)
    for s in scores:
        by_case[s.case_id].append(s)

    per_case = {}
    for cid, group in by_case.items():
        passes = sum(s.passed for s in group)
        per_case[cid] = {"category": category_of[cid], "passes": passes, "runs": len(group),
                         "label": classify_case(passes, len(group))}

    def view(selected: list[str] | None) -> dict[str, Any]:
        ids = [c for c in per_case if selected is None or category_of[c] == selected]
        runs = [s for s in scores if s.case_id in ids]
        labels = Counter(per_case[c]["label"] for c in ids)
        return {
            "run_level": _kn(sum(s.passed for s in runs), len(runs)),
            "case_level": _kn(labels["consistent"], len(ids)),
            "consistent": labels["consistent"], "flaky": labels["flaky"], "failing": labels["failing"],
        }

    categories = [c for c in EXPECTED_COUNTS if any(v["category"] == c for v in per_case.values())]
    categories += sorted({v["category"] for v in per_case.values()} - set(categories))

    check_totals = {}
    for check in CHECKS:
        applicable = [s.checks[check] for s in scores if s.checks.get(check) is not None]
        check_totals[check] = _kn(sum(applicable), len(applicable))

    failed = [s for s in scores if not s.passed]
    rec_by_key = {(r.case_id, r.run_index): r for r in records}
    infra = sum(1 for s in failed if rec_by_key[(s.case_id, s.run_index)].infrastructure)
    statuses = Counter(r.status for r in records)

    costs = [cost_usd(r.model, r.tokens_in, r.tokens_out, prices, r.tokens_thought) for r in records]
    cost_total = sum(costs) if records and all(c is not None for c in costs) else None
    tokens = {"in": sum(r.tokens_in for r in records), "out": sum(r.tokens_out for r in records),
              "thought": sum(r.tokens_thought for r in records)}

    def latency(attr: str) -> dict[str, float | None]:
        vals = [getattr(r, attr) for r in records]
        return {"median_ms": statistics.median(vals) if vals else None, "p95_ms": percentile_nearest_rank(vals, 95)}

    return {
        "overall": view(None),
        "by_category": {c: view(c) for c in categories},
        "per_case": per_case,
        "checks": check_totals,
        "failed_runs": {"total": len(failed), "infrastructure": infra, "other": len(failed) - infra},
        "run_status": dict(statuses),
        "tokens": tokens,
        "latency": {"run_ms_inside_child": latency("run_ms"), "total_ms_parent": latency("total_ms"),
                    "llm_ms": latency("llm_ms")},
        "attempts": {"total": sum(r.attempts for r in records), "runs_with_retries": sum(1 for r in records if r.attempts > max(r.steps, 1))},
        "cost_usd": cost_total,
        "cost_note": None if cost_total is not None else "not reported: price table empty or incomplete for the model (tokens only)",
    }
