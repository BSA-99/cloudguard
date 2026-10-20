"""Check that `cloudguard eval-retrieval --mode vector` still reproduces the frozen
vector-only baseline (evals/results/retrieval_20261020T011040Z.json) exactly.

This is the gate in evals/RESULTS.md: after the hybrid-retrieval refactor, vector mode must
reproduce the baseline's metrics before any hybrid number means anything. A mismatch means the
refactor changed vector behavior; stop and investigate, do not run hybrid.

  python -m cloudguard.evals.baseline_check                      # runs --mode vector, then compares
  python -m cloudguard.evals.baseline_check --candidate FILE     # compares an existing results file

Comparison is exact (==) on every compared value, per-query top-1 scores included. Nothing is
loosened silently: `--score-tol` exists for diagnosing float noise but defaults to 0, and the
report prints the largest score difference either way. Provenance fields that legitimately differ
between runs (timestamp, git_commit, index built_at) and any extra keys the candidate adds
(mode, vector cosines) are ignored.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

BASELINE_PATH = Path(__file__).resolve().parents[3] / "evals" / "results" / "retrieval_20261020T011040Z.json"

_BLOCK_KEYS = ("num_queries", "hit@1", "hit@3", "hit@5", "mrr", "section_hit@3", "section_eval_count")
_PER_QUERY_EXACT = (
    "query", "source", "category", "expected_articles", "hit_at_1", "hit_at_3", "hit_at_5",
    "reciprocal_rank", "top3_article_ids", "section_hit_at_3",
)


def compare_results(baseline: dict[str, Any], candidate: dict[str, Any], score_tol: float = 0.0) -> list[str]:
    """Every difference between the two results files, as human-readable lines. Empty = reproduces."""
    diffs: list[str] = []

    def eq(label: str, a: Any, b: Any) -> None:
        if a != b:
            diffs.append(f"{label}: baseline={a!r} candidate={b!r}")

    def close(label: str, a: float | None, b: float | None) -> None:
        if a is None or b is None:
            eq(label, a, b)
        elif abs(a - b) > score_tol:
            diffs.append(f"{label}: baseline={a!r} candidate={b!r} (|diff|={abs(a - b):.3g})")

    mode = candidate.get("mode", "vector")
    if mode != "vector":
        diffs.append(f"mode: candidate was run with mode={mode!r}, expected 'vector'")

    eq("k", baseline["k"], candidate.get("k"))
    eq("model_name", baseline["model_name"], candidate.get("model_name"))
    eq("index_meta.manifest_hash", baseline["index_meta"]["manifest_hash"], candidate.get("index_meta", {}).get("manifest_hash"))
    eq("index_meta.chunk_count", baseline["index_meta"]["chunk_count"], candidate.get("index_meta", {}).get("chunk_count"))

    for key in _BLOCK_KEYS:
        eq(f"overall.{key}", baseline["overall"].get(key), candidate.get("overall", {}).get(key))
    eq("by_source keys", list(baseline["by_source"]), list(candidate.get("by_source", {})))
    for source, block in baseline["by_source"].items():
        for key in _BLOCK_KEYS:
            eq(f"by_source[{source}].{key}", block.get(key), candidate.get("by_source", {}).get(source, {}).get(key))

    eq("misses_top3", baseline["misses_top3"], candidate.get("misses_top3"))
    for key, value in baseline["top1_score_stats"].items():
        close(f"top1_score_stats.{key}", value, candidate.get("top1_score_stats", {}).get(key))

    base_q, cand_q = baseline["per_query"], candidate.get("per_query", [])
    eq("per_query count", len(base_q), len(cand_q))
    for b, c in zip(base_q, cand_q):
        label = f"per_query[{b['query'][:50]!r}]"
        for key in _PER_QUERY_EXACT:
            eq(f"{label}.{key}", b.get(key), c.get(key))
        close(f"{label}.top1_score", b.get("top1_score"), c.get("top1_score"))

    base_o, cand_o = baseline["out_of_scope"], candidate.get("out_of_scope", [])
    eq("out_of_scope count", len(base_o), len(cand_o))
    for b, c in zip(base_o, cand_o):
        eq(f"out_of_scope[{b['query'][:50]!r}].query", b["query"], c.get("query"))
        close(f"out_of_scope[{b['query'][:50]!r}].top1_score", b["top1_score"], c.get("top1_score"))
    return diffs


def max_score_diff(baseline: dict[str, Any], candidate: dict[str, Any]) -> float:
    pairs = [(b.get("top1_score"), c.get("top1_score")) for b, c in zip(baseline["per_query"], candidate.get("per_query", []))]
    return max((abs(a - b) for a, b in pairs if a is not None and b is not None), default=0.0)


def _run_vector_mode() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, "-m", "cloudguard.cli", "eval-retrieval", "--mode", "vector", "--results-dir", tmp],
            check=True,
        )
        (path,) = sorted(Path(tmp).glob("retrieval_*.json"))
        return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m cloudguard.evals.baseline_check")
    p.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    p.add_argument("--candidate", type=Path, default=None, help="an existing results file; default: run --mode vector")
    p.add_argument("--score-tol", type=float, default=0.0, help="diagnostic only; the gate is exact (0)")
    args = p.parse_args(argv)

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8")) if args.candidate else _run_vector_mode()

    diffs = compare_results(baseline, candidate, score_tol=args.score_tol)
    print(f"baseline: {args.baseline.name}   max |per-query top-1 score diff| = {max_score_diff(baseline, candidate):.3g}")
    if diffs:
        print(f"MISMATCH ({len(diffs)}): vector mode does not reproduce the baseline. Stop and investigate before running hybrid.")
        for d in diffs:
            print(f"  {d}")
        return 1
    print("OK: vector mode reproduces the baseline exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
