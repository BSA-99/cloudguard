"""Command line entry point.

  cloudguard --mock tool list_unattached_disks
  cloudguard --mock tool get_resource --arg resource_id=/subscriptions/.../resourceGroups/cg-dev-sandbox
  cloudguard --mock audit                 # run all detectors, print findings
  cloudguard --mock check                 # run all detectors, score against ground_truth.json
  cloudguard snapshot -o mock/live.json   # live only: record the subscription for mock mode
  cloudguard llm-ping                     # agent extra: one test call to Gemini
  cloudguard --mock agent "audit this subscription" [--max-steps N]
  cloudguard build-index                  # agent extra: embed the RAG corpus into corpus/index/
  cloudguard eval-retrieval [--k 5]        # agent extra: score retrieval against evals/retrieval_golden.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cloudguard.agent import AgentRun, DEFAULT_MAX_STEPS, run_agent
from cloudguard.backends import MockBackend, ResourceGraphBackend
from cloudguard.evals import retrieval as eval_retrieval
from cloudguard.llm import GeminiClient, LLMClient, LLMError
from cloudguard.models import GroundTruth
from cloudguard.rag import index as rag_index
from cloudguard.tools import DETECTORS, REPO_ROOT, TOOL_SPECS, TOOLS

DEFAULT_SNAPSHOT = REPO_ROOT / "mock" / "subscription_snapshot.json"
DEFAULT_GROUND_TRUTH = REPO_ROOT / "ground_truth.json"


def make_backend(args: argparse.Namespace):
    if args.mock:
        return MockBackend(args.snapshot)
    sub = os.environ.get("AZURE_SUBSCRIPTION_ID")
    if not sub:
        sys.exit("Live mode needs AZURE_SUBSCRIPTION_ID (or pass --mock).")
    return ResourceGraphBackend(sub)


def score(found_ids: set[str], truth: GroundTruth) -> dict:
    """Detector-level precision/recall. IDs compared case-insensitively (ARM IDs are)."""
    expected = {i.resource_id.lower() for i in truth.expected_findings}
    negatives = {i.resource_id.lower() for i in truth.expected_negatives}
    found = {i.lower() for i in found_ids}
    tp, fp, fn = found & expected, found - expected, expected - found
    return {
        "expected": len(expected), "found": len(found),
        "true_positives": len(tp), "false_positives": sorted(fp), "missed": sorted(fn),
        "negatives_wrongly_flagged": sorted(found & negatives),
        "precision": round(len(tp) / len(found), 4) if found else 0.0,
        "recall": round(len(tp) / len(expected), 4) if expected else 0.0,
    }


def cmd_llm_ping() -> int:
    """One real call to the live LLM: sanity-checks the key, model and tool-call wiring."""
    tool_spec = next(s for s in TOOL_SPECS if s["name"] == "list_unattached_disks")
    messages = [{"role": "user", "content": "Check this subscription for unattached disks"}]

    try:
        client = GeminiClient()
    except RuntimeError as e:
        sys.exit(str(e))

    start = time.perf_counter()
    response = client.generate(messages, [tool_spec])
    latency_ms = (time.perf_counter() - start) * 1000

    print(f"model: {client.model}")
    print(f"response: {'tool_call' if response.tool_calls else 'text'}")
    if response.text:
        print(f"text: {response.text}")
    for call in response.tool_calls:
        print(f"tool_call: {call.name}({call.arguments})")
    print(f"usage: input_tokens={response.usage.input_tokens} output_tokens={response.usage.output_tokens}")
    print(f"attempts: {response.attempts}")
    print(f"latency_ms: {latency_ms:.0f} total, {latency_ms / response.attempts:.0f} per attempt")
    return 0


def _print_agent_run(run: AgentRun, llm: LLMClient) -> None:
    print("== Findings ==")
    if not run.findings:
        print("(none)")
    else:
        by_type: dict[str, list[Any]] = {}
        for f in run.findings:
            by_type.setdefault(f.type.value, []).append(f)
        for type_name, findings in by_type.items():
            print(f"\n{type_name} ({len(findings)})")
            for f in findings:
                flag = f"  !! untrusted text: {', '.join(f.untrusted_text_flags)}" if f.untrusted_text_flags else ""
                print(f"  [{f.severity.value:6}] {f.resource_id}{flag}")

    missing_detectors = [name for name, ran in run.coverage.items() if not ran]
    if missing_detectors:
        print(f"\nDetectors not run in this audit: {', '.join(missing_detectors)}")

    print("\n== Answer ==")
    print(run.final_answer or "(no final answer — stopped at max_steps)")

    print("\n== Grounding ==")
    if run.grounding_ok is None:
        print("not checked (no final answer)")
    elif run.grounding_ok:
        print("ok")
    else:
        print("FAILED")
        for rid in run.ungrounded_ids:
            print(f"  ungrounded: {rid}")

    print("\n== Proposals ==")
    seen_proposal_ids: set[str] = set()
    for tc in run.tool_calls:
        if tc["name"] != "propose_lock" or not tc["result"].ok:
            continue
        proposal = tc["result"].data
        if proposal["proposal_id"] in seen_proposal_ids:
            continue
        seen_proposal_ids.add(proposal["proposal_id"])
        print(f"  [{proposal['status']:9}] {proposal['resource_group_id']}")
        print(f"      {proposal['suggested_command']}")
    if not seen_proposal_ids:
        print("(none)")

    print("\n== Run stats ==")
    print(f"run_id: {run.run_id}")
    print(f"model: {getattr(llm, 'model', llm.name)}")
    print(f"steps: {run.steps}")
    print(f"stopped_reason: {run.stopped_reason}")
    print(f"tokens: {run.total_input_tokens} in / {run.total_output_tokens} out")
    print(f"attempts: {run.total_attempts}")
    print(f"latency_ms: {run.total_latency_ms:.0f}")


def cmd_agent(args: argparse.Namespace, backend, llm: LLMClient | None = None) -> int:
    if llm is None:
        try:
            llm = GeminiClient()
        except RuntimeError as e:
            sys.exit(str(e))

    try:
        run = run_agent(llm, backend, args.message, max_steps=args.max_steps)
    except LLMError as e:
        print(f"LLM call failed: {e.status} {e.message} (model: {e.model})")
        return 2

    _print_agent_run(run, llm)
    return 1 if (run.grounding_ok is False or run.stopped_reason == "max_steps") else 0


def cmd_build_index(args: argparse.Namespace) -> int:
    from cloudguard.rag.embed import FastEmbedder
    from cloudguard.rag.index import build_index

    embedder = FastEmbedder(model_name=args.embed_model)
    result = build_index(args.manifest, args.raw_dir, embedder, args.out_dir)

    print(f"model: {result['model_name']}")
    print(f"chunks: {result['chunk_count']}")
    print(f"built_at: {result['built_at']}")
    over_warn = result["over_warn_chunks"]
    if over_warn:
        for c in over_warn:
            print(f"OVER {result['warn_limit_words']} WORDS: {c['chunk_id']} ({c['word_count']} words)")
        print(f"WARNING: {len(over_warn)} chunk(s) exceed "
              f"{result['warn_limit_words']} words - close to {embedder.name}'s truncation point")
    print(f"wrote {args.out_dir}")
    return 0


def cmd_eval_retrieval(args: argparse.Namespace) -> int:
    from cloudguard.evals.retrieval import load_golden, run_eval
    from cloudguard.rag.embed import FastEmbedder

    golden = load_golden(args.golden)
    embedder = FastEmbedder()
    report = run_eval(golden, embedder, rag_index.DEFAULT_MANIFEST_PATH, rag_index.DEFAULT_INDEX_DIR, k=args.k)

    index_meta = json.loads((rag_index.DEFAULT_INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
    corpus_manifest = json.loads(rag_index.DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    ).stdout.strip()

    def _fmt(block) -> dict:
        return {
            "num_queries": block.num_queries, "hit@1": round(block.hit_at_1, 4),
            "hit@3": round(block.hit_at_3, 4), "hit@5": round(block.hit_at_5, 4),
            "mrr": round(block.mrr, 4),
            "section_hit@3": round(block.section_hit_at_3, 4) if block.section_hit_at_3 is not None else None,
            "section_eval_count": block.section_eval_count,
        }

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "k": args.k,
        "model_name": embedder.name,
        "index_meta": index_meta,
        "corpus_repos": corpus_manifest["repos"],
        "git_commit": git_commit,
        "overall": _fmt(report.overall),
        "by_source": {source: _fmt(block) for source, block in report.by_source.items()},
        "misses_top3": report.misses_top3,
        "top1_score_stats": report.top1_score_stats,
        "out_of_scope": report.out_of_scope,
        "per_query": report.per_query,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.results_dir / f"retrieval_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    o = report.overall
    print(f"queries: {o.num_queries}")
    print(f"hit@1={o.hit_at_1:.2f}  hit@3={o.hit_at_3:.2f}  hit@5={o.hit_at_5:.2f}  mrr={o.mrr:.3f}")
    if o.section_hit_at_3 is not None:
        print(f"section_hit@3={o.section_hit_at_3:.2f} (n={o.section_eval_count})")
    for source, block in report.by_source.items():
        print(f"  [{source}] n={block.num_queries}  hit@1={block.hit_at_1:.2f}  "
              f"hit@3={block.hit_at_3:.2f}  hit@5={block.hit_at_5:.2f}  mrr={block.mrr:.3f}")
    if report.misses_top3:
        print(f"\nQueries missing their expected article in top 3 ({len(report.misses_top3)}):")
        for m in report.misses_top3:
            print(f"  {m['query']!r} -> expected {m['expected_articles']}, got {m['top3_article_ids']}")

    stats = report.top1_score_stats
    if stats["min"] is not None:
        print(f"\ntop-1 score (in-scope): min={stats['min']:.4f}  median={stats['median']:.4f}  "
              f"max={stats['max']:.4f}")

    if report.out_of_scope:
        print(f"\nOut-of-scope queries ({len(report.out_of_scope)}):")
        for o in report.out_of_scope:
            score_str = f"{o['top1_score']:.4f}" if o["top1_score"] is not None else "n/a"
            print(f"  {o['query']!r} -> top1_score={score_str}")

    print(f"\nwrote {out_path}")
    return 0


def main(argv: list[str] | None = None, llm: LLMClient | None = None) -> int:
    p = argparse.ArgumentParser(prog="cloudguard")
    p.add_argument("--mock", action="store_true", default=os.environ.get("CLOUDGUARD_MODE") == "mock",
                   help="read from a saved JSON snapshot instead of Azure")
    p.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tool", help="call one tool")
    t.add_argument("name", choices=sorted(TOOLS))
    t.add_argument("--arg", action="append", default=[], help="key=value")

    sub.add_parser("audit", help="run every detector")

    c = sub.add_parser("check", help="run every detector and score against ground truth")
    c.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)

    s = sub.add_parser("snapshot", help="(live) record the subscription to JSON for mock mode")
    s.add_argument("-o", "--out", type=Path, required=True)

    sub.add_parser("llm-ping", help="send one test message to Gemini and print how it responded")

    ag = sub.add_parser("agent", help="run the tool-calling agent loop")
    ag.add_argument("message")
    ag.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)

    bi = sub.add_parser("build-index", help="embed docs/corpus_manifest.json's corpus into corpus/index/")
    bi.add_argument("--manifest", type=Path, default=rag_index.DEFAULT_MANIFEST_PATH)
    bi.add_argument("--raw-dir", type=Path, default=rag_index.DEFAULT_RAW_DIR)
    bi.add_argument("--out-dir", type=Path, default=rag_index.DEFAULT_INDEX_DIR)
    bi.add_argument("--embed-model", default=None, help="defaults to CLOUDGUARD_EMBED_MODEL or bge-small-en-v1.5")

    er = sub.add_parser("eval-retrieval", help="score rag.index.search() against evals/retrieval_golden.jsonl")
    er.add_argument("--k", type=int, default=5)
    er.add_argument("--golden", type=Path, default=eval_retrieval.DEFAULT_GOLDEN_PATH)
    er.add_argument("--results-dir", type=Path, default=eval_retrieval.DEFAULT_RESULTS_DIR)

    args = p.parse_args(argv)

    if args.cmd == "llm-ping":
        return cmd_llm_ping()

    if args.cmd == "build-index":
        return cmd_build_index(args)

    if args.cmd == "eval-retrieval":
        return cmd_eval_retrieval(args)

    if args.cmd == "snapshot":
        if args.mock:
            sys.exit("snapshot records a live subscription; drop --mock.")
        backend = make_backend(args)
        args.out.write_text(json.dumps(backend.snapshot(), indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
        return 0

    backend = make_backend(args)

    if args.cmd == "agent":
        return cmd_agent(args, backend, llm=llm)

    if args.cmd == "tool":
        kwargs = dict(a.split("=", 1) for a in args.arg)
        if "k" in kwargs:
            kwargs["k"] = int(kwargs["k"])
        print(TOOLS[args.name](backend, **kwargs).model_dump_json(indent=2))
        return 0

    results = [TOOLS[name](backend) for name in DETECTORS]
    if args.cmd == "audit":
        for r in results:
            print(f"\n== {r.tool} ({r.count}) ==")
            for f in r.findings:
                flag = f"  !! untrusted text: {', '.join(f.untrusted_text_flags)}" if f.untrusted_text_flags else ""
                print(f"[{f.severity.value:6}] {f.resource_id}{flag}")
        return 0

    truth = GroundTruth.model_validate_json(args.ground_truth.read_text(encoding="utf-8"))
    found = {f.resource_id for r in results for f in r.findings}
    report = score(found, truth)
    print(json.dumps(report, indent=2))
    return 0 if report["recall"] == 1.0 and not report["false_positives"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
