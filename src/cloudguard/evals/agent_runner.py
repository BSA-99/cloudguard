"""Runs the agent golden set (mock backend, real or injected LLM) and writes
evals/results/agent_<UTC timestamp>.json. Protocol: "Agent eval protocol (pre-registered)" in
evals/RESULTS.md. Scoring is in agent_eval.py; this module only executes, records and reports.

Each run executes in its own spawned child process with its own temporary proposals file and audit
log, so (a) pending proposals can't leak between runs and (b) a hung run can be killed. Timing is
recorded twice: `run_ms` inside the child around run_agent only, and `total_ms` by the parent from
before the child starts until it exits (process startup included). The 300 s timeout (RUN_TIMEOUT_S) is the parent's
view of the child.

Only the full selection (every golden case, 3 runs) is an "official" run; anything else is a smoke
run and is written to agent_smoke_<ts>.json, which is git-ignored and not reported.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import importlib
import json
import math
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cloudguard.evals import agent_golden
from cloudguard.evals.agent_eval import (
    CHECKS, FROZEN_GOLDEN_SHA256, INFRA_STATUSES, RUN_TIMEOUT_S, RunRecord, RunScore, build_report, load_prices, score_run,
)
from cloudguard.evals.agent_golden import AgentGoldenCase
from cloudguard.evals.retrieval import load_golden
from cloudguard.models import CorpusManifest, GroundTruth
from cloudguard.rag import index as rag_index

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = ROOT / "evals" / "results"
DEFAULT_PRICES_PATH = ROOT / "evals" / "prices.json"
DEFAULT_SNAPSHOT = ROOT / "mock" / "subscription_snapshot.json"
DEFAULT_RUNS = 3
DEFAULT_DELAY_S = 1.0
RUN_ORDER = "run-major"  # run 1 of every case, then run 2, then run 3: an outage hits different cases
DEFAULT_LLM_FACTORY = "cloudguard.evals.agent_runner:default_llm_factory"

Executor = Callable[[dict[str, Any]], RunRecord]


def default_llm_factory(case_id: str, run_index: int):
    """Production client with its production default retry settings; the args are for test factories."""
    from cloudguard.llm import GeminiClient

    return GeminiClient()


def _load_factory(path: str) -> Callable[..., Any]:
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)


# --- one run -----------------------------------------------------------------------------------

@contextlib.contextmanager
def _isolated_env(tmp: str):
    keys = ("CLOUDGUARD_PROPOSALS", "CLOUDGUARD_AUDIT_LOG")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["CLOUDGUARD_PROPOSALS"] = str(Path(tmp) / "proposals.jsonl")
    os.environ["CLOUDGUARD_AUDIT_LOG"] = str(Path(tmp) / "audit_log.jsonl")
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_one(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one run in the CURRENT process (the child calls this; tests may too). Returns a RunRecord
    dict. Never raises: any failure becomes an error record."""
    from cloudguard import proposals
    from cloudguard.agent import run_agent
    from cloudguard.backends import MockBackend
    from cloudguard.llm import LLMError

    rec: dict[str, Any] = {"case_id": spec["case_id"], "run_index": spec["run_index"], "status": "error"}
    with tempfile.TemporaryDirectory() as tmp, _isolated_env(tmp):
        try:
            llm = _load_factory(spec["llm_factory"])(spec["case_id"], spec["run_index"])
            rec["model"] = str(getattr(llm, "model", llm.name))
            backend = MockBackend(spec["snapshot"])
            start = time.perf_counter()
            try:
                run = run_agent(llm, backend, spec["request"], max_steps=spec["max_steps"])
            finally:
                rec["run_ms"] = (time.perf_counter() - start) * 1000
            rec.update(
                status="ok" if run.stopped_reason == "final_answer" else "max_steps",
                final_answer=run.final_answer, stopped_reason=run.stopped_reason, steps=run.steps,
                tool_calls=[{"name": tc["name"], "arguments": tc["arguments"], "result": tc["result"].model_dump(mode="json")}
                            for tc in run.tool_calls],
                pending_proposals=[p["resource_group_id"] for p in proposals.list_proposals() if p["status"] == "pending"],
                tokens_in=run.total_input_tokens, tokens_out=run.total_output_tokens,
                tokens_thought=run.total_thought_tokens,
                attempts=run.total_attempts, llm_ms=run.total_latency_ms, llm_call_ms=run.llm_call_ms,
            )
        except LLMError as e:
            rec.update(error=f"LLMError: {e.status}: {e.message}", infrastructure=e.status in INFRA_STATUSES)
        except Exception as e:  # a harness or agent bug is a failed run, not a crash of the eval
            rec.update(error=f"{type(e).__name__}: {e}", traceback_tail=traceback.format_exc().splitlines()[-3:])
    return rec


def _child_main(spec: dict[str, Any], out_path: str) -> None:
    Path(out_path).write_text(json.dumps(run_one(spec)), encoding="utf-8")


def subprocess_executor(spec: dict[str, Any], timeout_s: float = RUN_TIMEOUT_S) -> RunRecord:
    """One run in a spawned child, killed at timeout_s (the parent's view). total_ms covers process startup."""
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "record.json"
        start = time.perf_counter()
        proc = ctx.Process(target=_child_main, args=(spec, str(out)))
        proc.start()
        proc.join(timeout_s)
        timed_out = proc.is_alive()
        if timed_out:
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()
        total_ms = (time.perf_counter() - start) * 1000

        base = {"case_id": spec["case_id"], "run_index": spec["run_index"]}
        if timed_out:
            data = {**base, "status": "timeout", "error": f"killed after {timeout_s:g}s wall-clock"}
        elif out.exists():
            data = json.loads(out.read_text(encoding="utf-8"))
        else:
            data = {**base, "status": "error", "error": f"child exited with code {proc.exitcode} without a record"}
    return RunRecord.model_validate({**data, "total_ms": total_ms})


# --- the whole eval --------------------------------------------------------------------------

def run_eval_agent(
    cases: list[AgentGoldenCase], runs: int, *, executor: Executor, make_spec: Callable[[AgentGoldenCase, int], dict[str, Any]],
    delay_s: float = 0.0, sleep: Callable[[float], None] = time.sleep, on_run: Callable[[RunRecord], None] | None = None,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    first = True
    for run_index in range(1, runs + 1):  # run-major
        for case in cases:
            if not first and delay_s > 0:
                sleep(delay_s)
            first = False
            rec = executor(make_spec(case, run_index))
            records.append(rec)
            if on_run:
                on_run(rec)
    return records


# --- estimates, provenance, redaction ---------------------------------------------------------

def estimate_tokens(cases: list[AgentGoldenCase], runs: int) -> dict[str, Any]:
    """A rough floor, not a measurement: (system prompt + tool specs + request) / 4 chars per token,
    for one LLM call. A run makes several calls and each carries the earlier tool results."""
    from cloudguard.agent import SYSTEM_PROMPT
    from cloudguard.tools import TOOL_SPECS

    fixed = len(SYSTEM_PROMPT) + len(json.dumps(TOOL_SPECS))
    per_call = [math.ceil((fixed + len(c.request)) / 4) for c in cases]
    one_pass = sum(per_call)
    return {"per_call_avg": round(one_pass / max(len(cases), 1)), "input_floor_2_calls": 2 * one_pass * runs,
            "input_floor_4_calls": 4 * one_pass * runs}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()


def git_state() -> tuple[str, bool]:
    return _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))


_KEY_SHAPE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


def redact(obj: Any, secrets: list[str]) -> Any:
    """Defensive: no API key may reach the results file, even inside an error message."""
    if isinstance(obj, str):
        for s in secrets:
            if s:
                obj = obj.replace(s, "[redacted]")
        return _KEY_SHAPE.sub("[redacted]", obj)
    if isinstance(obj, list):
        return [redact(x, secrets) for x in obj]
    if isinstance(obj, dict):
        return {k: redact(v, secrets) for k, v in obj.items()}
    return obj


# --- CLI command -------------------------------------------------------------------------------

def _select(cases: list[AgentGoldenCase], ids: list[str], limit: int | None) -> list[AgentGoldenCase]:
    if ids:
        unknown = [i for i in ids if i not in {c.id for c in cases}]
        if unknown:
            raise SystemExit(f"unknown case id(s): {unknown}")
        cases = [c for c in cases if c.id in set(ids)]
    return cases[:limit] if limit is not None else cases


def _print_report(report: dict[str, Any]) -> None:
    o = report["overall"]
    print(f"\nruns passed: {o['run_level']['text']}   cases consistent: {o['case_level']['text']}   "
          f"(consistent {o['consistent']}, flaky {o['flaky']}, failing {o['failing']})")
    for cat, v in report["by_category"].items():
        print(f"  {cat:16} runs {v['run_level']['text']:>6}   cases {v['case_level']['text']:>5}   "
              f"({v['consistent']} consistent, {v['flaky']} flaky, {v['failing']} failing)")
    print("checks (passed/applicable): " + ", ".join(f"{k} {v['text']}" for k, v in report["checks"].items()))
    f = report["failed_runs"]
    print(f"failed runs: {f['total']} ({f['infrastructure']} infrastructure, {f['other']} other); status counts {report['run_status']}")
    t, lat = report["tokens"], report["latency"]
    print(f"tokens: {t['in']} in / {t['out']} out / {t['thought']} thought   attempts: {report['attempts']['total']} "
          f"({report['attempts']['runs_with_retries']} runs had retries)")
    for name, key in (("inside child (run_agent)", "run_ms_inside_child"), ("parent total, incl. startup", "total_ms_parent")):
        m = lat[key]
        if m["median_ms"] is not None:
            print(f"latency {name}: median {m['median_ms']:.0f} ms, p95 {m['p95_ms']:.0f} ms")
    print(f"cost: {'$%.4f' % report['cost_usd'] if report['cost_usd'] is not None else report['cost_note']}")


def cmd_eval_agent(args, *, executor: Executor | None = None, llm_factory: str = DEFAULT_LLM_FACTORY,
                   sleep: Callable[[float], None] = time.sleep) -> int:
    try:
        gt = GroundTruth.model_validate_json(args.ground_truth.read_text(encoding="utf-8"))
        manifest = CorpusManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
        retrieval_golden = load_golden(args.retrieval_golden)
        golden_bytes = args.golden.read_bytes()
    except (OSError, ValueError) as e:
        print(f"cannot read inputs: {e}", file=sys.stderr)
        return 2

    from cloudguard.tools import TOOL_SPECS, TOOLS

    errors, all_cases = agent_golden.validate_cases(
        golden_bytes.decode("utf-8").splitlines(), gt, retrieval_golden, {a.id for a in manifest.articles},
        set(TOOLS) & {s["name"] for s in TOOL_SPECS})
    if errors:
        print(f"golden set is invalid ({len(errors)} problem(s)); run `cloudguard validate-agent-golden`", file=sys.stderr)
        return 2

    cases = _select(all_cases, args.case, args.limit)
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2
    runs = args.runs
    n_runs = len(cases) * runs
    official = not args.case and args.limit is None and runs == DEFAULT_RUNS
    golden_sha = hashlib.sha256(golden_bytes).hexdigest()
    commit, dirty = git_state()

    if args.dry_run:
        est = estimate_tokens(cases, runs)
        print(f"dry run: nothing is called, no API key needed\n"
              f"cases: {len(cases)}  runs per case: {runs}  total runs: {n_runs}  ({'OFFICIAL' if official else 'smoke'} selection)\n"
              f"per category: {dict(sorted(collections.Counter(c.category for c in cases).items()))}\n"
              f"timeout per run: {RUN_TIMEOUT_S}s   delay between runs: {args.delay:g}s (adds at least {max(n_runs - 1, 0) * args.delay:.0f}s)\n"
              f"estimated input tokens (rough floor, chars/4 of prompt + tool specs + request; ~{est['per_call_avg']} per LLM call): "
              f"{est['input_floor_2_calls']:,} if each run makes 2 calls, {est['input_floor_4_calls']:,} if 4. "
              f"Real input grows with each tool result; output tokens are not estimated.\n"
              f"official-run guards: working tree {'DIRTY' if dirty else 'clean'}; golden sha256 "
              f"{'matches' if golden_sha == FROZEN_GOLDEN_SHA256 else 'DOES NOT MATCH'} the pre-registered one")
        return 0

    if official and (dirty or golden_sha != FROZEN_GOLDEN_SHA256):
        why = "the working tree is not clean" if dirty else "the golden set differs from the pre-registered, frozen one"
        print(f"refusing an official run: {why}. (Smoke runs, using --case/--limit/--runs, are exempt.)", file=sys.stderr)
        return 2

    if executor is None and llm_factory == DEFAULT_LLM_FACTORY:
        with contextlib.suppress(ImportError):
            from dotenv import load_dotenv

            load_dotenv()
        if not os.environ.get("GEMINI_API_KEY"):
            print("GEMINI_API_KEY is not set (env var or .env file)", file=sys.stderr)
            return 2

    print(f"{'OFFICIAL' if official else 'smoke'} run: {len(cases)} cases x {runs} runs = {n_runs} runs, order {RUN_ORDER}")
    case_by_id = {c.id: c for c in cases}
    exec_fn = executor or subprocess_executor
    article_urls = {a.id: a.learn_url for a in manifest.articles}

    def make_spec(case: AgentGoldenCase, run_index: int) -> dict[str, Any]:
        return {"case_id": case.id, "run_index": run_index, "request": case.request, "max_steps": args.max_steps,
                "snapshot": str(args.snapshot), "llm_factory": llm_factory}

    done = 0

    def on_run(rec: RunRecord) -> None:
        nonlocal done
        done += 1
        s = score_run(case_by_id[rec.case_id], rec, article_urls)
        verdict = "pass" if s.passed else "FAIL " + (", ".join(s.failed_checks) or rec.status)
        extra = f" [{rec.status}{': ' + rec.error if rec.error else ''}]" if rec.status != "ok" else ""
        print(f"[{done}/{n_runs}] {rec.case_id} run {rec.run_index}: {verdict}{extra}", flush=True)

    started = time.perf_counter()
    records = run_eval_agent(cases, runs, executor=exec_fn, make_spec=make_spec, delay_s=args.delay, sleep=sleep, on_run=on_run)
    scores = [score_run(case_by_id[r.case_id], r, article_urls) for r in records]
    report = build_report(cases, records, scores, load_prices(args.prices))
    _print_report(report)

    models = sorted({r.model for r in records if r.model})
    index_meta_path = args.index_dir / "meta.json"
    out = {
        "kind": "agent_eval",
        "official": official,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "git_dirty": dirty,
        "golden": {"path": str(args.golden.relative_to(ROOT)) if args.golden.is_relative_to(ROOT) else args.golden.name,
                   "sha256": golden_sha, "frozen_sha256_matches": golden_sha == FROZEN_GOLDEN_SHA256, "num_cases": len(all_cases)},
        "model": models[0] if len(models) == 1 else models,
        "retrieval": {"mode": rag_index.DEFAULT_MODE,
                      "index_meta": json.loads(index_meta_path.read_text(encoding="utf-8")) if index_meta_path.exists() else None,
                      "corpus_repos": [r.model_dump() for r in manifest.repos]},
        "config": {"runs": runs, "max_steps": args.max_steps, "timeout_s": RUN_TIMEOUT_S, "delay_s": args.delay,
                   "order": RUN_ORDER, "selection": {"cases": args.case or None, "limit": args.limit},
                   "wall_seconds": round(time.perf_counter() - started, 1)},
        "runs": [r.model_dump() for r in records],
        "scores": [{"case_id": s.case_id, "run_index": s.run_index, "passed": s.passed, "failed_run": s.failed_run,
                    "checks": s.checks, "recall_fraction": s.recall_fraction, "false_positive_count": s.false_positive_count,
                    "hallucinated_count": s.hallucinated_count, "article_fraction": s.article_fraction} for s in scores],
        "report": report,
    }
    out = redact(out, [os.environ.get("GEMINI_API_KEY", "")])
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.results_dir / f"agent_{'' if official else 'smoke_'}{stamp}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")
    return 0
