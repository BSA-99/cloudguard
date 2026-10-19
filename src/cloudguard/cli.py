"""Command line entry point.

  cloudguard --mock tool list_unattached_disks
  cloudguard --mock tool get_resource --arg resource_id=/subscriptions/.../resourceGroups/cg-dev-sandbox
  cloudguard --mock audit                 # run all detectors, print findings
  cloudguard --mock check                 # run all detectors, score against ground_truth.json
  cloudguard snapshot -o mock/live.json   # live only: record the subscription for mock mode
  cloudguard llm-ping                     # agent extra: one test call to Gemini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from cloudguard.backends import MockBackend, ResourceGraphBackend
from cloudguard.llm import GeminiClient
from cloudguard.models import GroundTruth
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


def main(argv: list[str] | None = None) -> int:
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

    args = p.parse_args(argv)

    if args.cmd == "llm-ping":
        return cmd_llm_ping()

    if args.cmd == "snapshot":
        if args.mock:
            sys.exit("snapshot records a live subscription; drop --mock.")
        backend = make_backend(args)
        args.out.write_text(json.dumps(backend.snapshot(), indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
        return 0

    backend = make_backend(args)

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
