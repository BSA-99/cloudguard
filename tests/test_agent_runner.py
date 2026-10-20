"""The agent-eval runner: selection, dry run, results file, isolation, timeout, crash, guards. Most tests use an
in-process executor with scripted FakeLLM factories (tests/agent_fakes.py); a few run the real spawned child
process (timeout, crash, startup timing, isolation across processes). No network, no real model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from cloudguard.evals import agent_golden as ag
from cloudguard.evals import agent_runner as ar
from cloudguard.evals.agent_eval import FROZEN_GOLDEN_SHA256, RunRecord

FAKES = "tests.agent_fakes:"
SUB = "00000000-0000-0000-0000-000000000000"
RG_DEV = f"/subscriptions/{SUB}/resourceGroups/cg-dev-sandbox"


def _spec(case_id="x", factory="scripted", run_index=1, request="r") -> dict:
    return {"case_id": case_id, "run_index": run_index, "request": request, "max_steps": 8,
            "snapshot": str(ar.DEFAULT_SNAPSHOT), "llm_factory": FAKES + factory}


def in_process(spec: dict) -> RunRecord:
    return RunRecord.model_validate({**ar.run_one(spec), "total_ms": 1.0})


def _args(tmp_path: Path, **kw) -> argparse.Namespace:
    base = dict(case=[], limit=None, runs=3, delay=0.0, dry_run=False, max_steps=8, golden=ag.DEFAULT_GOLDEN_PATH,
                ground_truth=ag.DEFAULT_GROUND_TRUTH_PATH, retrieval_golden=ag.DEFAULT_RETRIEVAL_GOLDEN_PATH,
                manifest=ag.DEFAULT_MANIFEST_PATH, snapshot=ar.DEFAULT_SNAPSHOT, prices=tmp_path / "no_prices.json",
                index_dir=tmp_path / "no_index", results_dir=tmp_path / "results")
    return argparse.Namespace(**(base | kw))


@pytest.fixture(autouse=True)
def clean_tree(monkeypatch):
    monkeypatch.setattr(ar, "git_state", lambda: ("a" * 40, False))


def _only_file(directory: Path) -> Path:
    (path,) = list(directory.glob("*.json"))
    return path


# --- dry run and selection ------------------------------------------------------------------

def _boom(*a, **k):
    raise AssertionError("must not be called in a dry run")


def test_dry_run_prints_counts_and_estimate_and_calls_nothing(tmp_path, capsys):
    code = ar.cmd_eval_agent(_args(tmp_path, dry_run=True), executor=_boom, llm_factory=FAKES + "nonexistent")
    out = capsys.readouterr().out
    assert code == 0
    assert "total runs: 99" in out and "OFFICIAL" in out and "nothing is called, no API key needed" in out
    assert "estimated input tokens" in out and "rough floor" in out and "output tokens are not estimated" in out
    assert "golden sha256 matches the pre-registered one" in out
    assert not (tmp_path / "results").exists()


def test_dry_run_with_selection_counts_runs_and_delay(tmp_path, capsys):
    ar.cmd_eval_agent(_args(tmp_path, dry_run=True, case=["sd-disks-1", "fa-1"], runs=2, delay=2.0), executor=_boom)
    out = capsys.readouterr().out
    assert "cases: 2  runs per case: 2  total runs: 4" in out and "smoke" in out and "at least 6s" in out
    ar.cmd_eval_agent(_args(tmp_path, dry_run=True, limit=5), executor=_boom)
    assert "cases: 5" in capsys.readouterr().out


def test_unknown_case_id_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="unknown case id"):
        ar.cmd_eval_agent(_args(tmp_path, case=["nope"], dry_run=True))


def test_estimate_is_deterministic_and_scales_with_runs():
    cases = ag.build_cases(*_golden_inputs())
    a, b = ar.estimate_tokens(cases, 1), ar.estimate_tokens(cases, 3)
    assert a == ar.estimate_tokens(cases, 1) and b["input_floor_2_calls"] == 3 * a["input_floor_2_calls"]
    assert a["input_floor_4_calls"] == 2 * a["input_floor_2_calls"]


def _golden_inputs():
    from cloudguard.evals.retrieval import load_golden
    from cloudguard.models import GroundTruth
    return GroundTruth.model_validate_json(ag.DEFAULT_GROUND_TRUTH_PATH.read_text()), load_golden(ag.DEFAULT_RETRIEVAL_GOLDEN_PATH)


# --- a smoke run, in process ------------------------------------------------------------------

def test_smoke_run_writes_a_smoke_file_with_the_expected_content(tmp_path, capsys):
    sleeps: list[float] = []
    args = _args(tmp_path, case=["sd-disks-1", "pl-named"], runs=2, delay=0.5)
    code = ar.cmd_eval_agent(args, executor=in_process, llm_factory=FAKES + "scripted", sleep=sleeps.append)

    assert code == 0
    path = _only_file(tmp_path / "results")
    assert path.name.startswith("agent_smoke_")
    data = json.loads(path.read_text())
    assert data["kind"] == "agent_eval" and data["official"] is False
    assert data["git_commit"] == "a" * 40 and data["git_dirty"] is False
    assert data["golden"]["sha256"] == FROZEN_GOLDEN_SHA256 and data["golden"]["frozen_sha256_matches"] is True
    assert data["config"]["runs"] == 2 and data["config"]["timeout_s"] == 300 and data["config"]["delay_s"] == 0.5
    assert data["config"]["order"] == "run-major" and data["config"]["selection"] == {"cases": ["sd-disks-1", "pl-named"], "limit": None}
    assert data["model"] == "fake" and data["retrieval"]["mode"] == "vector"
    assert [(r["case_id"], r["run_index"]) for r in data["runs"]] == [("sd-disks-1", 1), ("pl-named", 1), ("sd-disks-1", 2), ("pl-named", 2)]
    assert sleeps == [0.5, 0.5, 0.5]  # between runs, not before the first or after the last
    out = capsys.readouterr().out
    assert "smoke run: 2 cases x 2 runs = 4 runs" in out and "[4/4]" in out and "runs passed:" in out and "cost: not reported" in out

    first = data["runs"][0]
    assert first["tool_calls"][0]["name"] == "list_unattached_disks" and first["tokens_in"] == 20 and first["attempts"] == 3
    assert first["tokens_out"] == 7 and first["tokens_thought"] == 40  # recorded separately, not folded into tokens_out
    assert {"run_ms", "total_ms", "llm_ms", "llm_call_ms"} <= set(first)
    assert len(first["llm_call_ms"]) == first["steps"] == 2 and sum(first["llm_call_ms"]) == pytest.approx(first["llm_ms"])
    assert len(data["scores"]) == 4 and {"passed", "checks"} <= set(data["scores"][0])
    assert data["report"]["overall"]["run_level"]["n"] == 4


def test_pending_proposals_never_leak_between_runs_even_in_one_process(tmp_path):
    ar.cmd_eval_agent(_args(tmp_path, case=["pl-named", "sd-disks-1"], runs=2), executor=in_process, llm_factory=FAKES + "scripted")
    runs = json.loads(_only_file(tmp_path / "results").read_text())["runs"]
    pending = {(r["case_id"], r["run_index"]): r["pending_proposals"] for r in runs}
    assert pending[("pl-named", 1)] == [RG_DEV] and pending[("pl-named", 2)] == [RG_DEV]  # one each, not two
    assert pending[("sd-disks-1", 1)] == [] and pending[("sd-disks-1", 2)] == []


def test_no_api_key_reaches_the_results_file(tmp_path, monkeypatch):
    key = "AIzaSyFAKEFAKEFAKEFAKEFAKE0123456789"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    ar.cmd_eval_agent(_args(tmp_path, case=["sd-disks-1"], runs=1), executor=in_process, llm_factory=FAKES + "leaks_key")
    text = _only_file(tmp_path / "results").read_text()
    assert key not in text and "AIza" not in text and "[redacted]" in text


def test_redact():
    assert ar.redact({"a": ["k=SECRET1 AIzaSyABCDEFGHIJKLMNOPQRSTUV"]}, ["SECRET1"]) == {"a": ["k=[redacted] [redacted]"]}
    assert ar.redact(5, ["x"]) == 5


def test_missing_api_key_stops_before_any_run(tmp_path, capsys):
    assert ar.cmd_eval_agent(_args(tmp_path, case=["sd-disks-1"], runs=1)) == 2
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


# --- official-run guards -------------------------------------------------------------------------

def test_official_run_refuses_a_dirty_tree(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ar, "git_state", lambda: ("b" * 40, True))
    assert ar.cmd_eval_agent(_args(tmp_path), executor=_boom) == 2
    assert "not clean" in capsys.readouterr().err


def test_official_run_refuses_a_golden_set_that_differs_from_the_frozen_one(tmp_path, capsys):
    changed = tmp_path / "golden.jsonl"
    extra = json.loads(ag.DEFAULT_GOLDEN_PATH.read_text().splitlines()[0]) | {"id": "x-human-1", "source": "human"}
    changed.write_text(ag.DEFAULT_GOLDEN_PATH.read_text() + json.dumps(extra) + "\n")
    assert ar.cmd_eval_agent(_args(tmp_path, golden=changed), executor=_boom) == 2
    assert "differs from the pre-registered, frozen one" in capsys.readouterr().err


def test_smoke_runs_are_exempt_from_the_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "git_state", lambda: ("b" * 40, True))
    assert ar.cmd_eval_agent(_args(tmp_path, limit=1, runs=1), executor=in_process, llm_factory=FAKES + "scripted") == 0
    assert json.loads(_only_file(tmp_path / "results").read_text())["git_dirty"] is True


def test_a_clean_full_run_is_official_and_not_named_smoke(tmp_path):
    assert ar.cmd_eval_agent(_args(tmp_path), executor=in_process, llm_factory=FAKES + "scripted") == 0
    path = _only_file(tmp_path / "results")
    data = json.loads(path.read_text())
    assert not path.name.startswith("agent_smoke_") and path.name.startswith("agent_")
    assert data["official"] is True and len(data["runs"]) == 99 and data["config"]["selection"] == {"cases": None, "limit": None}


def test_an_invalid_golden_set_stops_the_run(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    assert ar.cmd_eval_agent(_args(tmp_path, golden=bad), executor=_boom) == 2
    assert "golden set is invalid" in capsys.readouterr().err


# --- error classification (in process) -------------------------------------------------------------

@pytest.mark.parametrize("factory, infra", [("raises_unavailable", True), ("raises_bad_request", False)])
def test_llm_errors_are_failed_runs_and_only_retried_statuses_are_infrastructure(factory, infra):
    rec = in_process(_spec(factory=factory))
    assert rec.status == "error" and rec.infrastructure is infra and "LLMError" in rec.error


def test_a_non_llm_exception_is_an_error_run_not_a_crash():
    rec = in_process(_spec(factory="leaks_key"))
    assert rec.status == "error" and not rec.infrastructure and "RuntimeError" in rec.error


def test_max_steps_is_recorded_as_its_own_status():
    spec = _spec(case_id="sd-disks-1") | {"max_steps": 1}
    rec = in_process(spec)
    assert rec.status == "max_steps" and rec.final_answer is None


# --- the real child process --------------------------------------------------------------------------

def test_child_process_run_records_both_latencies_with_startup_only_in_the_parent_total():
    rec = ar.subprocess_executor(_spec(case_id="sd-disks-1"), timeout_s=60)
    assert rec.status == "ok" and rec.tool_calls[0]["name"] == "list_unattached_disks" and rec.attempts == 3
    assert 0 < rec.run_ms < rec.total_ms  # process spawn + imports are outside run_agent
    assert rec.llm_ms <= rec.run_ms
    assert len(rec.llm_call_ms) == rec.steps == 2 and all(ms >= 0 for ms in rec.llm_call_ms)
    assert sum(rec.llm_call_ms) == pytest.approx(rec.llm_ms)


def test_child_processes_are_isolated_from_each_other():
    first = ar.subprocess_executor(_spec(case_id="pl-named"), timeout_s=60)
    second = ar.subprocess_executor(_spec(case_id="pl-named", run_index=2), timeout_s=60)
    assert first.pending_proposals == [RG_DEV] and second.pending_proposals == [RG_DEV]


def test_a_hung_child_is_killed_at_the_timeout_and_recorded_as_a_timeout():
    rec = ar.subprocess_executor(_spec(factory="hangs"), timeout_s=2)
    assert rec.status == "timeout" and "killed after 2s" in rec.error
    assert 2000 <= rec.total_ms < 30_000


def test_a_crashing_child_is_an_error_run_with_its_exit_code():
    rec = ar.subprocess_executor(_spec(factory="crashes"), timeout_s=60)
    assert rec.status == "error" and "exited with code 3" in rec.error


def test_the_default_timeout_is_300_seconds_per_the_2026_10_20_addendum():
    from cloudguard.evals.agent_eval import RUN_TIMEOUT_S
    assert RUN_TIMEOUT_S == 300


def test_dry_run_reports_the_300_second_timeout(tmp_path, capsys):
    ar.cmd_eval_agent(_args(tmp_path, dry_run=True), executor=_boom)
    assert "timeout per run: 300s" in capsys.readouterr().out


def test_the_child_executor_defaults_to_the_300_second_timeout():
    import inspect
    from cloudguard.evals.agent_eval import RUN_TIMEOUT_S
    assert inspect.signature(ar.subprocess_executor).parameters["timeout_s"].default == RUN_TIMEOUT_S == 300
