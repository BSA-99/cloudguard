import json
from pathlib import Path

from cloudguard import sandbox

ROOT = Path(__file__).resolve().parents[1]
SPEC = sandbox.load_spec(ROOT / "sandbox" / "seed_spec.json")


def test_committed_mock_files_are_in_sync_with_spec():
    """If someone edits seed_spec.json, they must regenerate (python sandbox/seed.py --mock-only)."""
    snap = json.loads((ROOT / "mock" / "subscription_snapshot.json").read_text())
    gt = json.loads((ROOT / "ground_truth.json").read_text())
    assert snap == sandbox.build_mock_snapshot(SPEC)
    assert gt == sandbox.build_ground_truth(SPEC, sandbox.PLACEHOLDER_SUB)


def test_spec_matches_the_brief():
    rgs = SPEC["resource_groups"]
    assert len(rgs) == 6 and sum(1 for r in rgs if not r["lock"]) == 4
    assert len(SPEC["disks"]) == 2
    assert sum(1 for s in SPEC["storage_accounts"] if s["weak"]) == 1


def test_rg_locks_are_created_after_resources():
    cmds = sandbox.az_seed_commands(SPEC)
    first_rg_lock = next(i for i, c in enumerate(cmds) if c[:3] == ["az", "lock", "create"] and "--resource" not in c)
    assert all(c[1] != "disk" and c[1] != "storage" for c in cmds[first_rg_lock:])


def test_ground_truth_drops_weak_tls_if_azure_refused_it():
    gt = sandbox.build_ground_truth(SPEC, sandbox.PLACEHOLDER_SUB, weak_tls_seeded=False)
    assert not any(i["type"] == "weak_tls" for i in gt["expected_findings"])
