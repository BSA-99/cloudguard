"""Detector correctness against the seeded ground truth (mock mode)."""

from cloudguard import tools
from cloudguard.cli import score
from cloudguard.models import FindingType


def _ids(result):
    return {f.resource_id.lower() for f in result.findings}


def _expected(truth, ftype):
    return {i.resource_id.lower() for i in truth.expected_findings if i.type == ftype}


def test_unlocked_resource_groups_match_ground_truth(backend, truth):
    r = tools.list_unlocked_resource_groups(backend)
    assert r.ok
    assert _ids(r) == _expected(truth, FindingType.UNLOCKED_RESOURCE_GROUP)


def test_unattached_disks_match_ground_truth(backend, truth):
    r = tools.list_unattached_disks(backend)
    assert _ids(r) == _expected(truth, FindingType.UNATTACHED_DISK)
    assert all(f.evidence["skuName"] == "Standard_LRS" for f in r.findings)


def test_weak_tls_matches_ground_truth(backend, truth):
    r = tools.find_weak_tls(backend)
    assert _ids(r) == _expected(truth, FindingType.WEAK_TLS)


def test_no_seeded_negative_is_flagged(backend, truth):
    found = set()
    for name in tools.DETECTORS:
        found |= _ids(tools.TOOLS[name](backend))
    negatives = {i.resource_id.lower() for i in truth.expected_negatives}
    assert negatives, "ground truth should contain look-alike negatives"
    assert not (found & negatives)


def test_overall_baseline_score(backend, truth):
    found = {f.resource_id for n in tools.DETECTORS for f in tools.TOOLS[n](backend).findings}
    report = score(found, truth)
    assert report["precision"] == 1.0 and report["recall"] == 1.0


def test_resource_level_lock_does_not_count_as_group_lock(backend):
    """Trap: cg-legacy-app holds a lock on one storage account, not on the group."""
    names = {f.name for f in tools.list_unlocked_resource_groups(backend).findings}
    assert "cg-legacy-app" in names


def test_naive_lock_logic_would_miss_the_trap(backend, truth):
    """Documents the bug the scope-aware query prevents: grouping locks by resourceGroup
    marks cg-legacy-app as locked and recall drops from 4/4 to 3/4."""
    locked_rgs = {r["resourceGroup"].lower() for r in backend.resources
                  if r["type"] == "microsoft.authorization/locks"}
    naive = {c["id"].lower() for c in backend.containers if c["name"].lower() not in locked_rgs}
    expected = _expected(truth, FindingType.UNLOCKED_RESOURCE_GROUP)
    assert len(naive & expected) == len(expected) - 1
