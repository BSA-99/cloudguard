"""Unit tests for the pure grounding check. No LLM, no agent loop - check_grounding() takes
answer text and a list of ToolResult objects built directly, exactly how the Day-4 eval
harness would reuse it.
"""

from __future__ import annotations

from cloudguard.grounding import check_grounding
from cloudguard.models import Finding, FindingType, Severity, ToolResult

SUB = "00000000-0000-0000-0000-000000000000"
DISK_1 = f"/subscriptions/{SUB}/resourceGroups/cg-dev-sandbox/providers/Microsoft.Compute/disks/cg-disk-orphan-01"
DISK_2 = f"/subscriptions/{SUB}/resourceGroups/cg-test-batch/providers/Microsoft.Compute/disks/cg-disk-orphan-02"
RG_DEV_SANDBOX = f"/subscriptions/{SUB}/resourceGroups/cg-dev-sandbox"
RG_PROD_WEB = f"/subscriptions/{SUB}/resourceGroups/cg-prod-web"


def _disk_finding(resource_id: str, name: str) -> Finding:
    return Finding(
        type=FindingType.UNATTACHED_DISK, resource_id=resource_id, name=name,
        severity=Severity.LOW, reason="test",
    )


def _unattached_disks_result(*disk_ids_and_names: tuple[str, str]) -> ToolResult:
    findings = [_disk_finding(rid, name) for rid, name in disk_ids_and_names]
    return ToolResult(tool="list_unattached_disks", ok=True, backend="mock", findings=findings, count=len(findings))


def test_correct_answer_citing_real_disk_ids_passes():
    tool_results = [_unattached_disks_result((DISK_1, "cg-disk-orphan-01"), (DISK_2, "cg-disk-orphan-02"))]
    answer = f"I found two unattached disks: {DISK_1} and {DISK_2}."

    report = check_grounding(answer, tool_results)

    assert report.grounding_ok is True
    assert set(report.grounded_ids) == {DISK_1, DISK_2}
    assert report.ungrounded_ids == []


def test_answer_inventing_a_disk_id_fails_with_exactly_that_id_listed():
    tool_results = [_unattached_disks_result((DISK_1, "cg-disk-orphan-01"))]
    invented = f"/subscriptions/{SUB}/resourceGroups/cg-dev-sandbox/providers/Microsoft.Compute/disks/cg-disk-orphan-03"
    answer = f"Found disks {DISK_1} and {invented}."

    report = check_grounding(answer, tool_results)

    assert report.grounding_ok is False
    assert report.ungrounded_ids == [invented]
    assert report.grounded_ids == [DISK_1]


def test_real_id_cited_in_different_capitalization_passes():
    tool_results = [_unattached_disks_result((DISK_1, "cg-disk-orphan-01"))]
    answer = f"Found disk {DISK_1.upper()}."

    report = check_grounding(answer, tool_results)

    assert report.grounding_ok is True
    assert report.grounded_ids == [DISK_1.upper()]


def test_id_wrapped_in_backticks_with_trailing_period_inside_is_extracted_correctly():
    """The period sits right after the id, before the closing backtick - this is what
    actually exercises the trailing-punctuation stripping, not just the backtick."""
    tool_results = [_unattached_disks_result((DISK_1, "cg-disk-orphan-01"))]
    answer = f"The orphaned disk is `{DISK_1}.`"

    report = check_grounding(answer, tool_results)

    assert report.grounding_ok is True
    assert report.grounded_ids == [DISK_1]
    assert report.ungrounded_ids == []


def test_not_found_get_resource_lookup_is_ungrounded():
    fake_id = f"/subscriptions/{SUB}/resourceGroups/cg-dev-sandbox/providers/Microsoft.Compute/disks/does-not-exist"
    not_found = ToolResult(tool="get_resource", ok=False, backend="mock", error=f"resource not found: {fake_id}")
    answer = f"I checked {fake_id} and it appears to exist."

    report = check_grounding(answer, [not_found])

    assert report.grounding_ok is False
    assert report.ungrounded_ids == [fake_id]


def test_answer_with_no_ids_is_grounding_ok_with_empty_grounded_ids():
    tool_results = [_unattached_disks_result((DISK_1, "cg-disk-orphan-01"))]
    answer = "There are no issues to report."

    report = check_grounding(answer, tool_results)

    assert report.grounding_ok is True
    assert report.grounded_ids == []
    assert report.ungrounded_ids == []


def test_parent_resource_group_of_a_grounded_disk_passes():
    """The RG is implied by the disk's own ID, not invented, even though
    list_unattached_disks never returned the RG itself as a finding."""
    tool_results = [_unattached_disks_result((DISK_1, "cg-disk-orphan-01"))]
    answer = f"The resource group {RG_DEV_SANDBOX} contains an unattached disk."

    report = check_grounding(answer, tool_results)

    assert report.grounding_ok is True
    assert report.grounded_ids == [RG_DEV_SANDBOX]


def test_trailing_slash_is_ignored_when_comparing():
    tool_results = [_unattached_disks_result((DISK_1, "cg-disk-orphan-01"))]
    answer = f"Found disk {DISK_1}/."

    report = check_grounding(answer, tool_results)

    assert report.grounding_ok is True
    assert report.ungrounded_ids == []


def test_a_different_unseen_resource_group_still_fails():
    """The derived-parent-RG rule must not leak into other, untouched resource groups."""
    tool_results = [_unattached_disks_result((DISK_1, "cg-disk-orphan-01"))]
    answer = f"The resource group {RG_PROD_WEB} is unlocked."

    report = check_grounding(answer, tool_results)

    assert report.grounding_ok is False
    assert report.ungrounded_ids == [RG_PROD_WEB]
