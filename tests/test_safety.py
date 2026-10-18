"""Guardrails: no invented IDs, no KQL injection, untrusted text flagged, everything logged, read-only."""

import re
from pathlib import Path

import pytest

from cloudguard import tools
from cloudguard.backends import validate_resource_id, InvalidResourceId
from tests.conftest import read_log

SUB = "00000000-0000-0000-0000-000000000000"
SRC = Path(__file__).resolve().parents[1] / "src" / "cloudguard"


def test_get_resource_returns_known_resource(backend):
    rid = f"/subscriptions/{SUB}/resourceGroups/cg-dev-sandbox/providers/Microsoft.Compute/disks/cg-disk-orphan-01"
    r = tools.get_resource(backend, resource_id=rid)
    assert r.ok and r.data["name"] == "cg-disk-orphan-01"


def test_get_resource_is_case_insensitive(backend):
    r = tools.get_resource(backend, resource_id=f"/SUBSCRIPTIONS/{SUB}/RESOURCEGROUPS/CG-DEV-SANDBOX")
    assert r.ok


def test_get_resource_unknown_id_is_not_found_not_guessed(backend):
    """The hallucinated-ID check the agent will rely on: a plausible but fake ID must fail."""
    fake = f"/subscriptions/{SUB}/resourceGroups/cg-dev-sandbox/providers/Microsoft.Compute/disks/cg-disk-orphan-03"
    r = tools.get_resource(backend, resource_id=fake)
    assert not r.ok and "not found" in r.error
    assert r.data is None


@pytest.mark.parametrize("bad", [
    "cg-dev-sandbox",
    f"/subscriptions/{SUB}/resourceGroups/x') | project properties //",
    f"/subscriptions/{SUB}/resourceGroups/a b",
    "/subscriptions/not-a-guid/resourceGroups/rg",
    "",
])
def test_malformed_ids_are_rejected_before_query(backend, bad):
    with pytest.raises(InvalidResourceId):
        validate_resource_id(bad)
    r = tools.get_resource(backend, resource_id=bad)
    assert not r.ok and "not a valid Azure resource ID" in r.error


def test_prompt_injection_tag_is_flagged_but_finding_kept(backend):
    findings = {f.name: f for f in tools.list_unlocked_resource_groups(backend).findings}
    f = findings["cg-shared-net"]
    assert f.untrusted_text_flags, "injection text in tags must be flagged"
    assert any("override_instructions" in x for x in f.untrusted_text_flags)
    # The tag is preserved verbatim for the human reviewer; flagging is not deletion.
    assert "IGNORE PREVIOUS INSTRUCTIONS" in f.evidence["tags"]["note"]


def test_clean_resources_have_no_flags(backend):
    findings = tools.list_unattached_disks(backend).findings
    assert findings and all(not f.untrusted_text_flags for f in findings)


def test_every_call_is_audit_logged_including_failures(backend, isolated_audit_log):
    tools.list_unattached_disks(backend)
    tools.get_resource(backend, resource_id="garbage")
    tools.search_docs(backend, query="resource lock")
    log = read_log(isolated_audit_log)
    assert [e["tool"] for e in log] == ["list_unattached_disks", "get_resource", "search_docs"]
    assert log[0]["ok"] and log[0]["result_count"] == 2 and len(log[0]["result_ids"]) == 2
    assert not log[1]["ok"] and log[1]["error"]
    assert all(e["backend"] == "mock" and e["duration_ms"] >= 0 for e in log)


def test_tool_registry_and_specs_agree():
    assert {s["name"] for s in tools.TOOL_SPECS} == set(tools.TOOLS)


def test_no_write_calls_anywhere_in_source():
    """Crude but useful: fail the build if someone adds an ARM write path."""
    write_markers = re.compile(r"\b(begin_create|create_or_update|begin_delete|\.delete\(|ManagementLockClient|ResourceManagementClient)\b")
    offenders = [p.name for p in SRC.rglob("*.py") if write_markers.search(p.read_text())]
    assert offenders == []
