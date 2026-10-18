"""Typed shapes for everything that crosses a boundary (tool output, audit log, ground truth).

Using pydantic here means a malformed tool result fails loudly in tests instead of
silently reaching the LLM on Day 2.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FindingType(str, Enum):
    UNLOCKED_RESOURCE_GROUP = "unlocked_resource_group"
    UNATTACHED_DISK = "unattached_disk"
    WEAK_TLS = "weak_tls"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Finding(BaseModel):
    """One detected risk, always anchored to a real resource ID returned by Resource Graph."""

    type: FindingType
    resource_id: str
    name: str
    resource_group: str | None = None
    severity: Severity
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    # Names and tags are attacker-controllable. We never strip them (the auditor needs
    # to see them), but we flag text that looks like instructions aimed at an LLM.
    untrusted_text_flags: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    tool: str
    ok: bool
    backend: str
    findings: list[Finding] = Field(default_factory=list)
    data: dict[str, Any] | None = None
    error: str | None = None
    count: int = 0


class GroundTruthItem(BaseModel):
    type: FindingType
    resource_id: str
    note: str = ""


class GroundTruth(BaseModel):
    subscription_id: str
    description: str
    expected_findings: list[GroundTruthItem]
    # Resources deliberately seeded as look-alikes that must NOT be flagged.
    expected_negatives: list[GroundTruthItem] = Field(default_factory=list)
