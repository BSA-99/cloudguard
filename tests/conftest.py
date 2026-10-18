import json
from pathlib import Path

import pytest

from cloudguard.backends import MockBackend
from cloudguard.models import GroundTruth

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_audit_log(tmp_path, monkeypatch):
    """Every test writes to its own audit log, never the repo's."""
    path = tmp_path / "audit_log.jsonl"
    monkeypatch.setenv("CLOUDGUARD_AUDIT_LOG", str(path))
    return path


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend(ROOT / "mock" / "subscription_snapshot.json")


@pytest.fixture
def truth() -> GroundTruth:
    return GroundTruth.model_validate_json((ROOT / "ground_truth.json").read_text())


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
