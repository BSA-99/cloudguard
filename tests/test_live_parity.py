"""Runs only against your real seeded sandbox:

  CLOUDGUARD_LIVE_TESTS=1 AZURE_SUBSCRIPTION_ID=<id> pytest -m live

Checks that the real KQL finds exactly what was seeded (ground_truth.live.json). This is
what proves the mock implementations in backends.py didn't drift from the KQL.
"""

import os
from pathlib import Path

import pytest

from cloudguard import tools
from cloudguard.cli import score
from cloudguard.models import GroundTruth

ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("CLOUDGUARD_LIVE_TESTS") != "1", reason="live tests disabled"),
]


def test_live_detectors_match_seeded_ground_truth():
    from cloudguard.backends import ResourceGraphBackend

    backend = ResourceGraphBackend(os.environ["AZURE_SUBSCRIPTION_ID"])
    truth = GroundTruth.model_validate_json((ROOT / "ground_truth.live.json").read_text())
    found = {f.resource_id for n in tools.DETECTORS for f in tools.TOOLS[n](backend).findings}
    # Only count resources inside the sandbox groups; your subscription may hold others.
    sandbox_found = {i for i in found if "/resourcegroups/cg-" in i.lower()}
    report = score(sandbox_found, truth)
    assert report["missed"] == [] and report["false_positives"] == [], report
