"""One seed spec -> three outputs: the az commands that build the sandbox, the ground truth,
and a synthetic mock snapshot.

Why ground truth comes from the spec and NOT from running the tools: if we generated
"expected findings" by running our own detectors, the eval would be circular and always
score 100%. The spec states what we *intentionally broke*; the tools are graded against it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PLACEHOLDER_SUB = "00000000-0000-0000-0000-000000000000"


def load_spec(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sa_name(sa: dict[str, Any], suffix: str) -> str:
    name = f"{sa['name_prefix']}{suffix}".lower()
    if not (3 <= len(name) <= 24 and name.isalnum()):
        raise ValueError(f"invalid storage account name {name!r} (3-24 lowercase letters/digits)")
    return name


def rg_id(sub: str, rg: str) -> str:
    return f"/subscriptions/{sub}/resourceGroups/{rg}"


def disk_id(sub: str, d: dict[str, Any]) -> str:
    return f"{rg_id(sub, d['resource_group'])}/providers/Microsoft.Compute/disks/{d['name']}"


def sa_id(sub: str, sa: dict[str, Any], suffix: str) -> str:
    return f"{rg_id(sub, sa['resource_group'])}/providers/Microsoft.Storage/storageAccounts/{sa_name(sa, suffix)}"


def build_ground_truth(spec: dict[str, Any], sub: str, weak_tls_seeded: bool = True) -> dict[str, Any]:
    suffix = spec["suffix"]
    expected, negatives = [], []
    for rg in spec["resource_groups"]:
        item = {"type": "unlocked_resource_group", "resource_id": rg_id(sub, rg["name"]), "note": rg.get("trap", "")}
        (negatives if rg["lock"] else expected).append(item)
    for d in spec["disks"]:
        expected.append({"type": "unattached_disk", "resource_id": disk_id(sub, d), "note": f"{d['sku']} {d['size_gb']} GiB"})
    for sa in spec["storage_accounts"]:
        item = {"type": "weak_tls", "resource_id": sa_id(sub, sa, suffix), "note": sa.get("trap", f"min TLS {sa['min_tls']}")}
        if sa["weak"] and weak_tls_seeded:
            expected.append(item)
        elif not sa["weak"]:
            negatives.append(item)
    return {
        "subscription_id": sub,
        "description": spec["description"] + ("" if weak_tls_seeded else " (weak-TLS account could not be created; excluded.)"),
        "expected_findings": expected,
        "expected_negatives": negatives,
    }


def build_mock_snapshot(spec: dict[str, Any], sub: str = PLACEHOLDER_SUB) -> dict[str, Any]:
    """Synthetic snapshot in the same row shape Resource Graph returns."""
    loc, suffix = spec["location"], spec["suffix"]
    containers, resources = [], []
    for rg in spec["resource_groups"]:
        containers.append({
            "id": rg_id(sub, rg["name"]), "name": rg["name"],
            "type": "microsoft.resources/subscriptions/resourcegroups",
            "location": loc, "subscriptionId": sub, "tags": rg.get("tags", {}),
            "properties": {"provisioningState": "Succeeded"},
        })
        if rg["lock"]:
            resources.append(_lock(sub, rg["name"], rg_id(sub, rg["name"]), "cg-rg-lock", rg["lock"]))
    for d in spec["disks"]:
        resources.append({
            "id": disk_id(sub, d), "name": d["name"], "type": "microsoft.compute/disks",
            "resourceGroup": d["resource_group"], "location": loc, "subscriptionId": sub, "tags": {},
            "sku": {"name": d["sku"], "tier": "Standard"}, "managedBy": None,
            "properties": {"diskState": "Unattached", "diskSizeGB": d["size_gb"], "provisioningState": "Succeeded"},
        })
    for sa in spec["storage_accounts"]:
        sid = sa_id(sub, sa, suffix)
        resources.append({
            "id": sid, "name": sa_name(sa, suffix), "type": "microsoft.storage/storageaccounts",
            "resourceGroup": sa["resource_group"], "location": loc, "subscriptionId": sub, "tags": {},
            "sku": {"name": "Standard_LRS", "tier": "Standard"}, "managedBy": None,
            "properties": {"minimumTlsVersion": sa["min_tls"], "supportsHttpsTrafficOnly": True},
        })
        if sa.get("resource_lock"):
            resources.append(_lock(sub, sa["resource_group"], sid, "cg-sa-lock", sa["resource_lock"]))
    return {"subscriptionId": sub, "resource_containers": containers, "resources": resources}


def _lock(sub: str, rg: str, scope_id: str, name: str, level: str) -> dict[str, Any]:
    return {
        "id": f"{scope_id}/providers/Microsoft.Authorization/locks/{name}", "name": name,
        "type": "microsoft.authorization/locks", "resourceGroup": rg, "location": "global",
        "subscriptionId": sub, "tags": {}, "sku": None, "managedBy": None,
        "properties": {"level": level, "notes": "CloudGuard sandbox"},
    }


def az_seed_commands(spec: dict[str, Any]) -> list[list[str]]:
    loc, suffix = spec["location"], spec["suffix"]
    cmds: list[list[str]] = []
    for rg in spec["resource_groups"]:
        cmd = ["az", "group", "create", "-n", rg["name"], "-l", loc]
        if rg.get("tags"):
            cmd += ["--tags", *[f"{k}={v}" for k, v in rg["tags"].items()]]
        cmds.append(cmd)
    for d in spec["disks"]:
        cmds.append(["az", "disk", "create", "-g", d["resource_group"], "-n", d["name"],
                     "--size-gb", str(d["size_gb"]), "--sku", d["sku"], "-l", loc])
    for sa in spec["storage_accounts"]:
        cmds.append(["az", "storage", "account", "create", "-g", sa["resource_group"], "-n", sa_name(sa, suffix),
                     "-l", loc, "--sku", "Standard_LRS", "--kind", "StorageV2", "--min-tls-version", sa["min_tls"]])
        if sa.get("resource_lock"):
            cmds.append(["az", "lock", "create", "-n", "cg-sa-lock", "-t", sa["resource_lock"],
                         "-g", sa["resource_group"], "--resource", sa_name(sa, suffix),
                         "--resource-type", "Microsoft.Storage/storageAccounts"])
    # RG locks last, so creating resources inside the groups isn't blocked.
    for rg in spec["resource_groups"]:
        if rg["lock"]:
            cmds.append(["az", "lock", "create", "-n", "cg-rg-lock", "-t", rg["lock"], "-g", rg["name"]])
    return cmds


def az_destroy_commands(spec: dict[str, Any]) -> list[list[str]]:
    suffix = spec["suffix"]
    cmds: list[list[str]] = []
    for rg in spec["resource_groups"]:
        if rg["lock"]:
            cmds.append(["az", "lock", "delete", "-n", "cg-rg-lock", "-g", rg["name"]])
    for sa in spec["storage_accounts"]:
        if sa.get("resource_lock"):
            cmds.append(["az", "lock", "delete", "-n", "cg-sa-lock", "-g", sa["resource_group"],
                         "--resource", sa_name(sa, suffix), "--resource-type", "Microsoft.Storage/storageAccounts"])
    for rg in spec["resource_groups"]:
        cmds.append(["az", "group", "delete", "-n", rg["name"], "--yes", "--no-wait"])
    return cmds
