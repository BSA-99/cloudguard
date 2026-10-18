"""Where the data comes from: a live subscription (Resource Graph) or a saved JSON snapshot.

Both backends expose the same four read methods and return rows in the same shape, so the
tools layer (and later the agent) cannot tell them apart. The mock methods are Python
re-implementations of the KQL in queries.py; tests/test_parity.py (live-only) checks that
both produce identical IDs against the real sandbox.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from cloudguard import queries

# /subscriptions/{guid}/resourceGroups/{rg}[/providers/{ns}/{type}/{name}[/...]]
ARM_ID_PATTERN = re.compile(
    r"^/subscriptions/[0-9a-fA-F-]{36}/resourceGroups/[-\w.()]{1,90}"
    r"(/providers/[\w.]+(/[\w.-]+/[-\w.()]+)+)?$",
    re.IGNORECASE,
)

_LOCK_SCOPE = re.compile(r"^(.*)/providers/microsoft\.authorization/locks/[^/]+$", re.IGNORECASE)


class InvalidResourceId(ValueError):
    pass


def validate_resource_id(resource_id: str) -> str:
    """Reject anything that isn't a well-formed ARM ID before it touches a KQL string.

    This is the KQL-injection guard: an LLM (or a prompt-injected tag) could pass
    `x') | project secrets //` as an "ID". The regex forbids quotes, pipes and spaces.
    """
    if not isinstance(resource_id, str) or not ARM_ID_PATTERN.match(resource_id.strip()):
        raise InvalidResourceId(f"not a valid Azure resource ID: {resource_id!r}")
    return resource_id.strip()


class Backend(Protocol):
    name: str

    def unlocked_resource_groups(self) -> list[dict[str, Any]]: ...
    def unattached_disks(self) -> list[dict[str, Any]]: ...
    def weak_tls_storage(self) -> list[dict[str, Any]]: ...
    def get_resource(self, resource_id: str) -> dict[str, Any] | None: ...


class MockBackend:
    """Serves a recorded subscription snapshot. No network, no credentials."""

    name = "mock"

    def __init__(self, snapshot_path: str | Path):
        self.snapshot_path = Path(snapshot_path)
        data = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        self.subscription_id: str = data["subscriptionId"]
        self.containers: list[dict[str, Any]] = data["resource_containers"]
        self.resources: list[dict[str, Any]] = data["resources"]

    def _of_type(self, rtype: str) -> list[dict[str, Any]]:
        return [r for r in self.resources if r.get("type", "").lower() == rtype]

    def unlocked_resource_groups(self) -> list[dict[str, Any]]:
        locked_scopes = set()
        for lock in self._of_type("microsoft.authorization/locks"):
            m = _LOCK_SCOPE.match(lock["id"].lower())
            if m:
                locked_scopes.add(m.group(1))
        rows = [
            {k: c.get(k) for k in ("id", "name", "location", "subscriptionId", "tags")}
            for c in self.containers
            if c.get("type", "").lower() == "microsoft.resources/subscriptions/resourcegroups"
            and c["id"].lower() not in locked_scopes
        ]
        return sorted(rows, key=lambda r: r["name"])

    def unattached_disks(self) -> list[dict[str, Any]]:
        rows = []
        for d in self._of_type("microsoft.compute/disks"):
            props = d.get("properties") or {}
            if str(props.get("diskState", "")).lower() == "unattached":
                rows.append({
                    "id": d["id"], "name": d["name"], "resourceGroup": d.get("resourceGroup"),
                    "location": d.get("location"), "tags": d.get("tags"),
                    "skuName": (d.get("sku") or {}).get("name"),
                    "diskSizeGB": props.get("diskSizeGB"),
                    "diskState": props.get("diskState"), "managedBy": d.get("managedBy"),
                })
        return sorted(rows, key=lambda r: r["name"])

    def weak_tls_storage(self) -> list[dict[str, Any]]:
        rows = []
        for s in self._of_type("microsoft.storage/storageaccounts"):
            tls = str((s.get("properties") or {}).get("minimumTlsVersion") or "")
            if tls == "" or tls.upper() in ("TLS1_0", "TLS1_1"):
                rows.append({
                    "id": s["id"], "name": s["name"], "resourceGroup": s.get("resourceGroup"),
                    "location": s.get("location"), "tags": s.get("tags"),
                    "minimumTlsVersion": tls,
                })
        return sorted(rows, key=lambda r: r["name"])

    def get_resource(self, resource_id: str) -> dict[str, Any] | None:
        rid = validate_resource_id(resource_id).lower()
        for r in [*self.resources, *self.containers]:
            if r["id"].lower() == rid:
                return r
        return None


class ResourceGraphBackend:
    """Live mode: runs the KQL in queries.py through Azure Resource Graph.

    Auth is DefaultAzureCredential, which resolves to the managed identity when deployed
    on Container Apps, the OIDC federated credential in GitHub Actions, and `az login`
    on a laptop. The identity only ever needs the built-in Reader role.
    """

    name = "live"

    def __init__(self, subscription_id: str):
        # Imported lazily so mock mode and CI don't need the Azure SDK installed.
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.resourcegraph import ResourceGraphClient

        self.subscription_id = subscription_id
        self._client = ResourceGraphClient(DefaultAzureCredential())

    def run(self, kql: str) -> list[dict[str, Any]]:
        from azure.mgmt.resourcegraph.models import QueryRequest, QueryRequestOptions

        rows: list[dict[str, Any]] = []
        skip_token = None
        while True:
            req = QueryRequest(
                subscriptions=[self.subscription_id],
                query=kql,
                options=QueryRequestOptions(result_format="objectArray", skip_token=skip_token),
            )
            resp = self._client.resources(req)
            rows.extend(resp.data or [])
            skip_token = resp.skip_token
            if not skip_token:
                return rows

    def unlocked_resource_groups(self) -> list[dict[str, Any]]:
        return self.run(queries.UNLOCKED_RESOURCE_GROUPS)

    def unattached_disks(self) -> list[dict[str, Any]]:
        return self.run(queries.UNATTACHED_DISKS)

    def weak_tls_storage(self) -> list[dict[str, Any]]:
        return self.run(queries.WEAK_TLS_STORAGE)

    def get_resource(self, resource_id: str) -> dict[str, Any] | None:
        rid = validate_resource_id(resource_id).replace("'", "")
        rows = self.run(queries.GET_RESOURCE.format(resource_id=rid))
        return rows[0] if rows else None

    def snapshot(self) -> dict[str, Any]:
        return {
            "subscriptionId": self.subscription_id,
            "resource_containers": self.run(queries.SNAPSHOT_CONTAINERS),
            "resources": self.run(queries.SNAPSHOT_RESOURCES),
        }
