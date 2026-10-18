"""KQL for Azure Resource Graph. Every query here is a read (Resource Graph has no write API).

Kept as plain strings in one file so they can be reviewed, copied into the portal's
Resource Graph Explorer, and diffed against the mock implementations in backends.py.
"""

# Resource Graph's `resources` table does not index microsoft.authorization/locks
# (confirmed empirically: az lock list shows locks that this table never returns), so lock
# scope can't be resolved in KQL. This query only enumerates resource groups; lock data and
# the scope-from-ID derivation (a lock on a single resource inside the RG must not count as
# a group-scope lock, per the seeded `cg-legacy-app` trap) live in backends.py, sourced from
# the read-only ARM Management Locks list API (ManagementLockClient).
RESOURCE_GROUPS = r"""
resourcecontainers
| where type =~ 'microsoft.resources/subscriptions/resourcegroups'
| project id, name, location, subscriptionId, tags
| order by name asc
"""

UNATTACHED_DISKS = r"""
resources
| where type =~ 'microsoft.compute/disks'
| extend diskState = tostring(properties.diskState)
| where diskState =~ 'Unattached'
| project id, name, resourceGroup, location, tags,
          skuName = tostring(sku.name),
          diskSizeGB = toint(properties.diskSizeGB),
          diskState, managedBy
| order by name asc
"""

# Empty minimumTlsVersion is reported too: older accounts created without the property
# historically accepted TLS 1.0, so "unset" is treated as unknown rather than safe.
WEAK_TLS_STORAGE = r"""
resources
| where type =~ 'microsoft.storage/storageaccounts'
| extend minimumTlsVersion = tostring(properties.minimumTlsVersion)
| where isempty(minimumTlsVersion) or minimumTlsVersion in~ ('TLS1_0', 'TLS1_1')
| project id, name, resourceGroup, location, tags, minimumTlsVersion
| order by name asc
"""

# {resource_id} is validated against ARM_ID_PATTERN and quote-escaped before formatting.
GET_RESOURCE = r"""
union
    (resources | where tolower(id) == tolower('{resource_id}')),
    (resourcecontainers | where tolower(id) == tolower('{resource_id}'))
| project id, name, type, resourceGroup, location, subscriptionId, tags, sku, properties, managedBy
| take 1
"""

# Used by `cloudguard snapshot` to record a live subscription into mock JSON.
SNAPSHOT_CONTAINERS = r"""
resourcecontainers
| where type =~ 'microsoft.resources/subscriptions/resourcegroups'
| project id, name, type, location, subscriptionId, tags, properties
"""

# microsoft.authorization/locks is deliberately absent here: Resource Graph doesn't index
# it. ResourceGraphBackend.snapshot() merges lock rows in separately from the ARM Management
# Locks list API so the resulting mock JSON still matches the shape sandbox.py produces.
SNAPSHOT_RESOURCES = r"""
resources
| where type in~ ('microsoft.compute/disks', 'microsoft.storage/storageaccounts')
| project id, name, type, resourceGroup, location, subscriptionId, tags, sku, properties, managedBy
"""
