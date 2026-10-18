"""KQL for Azure Resource Graph. Every query here is a read (Resource Graph has no write API).

Kept as plain strings in one file so they can be reviewed, copied into the portal's
Resource Graph Explorer, and diffed against the mock implementations in backends.py.
"""

# A resource group counts as LOCKED only if a lock sits at exactly the RG scope.
# A lock on a single resource *inside* the RG has the same `resourceGroup` value, so a
# naive `summarize by resourceGroup` would wrongly mark the RG as locked. We derive the
# lock's scope from its ID instead. The seeded sandbox contains this exact trap.
# Subscription-scope locks are out of scope for v0.1 (the sandbox has none).
UNLOCKED_RESOURCE_GROUPS = r"""
resourcecontainers
| where type =~ 'microsoft.resources/subscriptions/resourcegroups'
| project rgId = tolower(id), id, name, location, subscriptionId, tags
| join kind=leftouter (
    resources
    | where type =~ 'microsoft.authorization/locks'
    | extend lockScope = tolower(extract(@'^(.*)/providers/microsoft\.authorization/locks/[^/]+$', 1, tolower(id)))
    | summarize lockCount = count(), lockLevels = make_set(tostring(properties.level)) by lockScope
) on $left.rgId == $right.lockScope
| where isnull(lockCount) or lockCount == 0
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

SNAPSHOT_RESOURCES = r"""
resources
| where type in~ ('microsoft.authorization/locks', 'microsoft.compute/disks', 'microsoft.storage/storageaccounts')
| project id, name, type, resourceGroup, location, subscriptionId, tags, sku, properties, managedBy
"""
