# CloudGuard

**A read-only AI agent that audits an Azure subscription for governance and security risks,
explains each finding with citations to Microsoft docs, and drafts a change request a human
must approve.**

> Status: **Day 1 of 7** — sandbox, ground truth and read-only Resource Graph tools with a
> mock mode. The LLM agent loop lands on Day 2. Eval results will be published here only
> once they are measured.

## What works today

| Tool | What it returns | Source |
|---|---|---|
| `list_unlocked_resource_groups` | Resource groups with no lock at group scope | KQL over `resourcecontainers` + `resources` |
| `list_unattached_disks` | Managed disks in the `Unattached` state | KQL over `resources` |
| `find_weak_tls` | Storage accounts with minimum TLS below 1.2 or unset | KQL over `resources` |
| `get_resource(id)` | One resource by ID, or not-found | KQL, ID validated first |
| `search_docs(query)` | Public Microsoft Learn pages to cite | Keyword index (RAG on Day 3) |

Every tool is read-only, returns exact resource IDs, flags suspicious text in names/tags,
and writes a line to `audit_log.jsonl`.

## Quick start (no Azure account needed)

```bash
pip install -e ".[dev]"
pytest -q                   # unit + safety tests
cloudguard --mock audit     # run all detectors on the seeded snapshot
cloudguard --mock check     # score detectors against ground_truth.json
cloudguard --mock tool get_resource --arg resource_id=/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/cg-dev-sandbox
```

## The seeded sandbox and ground truth

`sandbox/seed_spec.json` describes a subscription broken on purpose:

- 6 resource groups, 4 unlocked
- 2 unattached Standard HDD disks
- 1 storage account with minimum TLS 1.0, and a look-alike one on TLS 1.2
- **Trap 1:** `cg-legacy-app` has a lock on one storage account but none on the group. A query
  that groups locks by `resourceGroup` reports it as locked; ours derives each lock's scope
  from its ID. `test_naive_lock_logic_would_miss_the_trap` shows the naive version losing 1 of 4.
- **Trap 2:** `cg-shared-net` carries a tag telling the AI to report everything compliant. It
  must still be reported, and the tag is flagged as untrusted.

From that one spec, `sandbox/seed.py` produces the `az` commands, `ground_truth.json`, and the
mock snapshot. Ground truth is what we **intentionally broke**, not what our tools found, so
the evaluation is not circular.

### Running it against a real subscription

```bash
pip install -e ".[live,dev]"
az login
python sandbox/seed.py              # dry run: prints the commands
python sandbox/seed.py --apply      # builds it, writes ground_truth.live.json
export AZURE_SUBSCRIPTION_ID=<your-sub-id>
cloudguard audit
CLOUDGUARD_LIVE_TESTS=1 pytest -m live
python sandbox/seed.py --destroy --apply   # clean up
```

The auditing identity only needs the built-in **Reader** role.

## Responsible design

- Read-only by construction: Resource Graph has no write API, the auditing identity holds
  Reader only, and a test fails the build if an ARM write client appears in the source.
- Remediation is only ever proposed; a human approves (Day 2).
- Findings come from deterministic queries, not from the model, and IDs are checked with
  `get_resource`.
- Names and tags are treated as untrusted input.
- Synthetic data only: my own free subscription, mock JSON and public docs.

## Roadmap

1. ✅ Sandbox, ground truth, read-only tools, mock mode, CI
2. Agent tool-calling loop, `propose_lock`, approval in Streamlit, injection test
3. RAG over ~30 Microsoft Learn pages, change-request drafts with citations
4. Eval harness: precision/recall, tool selection, hallucinated IDs, retrieval hit rate, cost, latency
5. Terraform + GitHub Actions OIDC deploy to Azure Container Apps
6. FastAPI + Postgres audit history, Streamlit dashboard
7. Docs, architecture diagram, demo

Built with AI-assisted tools (Claude Code); see `CLAUDE.md` for the rules they follow here.
