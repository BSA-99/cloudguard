# CloudGuard

**A read-only AI agent that audits an Azure subscription for governance and security risks,
explains each finding with citations to Microsoft docs, and drafts a change request a human
must approve.**

> Status: **Day 2 of 7** — sandbox, read-only tools, the Gemini tool-calling agent loop
> (grounding check, `propose_lock`, human approval), and a Streamlit UI over it. RAG and
> change-request drafts land on Day 3. Eval numbers will be published here only once
> they're measured.

## What works today

| Tool | What it returns | Source |
|---|---|---|
| `list_unlocked_resource_groups` | Resource groups with no lock at group scope | KQL over `resourcecontainers` (groups) + ARM Management Locks list API (locks — Resource Graph doesn't index them) |
| `list_unattached_disks` | Managed disks in the `Unattached` state | KQL over `resources` |
| `find_weak_tls` | Storage accounts with minimum TLS below 1.2 or unset | KQL over `resources` |
| `get_resource(id)` | One resource by ID, or not-found | KQL, ID validated first |
| `search_docs(query)` | Public Microsoft Learn pages to cite | Keyword index (RAG on Day 3) |
| `propose_lock(rg, reason)` | Writes a pending lock proposal for a human to approve | `proposals.jsonl`; **makes no Azure call** |

Every tool is read-only against Azure, returns exact resource IDs, flags suspicious text in
names/tags, and writes a line to `audit_log.jsonl`. `propose_lock` is the one tool that
writes anything at all, and what it writes is a proposal record, never an ARM call — only
for a resource group `list_unlocked_resource_groups` already confirmed unlocked in the same
run. A human approves or rejects via `proposals.decide()`, which is deliberately not a tool.

## Quick start (no Azure account needed)

```bash
pip install -e ".[dev]"
pytest -q                   # unit + safety tests
cloudguard --mock audit     # run all detectors on the seeded snapshot
cloudguard --mock check     # score detectors against ground_truth.json
cloudguard --mock tool get_resource --arg resource_id=/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/cg-dev-sandbox
```

## Agent loop (Day 2)

`src/cloudguard/llm.py` defines a provider-neutral `LLMClient` protocol. `FakeLLM` replays
scripted responses, so tests and CI need no API key. `GeminiClient` (the live implementation,
needs `pip install -e ".[agent]"`) reads `GEMINI_API_KEY` and `CLOUDGUARD_MODEL` from the
environment or a local `.env`, never from the repo.

`src/cloudguard/agent.py` runs the tool-calling loop: every tool call goes through `TOOLS`
(audit-logged), results come back to the model inside `<tool_result>` markers, and the run
stops on a final answer or `max_steps`. After a final answer, `grounding.check_grounding`
verifies every resource ID the model cited actually came from a tool result this run —
if not, `grounding_ok=False` and the run counts as failed even though the model answered.
`AgentRun.findings` holds the deduplicated, deterministic findings from this run's detector
tools, independent of whatever the model's prose claims.

```bash
cloudguard --mock agent "audit this subscription"             # needs GEMINI_API_KEY / .env
cloudguard --mock agent "audit this subscription" --max-steps 4
```

Prints findings (grouped by type, with untrusted-text warnings), the model's answer, the
grounding verdict, any proposals created, and run stats (`run_id`, model, steps, tokens,
attempts, latency). Exits non-zero if grounding failed or the run hit `max_steps`.

### Streamlit UI

```bash
pip install -e ".[agent]"
streamlit run app/streamlit_app.py
```

Sidebar: mock/live toggle, max steps, the request text, a "Run audit" button, and the
active model name. The run only happens on that click — Streamlit reruns the whole script
on every interaction, so the `AgentRun` lives in `st.session_state` and approving or
rejecting a proposal never calls the LLM again. Findings, the answer, the grounding banner,
and pending/decided proposals (read fresh from `proposals.jsonl` on every render, so
decisions survive a restart) follow the same rule as the CLI: anything that came from Azure
data, the model, or a human-entered reason/note renders as plain text or a code block
(`st.text`/`st.code`), never `st.markdown`, so it can't be misread as formatting or acted on.
All of that shaping lives in `app/view.py` as plain, independently unit-tested functions —
`app/streamlit_app.py` itself only wires widgets to it.

"Decided by" is a free-text field in this version — it's self-asserted, not authenticated.
A deployed version would gate approval behind Entra ID sign-in instead.

## The seeded sandbox and ground truth

`sandbox/seed_spec.json` describes a subscription broken on purpose:

- 6 resource groups, 4 unlocked
- 2 unattached Standard HDD disks
- 1 storage account with minimum TLS 1.0, and a look-alike one on TLS 1.2 — the mock's
  TLS 1.0 account is a stand-in for a legacy account provisioned before Azure retired
  TLS 1.0/1.1; `az storage account create --min-tls-version TLS1_0` now exits 0 but Azure
  silently stores TLS1_2, so this account can no longer be recreated against a live
  subscription (see "Running it against a real subscription" below).
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

The auditing identity only needs the built-in **Reader** role. Locks come from the ARM
Management Locks list API (`ManagementLockClient.management_locks.list_at_subscription_level`),
not Resource Graph — Resource Graph does not index `microsoft.authorization/locks`, so a
resource group's lock status has to be resolved from that separate, still read-only, API.

`seed.py` doesn't trust the `create` command's exit code for `weak_tls` ground truth: after
`--apply` it reads back each storage account's `minimumTlsVersion` with `az storage account
show` and only marks the account weak in `ground_truth.live.json` if Azure actually stored a
weak value, printing a warning whenever the stored value differs from the spec. This is why
the weak-TLS account can't be reproduced live — current Azure accepts the request but stores
TLS1_2 regardless.

## RAG corpus (Day 3)

`docs/corpus_manifest.json` pins 30 public Microsoft Learn articles across the 10 topics
`search_docs` needs to cite (locks, policy, Resource Graph, managed disks, storage TLS,
storage security, RBAC, identity, tagging/governance, cost). Each entry names its source
GitHub repo, the exact `source_path`, and the commit SHA it was fetched at — never fetched
from memory, every path and URL verified live before being added.

```bash
python scripts/fetch_corpus.py --resolve   # re-resolve each repo's current commit SHA (git ls-remote, no clone)
python scripts/fetch_corpus.py             # fetch articles + one level of [!INCLUDE ...] files at the pinned SHAs
pytest tests/test_corpus_manifest.py       # schema, unique ids, allowed topics, learn_url prefix, files present
```

`corpus/raw/` (committed, ~420 KB) holds the fetched article Markdown plus
`corpus/raw/includes/<repo>/...` for same-repo includes — much of the real content in these
docs lives in includes, not the article body. `~/`-prefixed includes point at a different
pinned repo (`MicrosoftDocs/reusable-content`) and are generic boilerplate (free-trial
banners, GDPR notices), not topic content, so the script logs and skips them rather than
pinning a fifth repo for them. The one known, pre-existing exception is listed in the
manifest's `known_issues`: `disks-types.md` references an include with the wrong case
(`...-GA-...` vs. the real lowercase `...-ga-...` filename) — an upstream bug in
`azure-compute-docs`, confirmed by listing the real file tree, not guessed. The fetch script
reports it as a failed path and exits non-zero; it does not skip it silently.

## NOTICE — third-party content

`corpus/raw/` contains documentation fetched from these public Microsoft repositories, under
the license stated in **that repository's own `LICENSE` file** (checked directly, not
assumed — `entra-docs` turned out to differ from its sibling repos):

| Repo | Pinned commit | License |
|---|---|---|
| [MicrosoftDocs/azure-docs](https://github.com/MicrosoftDocs/azure-docs) | `8f89cc8a3023d185c2d965e924d5b4d22159ef45` | CC-BY-4.0 |
| [MicrosoftDocs/azure-compute-docs](https://github.com/MicrosoftDocs/azure-compute-docs) | `c21e8e8229e33073fe93faa58a415b64279f3a44` | CC-BY-4.0 |
| [MicrosoftDocs/cloud-adoption-framework](https://github.com/MicrosoftDocs/cloud-adoption-framework) | `5c44a2accff2018eba5dd738b648eef4c9db179b` | CC-BY-4.0 |
| [MicrosoftDocs/entra-docs](https://github.com/MicrosoftDocs/entra-docs) | `8d7c7ea1c9a7bc547e6077cb3703227f2e08b043` | MIT |

© Microsoft Corporation. Fetched verbatim at the pinned commit; no modifications made to the
article text. Full attribution strings (with per-repo LICENSE links) are in
`docs/corpus_manifest.json`.

## Responsible design

- Read-only by construction: Resource Graph has no write API and the ARM Management Locks
  client is only ever called with `list_at_subscription_level`; the auditing identity holds
  Reader only, and a test fails the build if a write-shaped ARM method call appears in the source.
- Remediation is only ever proposed; a human approves (Day 2).
- Findings come from deterministic queries, not from the model, and IDs are checked with
  `get_resource`.
- Names and tags are treated as untrusted input.
- Synthetic data only: my own free subscription, mock JSON and public docs.

## Roadmap

1. ✅ Sandbox, ground truth, read-only tools, mock mode, CI
2. ✅ Agent tool-calling loop, `propose_lock`, approval in Streamlit, injection test
3. RAG over ~30 Microsoft Learn pages, change-request drafts with citations
4. Eval harness: precision/recall, tool selection, hallucinated IDs, retrieval hit rate, cost, latency
5. Terraform + GitHub Actions OIDC deploy to Azure Container Apps
6. FastAPI + Postgres audit history, Streamlit dashboard
7. Docs, architecture diagram, demo

Built with AI-assisted tools (Claude Code); see `CLAUDE.md` for the rules they follow here.
