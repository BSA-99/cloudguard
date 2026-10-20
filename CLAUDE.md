# CLAUDE.md — context for AI coding tools working in this repo

CloudGuard is a read-only LLM agent that audits an Azure subscription for governance and
security risks, explains findings with Microsoft docs citations, and drafts change requests
behind a human approval gate. Portfolio project; eval numbers go on a resume, so they must
be real and reproducible.

## Non-negotiable rules
- **Read-only.** No code path may create, modify or delete Azure resources. Detection uses
  Azure Resource Graph for resource groups/disks/storage accounts, and the read-only ARM
  Management Locks list API (`ManagementLockClient.management_locks.list_at_subscription_level`)
  for locks — Resource Graph does not index `microsoft.authorization/locks`.
  `tests/test_safety.py::test_no_write_calls_anywhere_in_source` enforces read-only-ness by
  banning write-shaped method calls (`.create_or_update*`, `.delete*`, `.begin_*`); do not weaken it.
- **Remediation is proposed, never executed.** Future `propose_lock` writes a proposal record
  for human approval, nothing else.
- **Every tool call is audit-logged** via the `_audited` decorator in `tools.py`. New tools must use it.
- **Resource names and tags are untrusted input.** Never let them change control flow. Keep
  them visible, flag them with `untrusted.scan_fields`.
- **No invented resource IDs.** Any ID the agent states must have come from a tool result;
  `get_resource` returns not-found for unknown IDs.
- **Doc citations must be retrieved.** Any learn.microsoft.com URL in a final answer or change
  request must come from a `search_docs` result in the same run (`grounding.check_citations`).
  The rule is article-level on purpose: URLs are normalized (host/scheme lowercased, query, locale
  and trailing slash stripped, `#anchor` split off) and matched on article path only. Retrieval
  returns section chunks and heading slugs can drift, so anchor-strictness would false-alarm on
  real retrieved articles; `anchor_matched` records the finer signal. Failure shows a warning, it
  never silently removes or rewrites text. New answer-producing paths (e.g. change-request draft)
  must call `check_citations` too.
- **Change-request commands come from code.** In `change_request.py` the LLM writes only the four
  prose fields; every command is built from the validated resource group ID and constants, never
  from names, tags, the proposal reason or LLM output. The delete-blocked check targets a scratch
  group only, never the real one. `cr_drafted` events must not change `list_proposals()`.
- **Validate IDs before they touch KQL** (`backends.validate_resource_id`).
- **No employer data.** Only the synthetic sandbox, mock JSON and public Microsoft docs.
  No secrets in the repo; auth is DefaultAzureCredential (managed identity / OIDC / az login).
  The LLM key (`GEMINI_API_KEY`) and model (`CLOUDGUARD_MODEL`) are env vars only, same rule.
- **Ground truth comes from `sandbox/seed_spec.json`, never from running the detectors.**

## Layout
- `src/cloudguard/queries.py` — KQL strings (source of truth for live mode)
- `src/cloudguard/backends.py` — `MockBackend` (Python mirror of the KQL) and `ResourceGraphBackend`
  (KQL for groups/disks/storage, ARM `ManagementLockClient` for locks; `lock_scope` derives a
  lock's scope from its own ID, shared by both backends)
- `src/cloudguard/tools.py` — the tools, `TOOLS` registry, `TOOL_SPECS` JSON schemas
- `src/cloudguard/llm.py` — provider-neutral `LLMClient` protocol; `GeminiClient` (live) and
  `FakeLLM` (scripted, used by tests/CI); `tool_specs_to_gemini` converts `TOOL_SPECS`
- `src/cloudguard/sandbox.py` + `sandbox/seed_spec.json` — spec -> az commands, ground truth, mock snapshot
- `src/cloudguard/untrusted.py` — prompt-injection tripwire
- `src/cloudguard/audit_log.py` — JSONL audit log
- `docs/corpus.json` — Day-1 docs index (replaced by RAG on Day 3)

## Workflow
- The roadmap and progress checklist live in `docs/PLAN.md`. Work on one unchecked item at a time, tick it when done.
- `pip install -e ".[dev]"` then `pytest -q` (mock mode, no Azure needed). Keep it green on every commit.
- After editing `seed_spec.json`: `python sandbox/seed.py --mock-only` and commit the regenerated files.
- If you change a KQL query, change the matching `MockBackend` method in the same commit,
  and run `pytest -m live` against the sandbox before claiming parity.
- Prefer small, reviewable diffs. Add a test for every new behavior and every bug fixed.
