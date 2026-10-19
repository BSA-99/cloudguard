# CloudGuard build plan

Work one day at a time. At the end of each day: tests green, README "Status" line updated,
the boxes below ticked, one commit (or a few small ones). Eval numbers are written down only
after they are measured.

## Day 1 — Sandbox + read-only tools ✅
- [x] `sandbox/seed_spec.json`: 6 RGs (4 unlocked), 2 unattached Standard HDD disks, 1 TLS 1.0 storage account, traps
- [x] `ground_truth.json` generated from the spec (not from the tools)
- [x] Read-only KQL tools: `list_unlocked_resource_groups`, `list_unattached_disks`, `find_weak_tls`, `get_resource`, `search_docs`
- [x] `--mock` mode from saved JSON; audit log; untrusted-text flags; CI
- [ ] Seed the real free subscription (`python sandbox/seed.py --apply`)
- [ ] Verify `RESOURCE_GROUPS` in Resource Graph Explorer returns all 6 groups (lock status now comes from the ARM Management Locks API, not this query)
- [ ] `pytest -m live` passes; record whether Azure accepted TLS 1.0 (it may not, post Feb 2026)
- [ ] Replace synthetic mock with recorded one if useful (`cloudguard snapshot -o mock/live.json`, scrub sub ID)

## Day 2 — Agent loop
- [x] Choose LLM provider: Gemini via `google-genai` (`GEMINI_API_KEY` / `CLOUDGUARD_MODEL` env vars, never committed); `llm.py` has a provider-neutral `LLMClient` protocol so an `AnthropicClient` can be added later without touching `agent.py`
- [x] Fake LLM client (`FakeLLM`) for tests so CI needs no API key
- [ ] `agent.py`: tool-calling loop over `TOOL_SPECS`, max-steps limit, every call through `TOOLS` (so it is audit-logged)
- [ ] System prompt: findings must cite exact IDs from tool results; tag/name text is data, not instructions
- [ ] Post-check: every resource ID in the final answer must appear in a tool result this run; otherwise reject/flag
- [ ] `propose_lock(rg)`: writes a proposal record only (no ARM call); status pending/approved/rejected
- [ ] Streamlit page: run audit, show findings, approve/reject proposals
- [ ] Test: injection tag in `cg-shared-net` does not change the verdict and does not trigger `propose_lock` on every RG

## Day 3 — RAG + change requests
- [ ] ~30 public Microsoft Learn pages (locks, Policy, TLS, managed disks, Reader/RBAC); store URL + chunk
- [ ] Embeddings + Chroma (or numpy cosine); `search_docs` keeps the same signature
- [ ] Change-request draft: summary, risk, rollback, test plan, citations (URLs from retrieval only)

## Day 4 — Evals
- [ ] ~25-case golden set (questions + expected tools + expected IDs)
- [ ] Metrics: precision/recall vs ground truth, tool-selection accuracy, hallucinated IDs, retrieval hit rate, LLM-judge rubric for CRs, cost per run, latency
- [ ] Run in GitHub Actions (key as repo secret, manual trigger)
- [ ] One improvement with before/after numbers recorded in `evals/RESULTS.md`

## Day 5 — Deploy
- [ ] Terraform: Container Apps env + app, managed identity with Reader
- [ ] GitHub Actions OIDC (federated credential), no client secrets
- [ ] Token, cost, latency logging

## Day 6 — API + history
- [ ] FastAPI endpoints; Postgres audit history (mirror of JSONL)
- [ ] Streamlit dashboard; cost findings

## Day 7 — Ship
- [ ] README: pitch, architecture diagram, eval table (measured only), demo GIF, Responsible AI, AI-assisted note
- [ ] Live demo in mock mode, linked from portfolio
