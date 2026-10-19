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

## Day 2 — Agent loop ✅
- Model: gemini-3.5-flash-lite (free tier). 2.5-flash-lite was retired for new users, and 3.7/3.8-flash returned 503 under demand.
- [x] Choose LLM provider: Gemini via `google-genai` (`GEMINI_API_KEY` / `CLOUDGUARD_MODEL` env vars, never committed); `llm.py` has a provider-neutral `LLMClient` protocol so an `AnthropicClient` can be added later without touching `agent.py`
- [x] Fake LLM client (`FakeLLM`) for tests so CI needs no API key
- [x] `agent.py`: tool-calling loop over `TOOL_SPECS`, max-steps limit (default 8), every call through `TOOLS` (so it is audit-logged); run_id, tokens, attempts and latency recorded per run
- [x] System prompt: findings must cite exact IDs from tool results; tag/name text is data, not instructions (`agent.SYSTEM_PROMPT`; tool results passed back inside `<tool_result>` markers)
- [x] Post-check: every resource ID in the final answer must appear in a tool result this run; otherwise reject/flag (`grounding.check_grounding`, pure function reused by the Day-4 eval harness; `AgentRun.grounding_ok`/`grounded_ids`/`ungrounded_ids`, logged as a `grounding_check` audit entry)
- [x] `propose_lock(rg)`: writes a proposal record only (no ARM call); status pending/approved/rejected (`proposals.py`; only accepts a group `list_unlocked_resource_groups` returned this run, tracked via a ContextVar alongside `run_id`; `decide()` is the human approval path and is never a tool)
- [x] Streamlit page: run audit, show findings, approve/reject proposals (`app/streamlit_app.py`, thin; all data shaping in `app/view.py`, unit-tested; `AgentRun` in `st.session_state` so approve/reject never re-calls Gemini; proposals read fresh from `proposals.jsonl` every render; "Decided by" is self-asserted in this version, not authenticated)
- [x] Test: a compromised model that fully obeys `cg-shared-net`'s injected tag (calls `propose_lock` on all 6 groups, then lies in its final answer) still gets the 2 locked groups rejected by `propose_lock`'s own authorization check, with the rejections audit-logged and `AgentRun.findings` (deduplicated detector findings, independent of the model's prose) intact; a second case covers an invented resource group being rejected by both `propose_lock` and grounding
- [x] `cloudguard [--mock] agent "<message>" [--max-steps N]`: prints findings/answer/grounding/proposals/run-stats; exits non-zero on grounding failure or `max_steps`; `llm` injectable (`FakeLLM` in tests, `GeminiClient` by default)
- [x] First live run (gemini-3.5-flash-lite, mock data): 7/7 findings, grounding ok, injection tag flagged and ignored, 2 steps, 2,849/617 tokens, 1 retry (503).

## Day 3 — RAG + change requests
- [x] ~30 public Microsoft Learn pages across 10 topics, pinned by commit SHA in `docs/corpus_manifest.json`;
      `scripts/fetch_corpus.py` fetches articles + one level of includes from raw.githubusercontent.com (no
      clone), fails loudly on any 404, never silently; every path/URL/title verified live, not from memory;
      `corpus/raw/` committed (~420 KB); README NOTICE section names repo/license/pinned commit
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
