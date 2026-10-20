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
- [x] `src/cloudguard/rag/`: `chunk.py` strips front-matter/DocFX syntax, splices includes, splits by
      H2/H3 into 180-320-word chunks (never inside a fenced code block) with small overlap; `embed.py`
      (`FastEmbedder`/fastembed, `FakeEmbedder` for tests); `index.py` (`build_index`/`search`, numpy
      cosine, stale-index check on manifest hash + model name, 2-chunks-per-article cap). `search_docs`
      keeps the same name/args/audit-logging; `cloudguard build-index` added; Day-1 keyword index removed.
      Follow-up: relative markdown links rewritten to absolute Learn URLs (images/unresolvable links become
      plain text); oversized (>380-word) chunks that aren't a table/code block are split further at a
      paragraph or list-item boundary, with remaining ones (genuine tables, or a boundary-less nested list)
      printed by `build-index`
- [x] Retrieval eval: `evals/retrieval_golden.jsonl` (21 queries, proposed by Claude, reviewed/edited by the
      human before scoring, every id/slug verified against `corpus/index/chunks.jsonl`; `source` field
      (free-form label, e.g. `claude-drafted`/`human`/`human-edited`; by-source metrics group on whatever appears) and optional `category: out_of_scope` tracked per line); `cloudguard
      eval-retrieval [--k 5]` reports article-level hit@1/3/5 + MRR (deduped by `article_id` first) and
      section-level hit@3 where `expected_sections` is given, split by `source`; excludes `out_of_scope`
      queries from those metrics and reports them separately with their top-1 score, plus the
      min/median/max top-1 score across in-scope queries (for picking a score threshold); lists queries
      missing their expected article from the top 3; writes `evals/results/retrieval_<ts>.json` with model
      name, index meta, corpus SHAs and git commit; metric math (`score_ranking`/`aggregate`) is pure and
      unit-tested on hand-made rankings. Measured once on the committed index (21/21 `claude-drafted`, no
      `human`/`out_of_scope` queries yet): hit@1=0.81, hit@3=1.00, hit@5=1.00, MRR=0.897, section
      hit@3=0.58 (n=12), top-1 score min/median/max=0.64/0.79/0.87.
- [x] Citation grounding: any learn.microsoft.com URL in a final answer must come from a `search_docs` result
      in the same run. Article-level rule (`grounding.normalize_doc_url`: lowercase scheme/host, strip query,
      locale, trailing slash, split `#anchor`; anchor never affects grounding, `anchor_matched` is a separate
      precision signal). `check_citations` (pure, also called by `check_grounding`) reports `cited_urls`,
      `ungrounded_urls`, `external_urls` (non-Learn hosts, listed but not graded) and `citations_ok`; these go
      on `AgentRun` and into the `grounding_check` audit entry. Failure is a visible CLI warning under the
      unchanged answer, not an exit-code change or a rewrite. Reusable for the change-request draft below.
- [x] Change-request draft (`change_request.py`, `cloudguard draft-cr <proposal_id>`): code writes the resource ID,
      lock level, risk level, implementation/rollback/test commands (`shlex`-quoted); the LLM writes only summary,
      business_justification, risk_narrative and impact via `response_schema` (one retry, then a clear error).
      Untrusted text is delimited in the prompt and the tripwire scans the output; URLs the model writes are checked
      by `check_citations`; "Reference documentation" lists the retrieved context. Stored as an append-only
      `cr_drafted` event that `list_proposals()` ignores; Markdown output escapes LLM text; the delete-blocked check
      is optional, sandbox-only, scratch group with cleanup trap.

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
