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
| `search_docs(query)` | Chunks of public Microsoft Learn pages to cite, with URL + score | Embedding search over `corpus/index/` |
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
Any `learn.microsoft.com` URL in the answer must likewise have been returned by `search_docs`
in the same run (`citations_ok`; article-level, anchors don't matter, see the `grounding.py`
docstring). A failure prints a visible warning under the unchanged answer; non-Microsoft URLs are
listed as `external_urls` and not graded.
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
reports this as `KNOWN` (not `FAILED`) and still exits 0; any *other* 404 prints `FAILED` and
exits 1, and if this known issue ever starts succeeding (Microsoft fixes the casing), the
script prints a notice to remove the now-stale `known_issues` entry.

### RAG index

`src/cloudguard/rag/` turns `corpus/raw/` into a searchable index:

- **`chunk.py`** strips YAML front-matter (keeping `title`/`ms.date`), splices each
  `[!INCLUDE]` inline from `corpus/raw/includes/`, rewrites relative markdown links
  (`[text](../other/page.md#anchor)`) to absolute Learn URLs resolved against the article's
  own `learn_url` — an image link becomes just its alt text, anything else unresolvable
  (a relative link to a non-`.md` asset) becomes just its link text — then drops DocFX syntax
  with no retrieval value (`:::zone:::`/`:::image:::`/tab-group markers become nothing;
  `> [!NOTE]` becomes plain text; boilerplate trailer sections like "Next steps" are always
  dropped), and splits by H2/H3 heading into 180–320-word chunks with a small overlap —
  never inside a fenced code block. bge-small-en-v1.5 truncates at 512 tokens (~384 words),
  so that range leaves headroom for the `"{title} > {heading}"` context line prefixed onto
  the chunk before embedding (the stored/returned chunk text itself stays bare, for
  snippets). A chunk still over 380 words after normal packing is split further at a
  paragraph (or, failing that, a top-level list-item) boundary — unless it's a table or a
  fenced code block, which are left whole; `cloudguard build-index` prints the `chunk_id`
  and word count of anything still over the limit afterward. Each chunk's `chunk_id` is
  built from the article id and its *full* heading path, so two sections with the same
  subheading (e.g. two "Azure CLI" tabs under different H2s) never collide.
- **`embed.py`**: an `Embedder` protocol, `FastEmbedder` (fastembed, model from
  `CLOUDGUARD_EMBED_MODEL`, default `BAAI/bge-small-en-v1.5`, lazily imported so mock-mode
  tests never need it), and a deterministic `FakeEmbedder` (hashed bag-of-words) for tests.
- **`index.py`**: `build_index()` writes `corpus/index/{vectors.npy, chunks.jsonl,
  meta.json}` (`meta.json` records the manifest's hash and the embedding model name);
  `search()` cosine-ranks and caps results at 2 chunks per article, so one long page can't
  dominate. Loading an index whose manifest or model no longer matches `meta.json` raises
  `StaleIndexError` ("run `cloudguard build-index`") instead of silently searching stale data.

```bash
pip install -e ".[agent]"
cloudguard build-index     # rebuild corpus/index/ after editing the corpus or changing CLOUDGUARD_EMBED_MODEL
```

`corpus/index/` is committed (small — a few hundred KB) so CI and Streamlit work without
rebuilding it. `search_docs` keeps the same tool name, arguments, and audit logging; each
result is `{chunk_id, title, heading, url, snippet (~600 chars), score}`.

### Retrieval eval

`evals/retrieval_golden.jsonl` is a 21-query golden set (spread across locks, policy,
Resource Graph, disks, TLS, RBAC, identity, tagging/cost; a few deliberately word it
differently from the docs, e.g. "stop someone deleting prod by mistake") — each line is
`{query, expected_articles, expected_sections?, source, category?}`, every id/slug
verified against `corpus/index/chunks.jsonl`, not written from memory. `source` is
a free-form label such as `claude-drafted`, `human` or `human-edited` (the by-source report groups on whatever labels appear), so scores can be split by who wrote the query. `category`
defaults to `in_scope`; a line can instead be `"category": "out_of_scope"` with
`expected_articles: []` for a question the agent shouldn't be answering from this corpus
at all (e.g. something with no governance/security content) — those are excluded from
hit@k/MRR entirely rather than scored as misses, and reported separately by their top-1
similarity score.

```bash
cloudguard eval-retrieval [--k 5]   # score corpus/index/ against the golden set
```

`score_ranking()`/`aggregate()` are pure (no `search()`, no embedder), so the metric math
is unit-tested on hand-made rankings in `tests/test_evals_retrieval.py` independent of
anything real retrieval returns. Article-level hit@k dedupes by `article_id` first, so a
page that took 2 of the per-article chunk slots can't inflate its own rank. Each real run
writes `evals/results/retrieval_<UTC timestamp>.json` with the model name,
`corpus/index/meta.json`, the pinned corpus repo SHAs, and this repo's own
`git rev-parse HEAD`, so a score is always reproducible to the exact code + corpus + index
that produced it. The report also includes the min/median/max top-1 similarity score
across in-scope queries, to help pick a score threshold that separates in-scope from
out-of-scope questions.

Measured once, on the committed index, `bge-small-en-v1.5`, k=5 (21/21 `claude-drafted`,
no `human` or `out_of_scope` queries yet):

| metric | value |
|---|---|
| hit@1 | 0.81 |
| hit@3 | 1.00 |
| hit@5 | 1.00 |
| MRR | 0.897 |
| section hit@3 (n=12) | 0.58 |
| top-1 score: min / median / max | 0.64 / 0.79 / 0.87 |

No misses outside top 3 on this set. (An earlier run, before article-level hit@k deduped
by `article_id`, showed "How can a GitHub Actions pipeline authenticate to Azure without
storing a secret?" missing top 3 — that was a duplicate-chunk artifact of the same page
occupying two rank slots, not a real retrieval gap, and the dedup fix resolved it.)

## Change-request drafts (Day 3)

```bash
cloudguard --mock draft-cr <proposal_id>      # agent extra: needs GEMINI_API_KEY and a built index
```

Renders a Markdown change request for a lock proposal and saves it as a `cr_drafted` event in
`proposals.jsonl` (append-only; the latest draft wins; `list_proposals()` ignores these events, so
drafting never changes a proposal's status). Nothing is executed: the CR is marked **draft,
requires human approval** and every `az` command in it is text.

- **Code writes** the resource ID, lock level, risk level (from the finding's severity), the
  implementation command (the proposal's own `suggested_command`, which is re-derived and compared
  first), the rollback command and the verification command, all `shlex`-quoted.
- **The LLM writes** only `summary`, `business_justification`, `risk_narrative` and `impact`, as
  JSON matching a response schema, with retrieved docs as context. Invalid output is retried once,
  then the draft fails with a clear error.
- **Untrusted text** (names, tags, the proposal's reason) goes into the prompt only inside a delimited
  block. The injection tripwire runs over the LLM's output, and flagged resources get a warning line.
- **Reference documentation** lists what the model was given, not what it relied on. URLs the model
  writes itself are checked against that same retrieved set (`grounding.check_citations`); ungrounded
  ones are listed in a warning, never dropped.
- The delete-blocked check is **optional, sandbox only**: it runs against a throwaway group with a
  random name and an `EXIT` trap for cleanup, and is kept out of the implementation steps.
- LLM-written text is escaped so it can't inject Markdown. `draft-cr` exits 0 even with warnings.

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
