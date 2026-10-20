# Eval results

Written down before the experiment it describes. Sections below are append-only: results are
added under their own heading, and a pre-registered section is never edited after the fact.

## Pre-registered hypothesis: hybrid retrieval (BM25 + vector)

**Hypothesis.** Hybrid retrieval improves section hit@3 over vector-only retrieval.

**Primary metric.** Section hit@3, reported as counts (e.g. 7/13) as well as percentages. The
denominator is the number of in-scope queries that have `expected_sections`.

**Guardrails.**
1. Article-level hit@3 stays at 1.00.
2. Article-level hit@1 drops by no more than 1 query relative to the vector-only baseline.

**Decision rule.** Hybrid becomes the default only if section hit@3 improves by at least 2 queries
and both guardrails hold. Otherwise vector-only stays the default and the result is recorded as
null, with the numbers, under its own heading below.

**Fixed parameters (not tuned on the golden set).**
- BM25: k1 = 1.5, b = 0.75.
- Fusion: reciprocal rank fusion with k = 60.
- No per-field boosts.

**Frozen inputs.**
- Golden set: `evals/retrieval_golden.jsonl`, v1.2, 28 queries (frozen at commit `5490d02`).
- Baseline: `evals/results/retrieval_20261020T011040Z.json` (golden set v1.2, vector-only). All baseline
  numbers are read from that file; none are copied into this document.
- Neither is changed during this experiment. If either has to change, this experiment is void and
  restarts with a new pre-registration.

**Gate: vector mode must reproduce the baseline before hybrid is run.** Hybrid retrieval needs a
refactor of the retrieval path (a `--mode` flag, a shared ranking interface). After that refactor,
and before any bm25 or hybrid run, `cloudguard eval-retrieval --mode vector` must reproduce the
baseline file's metrics **exactly**:
- overall and per-source hit@1, hit@3, hit@5, MRR, section hit@3 (with its denominator);
- every per-query result: hit flags, reciprocal rank, top-3 article ids, section hit;
- every per-query top-1 score, and the out-of-scope top-1 scores.

If anything differs, the refactor changed vector behavior: stop and investigate, and do not run
hybrid. A difference confined to the last digits of a float score is still a difference to report,
not something to loosen silently; any tolerance is a decision recorded in this file, not applied by
default. Timestamps, git commit and index `built_at` are provenance and are not compared.

The check is `python -m cloudguard.evals.baseline_check` (runs `--mode vector` and compares; or
`--candidate FILE` to compare an existing results file). Its comparison logic is unit-tested in
`tests/test_baseline_check.py`; the real run is manual because tests never load the real embedding model.

## Addendum (before implementation), 2026-10-20

Written before any hybrid or BM25 code exists and before any bm25 or hybrid run. The original
pre-registration above is not edited. This addendum records choices the original did not specify,
so none of them can be made, or changed, after seeing results.

**Implementation choices.**
- **Tokenizer:** `re.findall(r"[a-z0-9]+", text.lower())`. No stemming. No camelCase splitting, so
  `CanNotDelete` is a single token `cannotdelete`. Punctuation and hyphens split tokens
  (`lock-resources` becomes `lock`, `resources`).
- **Stopwords (exact list):** `a an and are as at be by can do does for from how i in is it my of on or so that the this to was what when which with`
- **Document text per chunk:** `title + heading path + text`, as one bag of words, no field boosts.
- **IDF variant:** `ln(1 + (N - n + 0.5) / (n + 0.5))` (never negative), with N the number of
  chunks and n the number of chunks containing the term.
- **Query terms:** unique. A term repeated in the query is counted once.
- **BM25 ranking:** chunks with a BM25 score of 0 are not ranked. Ties break by `chunk_id`.
- **RRF input:** the full rankings are fused, with no truncation depth: the complete vector
  ranking (all chunks, ordered by cosine, ties by `chunk_id`) and the BM25 ranking (matching
  chunks only). Score = sum of `1/(60 + rank)` over rankings containing the chunk, 1-based ranks;
  ties break by `chunk_id`.
- **Per-article cap:** the existing cap of 2 chunks per article is applied after fusion, to the
  fused ranking, exactly as vector mode applies it to its own ranking; then truncation to k.
- **Vector mode** keeps its existing code path unchanged.

**Reporting.** In every mode each query also reports the top-1 vector cosine
(`vector_top1_cosine`) next to the mode's own top-1 score (cosine, BM25 or RRF). Scores from
different modes are on different scales and are never compared against each other or against a
single threshold.

**Run protocol additions.**
- The gate above (`baseline_check` on `--mode vector`) runs before any bm25 or hybrid run.
- Each of `--mode bm25` and `--mode hybrid` is run once, at k=5, with the parameters above.
- If anything is fixed after seeing bm25 or hybrid results (a bug, a wrong parameter, anything),
  it is recorded as a **second run** with the reason for the change. Both runs stay in this file,
  with their results files named, and nothing is silently re-run or overwritten.
- BM25-only results are reported for context and are not part of the decision rule.
- The decision rule is applied to the first run unless a recorded second run states, with reasons
  written down before its numbers are read, why it supersedes the first.

## Results: hybrid retrieval (BM25 + vector), 2026-10-20

**Outcome: null. Vector-only stays the default.** Hybrid did not improve section hit@3 and it failed
the hit@3 guardrail.

**Provenance.**
- Implementation commit: `7b3d2b03da2cd843bceff156c4d489782df0f294` (`7b3d2b0`).
- Gate: `python -m cloudguard.evals.baseline_check` was re-run on that commit before the bm25 and
  hybrid runs: vector mode reproduces the baseline exactly (max per-query top-1 score difference 0).
- Results files compared (all k=5, `BAAI/bge-small-en-v1.5`, golden set v1.2, 28 queries: 23 in-scope, 5 out-of-scope):

  | mode | file | `git_commit` recorded in the file |
  |---|---|---|
  | vector (baseline) | `evals/results/retrieval_20261020T011040Z.json` | `f1b9994` (pre-dates the implementation; frozen) |
  | bm25 | `evals/results/retrieval_20261020T013112Z.json` | `7b3d2b0` |
  | hybrid | `evals/results/retrieval_20261020T013113Z.json` | `7b3d2b0` |

- Each of bm25 and hybrid was run exactly once. No re-runs, no parameter changes, no second run.

**Metrics** (in-scope queries, n=23; section hit@3 is over the 13 queries that have `expected_sections`):

| metric | vector | hybrid | bm25 (context only) |
|---|---|---|---|
| **section hit@3 (primary)** | **7/13** | **7/13** | 4/13 |
| article hit@3 | 23/23 | 22/23 | 20/23 |
| article hit@1 | 18/23 | 19/23 | 17/23 |
| MRR | 0.8841 | 0.8913 | 0.8080 |

**Decision rule, applied mechanically.**

| condition | requirement | observed | result |
|---|---|---|---|
| section hit@3 improves | at least +2 queries over vector | 7/13 vs 7/13: +0 | **not met** |
| guardrail: hit@3 stays 1.00 | 23/23 | 22/23 (0.9565) | **fails** |
| guardrail: hit@1 drops by at most 1 query | drop <= 1 | 18/23 to 19/23: +1 | holds |

Hybrid becomes the default only if the improvement is met and both guardrails hold. Neither the
improvement nor the first guardrail holds, so the result is recorded as null and the default mode
is unchanged (`rag.index.DEFAULT_MODE = "vector"`).

**Queries that flipped, hybrid vs vector.**

Section hit@3 (net 0: one lost, one gained):
- lost: "What is a managed identity and when should I use one instead of a service principal?"
- gained: "How can a GitHub Actions pipeline authenticate to Azure without storing a secret?"

Article hit@1:
- gained: "Find the disks that are just sitting there costing us money for no reason"
- gained: "How do I make sure every resource has an owner tag so we can track cost accountability?"
- lost: "What permissions does the built-in Reader role actually grant?"

Article hit@3 (this is the guardrail failure):
- lost: "Make sure all the new resource groups created have a delete lock and apply the locks on the ones that currently exist" (source: `human`). Expected `policy-effect-deploy-if-not-exists` or `policy-remediate-resources`. Vector top 3: `locks-overview`, `policy-effect-deploy-if-not-exists`, `storage-security-blobs`. Hybrid top 3: `locks-overview`, `storage-security-blobs`, `tagging-tag-resources`.
- no query gained article hit@3 (vector already had 23/23).

**Queries that flipped, bm25 vs vector** (context only; bm25 is not part of the decision):
- article hit@1 lost: "Who has permission to create or remove a resource lock?"; "How do I enforce resource group locks automatically with Azure Policy?"; "Give me a sample Resource Graph query that lists storage accounts"; "What permissions does the built-in Reader role actually grant?"
- article hit@1 gained: "Find the disks that are just sitting there costing us money for no reason"; "How can a GitHub Actions pipeline authenticate to Azure without storing a secret?"; "How do I make sure every resource has an owner tag so we can track cost accountability?"
- article hit@3 lost (3 queries, so 20/23): "How do I stop someone deleting prod by mistake?"; "What permissions does the built-in Reader role actually grant?"; the same `human` delete-lock query as above.
- section hit@3 lost: "Who has permission to create or remove a resource lock?"; "How do I set the minimum TLS version required on a storage account?"; "What permissions does the built-in Reader role actually grant?"; "What is a managed identity and when should I use one instead of a service principal?"
- section hit@3 gained: "Why can't my pipeline delete old deployments in a resource group anymore?" (`human`)

**Out-of-scope queries (5), top-1 in each mode's own scale next to the vector cosine.** The baseline
file pre-dates `vector_top1_cosine`; its `top1_score` is the cosine.

| vector cosine (= vector top-1) | bm25 top-1 | hybrid top-1 (RRF) |
|---|---|---|
| 0.6533 | 9.3811 | 0.0264 |
| 0.6385 | 11.1678 | 0.0320 |
| 0.6442 | 12.3693 | 0.0304 |
| 0.6141 | 13.6640 | 0.0315 |
| 0.6837 | 12.1003 | 0.0302 |

The three scales are not comparable and no threshold is derived from this table.

**Caveats.** The section hit@3 denominator is 13, so one query is 7.7 points, and the `human` source
has only 1 section-eligible query. This is one run on one golden set with fixed parameters, as
pre-registered; it says nothing about other parameter settings, which were deliberately not tried.

## Agent eval protocol (pre-registered), 2026-10-20

Written before the agent eval runner exists and before any official agent run. This is a **baseline
measurement, not a hypothesis test**: there is no pass threshold and nothing to accept or reject.
Whatever the official run gives is reported as it is. Sections above are not edited.

**Frozen inputs.**
- Golden set: `evals/agent_golden.jsonl`, 33 cases (26 builder cases + 7 `human`), frozen at commit
  `4204ca84c5c46905de291fcf9e132a6a7bb4687c` (`4204ca8`), file sha256
  `d656d074224bb27de58231a0d6dcecee49376cabeeb29d6cd9678874d346c156`. It is not changed during this
  experiment. If it has to change, this experiment is void and restarts with a new pre-registration.
- The results file records the golden set's sha256 and the git commit the runner ran at.

**Setup.**
- Backend: the mock backend (`mock/subscription_snapshot.json`). No Azure access.
- Model: the configured model (`CLOUDGUARD_MODEL`, default in `llm.py`), recorded by name in the results file. Sampling settings are the client's defaults; none are set by the runner.
- **3 runs per case**, so 99 runs for 33 cases.
- **Retries: production defaults, and a case is never re-run.** The LLM client is built with its production default retry settings (`GeminiClient`: retry on 429 and 503, exponential backoff, default `max_retries`), not a special eval setting. The number of attempts is recorded per run. A run that still fails on a transport error after those retries is reported separately as an **infrastructure failure**, and is counted as a failed run in the primary figures. A whole case or run is never repeated because it failed.
- `max_steps` is the default (`agent.DEFAULT_MAX_STEPS`, currently 8). Retrieval for `search_docs` uses the default mode (`vector`); the index metadata and corpus pins are recorded.
- **Results are read from the run files only.** Every metric is computed from what the run file recorded (tools called and their results, the final answer, proposals, tokens, latency), never from live objects or console output, so a result can be recomputed later from the file alone.

**Isolation.** Each run uses a fresh temporary proposals file and audit log (`CLOUDGUARD_PROPOSALS`, `CLOUDGUARD_AUDIT_LOG`), created empty and discarded after the run's record is saved. Pending proposals from other runs cannot reach the `expected_proposals` check. Runs do not share state.

**Metrics per run (all rule-based).** A check is applicable only where stated; it passes or fails.

| check | applies to | pass condition |
|---|---|---|
| Tool selection | every case | `required_tools` is a subset of the tools called, and `forbidden_tools` intersect tools called is empty |
| Recall | cases with non-empty `expected_ids` | every expected ID is among the IDs extracted from the answer (fraction reported too) |
| False positives | cases with non-empty `forbidden_ids` | zero forbidden IDs among the extracted IDs (count reported too) |
| Hallucinated IDs | every case | zero IDs in the answer that no tool result of this run grounds |
| Proposals | every case | the set of resource groups with a pending proposal at the end of the run equals `expected_proposals` exactly (empty means none) |
| Citations | every case | `citations_ok` is true |
| Docs cases | cases with non-empty `expected_articles` | at least one expected article was returned by `search_docs` in this run **and** is cited in the answer at article level (via `normalize_doc_url`, matched to the article's `learn_url` in the corpus manifest); the fraction of expected articles satisfying both is reported too |
| Abstain cases | `should_abstain` | the answer contains no `learn.microsoft.com` URL, and no resource tool was called (anything other than `search_docs`) |

- IDs are compared as **exact extracted IDs** (`grounding.extract_resource_ids`, case-insensitive, trailing slash ignored), never as substrings. A sub-resource ID does not count as containing its resource group's ID.
- "Grounded" for the hallucinated-ID check is the **production grounding rule** (`grounding.check_grounding`), not a stricter one: an ID returned by a successful tool result of this run (a finding, or `get_resource` data) is grounded, and so is the parent resource group of such an ID. An answer that names the resource group of a grounded disk is therefore not a hallucination.
- Recall, false-positive and hallucination are binary for pass/fail (recall must be 1.0; the other two must be 0); the continuous values are reported as secondary figures.
- **Run failures count as failures: an exception or LLM error, `max_steps` reached without a final answer, and a timeout.** Nothing is dropped, and a failed run fails every applicable check. Transport errors that survive the client's retries are labeled infrastructure failures in the report (counted separately, e.g. "5 of the 21 failed runs were infrastructure failures") but stay in the primary pass/fail figures.
- **Per-run timeout: 120 seconds wall-clock.** It is a runaway guard, not a performance target, and it was fixed here before any results existed. A run that exceeds it is a failed run.

**Case outcome.** A run passes only if every applicable check passes. Per case, report passes out of 3:
- **consistent** = 3/3, **flaky** = 1/3 or 2/3, **failing** = 0/3, with counts of cases in each.
- Results are reported per category and overall in two views, each as k/n: run-level (runs passed / runs) and case-level (consistent cases / cases).

**Not scored here.** The wording of refusals and abstentions (`should_refuse`, and the quality of a `should_abstain` answer beyond the behavioural checks above). Those need a judge in a later step. For these cases only the behavioural checks apply: forbidden tools not called, proposals exactly as expected, no learn URL and no resource tools for abstain cases, and the ID checks where IDs are listed. A `should_refuse` case therefore passes on behaviour even if the refusal is badly worded; this is a known limit of this step.

**Cost and latency.**
- Tokens in and out are measured per run and reported (per run, per case and in total).
- Latency is wall-clock time per run, end to end; the LLM-call-only total is reported alongside. Report median and p95 over runs (nearest-rank p95).
- Dollar cost = measured tokens x a price table in config. **It is reported only if the price table is filled with a source URL and a retrieval date** taken from the provider's official pricing page. Prices are not invented, estimated from memory, or filled from anywhere else. Otherwise only tokens are reported, and no dollar figure appears.

**Run rule.**
- Smoke tests on 2-3 cases are allowed for debugging the harness. Their results are not reported anywhere, including in this file.
- The official run is all 33 cases x 3 runs, executed once, at a recorded commit with a clean working tree.
- Any fix after the official run (a bug, a wrong check, anything) is recorded as a **second run** with the reason. Both runs stay in this file with their results files named, and nothing is silently re-run or overwritten. The first run is the headline result unless a recorded second run states, with reasons written before its numbers are read, why it supersedes it.

### Addendum to the agent eval protocol (before the official run), 2026-10-20: thinking tokens and the price table

Written after the harness smoke test and before the official run. The protocol above is not edited.

**Thinking tokens.** The Gemini API reports thinking tokens separately from visible output: the SDK's
usage metadata has `candidates_token_count` (visible output) and `thoughts_token_count` (thinking) as
separate fields, and Google's thinking guide shows total tokens as input + output + thought. The
runner originally recorded only `candidates_token_count` as `tokens_out`, so **`tokens_out` excluded
thinking tokens**. Gemini 3.5 Flash-Lite has thinking on by default ("On (minimal)"), and the runner sets no
thinking configuration, so the default applies. A two-call probe of the live model before this change
reported no thought tokens for either call (`thoughts_token_count` absent, total = input + visible output), so
the size of the gap on real agent runs is not known until the official run records it.

Fix, made before the official run and before any official result exists:
- Each run records `tokens_thought` as its own field next to `tokens_in` and `tokens_out`; `tokens_out`
  stays visible output only. Run records without the field (such as the unreported smoke file) read as 0.
- The report totals in, out and thought separately.
- **Cost = `tokens_in` x input price + (`tokens_out` + `tokens_thought`) x output price.** Google bills thinking
  tokens at the output rate ("response pricing is the sum of output tokens and thinking tokens").

**Price table (`evals/prices.json`).** Model `gemini-3.5-flash-lite`, Standard tier, paid: input 0.30 USD per 1M
tokens (the same for text, image, video and audio), output 2.50 USD per 1M tokens, which the pricing page labels
"including thinking tokens". Source: https://ai.google.dev/gemini-api/docs/pricing, retrieved 2026-10-20. The rule
above still holds: a dollar figure is reported only because this entry is complete, with prices, a source URL and a date.
Prices on that page can change; the retrieval date is recorded so the figure can be read in context.

### Addendum to the agent eval protocol (before the official run), 2026-10-20: per-run timeout 120 s to 300 s, and per-call latency

Written before the official run. The protocol and the earlier addendum are not edited.

**Timeout.** The per-run timeout (the parent's wall-clock view of the child) changes from **120 seconds to
300 seconds**. It remains a runaway guard, not a performance target.

Reason: the harness smoke runs (not reported) showed 7.5 to 21 seconds per LLM call on average, with
**zero retries** (attempts equal steps in every run), so the time is plain call latency, not backoff. At the
default `max_steps` of 8, eight calls at the slowest observed average (about 21 s) reach about 170 seconds, which
would exceed 120 seconds for a healthy run that simply uses all its steps. A timeout in that situation would
measure API speed, not the agent. A follow-up timed probe of four single calls with the agent's real system prompt and
tool specs agreed: 7.5 to 18.8 seconds per call, one attempt each, no thought tokens reported, and 14 to 41 visible
output tokens per call.

**This change was made before any official result existed, and it was not based on any pass or fail outcome.**
It rests only on observed latency. The smoke runs are not reported anywhere, and no official run has happened. A
run that still exceeds 300 seconds is a failed run (a timeout), exactly as the protocol says.

**Per-call latency (extra detail).** Each run record now also has `llm_call_ms`, the wall time of every
LLM call in call order (retries and backoff inside a call included); the list sums to `llm_ms`. This is extra detail
only. The pre-registered latency metrics are unchanged: wall-clock per run, with the LLM-call-only total alongside,
reported as median and p95 over runs. Records without the field read as an empty list.
