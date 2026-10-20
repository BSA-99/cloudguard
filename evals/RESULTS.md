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
