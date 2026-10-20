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
