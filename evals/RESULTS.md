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
