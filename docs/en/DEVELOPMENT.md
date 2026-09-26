# Development record

[简体中文](../zh/DEVELOPMENT.md)

I record the actual local stages here so that the implementation and its evidence can be revisited. This was one continuous delivery session on 2026-09-27 in Asia/Shanghai (raw timestamps use UTC); no multi-week research history or production deployment is implied.

1. Inspected the empty non-Git workspace, CPU/RAM/disk, compiler/Python, public download access and Docker absence. Created only this isolated project directory and local dependency environments. A previously used public GitHub identity was verified; the repository uses that existing public name and privacy email, without changing global Git identity or publishing the global private email.
2. Compared three directions, verified dated upstream sources, and committed scope and numeric targets before implementation. The core mechanism is established COW combined with explicit transactional lifecycle contracts, not a new attention algorithm.
3. Implemented the bounded store and practical dense baseline. Review exposed allocation-after-reservation cleanup, direct/shallow-copy handle ownership and unlimited empty-sequence metadata; added cleanup, registry identity checks and max_sequences. Added meaningful failure-injection regressions alongside real-array and real-thread tests.
4. Added CLI, NPZ workflow, seeded randomized benchmark and separate fresh-process RSS observations. The first benchmark completed and was retained. Review then found page-rounded budget estimation, worker failure recording and missing-sample validation gaps; fixed them and reran the original workload fully. The original numerical targets were not altered.
5. Downloaded fixed-revision safetensors and validated real GPT-2 inference. Native-cache logits were bitwise equal in 45 comparisons. Full-recompute float32 error exceeded the strict bound in 41 cases; kept the raw failure and exit status rather than changing the threshold.
6. Generated tables/plots directly from raw JSON, wrote matched English/Chinese documentation and checked installation, packaging, examples, types, formatting, citations and public content. Publication and hosted CI have their own evidence, separate from local completion.

## Actual commits at documentation assembly

| Commit | Subject |
|---|---|
| 8ac7d75 | docs: freeze research scope and benchmark targets |
| f10e76c | feat: implement bounded transactional copy-on-write KV storage |
| 8aa4a0e | test: verify cache invariants and expose bilingual workflows |
| 4265c9b | build: add reproducible benchmark and acceptance tooling |
| c7931f5 | perf: record cache benchmarks and pretrained equivalence evidence |

These hashes came from git log, not a reconstructed narrative. Later documentation/acceptance/publication commits are visible in the actual repository history. Commit bodies contain corresponding Chinese descriptions and observed validation.

## Validation and bounded conclusions

The complete offline suite reached 151 passing tests before release checks; no GPU tests were skipped to imply support. Tests include independent random state trajectories, NumPy reference comparisons, input/output alias isolation, concurrent ticket/CAS races and recoverable allocation/copy failures. The store is not a crash-consistent database or malicious-process security boundary.

[Final benchmark](../../results/benchmark.json) and [initial benchmark](../../results/benchmark-initial.json) retain all valid data. [Model results](../../results/model.json) preserve the stronger diagnostic failure. [Acceptance](../../results/acceptance/checks.json) and [supplemental checks](../../results/acceptance/supplemental.json) record actual commands, exit states and source hashes. Local Docker remained unavailable; hosted container verification, if completed, is recorded independently in [Release](RELEASE.md).

There is no measured cost/token-saving claim or fabricated personal time history. Future changes must rerun affected tests, preserve old evidence and update both languages; see [AGENTS](../../AGENTS.md).
