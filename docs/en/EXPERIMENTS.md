# Experiments

[简体中文](../zh/EXPERIMENTS.md)

## Frozen hypotheses and actual conditions

The targets in [PLAN](PLAN.md) were committed as `8ac7d75` before implementation and timing. No target, baseline, workload or acceptance tolerance was lowered after observing results. The final cache run is `bc2deb1c-429c-420a-82ac-b6e003fc2b77`, started `2026-09-26T16:55:18.930061+00:00` and completed `2026-09-26T16:55:59.563741+00:00`. The raw record holds the original command, full config, revision, per-file SHA-256, versions and every sample. Its revision can precede the evidence commit: the per-file hashes bind the tested working-tree code.

Hardware: Apple M1 Pro, 10 logical CPUs, 16 GiB shared memory; macOS arm64, Python 3.12.2, NumPy 2.3.3. Benchmark operations run on CPU with numerical worker environment variables fixed to one before import. The project does not call BLAS-heavy attention kernels here. Metal was discovered and a separate PyTorch MPS sum smoke check passed; this does not establish a supported project GPU backend. CUDA was unavailable. Docker was not installed locally. Public package and model downloads succeeded; no paid API was used.

The host was not isolated from other applications. No affinity, thermal lock or power instrumentation was available. Seeded method-order randomization and IQR describe variability but cannot remove shared-host interference. The two recorded development runs differ in timing and harness hardening; neither is discarded, and only the final run is used for claims.

## Fair work and cost boundaries

Each case preloads one parent. It creates all child branches while keeping the parent and siblings alive, appends each branch's distinct values, truncates to the accepted length, then releases every child, the parent and the pool. These are independent accepted branches; they are **not eight successful transaction commits against the same parent epoch**. Transaction publication is tested separately by unit tests, demo and model integration.

The practical dense baseline allocates a contiguous capacity of prefix+append tokens per sequence and copies only visible prefix bytes on fork. Appends never concatenate old history. The eager-paged ablation uses the same pool and lifecycle as sharing but copies pages on fork. All methods use identical data, dtype, input validation and completed work, the same total byte budget, and exclusive-copy materialization semantics. The budget is a ceiling, not a demand to consume equal bytes. Dense has fixed per-sequence capacity, while paged storage allocates lazily; that is an intentional architectural difference.

Initialization and prefix copying are timed as `preload_ns`, outside `elapsed_ns`. The timed management region includes allocation, validation, fork, append, truncate, release and pool destruction. Input generation is outside timing for all methods. The second scope also materializes every branch into independent contiguous output arrays; it is a complete **storage** lifecycle, not end-to-end inference. A native dense consumer could read internal storage without a copy, but this API deliberately promises owned copies. Do not interpret the materialization comparison as an unavoidable inference cost.

All geometry is 4 layers × 4 KV heads × 64 dimensions, float32, page size 16; suffix=16 tokens, accepted=8. The seven workloads in [config](../../configs/benchmark.json) cover prefixes 0,32,256,2047,2048,4096 and 1,4,8 branches. `read_heavy` performs four reads of each child. The aligned primary case is favorable to sharing; `partial_page` explicitly exercises a copied shared tail.

There are 3 warmup and 21 measured repetitions per method/case/scope, giving 1008 stored samples, including 126 warmups and 882 timed observations used in aggregates. Randomized within-repetition order, raw order indices, median and IQR are retained. Warmups are marked rather than deleted. No outliers are removed and no p95/p99 or statistical significance claim is made. Throughput in [summary](../../results/summary.json) means completed branches per second for that storage lifecycle, not tokens per second or model requests per second. Cold process/import costs are not steady-state latency; preload and the separate RSS process are labeled accordingly.

## Target versus measured

| Target / 目标 | Measured / 实测 | Status / 状态 |
|---|---|---|
| Primary management ≥2.0× | 5.02×; 22.587 → 4.502 ms | passed / 通过 |
| Managed live payload ≤30% | 11.7%; 152,174,592 → 17,825,792 bytes | passed / 通过 |
| Exact K/V, every benchmark case/method | 21/21 case-method checks | passed / 通过 |
| Native GPT-2 logits ≤1e-5 + matching greedy IDs | 45/45; maximum error 0; bitwise identical | passed / 通过 |
| Stronger full-recompute logits ≤1e-5 | 4 passed, 41 failed; max 0.000701904296875 | failed / 失败 |


Primary management median/IQR: dense 22.587 ms [20.199,24.428], shared 4.502 ms [2.970,5.144]. Including materialization gives 40.748 versus 25.481 ms, only 1.60×. The read-heavy case drops to 0.83×, so the optimization loses in a relevant adverse workload. Short-prefix materialization also has no reliable improvement. The successful primary target does not imply universal acceleration.

Full generated results: [English tables](../../results/TABLE_en.md), [Chinese tables](../../results/TABLE_zh.md), [raw JSON](../../results/benchmark.json), [summary JSON](../../results/summary.json), [SVG](../../results/benchmark.svg), [PNG](../../results/benchmark.png). Eager-paged is slower than shared and generally slower than dense: paging overhead alone is not the source of the benefit. Sharing avoids prefix copying; Python page-table work and gathering are costs, not free operations.

## Three distinct memory measurements

`managed_peak_bytes` is the maximum sum of K/V page capacities with live ownership, including reservation peaks and the parent; dense counts live preallocated array capacity. It is a mechanism counter derived from real owned allocations, not sampled RSS. `allocated_bytes_at_branch_peak` additionally reports retained backing arrays at the branch-peak observation, but is not a sampled process peak; freed pages can remain allocated until pool close. Neither includes Python metadata, input arrays, temporary finite-check arrays or returned materializations.

Fresh worker process high-water RSS uses `resource.getrusage`: macOS bytes, Linux KiB converted to bytes. It includes interpreter/imports, inputs, prefix construction, read workflow and allocator behavior. One observation per case/method is descriptive, not a distribution. Primary RSS: dense 225,574,912 bytes; eager 226,099,200; shared 93,995,008. Do not substitute the 11.7% live-payload ratio for an RSS ratio. Counters for copied bytes include preload plus successful append/fork/COW, exclude materialization and failed partial copies, and are not hardware memory-traffic counters.

## Pretrained validation and a preserved failed hypothesis

[GPT-2 raw report](../../results/model.json) uses 124,439,808 pretrained parameters, `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, safetensors, CPU float32, one Torch thread and eager attention. The file lists every downloaded artifact SHA-256, original synthetic prompts, token IDs, all comparisons and cleanup. Three prompts each branch into two suffixes, with cached continuation steps; three additional scenarios exercise rollback and partial commit. This is real pretrained forward computation, not a random-weight model or cached fabricated output.

The native Transformers cache and the store adapter give exactly identical K/V and logits in all 45 comparisons. The stronger full-recomputation comparison exceeds the frozen 1e-5 absolute threshold in 41 cases; max is 0.000701904296875. Native-versus-full and adapted-versus-full errors match on every comparison, and all greedy IDs agree. This isolates zero additional storage error in the tested path; it does not establish identical distributions over longer generation or overall model quality. Differing float32 computation shapes are a plausible explanation, consistent with [PyTorch numerical-accuracy documentation](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html), rather than evidence for changing the tolerance. The stricter diagnostic remains failed and the script exits 1. No assertion was removed and no tolerance was widened.

Model downloads, hash verification and loading are separate diagnostic intervals. Per-forward/conversion times are recorded but not used for speedup claims. Model weights are excluded from Git.

## Development runs, failures and reproduction

[Initial cache run](../../results/benchmark-initial.json) and [initial model run](../../results/model-initial.json) are preserved intact. After the first cache run, review found a rounded page-budget guard and failure-recording/completeness checks that needed hardening. The final run reran the original unmodified workload; old samples were not merged or selectively removed. Known model failures are independent of those harness changes. Unit tests inject allocation/copy failures to verify recovery, while all normal array paths and thread tests execute real code.

```bash
uv sync --frozen
uv run python benchmark.py --output results/benchmark.json
uv run python scripts/analyze.py
uv run python scripts/update_readme.py
uv run python scripts/validate.py
uv sync --frozen --group model
uv run --group model python scripts/validate_model.py --output results/model.json
```

The final command currently returns 1 for the documented diagnostic failure. Re-running a benchmark intentionally creates a new run ID/time; preserve an earlier report before replacing it. Model offline mode is `--offline` after downloading the pinned snapshot. Do not run model validation alongside timing. [Acceptance logs](../../results/acceptance/checks.json) and the [supplemental record](../../results/acceptance/supplemental.json) bind commands and actual outcomes to source hashes. Unavailable hardware is not counted as passed or as a supported backend.
