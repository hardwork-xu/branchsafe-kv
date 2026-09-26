# Resume material and project explanation

[简体中文](../zh/RESUME.md)

I use these statements as evidence-backed descriptions of the repository. They describe implemented work, not a claim about learning already completed, production adoption, research acceptance or time invested.

## Four project bullets

- Built a bounded NumPy KV store with reference-counted prefix sharing, copy-on-write tail updates, checked write tickets and partial-acceptance transactions; verified visible-state preservation under capacity, allocation, copy and concurrency failures.
- Designed a preregistered comparison against preallocated dense-copy and eager-paged baselines; on Apple M1 Pro CPU float32 with 2048 prefix tokens and 8 branches, reduced median cache-management latency from 22.587 to 4.502 ms (5.02×), with identical stored K/V.
- Reduced live managed KV payload in that case from 152,174,592 to 17,825,792 bytes (11.7% of baseline); separately measured process RSS and showed a read-heavy workload slowed to 0.83×, avoiding an unsupported end-to-end inference claim.
- Integrated a pinned 124M-parameter pretrained GPT-2 cache validation with 45/45 bitwise-identical native-cache logit comparisons, while retaining 41 stricter full-recompute failures; delivered offline tests, frozen dependencies, bilingual documentation and reproducible build/CI configuration.

## Evidence index

| Statement | Implementation | Tests / inputs | Raw evidence |
|---|---|---|---|
| Ownership and failure contract | [core](../../src/branchsafe/core.py), [baseline](../../src/branchsafe/baseline.py) | [core tests](../../tests/test_core.py), [thread tests](../../tests/test_concurrency.py) | [acceptance](../../results/acceptance/checks.json) |
| Latency and fair ablation | [benchmark](../../benchmark.py), [analyzer](../../scripts/analyze.py) | [frozen config](../../configs/benchmark.json), [plan](PLAN.md), [harness tests](../../tests/test_harness.py) | [raw](../../results/benchmark.json), [summary](../../results/summary.json) |
| Memory and adverse scope | [counters](../../src/branchsafe/core.py), [RSS worker](../../benchmark.py) | `primary`, `read_heavy` in the same config | [memory/samples](../../results/benchmark.json) |
| Pretrained integration and delivery | [validator](../../scripts/validate_model.py), [CI](../../.github/workflows/ci.yml) | [adapter tests](../../tests/test_model_adapter.py), prompts/revision in model report | [model](../../results/model.json), [release](RELEASE.md) |

## Short explanation

Why this problem: branching continuations can duplicate long KV prefixes, while sharing creates lifecycle hazards. This week's upstream work supplied a concrete motivation; the underlying COW method is established.

How it works: logical sequences own page references. Fork increments ownership; an append to a shared partial tail first reserves a private page. A transaction writes to an isolated branch and only publishes accepted tokens if the parent version still matches. Capacity and sequence-count limits give explicit failure instead of uncontrolled growth.

Key tradeoff: less copied prefix data in exchange for Python page metadata, locking and expensive contiguous reads. A global lock simplifies the correctness argument and limits compute parallelism. Dense preallocation is deliberately a credible baseline.

Most important experiment: the paired management/read comparison. 5.02× management speedup becomes 1.60× after one materialization, and 0.83× under frequent reads. The storage-bound optimization does not automatically accelerate model inference.

Failure worth discussing: full-recompute float32 logits violated 1e-5 even though native and restored-cache logits were exactly identical. Keep the failed hypothesis and isolate the implementation's contribution rather than changing the metric.

Next research questions: can an attention consumer read pages directly, what is the crossover by page size and branch count, and can finer locking preserve the same failure contract? These are questions, not promised implemented features. There is no guarantee of PhD admission or hiring outcomes.
