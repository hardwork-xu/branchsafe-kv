# Research rationale

[简体中文](../zh/RESEARCH.md) · [Plan](PLAN.md) · [Experiments](EXPERIMENTS.md) · [Verified sources](../sources.json)

I chose branch storage because it exposes both a measurable data movement cost and a correctness question that can be tested on an ordinary development machine. A common prefix may be reused by several continuations; an abandoned continuation must release its resources without changing its parent. Saving copies is useful only when those lifecycle guarantees remain intact.

## Why this week

The research window is **21–27 September 2026**. On 21 September, vLLM merged [PR #56734](https://github.com/vllm-project/vllm/pull/56734), addressing speculative draft writes through stale block-table rows during dummy runs. Its merge is [d2983f2](https://github.com/vllm-project/vllm/commit/d2983f2). This is evidence of active work on cache lifetime correctness this week, not evidence that BranchSafe reproduces that GPU failure or repairs vLLM. The [22 September vLLM Metal announcement](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0) also discusses paged KV storage and explicit memory budgets on Apple Silicon. “Current topic” here means dated upstream engineering activity; no popularity ranking is claimed.

Three candidates were considered: full speculative decoding, finite-state token masking, and branch KV storage. Full decoding introduces draft-model quality and inference costs; masking has mature native engines, including [XGrammar's 24 September release](https://github.com/mlc-ai/xgrammar/releases/tag/v0.2.8). Branch storage offers a narrower independently testable contribution within the available CPU and memory budget. The decision and numerical targets were recorded before performance measurements in [PLAN.md](PLAN.md).

## Question, hypothesis and contribution

**Question:** Under what branch shapes does page sharing reduce storage-management latency and copied bytes, while rollback, rejected writes and resource failures preserve the visible parent state?

**Hypothesis:** Long shared prefixes and several short branches benefit from avoiding prefix copies. Tiny or empty prefixes, Python page-table overhead, and repeated contiguous materialization can erase that benefit. This is falsifiable with the short, empty-prefix, partial-page and read-heavy cases retained in the experiment configuration.

The contribution is an independently implemented, bounded NumPy store with checked sequence tickets, isolated append transactions, recoverable allocation paths, an eager-paged ablation, and a preallocated contiguous-copy baseline. Its research value is the explicit contract and evidence about tradeoffs. It is **not** a new attention algorithm, new COW scheme, inference server, or claim of superiority over vLLM.

## Related work and attribution

- [PagedAttention (SOSP 2023), §4.4–5.2](https://arxiv.org/pdf/2309.06180) already describes block sharing, reference counts, COW, and fork/append/free. BranchSafe reimplements these established storage ideas in a small CPU library; it does not implement the paper's GPU attention kernel or reproduce its throughput results.
- [Transformers' cache guide](https://huggingface.co/docs/transformers/kv_cache#prefill-a-cache-prefix-caching) demonstrates preallocated `StaticCache` and prefix reuse via copying. This motivates a practical dense-copy comparison. The local baseline is independent NumPy code, not a timing of Transformers internals.
- [Generational arena documentation](https://docs.rs/generational-arena/0.2.9/generational_arena/) illustrates established stale-reference defenses. BranchSafe checks sequence identity, epoch and length. It does not expose raw page handles or claim a new solution to the ABA problem.

These are conceptual references; third-party implementation code was not copied into the core. See [NOTICE](../../NOTICE.md) for dependency attribution.

## Storage model

Let `L` be layers, `H` KV heads, `D` head dimension, `s` bytes per scalar, `P` tokens per page, `T` visible prefix tokens, `A` newly appended tokens, and `B` child branches. All arrays have shape `[L,T,H,D]`. The payload for a K/V token pair is

\[
u = 2LHDs, \qquad \text{page bytes} = Pu.
\]

A sequence stores an ordered page table and a visible length. Logical token `t` maps to page `floor(t/P)` and offset `t mod P`. The table has exactly `ceil(T/P)` entries. A physical page's reference count equals the number of live sequence tables containing its slot. Bytes beyond the visible length are unspecified and never returned.

For an append to a shared partial page, the store reserves all required new slots first, copies the valid tail into a private page, copies the new tokens, then publishes the new table and length. Full prefix pages stay shared. If the tail is exclusive, appending uses its unused capacity directly.

For one retained parent and `B` children with equal suffix length `A`, ignoring metadata, let `r=T mod P`. The following is an **expected payload count**, not process memory measurement:

\[
\text{distinct pages} = \lceil T/P\rceil + B\lceil(r+A)/P\rceil
\]

when `A>0`, each child has appended, and all parents/children remain live. For `r=0`, the added pages contain only suffixes; for `r>0`, each child additionally owns its copied partial tail. With `A=0`, forks allocate no new payload pages. The dense baseline instead reserves `S` token slots per sequence, using `(B+1)Su` bytes. The benchmark sets the same final storage requirement and overall byte budget for all methods.

## Correctness argument and invariants

1. A new sequence has no visible tokens. Fork either copies the visible bytes or increases references to identical pages.
2. A shared page is never modified in its visible range. Append uses private storage when its tail is shared. Therefore a child's writes cannot alter another sequence's visible prefix.
3. Truncation changes visibility and releases whole unused pages. Subsequent writes still check ownership, so an earlier truncated tail cannot corrupt a sibling.
4. A transaction snapshots `(sequence_id, epoch, length)` and owns an isolated child. Commit succeeds only if the parent still matches that snapshot; it adopts the accepted prefix and increments the epoch. Rollback releases the child.
5. Allocation capacity is checked before mutation. Tested recoverable allocation/copy errors preserve visible data and reclaim live reservations. This is not a formal proof against arbitrary interpreter OOM, asynchronous termination, or hostile mutation of private fields.

Storage equality is bitwise: no compression or quantization occurs. A downstream model integration has its own numerical tolerance, frozen as `max_abs_logit_error <= 1e-5` with identical greedy IDs. Numerical agreement is a correctness check, not an output-quality improvement.

## Complexity and boundaries

| Operation | Shared paged store | Preallocated dense baseline |
|---|---|---|
| Fork | `O(ceil(T/P))` metadata, zero prefix payload copy | `O(TLHD)` prefix copying; reserves `S` capacity |
| Append | `O(ALHD + PLHD)` worst case for partial-tail COW, plus page-table copy | `O(ALHD)` into reserved capacity |
| Truncate | `O(number of released pages)` | `O(1)` visibility change |
| Materialize | `O(TLHD)` and a fresh output allocation | `O(TLHD)` and a fresh output allocation |
| Commit | Page-table ownership transfer plus reference releases | Array ownership transfer |

The Python paged append also copies the page-table list, costing `O(ceil(T/P))`; it is not constant-time at long context. A pool-wide lock serializes operations. `max_pages` limits retained NumPy payload, while `max_sequences` bounds live sequence count; neither is a total process RSS limit. The study covers conventional full-attention K/V tensors, not rolling or recurrent model state, device kernels, distributed leases, or malicious callers. Remaining research questions include direct paged-attention consumers, finer-grained locking, and scheduler admission policies; none is promised by this release.
