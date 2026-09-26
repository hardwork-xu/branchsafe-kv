# Preregistered plan

Frozen before implementation and timing on 2026-09-27 (Asia/Shanghai).

I chose BranchSafe KV to study a narrow systems question: can prefix sharing reduce branch-management copying while transaction and resource failures preserve every visible parent byte?

Candidates: speculative decoding requires draft-model quality/cost alignment and long experiments; finite-state token masking has mature highly optimized native competitors; transactional KV branch storage exposes a measurable CPU memory bottleneck and testable lifecycle invariants. The third option fits this 16 GiB Apple M1 Pro host.

Contribution: independently implemented bounded page store, checked sequence write tickets, failure-atomic append reservations, branch transactions and an auditable evaluation. Paging, reference counting and copy-on-write are established methods, not algorithmic novelty. No production vLLM integration, GPU kernel, model training, serving fleet or universal speedup claim.

## Target (fixed before timing)
- Primary: at least 2.0x lower median branch-management latency than a preallocated dense-copy baseline for prefix=2048, branches=8, append=16, accepted=8, geometry=(4 layers,4 KV heads,64 head dimension), float32, page=16.
- Managed live KV payload at peak at most 30% of dense baseline in that case. This is allocator-owned array nbytes, NOT RSS.
- Exact bitwise equality of stored K/V for all tests and workloads. Pretrained logits max absolute error <=1e-5, identical greedy token IDs.
- Zero parent mutation and zero leaked live pages on invalid input, capacity failure, rollback, stale write and repeated cleanup.

## Method
Dense baseline preallocates sequence capacity and copies only used prefix bytes on fork. Eager-paged ablation copies pages; shared-paged variant retains them and copies only an appended shared partial page. All use equal dtype, input arrays, final accepted lengths and single-thread execution. Include initialization/preload separately, warm-up (3), 21 measured repetitions per method, seeded randomized method order, no outlier removal. Report median, IQR, raw samples, operations/sec; no p99 claim from 21 runs. Report branch management and separately management plus materialization of every branch. Root input generation and external model loading are outside timed intervals; all append validation, allocation, fork, truncate, release and commit inside their respective paths. Raw results include revision and content hash, environment and all failures.

Cases: short prefix (32,1 branch), medium (256,4), primary (2048,8), long (4096,4), partial-page (2047,8), no sharing (0,8), read-heavy (2048,8). Boundary inputs include zero tokens, page edges, exhausted capacity, NaN/Inf, nested forks and concurrent stale tickets. Synthetic Gaussian data is labeled synthetic; optional real pretrained GPT-2 validation is reported separately, not production inference timing.

Expected: dense fork is O(L H D T), shared fork O(ceil(T/P)) metadata, shared append copies at most one page plus new tokens. Contiguous reads remain O(L H D T) and can erase sharing gains. Primary latency target is a hypothesis, not an achieved result.

Acceptance also requires offline unit/integration tests, CLI/API, locked clean installation, formatting, strict typing, package build, paired docs, raw evidence and release/privacy checks. Docker is absent: container config must be supplied and marked not locally run unless environment changes. Publication uses only inspected project files and a verified public Git identity; no model weights or private machine paths.
