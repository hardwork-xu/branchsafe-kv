# Code walkthrough

[简体中文](../zh/WALKTHROUGH.md) · [API](ARCHITECTURE.md) · [Research](RESEARCH.md)

I use the following reading route to keep the ownership rules visible while making changes. Start with one sequence and one partial page; add concurrency only after the single-branch invariants make sense. This guide describes the implementation, not a claim about the reader's prior experience.

## 1. Run the complete offline path

From the repository root:

```sh
uv sync --frozen
uv run branchsafe demo
uv run pytest -q tests/test_core.py tests/test_concurrency.py
```

[cli.py](../../src/branchsafe/cli.py) maps `demo` to `demo()`. It creates 17 prefix tokens, forks a child, accepts three of five appended child tokens, and discards an uncommitted parent transaction. It checks exact array equality and cleanup before returning JSON. All tensors in this default demo are synthetic; there is no model or network request.

The file interface is `branchsafe branch --input INPUT.npz --output OUTPUT.npz --accept N`. Input keys are `prefix_k`, `prefix_v`, `suffix_k`, `suffix_v`; each is float32 with shape `[L,T,H,D]`. The output keys are `k` and `v`, containing the prefix plus the first `N` suffix tokens. See the CLI `--help` for the page-budget flags and the [integration tests](../../tests/test_integration.py) for a file round trip.

## 2. Follow one transaction in Python

This example is small enough to inspect every byte:

```sh
uv run python - <<'PY'
import numpy as np
from branchsafe import CacheConfig, PagedCache

cfg = CacheConfig(layers=2, heads=2, head_dim=4, page_size=4, max_pages=16)
prefix_k = np.zeros((2, 5, 2, 4), dtype=np.float32)
prefix_v = np.ones_like(prefix_k)
tail_k = np.full((2, 3, 2, 4), 2, dtype=np.float32)
tail_v = np.full_like(tail_k, 3)
with PagedCache(cfg) as cache:
    parent = cache.create()
    parent.append(prefix_k, prefix_v)
    sibling = parent.fork()
    with parent.transaction() as tx:
        tx.append(tail_k, tail_v)
        tx.commit(accepted_tokens=2)
    k, v = parent.materialize()
    np.testing.assert_array_equal(k, np.concatenate((prefix_k, tail_k[:, :2]), axis=1))
    np.testing.assert_array_equal(v, np.concatenate((prefix_v, tail_v[:, :2]), axis=1))
    np.testing.assert_array_equal(sibling.materialize()[0], prefix_k)
    cache.check_invariants()
    print({"parent_tokens": parent.length, "sibling_tokens": sibling.length})
PY
```

The printed lengths are 7 and 5. The original tail contains one visible token of a four-token page. Transaction append copies that shared tail to a private page; the sibling keeps the original five-token history. A two-token acceptance is relative to the transaction start, so the parent becomes length `5+2`, not length 2.

## 3. Read the append path

In [core.py](../../src/branchsafe/core.py), read `CacheConfig`, `_validate`, `Sequence.append`, then `PagedCache._reserve`:

1. Validate ownership, optional ticket, shape, exact dtype and finite values under the pool lock.
2. Compute the partial-tail length, whether it is shared, and the complete reservation size.
3. Reserve capacity before publishing state. Fresh arrays are constructed locally before insertion into the pool.
4. Copy the visible shared tail if necessary, then copy each span of the new input to its page.
5. On a recoverable copy exception, release unpublished reservations and propagate the exception. On success, release the old shared tail reference and publish the new table, length, epoch and counters.

The active sequence length is the visibility boundary. A failed copy into unused positions need not erase those hidden bytes; `materialize()` never returns them, and a later append overwrites its own new visible range. This is ownership isolation, not secure data erasure.

## 4. Follow lifecycle state

`fork()` either shares references or eagerly copies the used prefix. `truncate()` releases pages after the new visible end. `close()` removes ownership once; repeated cleanup is harmless. A child remains valid if its parent is closed because references belong to each live sequence independently.

`WriteTicket` contains sequence identity, epoch and length. It is an optimistic consistency token, not a credential. If a writer changes the parent after transaction creation, `commit()` raises `StaleWriteError`; it cannot silently overwrite the new parent. Transactions fork at **creation**, before `__enter__`, and must be committed or closed explicitly if used without a context.

The [dense baseline](../../src/branchsafe/baseline.py) has the same public semantics. Read its `fork()` to confirm that it copies only the used prefix. It does not repeatedly concatenate the entire sequence on append. Its transaction commit transfers the child arrays to the parent; it does not perform a second full-prefix copy.

## 5. Connect code to evidence

| Evidence | What it checks |
|---|---|
| [test_core.py](../../tests/test_core.py) | Exact K/V, page boundaries, reference state machine, bounded resources, rollback, bad inputs and failure recovery |
| [test_concurrency.py](../../tests/test_concurrency.py) | Real threaded branch isolation, same-ticket races and competing commits |
| [test_model_adapter.py](../../tests/test_model_adapter.py) | Offline cache-layout conversion checks |
| [validate_model.py](../../scripts/validate_model.py) | Optional pinned GPT-2 native/adapted/full-prefix forwards with cache and logits checks |
| [benchmark.py](../../benchmark.py) | Dense/shared/eager comparison with complete per-repeat records |
| [benchmark.json](../../configs/benchmark.json) | Fixed workload and repetition configuration |

The benchmark's main management path is fork → append → truncate accepted suffix → close. It models branch management; it is not itself the transaction `commit()` path and is not an inference benchmark. A separate timed scope materializes every branch. Transactions are exercised by correctness tests, the demo and model validation. Model-validation diagnostic timings have different scopes and must not be turned into a speedup claim.

## 6. Diagnose a regression

```sh
uv run pytest -q tests/test_core.py -k 'capacity or failure or transaction'
uv run pytest -q tests/test_concurrency.py
uv run python benchmark.py
uv run python scripts/analyze.py
uv run ruff check .
uv run mypy
```

For a mismatch, record geometry, seed and operation order; call `check_invariants()` after each transition and compare against fresh independent arrays. Check partial-page sharing before inspecting numerical code. For unexpected memory, distinguish `live_bytes`, retained `allocated_bytes`, output materialization and measured RSS. For timing, inspect all raw samples and method order; preserve failed runs and check source hashes before comparing revisions. The authoritative status and measured values live in [EXPERIMENTS.md](EXPERIMENTS.md), not in this conceptual guide.
