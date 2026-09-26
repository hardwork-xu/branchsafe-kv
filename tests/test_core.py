"""Storage contracts against independent arrays. / 用独立数组验证存储合同。"""

from __future__ import annotations

import random

import numpy as np
import pytest

from branchsafe.baseline import DenseCache
from branchsafe.core import (
    CacheConfig,
    CacheError,
    CapacityError,
    ClosedError,
    PagedCache,
    StaleWriteError,
)


def make_config(**overrides):
    values = {"layers": 2, "heads": 2, "head_dim": 3, "page_size": 4, "max_pages": 64}
    values.update(overrides)
    return CacheConfig(**values)


def arrays(config, tokens, offset=0):
    shape = (config.layers, tokens, config.heads, config.head_dim)
    key = np.arange(np.prod(shape), dtype=config.dtype).reshape(shape) + offset
    return key, -key - 1


def assert_values(sequence, expected):
    actual = sequence.materialize()
    assert sequence.length == expected[0].shape[1]
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def concatenate(left, right):
    return tuple(np.concatenate((a, b), axis=1) for a, b in zip(left, right, strict=True))


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 8, 9, 31])
def test_fork_isolates_both_writers_at_page_boundaries(sharing, length):
    config = make_config()
    cache = PagedCache(config, sharing=sharing)
    parent = cache.create()
    prefix = arrays(config, length)
    parent.append(*prefix)
    child = parent.fork()
    parent_tail = arrays(config, 7, 1000)
    child_tail = arrays(config, 6, -1000)
    parent.append(*parent_tail)
    child.append(*child_tail)
    assert_values(parent, concatenate(prefix, parent_tail))
    assert_values(child, concatenate(prefix, child_tail))
    cache.check_invariants()
    parent.close()
    assert_values(child, concatenate(prefix, child_tail))
    child.close()
    assert cache.stats()["live_pages"] == 0
    cache.check_invariants()


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_dtype_and_empty_round_trip(dtype):
    config = make_config(dtype=dtype)
    cache = PagedCache(config)
    sequence = cache.create()
    empty = arrays(config, 0)
    ticket = sequence.ticket()
    sequence.append(*empty)
    assert sequence.ticket() == ticket
    assert_values(sequence, empty)
    data = arrays(config, 7)
    sequence.append(*data, ticket=ticket)
    actual = sequence.materialize()
    assert actual[0].dtype == np.dtype(dtype)
    assert_values(sequence, data)
    cache.close()


@pytest.mark.parametrize("field", ["layers", "heads", "head_dim", "page_size", "max_pages"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_resource_dimensions_are_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        PagedCache(make_config(**{field: value}))


@pytest.mark.parametrize("dtype", ["int32", "complex64", "object", "not-a-dtype"])
def test_unsupported_dtypes_are_rejected(dtype):
    with pytest.raises((TypeError, ValueError)):
        PagedCache(make_config(dtype=dtype))


@pytest.mark.parametrize(
    "bad_input", ["rank", "layers", "heads", "dim", "tokens", "dtype", "nan", "inf"]
)
def test_invalid_append_preserves_existing_state(bad_input):
    config = make_config()
    cache = PagedCache(config)
    sequence = cache.create()
    prefix = arrays(config, 3)
    sequence.append(*prefix)
    before = cache.stats().copy()
    ticket = sequence.ticket()
    key, value = arrays(config, 2, 50)
    if bad_input == "rank":
        key = key[0]
    elif bad_input == "layers":
        key = key[:1]
    elif bad_input == "heads":
        value = value[:, :, :1]
    elif bad_input == "dim":
        key = key[:, :, :, :1]
    elif bad_input == "tokens":
        value = value[:, :1]
    elif bad_input == "dtype":
        key = key.astype(np.float64)
    elif bad_input == "nan":
        key[0, 0, 0, 0] = np.nan
    else:
        value[0, 0, 0, 0] = np.inf
    with pytest.raises((TypeError, ValueError)):
        sequence.append(key, value)
    assert_values(sequence, prefix)
    assert sequence.ticket() == ticket
    assert cache.stats() == before
    cache.check_invariants()


def test_input_and_returned_arrays_cannot_mutate_storage():
    config = make_config()
    cache = PagedCache(config)
    parent = cache.create()
    supplied = arrays(config, 5)
    expected = tuple(item.copy() for item in supplied)
    parent.append(*supplied)
    child = parent.fork()
    supplied[0].fill(9000)
    supplied[1].fill(9000)
    exported = parent.materialize()
    exported[0].fill(-9000)
    exported[1].fill(-9000)
    assert_values(parent, expected)
    assert_values(child, expected)
    cache.check_invariants()


def test_noncontiguous_input_uses_values_without_aliasing():
    config = make_config()
    cache = PagedCache(config)
    sequence = cache.create()
    key, value = arrays(config, 12)
    slices = (key[:, ::2], value[:, ::2])
    expected = tuple(item.copy() for item in slices)
    sequence.append(*slices)
    key.fill(0)
    value.fill(0)
    assert_values(sequence, expected)


@pytest.mark.parametrize("sharing", [False, True])
def test_truncate_does_not_modify_shared_history(sharing):
    config = make_config()
    cache = PagedCache(config, sharing=sharing)
    parent = cache.create()
    prefix = arrays(config, 11)
    parent.append(*prefix)
    child = parent.fork()
    child.truncate(3)
    suffix = arrays(config, 5, 200)
    child.append(*suffix)
    assert_values(parent, prefix)
    assert_values(child, concatenate(tuple(a[:, :3] for a in prefix), suffix))
    parent.truncate(0)
    child.close()
    assert cache.stats()["live_pages"] == 0
    cache.check_invariants()


@pytest.mark.parametrize("length", [-1, 6, 1.5, True])
def test_invalid_truncate_is_atomic(length):
    config = make_config()
    cache = PagedCache(config)
    sequence = cache.create()
    expected = arrays(config, 5)
    sequence.append(*expected)
    before = cache.stats().copy()
    with pytest.raises((TypeError, ValueError)):
        sequence.truncate(length)
    assert_values(sequence, expected)
    assert cache.stats() == before


def test_capacity_failure_is_atomic_after_partially_filled_page():
    config = make_config(max_pages=2)
    cache = PagedCache(config)
    sequence = cache.create()
    prefix = arrays(config, 3)
    sequence.append(*prefix)
    ticket = sequence.ticket()
    before = cache.stats().copy()
    with pytest.raises(CapacityError):
        sequence.append(*arrays(config, 6, 500))
    assert_values(sequence, prefix)
    assert sequence.ticket() == ticket
    assert cache.stats() == before
    sequence.append(*arrays(config, 5, 50), ticket=ticket)
    assert sequence.length == 8
    cache.check_invariants()


def test_cow_requires_capacity_before_changing_parent_or_child():
    config = make_config(max_pages=1)
    cache = PagedCache(config)
    parent = cache.create()
    prefix = arrays(config, 3)
    parent.append(*prefix)
    child = parent.fork()
    before = cache.stats().copy()
    with pytest.raises(CapacityError):
        child.append(*arrays(config, 1, 100))
    assert_values(parent, prefix)
    assert_values(child, prefix)
    assert cache.stats() == before
    parent.close()
    child.append(*arrays(config, 1, 100))
    cache.check_invariants()


def test_allocator_memory_error_releases_reserved_pages_and_allows_retry(monkeypatch):
    config = make_config(max_pages=3)
    cache = PagedCache(config)
    parent = cache.create()
    prefix = arrays(config, 3)
    tail = arrays(config, 6, 100)
    parent.append(*prefix)
    original_empty = np.empty
    calls = 0

    def fail_second_page(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise MemoryError("injected allocation failure")
        return original_empty(*args, **kwargs)

    # Only allocation failure is injected; ownership/copy logic executes normally.
    # 仅注入分配失败，引用管理与复制逻辑保持真实执行。
    with monkeypatch.context() as context:
        context.setattr(np, "empty", fail_second_page)
        with pytest.raises(MemoryError):
            parent.append(*tail)
    assert_values(parent, prefix)
    assert cache.stats()["live_pages"] == 1
    cache.check_invariants()
    parent.append(*tail)
    assert_values(parent, concatenate(prefix, tail))
    cache.check_invariants()


@pytest.mark.parametrize("sharing", [False, True])
def test_payload_and_copy_counters_follow_documented_units(sharing):
    config = make_config()
    cache = PagedCache(config, sharing=sharing)
    parent = cache.create()
    parent.append(*arrays(config, 7))
    child = parent.fork()
    assert cache.stats()["live_pages"] == (2 if sharing else 4)
    child.append(*arrays(config, 2, 50))
    stats = cache.stats()
    expected_pages = 4 if sharing else 5
    expected_copied_tokens = 7 + 2 + (3 if sharing else 7)
    assert stats["live_pages"] == expected_pages
    assert stats["live_bytes"] == expected_pages * config.page_bytes
    assert stats["copied_bytes"] == expected_copied_tokens * config.token_bytes
    before = stats.copy()
    child.materialize()
    assert cache.stats() == before
    parent.close()
    child.close()
    assert cache.stats()["allocated_pages"] == expected_pages
    cache.close()
    assert cache.stats()["allocated_bytes"] == 0
    cache.check_invariants()


def test_eager_fork_capacity_failure_preserves_parent():
    config = make_config(max_pages=3)
    cache = PagedCache(config, sharing=False)
    parent = cache.create()
    prefix = arrays(config, 7)
    parent.append(*prefix)
    before = cache.stats().copy()
    with pytest.raises(CapacityError):
        parent.fork()
    assert_values(parent, prefix)
    assert cache.stats() == before
    cache.check_invariants()


def test_recycled_page_rejects_closed_owner_and_stale_ticket():
    config = make_config(max_pages=1)
    cache = PagedCache(config)
    old = cache.create()
    old.append(*arrays(config, 4))
    ticket = old.ticket()
    old.close()
    current = cache.create()
    expected = arrays(config, 3, 500)
    current.append(*expected)
    with pytest.raises(ClosedError):
        old.append(*arrays(config, 1), ticket=ticket)
    with pytest.raises(StaleWriteError):
        current.append(*arrays(config, 1), ticket=ticket)
    assert_values(current, expected)
    cache.check_invariants()


def test_ticket_is_bound_to_cache_sequence_and_revision():
    config = make_config()
    first = PagedCache(config)
    second = PagedCache(config)
    sequence = first.create()
    sibling = first.create()
    foreign = second.create()
    ticket = sequence.ticket()
    for wrong_owner in (sibling, foreign):
        with pytest.raises(StaleWriteError):
            wrong_owner.append(*arrays(config, 1), ticket=ticket)
        assert wrong_owner.length == 0
    sequence.append(*arrays(config, 2), ticket=ticket)
    with pytest.raises(StaleWriteError):
        sequence.append(*arrays(config, 1), ticket=ticket)
    current = sequence.ticket()
    sequence.truncate(1)
    with pytest.raises(StaleWriteError):
        sequence.append(*arrays(config, 1), ticket=current)
    with pytest.raises(TypeError):
        sequence.append(*arrays(config, 1), ticket="wrong")
    first.check_invariants()
    second.check_invariants()


def test_noop_truncate_preserves_ticket_and_direct_transaction_can_roll_back():
    config = make_config()
    cache = PagedCache(config)
    parent = cache.create()
    prefix = arrays(config, 4)
    parent.append(*prefix)
    ticket = parent.ticket()
    parent.truncate(parent.length)
    assert parent.ticket() == ticket
    transaction = parent.transaction()
    transaction.append(*arrays(config, 5, 500))
    transaction.rollback()
    transaction.rollback()
    transaction.close()
    assert_values(parent, prefix)
    assert cache.stats()["live_pages"] == 1
    cache.check_invariants()


@pytest.mark.parametrize("accepted", [0, 1, 3, 4, 7, None])
def test_transaction_commits_only_accepted_new_tokens(accepted):
    config = make_config()
    cache = PagedCache(config)
    parent = cache.create()
    prefix = arrays(config, 3)
    tail = arrays(config, 7, 700)
    parent.append(*prefix)
    old_ticket = parent.ticket()
    with parent.transaction() as transaction:
        transaction.append(*tail)
        assert_values(parent, prefix)
        transaction.commit(accepted_tokens=accepted)
    count = 7 if accepted is None else accepted
    assert_values(parent, concatenate(prefix, tuple(a[:, :count] for a in tail)))
    with pytest.raises(StaleWriteError):
        parent.append(*arrays(config, 1), ticket=old_ticket)
    cache.check_invariants()
    parent.close()
    assert cache.stats()["live_pages"] == 0


@pytest.mark.parametrize("raises", [False, True])
def test_uncommitted_transaction_rolls_back_and_reclaims_pages(raises):
    config = make_config()
    cache = PagedCache(config)
    parent = cache.create()
    prefix = arrays(config, 3)
    parent.append(*prefix)
    live_before = cache.stats()["live_pages"]
    ticket = parent.ticket()

    def run():
        with parent.transaction() as transaction:
            transaction.append(*arrays(config, 13, 70))
            if raises:
                raise LookupError("injected application failure")

    if raises:
        with pytest.raises(LookupError):
            run()
    else:
        run()
    assert_values(parent, prefix)
    assert parent.ticket() == ticket
    assert cache.stats()["live_pages"] == live_before
    cache.check_invariants()


@pytest.mark.parametrize("accepted", [-1, 4, 1.5, True])
def test_invalid_acceptance_can_be_corrected_without_changing_parent(accepted):
    config = make_config()
    cache = PagedCache(config)
    parent = cache.create()
    prefix = arrays(config, 2)
    tail = arrays(config, 3, 90)
    parent.append(*prefix)
    with parent.transaction() as transaction:
        transaction.append(*tail)
        with pytest.raises((TypeError, ValueError)):
            transaction.commit(accepted_tokens=accepted)
        assert_values(parent, prefix)
        transaction.commit(accepted_tokens=2)
    assert_values(parent, concatenate(prefix, tuple(a[:, :2] for a in tail)))
    cache.check_invariants()


def test_transaction_conflict_preserves_newer_parent_state():
    config = make_config()
    cache = PagedCache(config)
    parent = cache.create()
    prefix = arrays(config, 3)
    concurrent_tail = arrays(config, 2, 900)
    parent.append(*prefix)
    with parent.transaction() as transaction:
        transaction.append(*arrays(config, 7, 700))
        parent.append(*concurrent_tail)
        with pytest.raises(StaleWriteError):
            transaction.commit()
        assert_values(parent, concatenate(prefix, concurrent_tail))
    cache.check_invariants()
    assert cache.stats()["live_pages"] == 2


def test_two_open_transactions_cannot_both_replace_parent():
    config = make_config()
    cache = PagedCache(config)
    parent = cache.create()
    prefix = arrays(config, 3)
    parent.append(*prefix)
    with parent.transaction() as first, parent.transaction() as second:
        first_tail = arrays(config, 1, 100)
        first.append(*first_tail)
        second.append(*arrays(config, 1, 200))
        first.commit()
        with pytest.raises(StaleWriteError):
            second.commit()
    assert_values(parent, concatenate(prefix, first_tail))
    cache.check_invariants()


def test_transaction_reentry_and_use_after_finish_are_rejected():
    config = make_config()
    cache = PagedCache(config)
    parent = cache.create()
    transaction = parent.transaction()
    with transaction:
        with pytest.raises(CacheError):
            transaction.__enter__()
        transaction.append(*arrays(config, 2))
        transaction.commit()
        with pytest.raises(CacheError):
            transaction.append(*arrays(config, 1))
        with pytest.raises(CacheError):
            transaction.commit()
    with pytest.raises(CacheError):
        transaction.__enter__()
    with pytest.raises(CacheError):
        transaction.append(*arrays(config, 1))
    cache.check_invariants()


def test_parent_close_keeps_fork_alive_and_blocks_transaction_commit():
    config = make_config()
    cache = PagedCache(config)
    parent = cache.create()
    expected = arrays(config, 7)
    parent.append(*expected)
    child = parent.fork()
    with parent.transaction() as transaction:
        transaction.append(*arrays(config, 1, 500))
        parent.close()
        with pytest.raises(ClosedError):
            transaction.commit()
    assert_values(child, expected)
    child.append(*arrays(config, 1, 600))
    child.close()
    assert cache.stats()["live_pages"] == 0
    cache.check_invariants()


def test_close_is_idempotent_and_invalidates_all_sequences():
    config = make_config()
    cache = PagedCache(config)
    parent = cache.create()
    parent.append(*arrays(config, 3))
    child = parent.fork()
    parent.close()
    parent.close()
    with pytest.raises(ClosedError):
        parent.ticket()
    cache.close()
    cache.close()
    for operation in (cache.create, child.materialize, child.fork, child.ticket):
        with pytest.raises(ClosedError):
            operation()
    child.close()


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("seed", [1, 11, 111])
def test_random_lifecycles_match_independent_numpy_reference(sharing, seed):
    """Use model state, never storage internals. / 独立状态模型，不检查内部实现。"""
    config = make_config(max_pages=1024)
    cache = PagedCache(config, sharing=sharing)
    rng = random.Random(seed)
    first = cache.create()
    active = [(first, arrays(config, 0))]
    for step in range(180):
        index = rng.randrange(len(active))
        sequence, expected = active[index]
        operation = rng.choice(("append", "truncate", "fork", "close", "commit", "rollback"))
        if operation == "append":
            tail = arrays(config, rng.randrange(8), step * 100)
            sequence.append(*tail)
            active[index] = (sequence, concatenate(expected, tail))
        elif operation == "truncate":
            length = rng.randrange(sequence.length + 1)
            sequence.truncate(length)
            active[index] = (sequence, tuple(a[:, :length].copy() for a in expected))
        elif operation == "fork" and len(active) < 8:
            active.append((sequence.fork(), tuple(a.copy() for a in expected)))
        elif operation == "close" and len(active) > 1:
            sequence.close()
            active.pop(index)
        elif operation in {"commit", "rollback"}:
            tail = arrays(config, rng.randrange(8), step * 100)
            with sequence.transaction() as transaction:
                transaction.append(*tail)
                if operation == "commit":
                    accepted = rng.randrange(tail[0].shape[1] + 1)
                    transaction.commit(accepted_tokens=accepted)
                    active[index] = (
                        sequence,
                        concatenate(expected, tuple(a[:, :accepted] for a in tail)),
                    )
        for candidate, reference in active:
            assert_values(candidate, reference)
        cache.check_invariants()
    for sequence, _ in active:
        sequence.close()
    assert cache.stats()["live_pages"] == 0
    cache.check_invariants()


@pytest.mark.parametrize("backend", ["dense", "eager", "shared"])
def test_backends_share_branch_transaction_and_cleanup_contracts(backend):
    config = make_config(max_pages=64)
    cache = (
        DenseCache(config, max_tokens=16)
        if backend == "dense"
        else PagedCache(config, sharing=backend == "shared")
    )
    parent = cache.create()
    prefix = arrays(config, 5)
    parent.append(*prefix)
    child = parent.fork()
    with parent.transaction() as transaction:
        tail = arrays(config, 6, 100)
        transaction.append(*tail)
        transaction.commit(accepted_tokens=2)
    expected = concatenate(prefix, tuple(a[:, :2] for a in tail))
    assert_values(parent, expected)
    assert_values(child, prefix)
    with parent.transaction() as transaction:
        transaction.append(*arrays(config, 5, 800))
    assert_values(parent, expected)
    with parent.transaction() as transaction:
        transaction.append(*arrays(config, 1, 400))
        parent.truncate(3)
        with pytest.raises(StaleWriteError):
            transaction.commit()
    assert_values(parent, tuple(a[:, :3] for a in prefix))
    parent.close()
    child.append(*arrays(config, 1, 900))
    assert_values(child, concatenate(prefix, arrays(config, 1, 900)))
    child.close()
    assert cache.stats()["live_bytes"] == 0
    cache.check_invariants()
    cache.close()
    assert cache.stats()["allocated_bytes"] == 0


def test_dense_capacity_failures_are_atomic_and_recoverable():
    config = make_config(max_pages=2)
    cache = DenseCache(config, max_tokens=8)
    parent = cache.create()
    prefix = arrays(config, 5)
    parent.append(*prefix)
    before = cache.stats().copy()
    for operation in (parent.fork, parent.transaction, cache.create):
        with pytest.raises(CapacityError):
            operation()
        assert cache.stats() == before
        assert_values(parent, prefix)
    with pytest.raises(CapacityError):
        parent.append(*arrays(config, 4))
    assert cache.stats() == before
    assert_values(parent, prefix)
    parent.close()
    replacement = cache.create()
    replacement.append(*arrays(config, 8))
    assert replacement.length == 8
    cache.check_invariants()


@pytest.mark.parametrize("max_tokens", [0, -1, True, 2.5])
def test_dense_invalid_max_tokens_is_rejected(max_tokens):
    with pytest.raises(ValueError):
        DenseCache(make_config(), max_tokens=max_tokens)


def test_dense_visible_prefix_copy_accounting_and_output_isolation():
    config = make_config()
    cache = DenseCache(config, max_tokens=32)
    parent = cache.create()
    prefix = arrays(config, 3)
    parent.append(*prefix)
    child = parent.fork()
    assert cache.stats()["copied_bytes"] == 6 * config.token_bytes
    assert cache.stats()["live_bytes"] == 64 * config.token_bytes
    exported = child.materialize()
    exported[0].fill(500)
    exported[1].fill(500)
    child.append(*arrays(config, 1, 70))
    assert cache.stats()["copied_bytes"] == 7 * config.token_bytes
    assert_values(parent, prefix)
    assert_values(child, concatenate(prefix, arrays(config, 1, 70)))
    cache.check_invariants()


@pytest.mark.parametrize("backend", ["dense", "eager", "shared"])
def test_sequence_budget_covers_empty_forks_and_transactions(backend):
    config = make_config(max_sequences=2)
    cache = (
        DenseCache(config, max_tokens=8)
        if backend == "dense"
        else PagedCache(config, sharing=backend == "shared")
    )
    parent = cache.create()
    child = parent.fork()
    before = cache.stats().copy()
    for operation in (cache.create, parent.fork, parent.transaction):
        with pytest.raises(CapacityError):
            operation()
        assert cache.stats() == before
        assert parent.length == 0
        assert child.length == 0
    child.close()
    with parent.transaction() as transaction:
        transaction.append(*arrays(config, 1))
        transaction.commit()
    assert parent.length == 1
    assert cache.stats()["sequences"] == 1
    cache.check_invariants()


@pytest.mark.parametrize("backend", ["dense", "eager", "shared"])
def test_copy_failure_keeps_visible_values_and_recovers(backend, monkeypatch):
    config = make_config()
    cache = (
        DenseCache(config, max_tokens=16)
        if backend == "dense"
        else PagedCache(config, sharing=backend == "shared")
    )
    parent = cache.create()
    prefix = arrays(config, 3)
    tail = arrays(config, 6, 100)
    parent.append(*prefix)
    child = parent.fork()
    live_before = cache.stats()["live_bytes"]
    ticket = child.ticket()
    original_copy = np.copyto
    calls = 0

    def fail_copy(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected copy failure")
        return original_copy(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(np, "copyto", fail_copy)
        with pytest.raises(RuntimeError, match="injected copy failure"):
            child.append(*tail)
    assert_values(parent, prefix)
    assert_values(child, prefix)
    assert child.ticket() == ticket
    assert cache.stats()["live_bytes"] == live_before
    cache.check_invariants()
    child.append(*tail)
    assert_values(child, concatenate(prefix, tail))
    cache.check_invariants()


@pytest.mark.parametrize("backend", ["dense", "eager"])
def test_eager_fork_copy_failure_reclaims_child(backend, monkeypatch):
    config = make_config()
    cache = DenseCache(config, max_tokens=16) if backend == "dense" else PagedCache(config, False)
    parent = cache.create()
    prefix = arrays(config, 7)
    parent.append(*prefix)
    before = cache.stats().copy()

    def fail_copy(*args, **kwargs):
        raise RuntimeError("injected fork copy failure")

    with monkeypatch.context() as context:
        context.setattr(np, "copyto", fail_copy)
        with pytest.raises(RuntimeError, match="injected fork copy failure"):
            parent.fork()
    assert_values(parent, prefix)
    assert cache.stats()["live_bytes"] == before["live_bytes"]
    assert cache.stats()["sequences"] == before["sequences"]
    cache.check_invariants()
    child = parent.fork()
    assert_values(child, prefix)
