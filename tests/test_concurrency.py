"""Real threaded ownership transitions. / 真实线程中的所有权状态变化。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from branchsafe.core import CacheConfig, PagedCache, StaleWriteError


def batch(config, value, tokens=1):
    shape = (config.layers, tokens, config.heads, config.head_dim)
    return np.full(shape, value, dtype=config.dtype), np.full(shape, -value, dtype=config.dtype)


def test_same_ticket_allows_exactly_one_concurrent_writer():
    config = CacheConfig(layers=1, heads=1, head_dim=2, page_size=4, max_pages=8)
    cache = PagedCache(config)
    sequence = cache.create()
    ticket = sequence.ticket()
    workers = 12
    barrier = Barrier(workers)

    def attempt(index):
        barrier.wait(timeout=10)
        try:
            sequence.append(*batch(config, index + 1), ticket=ticket)
        except StaleWriteError:
            return None
        return index + 1

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(attempt, range(workers)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    key, value = sequence.materialize()
    np.testing.assert_array_equal(key, batch(config, winners[0])[0])
    np.testing.assert_array_equal(value, batch(config, winners[0])[1])
    assert sequence.length == 1
    cache.check_invariants()


@pytest.mark.parametrize("sharing", [False, True])
def test_parallel_branch_appends_and_closes_preserve_parent(sharing):
    config = CacheConfig(layers=2, heads=2, head_dim=4, page_size=4, max_pages=128)
    cache = PagedCache(config, sharing=sharing)
    parent = cache.create()
    prefix = batch(config, 7, tokens=11)
    parent.append(*prefix)
    workers = 8
    barrier = Barrier(workers)

    def branch(index):
        child = parent.fork()
        barrier.wait(timeout=10)
        for _ in range(7):
            child.append(*batch(config, index + 20))
        expected_k = np.concatenate((prefix[0], batch(config, index + 20, 7)[0]), axis=1)
        expected_v = np.concatenate((prefix[1], batch(config, index + 20, 7)[1]), axis=1)
        actual = child.materialize()
        np.testing.assert_array_equal(actual[0], expected_k)
        np.testing.assert_array_equal(actual[1], expected_v)
        child.close()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(branch, range(workers)))
    actual = parent.materialize()
    np.testing.assert_array_equal(actual[0], prefix[0])
    np.testing.assert_array_equal(actual[1], prefix[1])
    assert cache.stats()["live_pages"] == 3
    cache.check_invariants()
    parent.close()
    assert cache.stats()["live_pages"] == 0


def test_parallel_transaction_commits_have_one_winner_without_leaks():
    config = CacheConfig(layers=1, heads=1, head_dim=2, page_size=4, max_pages=64)
    cache = PagedCache(config)
    parent = cache.create()
    prefix = batch(config, 10, tokens=3)
    parent.append(*prefix)
    workers = 8
    barrier = Barrier(workers)

    def attempt(index):
        with parent.transaction() as transaction:
            transaction.append(*batch(config, index + 100, tokens=5))
            barrier.wait(timeout=10)
            try:
                transaction.commit(accepted_tokens=2)
            except StaleWriteError:
                return None
            return index + 100

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(attempt, range(workers)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    actual = parent.materialize()
    tail = batch(config, winners[0], tokens=2)
    np.testing.assert_array_equal(actual[0], np.concatenate((prefix[0], tail[0]), axis=1))
    np.testing.assert_array_equal(actual[1], np.concatenate((prefix[1], tail[1]), axis=1))
    assert cache.stats()["live_pages"] == 2
    cache.check_invariants()
    parent.close()
    assert cache.stats()["live_pages"] == 0
