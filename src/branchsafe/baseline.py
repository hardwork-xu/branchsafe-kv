"""Preallocated dense-copy baseline. / 预分配连续复制基线。"""

from __future__ import annotations

from threading import RLock
from types import TracebackType
from uuid import uuid4

import numpy as np

from .core import (
    Array,
    CacheConfig,
    CacheError,
    CapacityError,
    ClosedError,
    StaleWriteError,
    WriteTicket,
    _integer,
    _validate,
)


class DenseCache:
    """One preallocated contiguous buffer per sequence. / 每序列一个预分配连续缓冲区。

    Fork copies only the visible prefix; append never concatenates the full prefix.
    The byte budget equals config.max_pages * config.page_bytes, as for PagedCache.
    分支只复制可见前缀；追加不拼接整个前缀；预算与 PagedCache 相同。
    """

    def __init__(self, config: CacheConfig, max_tokens: int) -> None:
        if not isinstance(config, CacheConfig):
            raise TypeError("config must be CacheConfig / 配置必须为 CacheConfig")
        _integer(max_tokens, "max_tokens", minimum=1)
        self.config = config
        self.max_tokens = max_tokens
        self._lock = RLock()
        self._sequences: dict[int, DenseSequence] = {}
        self._creation_token = object()
        self._next_id = uuid4().int << 64
        self._closed = False
        self._live = 0
        self._peak = 0
        self._copied = 0
        self._sequence_bytes = max_tokens * config.token_bytes

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedError("cache is closed / 缓存已关闭")

    def create(self) -> DenseSequence:
        """Reserve a full-capacity empty sequence. / 为完整容量预留空序列。"""
        with self._lock:
            self._ensure_open()
            if len(self._sequences) >= self.config.max_sequences:
                raise CapacityError("sequence budget exhausted / 序列数量达到上限")
            if self._live + self._sequence_bytes > self.config.max_pages * self.config.page_bytes:
                raise CapacityError("dense payload budget exhausted / 连续存储容量不足")
            shape = (self.config.layers, self.max_tokens, self.config.heads, self.config.head_dim)
            k, v = (
                np.empty(shape, dtype=self.config.dtype),
                np.empty(shape, dtype=self.config.dtype),
            )
            sequence = DenseSequence(self, self._next_id, k, v, self._creation_token)
            self._next_id += 1
            self._sequences[sequence._id] = sequence
            self._live += self._sequence_bytes
            self._peak = max(self._peak, self._live)
            return sequence

    def stats(self) -> dict[str, int]:
        """Live/peak ndarray capacity and storage-copy bytes, excluding materialize.

        Dense live_pages/peak_pages are rounded page-equivalent capacities only.
        连续存储 live_pages/peak_pages 仅为容量折算；复制计数不包含物化输出。
        """
        with self._lock:
            page_bytes = self.config.page_bytes
            return {
                "live_pages": (self._live + page_bytes - 1) // page_bytes,
                "peak_pages": (self._peak + page_bytes - 1) // page_bytes,
                "live_bytes": self._live,
                "peak_bytes": self._peak,
                "allocated_pages": (self._live + page_bytes - 1) // page_bytes,
                "allocated_bytes": self._live,
                "copied_bytes": self._copied,
                "sequences": len(self._sequences),
            }

    def check_invariants(self) -> None:
        """Check ownership, visible lengths and byte budget. / 检查所有权、长度及容量。"""
        with self._lock:
            assert self._live == len(self._sequences) * self._sequence_bytes
            assert 0 <= self._live <= self.config.max_pages * self.config.page_bytes
            for sequence in self._sequences.values():
                assert not sequence._closed
                assert 0 <= sequence._length <= self.max_tokens
                k, v = sequence._arrays()
                assert k.nbytes + v.nbytes == self._sequence_bytes
            if self._closed:
                assert not self._sequences

    def close(self) -> None:
        """Release all owned buffers. / 释放所有缓冲区。"""
        with self._lock:
            if not self._closed:
                for sequence in list(self._sequences.values()):
                    sequence.close()
                self._closed = True

    def __enter__(self) -> DenseCache:
        with self._lock:
            self._ensure_open()
            return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class DenseSequence:
    """Dense sequence with the same lifecycle and input contract. / 相同输入与生命周期合同。"""

    def __init__(
        self, pool: DenseCache, sequence_id: int, k: Array, v: Array, _token: object | None = None
    ) -> None:
        if _token is not pool._creation_token:
            raise TypeError("use cache.create() to construct sequences / 请使用 cache.create()")
        self._pool = pool
        self._id = sequence_id
        self._epoch = 0
        self._length = 0
        self._closed = False
        self._k: Array | None = k
        self._v: Array | None = v

    def _ensure_open(self) -> None:
        self._pool._ensure_open()
        if self._closed:
            raise ClosedError("sequence is closed / 序列已关闭")
        if self._pool._sequences.get(self._id) is not self:
            raise CacheError("sequence is not owned by this cache / 序列不属于本缓存")

    def _arrays(self) -> tuple[Array, Array]:
        assert self._k is not None and self._v is not None
        return self._k, self._v

    @property
    def length(self) -> int:
        """Visible token count. / 可见 token 数量。"""
        with self._pool._lock:
            self._ensure_open()
            return self._length

    def ticket(self) -> WriteTicket:
        """Return a checked-write snapshot. / 返回条件写入快照。"""
        with self._pool._lock:
            self._ensure_open()
            return WriteTicket(self._id, self._epoch, self._length)

    def _check_ticket(self, ticket: WriteTicket | None) -> None:
        if ticket is not None:
            if not isinstance(ticket, WriteTicket):
                raise TypeError("ticket must be WriteTicket / ticket 必须为 WriteTicket")
            if ticket != self.ticket():
                raise StaleWriteError(
                    "stale or foreign write ticket / 写入凭据已过期或属于其他序列"
                )

    def append(self, k: Array, v: Array, ticket: WriteTicket | None = None) -> None:
        """Copy only newly appended tokens. Do not concurrently mutate input arrays.

        仅复制新增 token；调用方不得并发修改输入数组。
        """
        with self._pool._lock:
            self._ensure_open()
            self._check_ticket(ticket)
            k, v = _validate(self._pool.config, k, v)
            count = k.shape[1]
            if self._length + count > self._pool.max_tokens:
                raise CapacityError("sequence max_tokens exhausted / 序列 token 容量不足")
            if count == 0:
                return
            target_k, target_v = self._arrays()
            end = self._length + count
            np.copyto(target_k[:, self._length : end], k)
            np.copyto(target_v[:, self._length : end], v)
            self._length = end
            self._epoch += 1
            self._pool._copied += count * self._pool.config.token_bytes

    def fork(self) -> DenseSequence:
        """Allocate capacity and copy the used prefix. / 分配容量并复制已使用前缀。"""
        with self._pool._lock:
            self._ensure_open()
            child = self._pool.create()
            try:
                k, v = self._arrays()
                target_k, target_v = child._arrays()
                np.copyto(target_k[:, : self._length], k[:, : self._length])
                np.copyto(target_v[:, : self._length], v[:, : self._length])
            except BaseException:
                child.close()
                raise
            child._length = self._length
            self._pool._copied += self._length * self._pool.config.token_bytes
            return child

    def truncate(self, length: int) -> None:
        """Reduce visible length without reallocating. / 减少可见长度，不重新分配。"""
        with self._pool._lock:
            self._ensure_open()
            _integer(length, "length")
            if length > self._length:
                raise ValueError("truncate cannot grow a sequence / 截断不能增长序列")
            if length != self._length:
                self._length = length
                self._epoch += 1

    def materialize(self) -> tuple[Array, Array]:
        """Return independent contiguous arrays of the visible prefix. / 返回独立连续可见前缀。"""
        with self._pool._lock:
            self._ensure_open()
            k, v = self._arrays()
            return k[:, : self._length].copy(), v[:, : self._length].copy()

    def transaction(self) -> DenseTransaction:
        """Fork an isolated append transaction. / 创建隔离追加事务。"""
        return DenseTransaction(self)

    def close(self) -> None:
        """Release capacity idempotently. / 释放容量，可重复调用。"""
        with self._pool._lock:
            if not self._closed:
                self._ensure_open()
                self._pool._sequences.pop(self._id, None)
                self._pool._live -= self._pool._sequence_bytes
                self._k = None
                self._v = None
                self._closed = True
                self._epoch += 1


class DenseTransaction:
    """Dense-copy branch transaction with epoch-checked commit. / 检查 epoch 的连续分支事务。"""

    def __init__(self, parent: DenseSequence) -> None:
        self._parent = parent
        with parent._pool._lock:
            self._snapshot = parent.ticket()
            self._child = parent.fork()
        self._done = False
        self._entered = False

    def _ensure_active(self) -> None:
        if self._done:
            raise ClosedError("transaction is closed / 事务已关闭")
        self._child._ensure_open()

    def append(self, k: Array, v: Array) -> None:
        """Append to the isolated child. / 向隔离分支追加。"""
        with self._parent._pool._lock:
            self._ensure_active()
            self._child.append(k, v)

    def commit(self, accepted_tokens: int | None = None) -> None:
        """Accept appended tokens if the parent snapshot is current. / 快照有效时接受追加。"""
        parent = self._parent
        with parent._pool._lock:
            self._ensure_active()
            parent._ensure_open()
            parent._check_ticket(self._snapshot)
            appended = self._child._length - self._snapshot.length
            accepted = appended if accepted_tokens is None else accepted_tokens
            _integer(accepted, "accepted_tokens")
            if accepted > appended:
                raise ValueError("accepted_tokens exceeds appended tokens / 接受数量超过追加数量")
            parent._k, parent._v = self._child._arrays()
            parent._length = self._snapshot.length + accepted
            parent._epoch += 1
            self._child.close()
            self._done = True

    def rollback(self) -> None:
        """Discard the isolated child idempotently. / 丢弃隔离分支，可重复调用。"""
        with self._parent._pool._lock:
            if not self._done:
                self._child.close()
                self._done = True

    def close(self) -> None:
        """Alias for rollback. / rollback 的别名。"""
        self.rollback()

    def __enter__(self) -> DenseTransaction:
        with self._parent._pool._lock:
            self._ensure_active()
            if self._entered:
                raise CacheError("transaction context cannot be reentered / 事务上下文不可重复进入")
            self._entered = True
            return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.rollback()
