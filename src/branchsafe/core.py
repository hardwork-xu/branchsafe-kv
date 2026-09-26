"""Bounded copy-on-write KV storage. / 有界写时复制 KV 存储。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from types import TracebackType
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]


class CacheError(RuntimeError):
    """Invalid cache lifecycle or state. / 缓存生命周期或状态错误。"""


class CapacityError(CacheError):
    """The configured memory budget cannot satisfy the operation. / 操作超出内存预算。"""


class StaleWriteError(CacheError):
    """A write ticket or transaction snapshot is stale. / 写入凭据或事务快照已过期。"""


class ClosedError(CacheError):
    """The cache, sequence or transaction is closed. / 缓存、序列或事务已关闭。"""


@dataclass(frozen=True)
class CacheConfig:
    """KV geometry and payload bound. / KV 形状与有效载荷上限。

    Arrays have shape (layers, tokens, heads, head_dim), with exact dtype.
    数组形状为 (layers, tokens, heads, head_dim)，dtype 必须精确匹配。
    """

    layers: int
    heads: int
    head_dim: int
    page_size: int = 16
    max_pages: int = 4096
    dtype: str = "float32"
    max_sequences: int = 4096

    def __post_init__(self) -> None:
        for name in ("layers", "heads", "head_dim", "page_size", "max_pages", "max_sequences"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer / 必须为正整数")
        if self.dtype not in ("float16", "float32", "float64"):
            raise ValueError("dtype must be float16/float32/float64 / 不支持此数据类型")

    @property
    def token_bytes(self) -> int:
        """Bytes for one K/V token across layers. / 所有层单个 K/V token 字节数。"""
        return 2 * self.layers * self.heads * self.head_dim * np.dtype(self.dtype).itemsize

    @property
    def page_bytes(self) -> int:
        """One K/V page pair's capacity in bytes. / 一对 K/V 页的容量字节数。"""
        return self.token_bytes * self.page_size


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum} / 参数必须为范围内整数")


def _validate(config: CacheConfig, k: Array, v: Array) -> tuple[Array, Array]:
    if not isinstance(k, np.ndarray) or not isinstance(v, np.ndarray):
        raise ValueError("K and V must be NumPy arrays / K 和 V 必须为 NumPy 数组")
    # Strip ndarray subclasses; no user callback runs during page writes.
    # 去除 ndarray 子类，避免页写入期间执行用户回调。
    k, v = np.asarray(k), np.asarray(v)
    if k.ndim != 4 or v.shape != k.shape:
        raise ValueError("K/V require matching rank-4 shapes / K/V 须为同形四维数组")
    if (k.shape[0], k.shape[2], k.shape[3]) != (
        config.layers,
        config.heads,
        config.head_dim,
    ):
        raise ValueError("K/V geometry does not match config / K/V 形状与配置不一致")
    if k.dtype != np.dtype(config.dtype) or v.dtype != np.dtype(config.dtype):
        raise ValueError("K/V dtype must exactly match config / K/V 数据类型须精确匹配配置")
    if not np.isfinite(k).all() or not np.isfinite(v).all():
        raise ValueError("K/V must contain only finite values / K/V 不得包含 NaN 或 Inf")
    return k, v


@dataclass(frozen=True)
class WriteTicket:
    """Checked optimistic-write snapshot, not an access secret. / 乐观写快照，非访问密钥。"""

    sequence_id: int
    epoch: int
    length: int


@dataclass
class _Page:
    k: Array
    v: Array
    refs: int = 0


class PagedCache:
    """Bounded, locked page pool with shared or eager forks. / 有界加锁页池。

    max_pages bounds allocated ndarray payload, including reusable free slots.
    max_pages 限制实际分配的数组载荷，包括保留用于复用的空闲页。
    """

    def __init__(self, config: CacheConfig, sharing: bool = True) -> None:
        if not isinstance(config, CacheConfig):
            raise TypeError("config must be CacheConfig / 配置必须为 CacheConfig")
        if not isinstance(sharing, bool):
            raise TypeError("sharing must be bool / sharing 必须为布尔值")
        self.config = config
        self.sharing = sharing
        self._lock = RLock()
        self._slots: list[_Page] = []
        self._free: list[int] = []
        self._sequences: dict[int, Sequence] = {}
        self._creation_token = object()
        self._next_id = uuid4().int << 64
        self._closed = False
        self._live = 0
        self._peak = 0
        self._copied = 0

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedError("cache is closed / 缓存已关闭")

    def _check_sequence_capacity(self) -> None:
        if len(self._sequences) >= self.config.max_sequences:
            raise CapacityError("sequence budget exhausted / 序列数量达到上限")

    def _register(self) -> Sequence:
        self._check_sequence_capacity()
        sequence = Sequence(self, self._next_id, self._creation_token)
        self._next_id += 1
        self._sequences[sequence._id] = sequence
        return sequence

    def create(self) -> Sequence:
        """Create an empty owned sequence. / 创建本池拥有的空序列。"""
        with self._lock:
            self._ensure_open()
            return self._register()

    def _reserve(self, count: int) -> list[int]:
        if count > self.config.max_pages - self._live:
            raise CapacityError("page budget exhausted / 页容量不足")
        free_count = min(count, len(self._free))
        shape = (
            self.config.layers,
            self.config.page_size,
            self.config.heads,
            self.config.head_dim,
        )
        # Allocate locally first: ndarray allocation failure leaves the pool untouched.
        # 先在局部分配；数组分配失败不修改页池。
        fresh = [
            _Page(
                np.empty(shape, dtype=self.config.dtype), np.empty(shape, dtype=self.config.dtype)
            )
            for _ in range(count - free_count)
        ]
        reserved = self._free[len(self._free) - free_count :] + list(
            range(len(self._slots), len(self._slots) + len(fresh))
        )
        self._slots.extend(fresh)
        if free_count:
            del self._free[-free_count:]
        for slot in reserved:
            self._slots[slot].refs = 1
        self._live += count
        self._peak = max(self._peak, self._live)
        return reserved

    def _release(self, slot: int) -> None:
        page = self._slots[slot]
        page.refs -= 1
        if page.refs == 0:
            self._live -= 1
            self._free.append(slot)

    def stats(self) -> dict[str, int]:
        """Array-capacity counters, not RSS. / 数组容量计数，不是进程 RSS。

        copied_bytes counts successful append, eager fork and valid COW tail bytes;
        it excludes materialize output copies and validation reads.
        copied_bytes 包含成功追加、复制分叉及 COW 有效尾部；不含失败操作、物化与校验。
        """
        with self._lock:
            return {
                "live_pages": self._live,
                "peak_pages": self._peak,
                "live_bytes": self._live * self.config.page_bytes,
                "peak_bytes": self._peak * self.config.page_bytes,
                "allocated_pages": len(self._slots),
                "allocated_bytes": len(self._slots) * self.config.page_bytes,
                "copied_bytes": self._copied,
                "sequences": len(self._sequences),
            }

    def check_invariants(self) -> None:
        """Check ownership, lengths, references and bounds. / 检查所有权、长度、引用及上限。"""
        with self._lock:
            counts = [0] * len(self._slots)
            for sequence in self._sequences.values():
                assert not sequence._closed, "registered closed sequence / 已关闭序列仍注册"
                expected = (sequence._length + self.config.page_size - 1) // self.config.page_size
                assert len(sequence._pages) == expected, "length/page mismatch / 长度与页数不符"
                assert len(set(sequence._pages)) == len(sequence._pages), "duplicate page / 重复页"
                for slot in sequence._pages:
                    counts[slot] += 1
            assert counts == [page.refs for page in self._slots], "bad refcounts / 引用计数错误"
            assert len(self._free) == len(set(self._free)), "duplicate free slot / 重复空闲槽"
            assert set(self._free) == {i for i, refs in enumerate(counts) if refs == 0}
            assert self._live == sum(refs > 0 for refs in counts)
            assert self._live <= len(self._slots) <= self.config.max_pages
            if self._closed:
                assert not self._sequences and not self._slots

    def close(self) -> None:
        """Close every sequence and release retained arrays. / 关闭所有序列并释放保留数组。"""
        with self._lock:
            if not self._closed:
                for sequence in list(self._sequences.values()):
                    sequence.close()
                self._slots.clear()
                self._free.clear()
                self._closed = True

    def __enter__(self) -> PagedCache:
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


class Sequence:
    """Owned KV sequence; create through a cache. / 由缓存创建并拥有的 KV 序列。"""

    def __init__(self, pool: PagedCache, sequence_id: int, _token: object | None = None) -> None:
        if _token is not pool._creation_token:
            raise TypeError("use cache.create() to construct sequences / 请使用 cache.create()")
        self._pool = pool
        self._id = sequence_id
        self._epoch = 0
        self._length = 0
        self._pages: list[int] = []
        self._closed = False

    def _ensure_open(self) -> None:
        self._pool._ensure_open()
        if self._closed:
            raise ClosedError("sequence is closed / 序列已关闭")
        if self._pool._sequences.get(self._id) is not self:
            raise CacheError("sequence is not owned by this cache / 序列不属于本缓存")

    @property
    def length(self) -> int:
        """Visible token count; closed sequences reject reads. / 可见 token 数，关闭后拒绝读取。"""
        with self._pool._lock:
            self._ensure_open()
            return self._length

    def ticket(self) -> WriteTicket:
        """Snapshot for conditional append. / 获取条件追加写入快照。"""
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
        """Append finite KV atomically on capacity failure. / 追加有限 KV，容量不足不改动序列。

        Inputs are copied; no writable page views escape. Empty writes are no-ops.
        Callers must not mutate inputs concurrently. This is not a security sandbox.
        复制输入且不暴露可写页视图；空写入不改变状态。
        调用方不得并发修改输入；本库不是安全沙箱。
        """
        pool = self._pool
        with pool._lock:
            self._ensure_open()
            self._check_ticket(ticket)
            k, v = _validate(pool.config, k, v)
            count = k.shape[1]
            if count == 0:
                return
            size = pool.config.page_size
            tail = self._length % size
            cow = bool(tail and pool._slots[self._pages[-1]].refs > 1)
            required = (self._length + count + size - 1) // size - len(self._pages)
            new_pages = self._pages.copy()
            reserved = pool._reserve(required + int(cow))
            try:
                cursor = 0
                if cow:
                    old = pool._slots[new_pages[-1]]
                    replacement = pool._slots[reserved[0]]
                    np.copyto(replacement.k[:, :tail], old.k[:, :tail])
                    np.copyto(replacement.v[:, :tail], old.v[:, :tail])
                    new_pages[-1] = reserved[0]
                    cursor = 1
                new_pages.extend(reserved[cursor:])
                position = self._length
                source = 0
                while source < count:
                    slot, offset = divmod(position, size)
                    take = min(size - offset, count - source)
                    page = pool._slots[new_pages[slot]]
                    np.copyto(page.k[:, offset : offset + take], k[:, source : source + take])
                    np.copyto(page.v[:, offset : offset + take], v[:, source : source + take])
                    source += take
                    position += take
            except BaseException:
                # Only unpublished tail positions were touched; return every reservation.
                # 仅写入尚未公开的尾部位置；回收全部预留页。
                for slot in reserved:
                    pool._release(slot)
                raise
            if cow:
                pool._release(self._pages[-1])
            self._pages = new_pages
            self._length += count
            self._epoch += 1
            pool._copied += (count + (tail if cow else 0)) * pool.config.token_bytes

    def fork(self) -> Sequence:
        """Fork with shared references or eager copies. / 通过共享引用或完整复制创建分支。"""
        pool = self._pool
        with pool._lock:
            self._ensure_open()
            pool._check_sequence_capacity()
            if pool.sharing:
                pages = self._pages.copy()
                child = pool._register()
                for slot in pages:
                    pool._slots[slot].refs += 1
            else:
                pages = pool._reserve(len(self._pages))
                try:
                    for i, (source, target) in enumerate(zip(self._pages, pages, strict=True)):
                        take = min(pool.config.page_size, self._length - i * pool.config.page_size)
                        np.copyto(pool._slots[target].k[:, :take], pool._slots[source].k[:, :take])
                        np.copyto(pool._slots[target].v[:, :take], pool._slots[source].v[:, :take])
                    child = pool._register()
                except BaseException:
                    for slot in pages:
                        pool._release(slot)
                    raise
                pool._copied += self._length * pool.config.token_bytes
            child._pages = pages
            child._length = self._length
            return child

    def truncate(self, length: int) -> None:
        """Keep a prefix without exposing removed tokens. / 保留前缀，不暴露已删除 token。"""
        with self._pool._lock:
            self._ensure_open()
            _integer(length, "length")
            if length > self._length:
                raise ValueError("truncate cannot grow a sequence / 截断不能增长序列")
            if length == self._length:
                return
            keep = (length + self._pool.config.page_size - 1) // self._pool.config.page_size
            for slot in self._pages[keep:]:
                self._pool._release(slot)
            del self._pages[keep:]
            self._length = length
            self._epoch += 1

    def materialize(self) -> tuple[Array, Array]:
        """Return independent contiguous K/V arrays. / 返回独立连续 K/V 数组。"""
        pool = self._pool
        with pool._lock:
            self._ensure_open()
            shape = (pool.config.layers, self._length, pool.config.heads, pool.config.head_dim)
            k, v = (
                np.empty(shape, dtype=pool.config.dtype),
                np.empty(shape, dtype=pool.config.dtype),
            )
            for i, slot in enumerate(self._pages):
                start = i * pool.config.page_size
                take = min(pool.config.page_size, self._length - start)
                np.copyto(k[:, start : start + take], pool._slots[slot].k[:, :take])
                np.copyto(v[:, start : start + take], pool._slots[slot].v[:, :take])
            return k, v

    def transaction(self) -> Transaction:
        """Create an isolated append branch with checked commit. / 创建隔离追加分支并校验提交。"""
        return Transaction(self)

    def close(self) -> None:
        """Release page references; repeated calls are safe. / 释放页引用，可重复调用。"""
        with self._pool._lock:
            if not self._closed:
                self._ensure_open()
                for slot in self._pages:
                    self._pool._release(slot)
                self._pages.clear()
                self._pool._sequences.pop(self._id, None)
                self._closed = True
                self._epoch += 1


class Transaction:
    """Append-only branch transaction; context exit rolls back. / 只追加分支事务，退出时回滚。

    Commit accepts a count relative to the starting parent length and checks
    its epoch. Invalid counts may be corrected and retried before context exit.
    提交数量相对父序列初始长度，并检查 epoch；非法数量可在退出前修改后重试。
    """

    def __init__(self, parent: Sequence) -> None:
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
        """Append only to the isolated branch. / 仅向隔离分支追加。"""
        with self._parent._pool._lock:
            self._ensure_active()
            self._child.append(k, v)

    def commit(self, accepted_tokens: int | None = None) -> None:
        """Publish accepted tokens if parent is unchanged. / 父序列未变时发布接受的 token。"""
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
            self._child.truncate(self._snapshot.length + accepted)
            for slot in parent._pages:
                parent._pool._release(slot)
            parent._pages = self._child._pages
            parent._length = self._child._length
            parent._epoch += 1
            self._child._pages = []
            self._child.close()
            self._done = True

    def rollback(self) -> None:
        """Discard the branch idempotently. / 丢弃分支，可重复调用。"""
        with self._parent._pool._lock:
            if not self._done:
                self._child.close()
                self._done = True

    def close(self) -> None:
        """Alias for rollback. / rollback 的别名。"""
        self.rollback()

    def __enter__(self) -> Transaction:
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
