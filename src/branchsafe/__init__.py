"""Bounded transactional KV storage. / 有界事务式KV存储。"""

from .core import CacheConfig, CapacityError, ClosedError, PagedCache, StaleWriteError

__version__ = "0.1.0"
__all__ = ["CacheConfig", "CapacityError", "ClosedError", "PagedCache", "StaleWriteError"]
