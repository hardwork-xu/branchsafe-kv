"""Bilingual local cache workflows. / 双语本地缓存流程。"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .core import CacheConfig, CacheError, PagedCache


def demo(seed: int = 7) -> dict[str, int | bool | str]:
    """Run fork, rollback and partial commit. / 执行分支、回滚和部分提交。"""
    rng = np.random.default_rng(seed)
    shape = (2, 17, 2, 8)
    k = rng.standard_normal(shape, dtype=np.float32)
    v = rng.standard_normal(shape, dtype=np.float32)
    tail_k = rng.standard_normal((2, 5, 2, 8), dtype=np.float32)
    tail_v = rng.standard_normal((2, 5, 2, 8), dtype=np.float32)
    with PagedCache(CacheConfig(layers=2, heads=2, head_dim=8, page_size=8, max_pages=20)) as pool:
        parent = pool.create()
        parent.append(k, v)
        child = parent.fork()
        with child.transaction() as tx:
            tx.append(tail_k, tail_v)
            tx.commit(accepted_tokens=3)
        with parent.transaction() as tx:
            tx.append(tail_k, tail_v)
        for actual, expected in zip(parent.materialize(), (k, v), strict=True):
            np.testing.assert_array_equal(actual, expected)
        for actual, pre, tail in zip(child.materialize(), (k, v), (tail_k, tail_v), strict=True):
            np.testing.assert_array_equal(actual, np.concatenate((pre, tail[:, :3]), axis=1))
        pool.check_invariants()
        output: dict[str, int | bool | str] = {
            "status": "passed",
            "data_kind": "synthetic",
            "parent_tokens": parent.length,
            "child_tokens": child.length,
            "exact_kv": True,
            "peak_managed_bytes": pool.stats()["peak_bytes"],
        }
        child.close()
        parent.close()
        output["live_bytes_after_close"] = pool.stats()["live_bytes"]
        return output


def branch_file(
    input_path: Path, output_path: Path, accept: int, page_size: int, max_pages: int
) -> dict[str, str | int]:
    """Load float32 NPZ, commit a suffix and save owned arrays. / 载入NPZ并保存接受前缀。

    Keys: prefix_k, prefix_v, suffix_k, suffix_v; all [L,T,H,D].
    字段为四个K/V张量，形状[L,T,H,D]；默认限制解压数据为256 MiB。
    """
    with zipfile.ZipFile(input_path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 256 * 1024**2:
            raise ValueError("NPZ exceeds 256 MiB / NPZ超过256 MiB")
    with np.load(input_path, allow_pickle=False) as data:
        expected = {"prefix_k", "prefix_v", "suffix_k", "suffix_v"}
        if set(data.files) != expected:
            raise ValueError("NPZ requires four documented K/V keys / NPZ须包含规定的四个K/V字段")
        arrays: list[NDArray[np.float32]] = [
            data[key] for key in ("prefix_k", "prefix_v", "suffix_k", "suffix_v")
        ]
    if arrays[0].ndim != 4:
        raise ValueError("expected [layers,tokens,heads,dim] / 需要四维KV张量")
    layers, _, heads, dim = arrays[0].shape
    cfg = CacheConfig(
        layers=layers, heads=heads, head_dim=dim, page_size=page_size, max_pages=max_pages
    )
    if 2 * layers * heads * dim * page_size * max_pages * 4 > 2 * 1024**3:
        raise ValueError("configured pool exceeds 2 GiB / 页池配置超过2 GiB")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must differ / 输入输出文件不可相同")
    with PagedCache(cfg) as pool:
        parent = pool.create()
        parent.append(arrays[0], arrays[1])
        with parent.transaction() as tx:
            tx.append(arrays[2], arrays[3])
            tx.commit(accepted_tokens=accept)
        k, v = parent.materialize()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as target:
            np.savez(target, k=k, v=v)
        return {"status": "passed", "tokens": parent.length}


def main(argv: list[str] | None = None) -> int:
    """CLI entry returning a process status. / 返回进程退出状态的命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_demo = commands.add_parser("demo", help="offline complete example / 离线完整示例")
    run_demo.add_argument("--seed", type=int, default=7, help="random seed / 随机种子")
    branch = commands.add_parser("branch", help="accept KV suffix from NPZ / 接受NPZ中的KV后缀")
    branch.add_argument("--input", type=Path, required=True, help="input NPZ / 输入NPZ")
    branch.add_argument("--output", type=Path, required=True, help="output NPZ / 输出NPZ")
    branch.add_argument(
        "--accept", type=int, required=True, help="accepted new tokens / 接受新增token数"
    )
    branch.add_argument("--page-size", type=int, default=16, help="tokens per page / 每页token数")
    branch.add_argument("--max-pages", type=int, default=1024, help="bounded live pool / 有界页数")
    args = parser.parse_args(argv)
    try:
        result = (
            demo(args.seed)
            if args.command == "demo"
            else branch_file(args.input, args.output, args.accept, args.page_size, args.max_pages)
        )
    except (ValueError, OSError, CacheError, zipfile.BadZipFile) as exc:
        print(f"error / 错误: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0
