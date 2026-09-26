#!/usr/bin/env python3
"""Run real cache lifecycle experiments. / 执行真实缓存生命周期实验。"""

from __future__ import annotations

# Thread pinning must precede NumPy import. / 线程固定必须在NumPy导入前。
# ruff: noqa: E402
import argparse
import json
import os

# Pin numerical workers before importing NumPy. / 导入NumPy之前固定线程。
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
import random
import resource
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from branchsafe.baseline import DenseCache
from branchsafe.core import CacheConfig, PagedCache
from branchsafe.provenance import environment, revision, source_manifest

METHODS = ("dense", "eager_paged", "shared_paged")
ROOT = Path(__file__).resolve().parent


def validate_config(config):
    """Reject oversized/malformed workloads before allocation. / 分配前验证负载。"""
    if set(config) != {"seed", "warmup", "repeats", "geometry", "cases"}:
        raise ValueError("invalid config keys / 配置字段不合法")
    for key in ("seed", "warmup", "repeats"):
        if type(config[key]) is not int or config[key] < (1 if key == "repeats" else 0):
            raise ValueError("invalid repetition or seed / 重复次数或种子不合法")
    if config["repeats"] > 10000 or config["warmup"] > 100:
        raise ValueError("run limit exceeded / 超过运行次数限制")
    geom = config["geometry"]
    if set(geom) != {"layers", "heads", "head_dim", "page_size"}:
        raise ValueError("invalid geometry / 张量形状字段不合法")
    if any(type(v) is not int or v <= 0 for v in geom.values()):
        raise ValueError("positive integer geometry required / 形状须为正整数")
    if not isinstance(config["cases"], list) or not 1 <= len(config["cases"]) <= 100:
        raise ValueError("expected 1..100 cases / 场景数量须为1..100")
    names = set()
    for case in config["cases"]:
        if set(case) != {"name", "prefix", "branches", "append", "accepted", "read_rounds"}:
            raise ValueError("invalid case keys / 场景字段不合法")
        if not isinstance(case["name"], str) or not case["name"].isidentifier():
            raise ValueError("invalid case name / 场景名称不合法")
        if case["name"] in names:
            raise ValueError("duplicate case / 场景重名")
        names.add(case["name"])
        if any(type(case[k]) is not int or case[k] < 0 for k in case if k != "name"):
            raise ValueError("nonnegative integer sizes required / 规模须为非负整数")
        if not 1 <= case["branches"] <= 128 or not 1 <= case["read_rounds"] <= 100:
            raise ValueError("invalid branch/read count / 分支或读取次数不合法")
        if case["accepted"] > case["append"] or case["prefix"] + case["append"] == 0:
            raise ValueError("invalid accepted length / 接受长度不合法")
        estimate = (case["branches"] + 2) * (case["prefix"] + case["append"]) * 2
        estimate *= geom["layers"] * geom["heads"] * geom["head_dim"] * 4
        if estimate > 2 * 1024**3:
            raise ValueError("2 GiB workload payload limit / 工作负载数据上限为2 GiB")
    return config


def inputs(geometry, case, seed):
    rng = np.random.default_rng(seed)

    def pair(tokens):
        shape = (geometry["layers"], tokens, geometry["heads"], geometry["head_dim"])
        return tuple(rng.standard_normal(shape, dtype=np.float32) for _ in range(2))

    return pair(case["prefix"]), [pair(case["append"]) for _ in range(case["branches"])]


def run_once(method, geometry, case, data, read):
    """Time preload and a full branch lifecycle separately. / 分别计时预载与分支生命周期。"""
    prefix, suffixes = data
    total = case["prefix"] + case["append"]
    page_size = geometry["page_size"]
    pages = (case["branches"] + 2) * ((total + page_size - 1) // page_size + 1)
    cfg = CacheConfig(**geometry, max_pages=pages)
    start = time.perf_counter_ns()
    pool = (
        DenseCache(cfg, max_tokens=total)
        if method == "dense"
        else PagedCache(cfg, sharing=method == "shared_paged")
    )
    parent = pool.create()
    parent.append(*prefix)
    preload_ns = time.perf_counter_ns() - start
    branches = []
    checks = []
    try:
        start = time.perf_counter_ns()
        for values in suffixes:
            child = parent.fork()
            branches.append(child)
            child.append(*values)
            child.truncate(case["prefix"] + case["accepted"])
        if read:
            for _ in range(case["read_rounds"]):
                for child in branches:
                    arrays = child.materialize()
                    # Read all bytes through the real materialization path. / 读取全部逻辑字节。
                    checks.append(float(arrays[0].reshape(-1)[0]) if arrays[0].size else 0.0)
                    del arrays
        peak = pool.stats().copy()
        for child in branches:
            child.close()
        parent.close()
        pool.close()
        elapsed_ns = time.perf_counter_ns() - start
        live_after = pool.stats()["live_bytes"]
        if live_after != 0:
            raise AssertionError("live allocation leak / 存活分配泄漏")
        return {
            "status": "passed",
            "elapsed_ns": elapsed_ns,
            "preload_ns": preload_ns,
            "managed_peak_bytes": peak["peak_bytes"],
            "copied_bytes": peak["copied_bytes"],
            "released_live_bytes": live_after,
            "read_count": len(checks),
        }
    finally:
        pool.close()


def correctness(method, geometry, case, data):
    prefix, suffixes = data
    total = case["prefix"] + case["append"]
    cfg = CacheConfig(
        **geometry, max_pages=(case["branches"] + 3) * ((total // geometry["page_size"]) + 2)
    )
    pool = (
        DenseCache(cfg, max_tokens=total)
        if method == "dense"
        else PagedCache(cfg, sharing=method == "shared_paged")
    )
    try:
        parent = pool.create()
        parent.append(*prefix)
        children = []
        for suffix in suffixes:
            child = parent.fork()
            child.append(*suffix)
            child.truncate(case["prefix"] + case["accepted"])
            children.append(child)
            for actual, pre, tail in zip(child.materialize(), prefix, suffix, strict=True):
                np.testing.assert_array_equal(
                    actual, np.concatenate((pre, tail[:, : case["accepted"]]), axis=1)
                )
        for actual, expected in zip(parent.materialize(), prefix, strict=True):
            np.testing.assert_array_equal(actual, expected)
        pool.check_invariants()
        return {"status": "passed", "exact_kv": True, "children": len(children)}
    finally:
        pool.close()


def memory_worker(config, case_index, method):
    case = config["cases"][case_index]
    data = inputs(config["geometry"], case, config["seed"] + case_index)
    result = run_once(method, config["geometry"], case, data, True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["process_peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
    result["scope"] = "fresh process peak; imports + inputs + preload + one read workflow"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark.json"),
        help="JSON workload config / JSON工作负载配置",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark.json"),
        help="raw evidence JSON / 原始证据JSON",
    )
    parser.add_argument(
        "--memory-worker", nargs=2, metavar=("CASE", "METHOD"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    config = validate_config(json.loads(args.config.read_text()))
    if args.memory_worker:
        print(json.dumps(memory_worker(config, int(args.memory_worker[0]), args.memory_worker[1])))
        return 0
    result = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "started_at": datetime.now(UTC).isoformat(),
        "command": ["python", "benchmark.py", *sys.argv[1:]],
        "git_revision": revision(ROOT),
        "source_manifest": source_manifest(ROOT),
        "environment": environment(),
        "config": config,
        "data_kind": "seeded synthetic Gaussian KV arrays; no model",
        "threads": 1,
        "methods": list(METHODS),
        "samples": [],
        "correctness": [],
        "memory": [],
    }
    order = random.Random(config["seed"])
    failed = False
    for index, case in enumerate(config["cases"]):
        data = inputs(config["geometry"], case, config["seed"] + index)
        for method in METHODS:
            try:
                check = correctness(method, config["geometry"], case, data)
            except Exception as exc:
                failed = True
                check = {"status": "failed", "error": type(exc).__name__ + ": " + str(exc)}
            result["correctness"].append({"case": case["name"], "method": method, **check})
        for read in (False, True):
            for iteration in range(-config["warmup"], config["repeats"]):
                methods = list(METHODS)
                order.shuffle(methods)
                for position, method in enumerate(methods):
                    try:
                        sample = run_once(method, config["geometry"], case, data, read)
                    except Exception as exc:
                        failed = True
                        sample = {"status": "failed", "error": type(exc).__name__ + ": " + str(exc)}
                    result["samples"].append(
                        {
                            "case": case["name"],
                            "method": method,
                            "scope": "management_and_read" if read else "management",
                            "iteration": iteration,
                            "warmup": iteration < 0,
                            "order": position,
                            **sample,
                        }
                    )
        for method in METHODS:
            command = [
                sys.executable,
                str(ROOT / "benchmark.py"),
                "--config",
                str(args.config),
                "--memory-worker",
                str(index),
                method,
            ]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if proc.returncode:
                failed = True
                memory = {
                    "status": "failed",
                    "exit_code": proc.returncode,
                    "error": "worker failed; rerun --memory-worker / 内存子进程失败，请单独复现",
                }
            else:
                memory = json.loads(proc.stdout)
            result["memory"].append({"case": case["name"], "method": method, **memory})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"completed / 已完成: {case['name']}", flush=True)
    result["completed_at"] = datetime.now(UTC).isoformat()
    result["status"] = "failed" if failed else "passed"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
