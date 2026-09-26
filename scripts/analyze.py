#!/usr/bin/env python3
"""Generate tables and plots from raw evidence. / 从原始证据生成表格与图。"""

import argparse
import json
from pathlib import Path

import numpy as np


def validate_evidence(data):
    """Reject incomplete, duplicated or failed evidence. / 拒绝不完整、重复或失败的证据。"""
    if data.get("status") != "passed":
        raise ValueError("incomplete/failed benchmark / 基准未完整通过")
    methods = data["methods"]
    names = [case["name"] for case in data["config"]["cases"]]
    if set(methods) != {"dense", "eager_paged", "shared_paged"} or len(methods) != 3:
        raise ValueError("invalid methods / 方法列表不合法")
    if not names or len(names) != len(set(names)):
        raise ValueError("invalid case names / 场景名称不合法")
    repeats, warmup = data["config"]["repeats"], data["config"]["warmup"]
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("invalid repetition counts / 重复次数不合法")
    expected = {
        (case, scope, method, iteration)
        for case in names
        for scope in ("management", "management_and_read")
        for method in methods
        for iteration in range(-warmup, repeats)
    }
    seen = set()
    orders = {}
    for sample in data["samples"]:
        key = (sample["case"], sample["scope"], sample["method"], sample["iteration"])
        if key not in expected or key in seen or sample["status"] != "passed":
            raise ValueError("missing/duplicate/failed samples / 样本缺失、重复或失败")
        if (
            type(sample["iteration"]) is not int
            or type(sample["warmup"]) is not bool
            or sample["warmup"] != (sample["iteration"] < 0)
            or type(sample["order"]) is not int
        ):
            raise ValueError("invalid iteration metadata / 迭代元数据无效")
        for field in ("elapsed_ns", "preload_ns", "managed_peak_bytes", "copied_bytes"):
            value = sample[field]
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError("invalid sample metric / 样本指标无效")
        if sample["elapsed_ns"] == 0:
            raise ValueError("zero latency is invalid / 耗时不得为零")
        seen.add(key)
        group = (sample["case"], sample["scope"], sample["iteration"])
        orders.setdefault(group, []).append(sample["order"])
    if seen != expected or any(sorted(order) != list(range(3)) for order in orders.values()):
        raise ValueError("incomplete samples or invalid run order / 样本不完整或运行顺序无效")
    expected_pairs = {(case, method) for case in names for method in methods}
    for section in ("correctness", "memory"):
        pairs = []
        for row in data[section]:
            pairs.append((row["case"], row["method"]))
            if row["status"] != "passed":
                raise ValueError("failed correctness or memory evidence / 正确性或内存证据失败")
            if section == "correctness" and row.get("exact_kv") is not True:
                raise ValueError("exact KV check missing / 缺少精确KV校验")
            if section == "memory" and (
                type(row.get("process_peak_rss_bytes")) is not int
                or row["process_peak_rss_bytes"] <= 0
            ):
                raise ValueError("invalid RSS evidence / RSS证据无效")
        if set(pairs) != expected_pairs or len(pairs) != len(expected_pairs):
            raise ValueError("incomplete evidence sections / 证据章节不完整")


def summarize(data):
    validate_evidence(data)
    rows = []
    for case in data["config"]["cases"]:
        for scope in ("management", "management_and_read"):
            for method in data["methods"]:
                samples = [
                    s
                    for s in data["samples"]
                    if s["case"] == case["name"]
                    and s["scope"] == scope
                    and s["method"] == method
                    and not s["warmup"]
                ]
                if not samples or any(s["status"] != "passed" for s in samples):
                    raise ValueError("missing or failed samples / 样本缺失或失败")
                ns = np.array([s["elapsed_ns"] for s in samples])
                rows.append(
                    {
                        "case": case["name"],
                        "scope": scope,
                        "method": method,
                        "n": len(ns),
                        "median_ms": float(np.median(ns) / 1e6),
                        "q25_ms": float(np.percentile(ns, 25) / 1e6),
                        "q75_ms": float(np.percentile(ns, 75) / 1e6),
                        "branches_per_second": float(case["branches"] * 1e9 / np.median(ns)),
                        "median_preload_ms": float(
                            np.median([s["preload_ns"] for s in samples]) / 1e6
                        ),
                        "managed_peak_bytes": max(s["managed_peak_bytes"] for s in samples),
                        "copied_bytes": max(s["copied_bytes"] for s in samples),
                        "allocated_bytes_at_branch_peak": (
                            max(s["allocated_bytes_at_branch_peak"] for s in samples)
                            if all("allocated_bytes_at_branch_peak" in s for s in samples)
                            else None
                        ),
                    }
                )
    return rows


def table(rows, scope):
    en = [
        "| Case | Dense ms | Eager paged ms | Shared paged ms | Speedup | Payload ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in dict.fromkeys(r["case"] for r in rows):
        r = {x["method"]: x for x in rows if x["case"] == name and x["scope"] == scope}
        d, e, s = (r[k] for k in ("dense", "eager_paged", "shared_paged"))
        en.append(
            f"| {name} | {d['median_ms']:.3f} | {e['median_ms']:.3f} | {s['median_ms']:.3f} | "
            f"{d['median_ms'] / s['median_ms']:.2f}x | "
            f"{100 * s['managed_peak_bytes'] / d['managed_peak_bytes']:.1f}% |"
        )
    return "\n".join(en)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if data.get("status") != "passed":
        raise ValueError("incomplete/failed benchmark / 基准未完整通过")
    rows = summarize(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"run_id": data["run_id"], "rows": rows}, indent=2) + "\n"
    )
    for lang in ("en", "zh"):
        title = "# Measured cache operations" if lang == "en" else "# 缓存操作实测"
        body = title + "\n\n" + table(rows, "management") + "\n\n"
        body += (
            "## Management + materialization" if lang == "en" else "## 管理加连续读取"
        ) + "\n\n"
        body += table(rows, "management_and_read") + "\n"
        if lang == "zh":
            body = body.replace(
                "Case | Dense ms | Eager paged ms | Shared paged ms | Speedup | Payload ratio",
                "场景 | Dense 毫秒 | Eager paged 毫秒 | Shared paged 毫秒 | 加速比 | 数据字节比例",
            )
        (args.output_dir / f"TABLE_{lang}.md").write_text(body)
    names = list(dict.fromkeys(r["case"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, scope in zip(axes, ("management", "management_and_read"), strict=True):
        for i, method in enumerate(data["methods"]):
            selected = [r for r in rows if r["method"] == method and r["scope"] == scope]
            med = np.array([r["median_ms"] for r in selected])
            lo = np.array([r["q25_ms"] for r in selected])
            hi = np.array([r["q75_ms"] for r in selected])
            ax.bar(
                np.arange(len(names)) + (i - 1) * 0.25,
                med,
                0.25,
                label=method,
                yerr=[med - lo, hi - med],
                capsize=2,
            )
        ax.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Median latency (ms), IQR; lower is better")
        ax.set_title(scope.replace("_", " "))
    axes[0].legend(fontsize=8)
    fig.savefig(args.output_dir / "benchmark.svg")
    fig.savefig(args.output_dir / "benchmark.png", dpi=160)
    plt.close(fig)
    print("tables + plots generated / 已生成表格和图表")


if __name__ == "__main__":
    main()
