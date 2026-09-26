#!/usr/bin/env python3
"""Generate tables and plots from raw evidence. / 从原始证据生成表格与图。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(data):
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
