"""Evidence schema and recoverable harness failures. / 证据格式与可恢复的运行器失败。"""

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from branchsafe.core import ClosedError, PagedCache

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load("benchmark_harness_test", "benchmark.py")
analyze = load("analyze_harness_test", "scripts/analyze.py")
acceptance = load("acceptance_harness_test", "scripts/validate.py")


@pytest.fixture
def config():
    return {
        "seed": 11,
        "warmup": 1,
        "repeats": 2,
        "geometry": {"layers": 1, "heads": 1, "head_dim": 4, "page_size": 4},
        "cases": [
            {
                "name": "small",
                "prefix": 7,
                "branches": 2,
                "append": 3,
                "accepted": 1,
                "read_rounds": 1,
            }
        ],
    }


@pytest.fixture
def evidence(config):
    """Schema fixture only; never written as measured results. / 仅格式测试，不作为实测结果。"""
    methods = list(bench.METHODS)
    samples = []
    for scope in ("management", "management_and_read"):
        for iteration in range(-config["warmup"], config["repeats"]):
            for order, method in enumerate(methods):
                samples.append(
                    {
                        "case": "small",
                        "scope": scope,
                        "method": method,
                        "iteration": iteration,
                        "warmup": iteration < 0,
                        "order": order,
                        "status": "passed",
                        "elapsed_ns": 100,
                        "preload_ns": 20,
                        "managed_peak_bytes": 64,
                        "copied_bytes": 32,
                    }
                )
    return {
        "status": "passed",
        "methods": methods,
        "config": config,
        "samples": samples,
        "correctness": [
            {"case": "small", "method": m, "status": "passed", "exact_kv": True} for m in methods
        ],
        "memory": [
            {"case": "small", "method": m, "status": "passed", "process_peak_rss_bytes": 64}
            for m in methods
        ],
    }


def test_rounded_page_capacity_guard(config):
    config["geometry"]["page_size"] = 10**9
    with pytest.raises(ValueError, match="2 GiB"):
        bench.validate_config(config)


def test_preload_failure_closes_real_pool(config, monkeypatch):
    created = []

    def traced_cache(cfg, sharing):
        pool = PagedCache(cfg, sharing=sharing)
        created.append(pool)
        return pool

    monkeypatch.setattr(bench, "PagedCache", traced_cache)
    case = config["cases"][0]
    data = bench.inputs(config["geometry"], case, 0)
    data[0][0][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        bench.run_once("shared_paged", config["geometry"], case, data, False)
    assert created[0].stats()["sequences"] == created[0].stats()["allocated_bytes"] == 0
    with pytest.raises(ClosedError):
        created[0].create()


def test_real_timing_includes_retained_counter(config):
    case = config["cases"][0]
    data = bench.inputs(config["geometry"], case, 0)
    result = bench.run_once("shared_paged", config["geometry"], case, data, True)
    assert result["status"] == "passed" and result["elapsed_ns"] > 0
    assert result["allocated_bytes_at_branch_peak"] >= result["managed_peak_bytes"]
    assert result["released_live_bytes"] == 0


@pytest.mark.parametrize(
    "code,timeout,expected",
    [
        ("import time; time.sleep(1)", 0.03, "timeout"),
        ("raise SystemExit(4)", 5, "failed"),
        ("print('invalid JSON')", 5, "JSON"),
        ("print('[]')", 5, "schema"),
        ('print(\'{"status": "passed", "process_peak_rss_bytes": 0}\')', 5, "schema"),
    ],
)
def test_memory_worker_real_subprocess_failures(code, timeout, expected):
    result = bench.run_memory_worker([sys.executable, "-c", code], timeout=timeout)
    assert result["status"] == "failed" and expected in result["error"]


def test_memory_worker_missing_executable(tmp_path):
    result = bench.run_memory_worker([str(tmp_path / "absent-executable")])
    assert result["status"] == "failed" and result["error"] == "FileNotFoundError"


def test_complete_evidence_summarizes_without_fabricated_retained_metric(evidence):
    rows = analyze.summarize(evidence)
    assert len(rows) == 6 and all(r["n"] == 2 for r in rows)
    assert all(r["allocated_bytes_at_branch_peak"] is None for r in rows)


@pytest.mark.parametrize("change", ["missing", "duplicate", "warmup", "order", "nan", "zero"])
def test_rejects_corrupted_samples(evidence, change):
    if change == "missing":
        evidence["samples"].pop()
    elif change == "duplicate":
        evidence["samples"].append(copy.deepcopy(evidence["samples"][0]))
    elif change == "warmup":
        evidence["samples"][0]["warmup"] = False
    elif change == "order":
        evidence["samples"][0]["order"] = 1
    elif change == "nan":
        evidence["samples"][0]["elapsed_ns"] = float("nan")
    else:
        evidence["samples"][0]["elapsed_ns"] = 0
    with pytest.raises(ValueError):
        analyze.summarize(evidence)


@pytest.mark.parametrize("section", ["correctness", "memory"])
def test_rejects_incomplete_supporting_evidence(evidence, section):
    evidence[section].pop()
    with pytest.raises(ValueError):
        analyze.summarize(evidence)


def test_additional_manifest_binds_check_and_container_sources(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_example.py").write_text("assert True\n")
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    first = acceptance.additional_manifest(tmp_path)
    assert set(first) == {"tests/test_example.py", "Dockerfile"}
    (tmp_path / "tests/test_example.py").write_text("assert False\n")
    assert acceptance.additional_manifest(tmp_path) != first


def test_docker_unavailable_explicitly_records_build_and_run(monkeypatch):
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: None)
    commands, records = acceptance.docker_plan()
    assert commands == []
    assert {r["name"] for r in records} == {"docker_build", "docker_run"}
    assert all(r["status"] == "not_run" and r["exit_code"] is None for r in records)


def test_docker_available_plans_real_build_and_run(monkeypatch):
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/tool/docker")
    monkeypatch.setattr(
        acceptance.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "", ""),
    )
    commands, records = acceptance.docker_plan()
    assert records == []
    assert [name for name, _ in commands] == ["docker_build", "docker_run"]
