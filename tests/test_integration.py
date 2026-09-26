import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from branchsafe.cli import branch_file, demo, main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_default_demo():
    result = demo()
    assert result["exact_kv"] and result["live_bytes_after_close"] == 0
    assert result["parent_tokens"] == 17 and result["child_tokens"] == 20


@pytest.mark.integration
def test_cli_and_npz(tmp_path):
    x = np.arange(24, dtype=np.float32).reshape(1, 6, 2, 2)
    inp, out = tmp_path / "in.npz", tmp_path / "out.npz"
    np.savez(inp, prefix_k=x, prefix_v=x, suffix_k=x[:, :3] + 99, suffix_v=x[:, :3] - 99)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "branchsafe",
            "branch",
            "--input",
            str(inp),
            "--output",
            str(out),
            "--accept",
            "2",
            "--page-size",
            "4",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["tokens"] == 8
    with np.load(out) as data:
        np.testing.assert_array_equal(data["k"], np.concatenate((x, x[:, :2] + 99), axis=1))
        np.testing.assert_array_equal(data["v"], np.concatenate((x, x[:, :2] - 99), axis=1))
    with pytest.raises(ValueError):
        branch_file(inp, inp, 2, 4, 1024)


def test_cli_errors(tmp_path, capsys):
    assert (
        main(
            [
                "branch",
                "--input",
                str(tmp_path / "missing.npz"),
                "--output",
                str(tmp_path / "o.npz"),
                "--accept",
                "1",
            ]
        )
        == 2
    )
    assert "错误" in capsys.readouterr().err
    assert main(["demo", "--seed", "-1"]) == 2


def test_help_bilingual():
    r = subprocess.run(
        [sys.executable, "-m", "branchsafe", "--help"], capture_output=True, text=True
    )
    assert r.returncode == 0 and "离线" in r.stdout


@pytest.mark.integration
def test_benchmark_and_analysis(tmp_path):
    cfg = json.loads((ROOT / "configs/benchmark.json").read_text())
    cfg["warmup"], cfg["repeats"] = 0, 2
    cfg["geometry"] = {"layers": 1, "heads": 1, "head_dim": 4, "page_size": 4}
    cfg["cases"] = [
        {"name": "smoke", "prefix": 7, "branches": 2, "append": 3, "accepted": 1, "read_rounds": 1}
    ]
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(cfg))
    raw = tmp_path / "raw.json"
    run = subprocess.run(
        [sys.executable, "benchmark.py", "--config", str(config), "--output", str(raw)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    data = json.loads(raw.read_text())
    assert data["status"] == "passed"
    assert len(data["samples"]) == 12
    assert len(data["correctness"]) == 3 and all(c["exact_kv"] for c in data["correctness"])
    assert len(data["memory"]) == 3 and data["source_manifest"]
    analyze = subprocess.run(
        [
            sys.executable,
            "scripts/analyze.py",
            "--input",
            str(raw),
            "--output-dir",
            str(tmp_path / "analysis"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert analyze.returncode == 0, analyze.stderr
    assert (tmp_path / "analysis/benchmark.svg").stat().st_size > 1000


def test_benchmark_rejects_invalid_config():
    spec = importlib.util.spec_from_file_location("bench", ROOT / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = json.loads((ROOT / "configs/benchmark.json").read_text())
    cfg["cases"][0]["accepted"] = 999
    with pytest.raises(ValueError):
        module.validate_config(cfg)
