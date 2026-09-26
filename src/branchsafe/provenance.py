"""Privacy-minimal evidence metadata. / 最小化隐私信息的证据元数据。"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path


def source_manifest(root: Path) -> dict[str, str]:
    """Hash relevant source, not local paths. / 哈希相关源码，不记录本地路径。"""
    files = sorted((root / "src").rglob("*.py"))
    files += [root / name for name in ("benchmark.py", "pyproject.toml", "uv.lock")]
    return {
        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in files
        if f.is_file()
    }


def revision(root: Path) -> str:
    """Return real commit or explicit unknown. / 返回真实提交或明确未知。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "uncommitted-source"


def environment() -> dict[str, str | int]:
    """Allowlist hardware/runtime facts. / 只记录允许公开的硬件运行时字段。"""
    import numpy as np

    result: dict[str, str | int] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "backend": "CPU NumPy",
        "timer": "perf_counter_ns",
    }
    if platform.system() == "Darwin":
        for key, label in (
            ("machdep.cpu.brand_string", "cpu"),
            ("hw.memsize", "ram_bytes"),
            ("hw.logicalcpu", "logical_cpus"),
        ):
            result[label] = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    else:
        import os

        result["logical_cpus"] = os.cpu_count() or 1
        result["cpu"] = platform.processor() or platform.machine()
    return result
