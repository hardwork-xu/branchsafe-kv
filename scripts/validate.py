#!/usr/bin/env python3
"""Record executed acceptance checks. / 记录实际执行的验收检查。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from branchsafe.provenance import revision, source_manifest

ROOT = Path(__file__).resolve().parents[1]


def clean_log(text):
    text = text.replace(str(ROOT), "<repo>").replace(str(Path.home()), "<home>")
    return re.sub(r"/private/var/folders/[^ \n:]+", "<temporary>", text)


def additional_manifest(root):
    """Bind the executed checks and build/CI recipes. / 绑定实际执行的检查与构建、CI配置。"""
    files = []
    for directory, pattern in (
        ("tests", "*.py"),
        ("scripts", "*.py"),
        ("examples", "*.py"),
        ("configs", "*.json"),
        (".github", "*.yml"),
    ):
        files.extend((root / directory).rglob(pattern))
    files.extend(root / name for name in ("Dockerfile", ".dockerignore", "Makefile"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
        if path.is_file()
    }


def docker_plan():
    """Run containers when a daemon is reachable; otherwise record why. / 检测容器运行前提。"""
    commands = [
        ("docker_build", ["docker", "build", "-t", "branchsafe-kv:0.1.0", "."]),
        ("docker_run", ["docker", "run", "--rm", "--network", "none", "branchsafe-kv:0.1.0"]),
    ]
    reason = None
    probe_code = None
    if shutil.which("docker") is None:
        reason = "Docker executable unavailable / 未安装Docker"
    else:
        try:
            probe = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=20)
            probe_code = probe.returncode
            if probe.returncode:
                reason = "Docker daemon unavailable / Docker守护进程不可用"
        except (OSError, subprocess.TimeoutExpired):
            reason = "Docker probe unavailable or timed out / Docker探测失败或超时"
    if reason is None:
        return commands, []
    return [], [
        {
            "name": name,
            "command": command,
            "status": "not_run",
            "exit_code": None,
            "probe_command": ["docker", "info"] if shutil.which("docker") else None,
            "probe_exit_code": probe_code,
            "summary": reason,
        }
        for name, command in commands
    ]


def main():
    out = ROOT / "results/acceptance"
    out.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    checks = [
        (
            "public_api",
            [
                python,
                "-c",
                "from branchsafe import CacheConfig,PagedCache; "
                "p=PagedCache(CacheConfig(1,1,4)); "
                "s=p.create(); assert s.length==0; p.close()",
            ],
        ),
        ("default_example", [python, "-m", "branchsafe", "demo"]),
        ("example_file", [python, "examples/branch.py"]),
        (
            "unit_tests",
            [python, "-m", "pytest", "-q", "-m", "not integration", "--junitxml=work/unit.xml"],
        ),
        (
            "integration_tests",
            [python, "-m", "pytest", "-q", "-m", "integration", "--junitxml=work/integration.xml"],
        ),
        ("lint", [python, "-m", "ruff", "check", "."]),
        ("format", [python, "-m", "ruff", "format", "--check", "."]),
        ("types", [python, "-m", "mypy"]),
        ("analysis", [python, "scripts/analyze.py"]),
        ("package_build", [python, "-m", "build"]),
        ("documentation_privacy", [python, "scripts/check_repo.py"]),
        ("git_diff", ["git", "diff", "--check"]),
    ]
    container_commands, container_unavailable = docker_plan()
    checks.extend(container_commands)
    manifest = source_manifest(ROOT)
    extra_manifest = additional_manifest(ROOT)
    record = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "git_revision": revision(ROOT),
        "source_manifest": manifest,
        "additional_manifest": extra_manifest,
        "checks": [],
    }
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(ROOT / "work/mpl")
    for name, command in checks:
        if name == "docker_run" and not any(
            check["name"] == "docker_build" and check["status"] == "passed"
            for check in record["checks"]
        ):
            record["checks"].append(
                {
                    "name": name,
                    "command": command,
                    "status": "not_run",
                    "exit_code": None,
                    "summary": "Docker build failed; run skipped / 构建失败，未运行旧镜像",
                }
            )
            continue
        start = datetime.now(UTC).isoformat()
        try:
            run = subprocess.run(
                command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600
            )
            code = run.returncode
            text = clean_log(run.stdout + run.stderr)
            status = "passed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            code, status, text = None, "failed", "timeout after 600 seconds / 600秒超时"
        except OSError as exc:
            code, status, text = None, "failed", type(exc).__name__
        log = out / f"{name}.log"
        log.write_text(text)
        record["checks"].append(
            {
                "name": name,
                "command": ["python" if c == python else c for c in command],
                "started_at": start,
                "exit_code": code,
                "status": status,
                "log": str(log.relative_to(ROOT)),
                "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                "summary": text.strip().splitlines()[-1] if text.strip() else status,
            }
        )
        (out / "checks.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"{name}: {status}", flush=True)
    record["checks"].extend(container_unavailable)
    if source_manifest(ROOT) != manifest or additional_manifest(ROOT) != extra_manifest:
        record["checks"].append(
            {
                "name": "source_stability",
                "command": ["sha256", "source-and-check-manifests"],
                "exit_code": None,
                "status": "failed",
                "summary": "source changed during validation / 验证期间源码发生变化",
            }
        )
    record["completed_at"] = datetime.now(UTC).isoformat()
    record["status"] = (
        "failed"
        if any(c["status"] == "failed" for c in record["checks"])
        else "passed_with_not_run"
        if any(c["status"] == "not_run" for c in record["checks"])
        else "passed"
    )
    (out / "checks.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return int(record["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
