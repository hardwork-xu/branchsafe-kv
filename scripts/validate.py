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
    manifest = source_manifest(ROOT)
    record = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "git_revision": revision(ROOT),
        "source_manifest": manifest,
        "checks": [],
    }
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(ROOT / "work/mpl")
    for name, command in checks:
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
    docker = shutil.which("docker")
    if docker is None:
        record["checks"].append(
            {
                "name": "local_docker",
                "command": ["docker", "build", "-t", "branchsafe-kv:0.1.0", "."],
                "status": "not_run",
                "exit_code": None,
                "summary": "Docker executable unavailable / 未安装Docker",
            }
        )
    record["completed_at"] = datetime.now(UTC).isoformat()
    record["status"] = (
        "failed"
        if any(c["status"] == "failed" for c in record["checks"])
        else "passed_with_not_run"
    )
    (out / "checks.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return int(record["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
