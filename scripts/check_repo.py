#!/usr/bin/env python3
"""Check relative links, paired docs and public content. / 检查相对链接、双语与公开内容。"""

import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ".git",
    ".venv",
    ".venv-model",
    ".venv-clean",
    ".cache",
    "work",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def main():
    files = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not any(
            part in EXCLUDED or part.startswith(".venv") for part in p.relative_to(ROOT).parts
        )
    ]
    problems = []
    links = 0
    english = {p.name for p in (ROOT / "docs/en").glob("*.md")}
    chinese = {p.name for p in (ROOT / "docs/zh").glob("*.md")}
    if english != chinese:
        problems.append("English/Chinese document names differ / 双语文档名称不一致")
    required = {
        "RESEARCH.md",
        "ARCHITECTURE.md",
        "EXPERIMENTS.md",
        "DEVELOPMENT.md",
        "WALKTHROUGH.md",
        "RELEASE.md",
        "RESUME.md",
        "PLAN.md",
    }
    if not required <= english:
        problems.append("required documents missing / 缺少必要文档")
    patterns = [
        r"/Users/[A-Za-z]",
        r"/home/[A-Za-z]",
        r"gh[pousr]_" + r"[A-Za-z0-9]{20,}",
        r"-----BEGIN " + r"(?:RSA |OPENSSH )?PRIVATE KEY-----",
        r"https://github.com/" + r"(?:your-name|your-username|example)/",
    ]
    for path in files:
        if path.suffix not in {".py", ".md", ".json", ".toml", ".yml", ".txt", ".cff", ".log"}:
            continue
        text = path.read_text()
        if any(re.search(pattern, text) for pattern in patterns):
            problems.append(f"public content scan failed: {path.relative_to(ROOT)}")
        if path.suffix == ".md":
            # Exclude code fences and external URLs. / 排除代码块及外部URL。
            body = re.sub(r"```.*?```", "", text, flags=re.S)
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", body):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = unquote(target.split("#")[0].strip("<>"))
                if not (path.parent / target).exists():
                    problems.append(f"broken link: {path.relative_to(ROOT)} -> {target}")
                links += 1
    raw = ROOT / "results/benchmark.json"
    if raw.exists():
        data = json.loads(raw.read_text())
        for relative, digest in data["source_manifest"].items():
            import hashlib

            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
                problems.append(f"benchmark source changed: {relative}")
    if problems:
        print("\n".join(problems))
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "files_scanned": len(files),
                "relative_links": links,
                "paired_documents": len(english),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
