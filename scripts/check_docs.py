"""Run documented Python examples and compare shell commands. / 执行文档示例并核对双语命令。"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    pairs = [(ROOT / "README.md", ROOT / "README_zh.md")]
    pairs += [(p, ROOT / "docs/zh" / p.name) for p in sorted((ROOT / "docs/en").glob("*.md"))]
    examples = 0
    for en, zh in pairs:
        texts = [p.read_text() for p in (en, zh)]
        commands = [re.findall(r"```(?:bash|sh)\n(.*?)```", text, re.S) for text in texts]
        if commands[0] != commands[1]:
            raise ValueError(f"shell command mismatch / 双语命令不一致: {en.name}")
        for text in texts:
            for code in re.findall(r"```python\n(.*?)```", text, re.S):
                subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=30)
                examples += 1
    print(f"passed / 通过: {len(pairs)} paired command sets; {examples} executed Python examples")


if __name__ == "__main__":
    main()
