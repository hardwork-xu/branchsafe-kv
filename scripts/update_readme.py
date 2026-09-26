"""Insert generated evidence tables in both READMEs. / 同步生成的双语结果表。"""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
for language, filename in (("en", "README.md"), ("zh", "README_zh.md")):
    path = root / filename
    content = path.read_text()
    table = (root / f"results/TABLE_{language}.md").read_text()
    before, remainder = content.split("<!-- benchmark:begin -->", 1)
    _, after = remainder.split("<!-- benchmark:end -->", 1)
    path.write_text(
        before + "<!-- benchmark:begin -->\n" + table + "<!-- benchmark:end -->" + after
    )
print("README tables synchronized / README结果表已同步")
