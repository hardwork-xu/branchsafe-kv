# Repository conventions / 仓库约定
- `src/branchsafe`: CPU NumPy storage and CLI; `tests`: offline tests; `benchmark.py`, `scripts`: evidence generation; `docs/en` and `docs/zh`: paired documentation.
- `uv sync --frozen`; `uv run pytest`; `uv run ruff check .`; `uv run ruff format --check .`; `uv run mypy`; `uv run python -m build`.
- Keep English and Simplified Chinese explanations aligned. / 中英说明同步；机器字段只用英文。
- Preserve every valid benchmark sample, original targets, failures, source hashes and measurement scope. / 不改写实测或事后移动目标。
- Never publish weights, local paths, credentials or private logs. / 不发布模型权重、本地路径、凭据或私人日志。
- Do not push, publish, spend money or change global configuration without task authorization. / 外写和付费须有当前任务授权。
- Changes are complete only after affected tests, lint/types, build and evidence links pass. / 变更须通过受影响验证。
