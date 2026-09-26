UV ?= uv
.PHONY: install demo test check benchmark analyze build model validate docker
install:
	$(UV) sync --frozen
demo:
	$(UV) run --frozen branchsafe demo
test:
	$(UV) run --frozen pytest
check:
	$(UV) run --frozen ruff check .
	$(UV) run --frozen ruff format --check .
	$(UV) run --frozen mypy
benchmark:
	$(UV) run --frozen python benchmark.py --output results/benchmark.json
analyze:
	$(UV) run --frozen python scripts/analyze.py
build:
	$(UV) run --frozen python -m build
model:
	$(UV) sync --frozen --group model
	$(UV) run --frozen --group model python scripts/validate_model.py
validate:
	$(UV) run --frozen python scripts/validate.py
docker:
	docker build -t branchsafe-kv:0.1.0 .
	docker run --rm branchsafe-kv:0.1.0
