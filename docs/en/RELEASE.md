# Release preparation — 0.1.0

[简体中文](../zh/RELEASE.md)

I am preparing BranchSafe KV as a small, inspectable systems project. This document is a release draft. At preparation time, no public repository, package-registry release, hosted service, or deployed container has been created. The intended repository name is `branchsafe-kv`; relative links remain usable before publication.

## Release notes draft

Version 0.1.0 provides bounded CPU NumPy KV storage with shared prefix pages, copy-on-write for a shared partial page, failure-atomic append reservations, stale-write detection, rollback, and partial acceptance of branch transactions. It includes a preallocated dense baseline, an eager-page ablation, an offline CLI, unit/integration tests, locked dependencies, bilingual documentation, and machine-readable experiment evidence.

Paging and copy-on-write are established techniques. The contribution is the inspectable implementation of their lifecycle and failure invariants, together with explicit measurement boundaries. This release makes no claim of a new attention algorithm, GPU support, production deployment, or universal inference speedup.

The optional pinned GPT-2 experiment records 45/45 native-cache comparisons with bitwise-identical logits and K/V. All greedy token IDs match, including full recomputation, and three rollback/partial-accept scenarios pass. However, 41/45 float32 full-recomputation comparisons exceed the preregistered `1e-5` absolute-error bound; the maximum is `0.000701904296875`. Native-cache versus full-recomputation discrepancies are identical to paged-cache versus full-recomputation discrepancies. The stricter diagnostic remains **failed**, and the validator returns exit code 1. These results support storage equivalence to the native cache, not the stronger full-recomputation error bound. See [the final result](../../results/model.json) and [the preserved initial result](../../results/model-initial.json).

## Installation and demonstration

The declared Python range is 3.11–3.13. The local validated path is Python 3.12 on macOS arm64, CPU, float32. A Linux GitHub Actions configuration is provided, but creating its YAML does not establish a successful remote run. CUDA, MPS execution, and Windows are not validated backends.

From a local checkout with `uv` available:

```sh
uv sync --frozen
uv run --frozen branchsafe demo
uv run --frozen python examples/branch.py
make test
make check
make benchmark
make analyze
make build
make validate
```

`make build` produces the versioned wheel and source distribution in `dist/`. Their existence is a local build result, not a registry upload. Install the wheel into a separate environment to verify the packaged API:

```sh
uv venv work/release-check
uv pip install --python work/release-check/bin/python dist/branchsafe_kv-0.1.0-py3-none-any.whl
work/release-check/bin/branchsafe demo
```

Optional real-model correctness validation:

```sh
uv sync --frozen --group model
uv run --frozen --group model python scripts/validate_model.py
uv run --frozen --group model python scripts/validate_model.py --offline --output results/model-recheck.json
```

The first command invoking the script can download approximately 548 MB of fixed-revision GPT-2 safetensors and additional tokenizer files. The offline command requires that snapshot to exist. Both preserve the strict diagnostic; exit code 1 is expected for the recorded full-recomputation failure. Do not run model validation concurrently with a performance benchmark. The default demo and test suite require neither weights nor commercial credentials.

## Container status

The [Dockerfile](../../Dockerfile) uses `python:3.12.11-slim-bookworm`, installs the frozen runtime dependency set, and runs the demo as a non-root user. Docker was unavailable on the development host, so container build and execution are **not locally verified**:

```sh
docker build -t branchsafe-kv:0.1.0 .
docker run --rm branchsafe-kv:0.1.0
```

The image serves the CPU demo path only. Host macOS results do not establish Linux container performance or optional model support inside the container.

## Public metadata draft

One-line introduction / GitHub About:

> Bounded transactional KV cache with copy-on-write prefix sharing, CPU benchmarks, and reproducible bilingual evidence.

Suggested topics: `kv-cache`, `copy-on-write`, `memory-management`, `transactions`, `llm-inference`, `numpy`, `benchmark`, `reproducible-research`.

Use the [MIT license](../../LICENSE), [third-party notice](../../NOTICE.md), [contribution guide](../../CONTRIBUTING.md), [security boundaries](../../SECURITY.md), and [software citation](../../CITATION.cff). The citation date records the local version snapshot. No DOI, institutional affiliation, or unverified author metadata is asserted.

## Publication boundary

Before publishing, review the final tracked files, source/evidence correspondence, documentation links, and sensitive-data scan. Publish only the reviewed source, tests, documentation, configurations, and small result artifacts. Exclude virtual environments, downloaded weights, caches, private machine logs, and unreviewed local work.

Creating a public repository, pushing commits, tagging a release, uploading wheel/source archives, and deploying a service are distinct external operations. They remain unexecuted in this preparation record. When one is actually performed, replace this status with the verified result and exact public link; do not prefill an assumed repository URL or a passing CI badge.
