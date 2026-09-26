# Contributing

[简体中文](CONTRIBUTING_zh.md)

I maintain BranchSafe KV as a focused CPU systems experiment. Contributions should improve cache correctness, explain a measured cost, or make an existing result easier to reproduce. A new backend or serving integration needs its own concrete acceptance criteria; adding an interface alone does not establish support.

## Local workflow

Use Python 3.11–3.13 and `uv`; the recorded local run uses Python 3.12 on macOS arm64. Run these commands from the repository root:

```sh
uv sync --frozen
make demo
make test
make check
make build
make validate
```

The default tests use synthetic arrays and run offline after dependencies are installed. Optional pretrained validation requires the `model` dependency group and approximately 548 MB of GPT-2 safetensors, plus tokenizer files. See the [release instructions](docs/en/RELEASE.md) before running it; its strict full-recomputation diagnostic currently returns a failure.

## What a change must preserve

- Keep stored K/V exact, shared prefixes isolated, capacity failures atomic, and repeated cleanup safe. Add a regression test that executes the actual failing path.
- Keep the default example small and offline. Validate public inputs and provide actionable English and Simplified Chinese errors.
- Update corresponding English and Chinese documentation together. Code names, CLI flags, schemas, formulas, examples, and result numbers must agree.
- Measure changes to hot paths against the preallocated dense baseline and the eager-page ablation under the same inputs and resource settings. Preserve unfavorable samples and original targets.
- Record the tested Git revision, relevant source hashes, configuration, seed, and all repetitions. Generate result tables with `make analyze`; do not manually improve numbers in documentation.

For a performance change:

```sh
make benchmark
make analyze
make validate
```

Run benchmarks without concurrent model validation or other heavy jobs. The benchmark measures cache management and materialization; it does not measure end-to-end language-model serving.

## Review and attribution

Use a focused change with an explanation of the problem, observable behavior, tests, and limitations. Commit summaries use Conventional Commits in English; commit bodies include a matching Chinese explanation and actual validation performed. Do not claim checks passed when they were skipped or not run.

I accept contributions under the repository's [MIT license](LICENSE). Preserve third-party attribution and dependency licenses; disclose copied or adapted material and its source. Do not submit credentials, private prompts, model weights, machine-specific paths, or personal logs. Public model outputs can also expose supplied inputs, so use reviewed synthetic prompts for shared evidence.

Publication, package upload, and deployment are separate operations. Local build success does not establish that any external release exists. See [AGENTS.md](AGENTS.md), [NOTICE.md](NOTICE.md), and [SECURITY.md](SECURITY.md) for the repository boundaries.
