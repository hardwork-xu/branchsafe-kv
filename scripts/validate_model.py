"""Validate real GPT-2 cache interoperability. / 验证真实 GPT-2 缓存互操作。

This optional CPU experiment downloads pinned MIT-licensed weights when needed.
该可选 CPU 实验在需要时下载固定版本的 MIT 许可模型权重。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

MODEL_ID = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
THRESHOLD = 1e-5
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
)
PROMPTS = (
    "A reliable cache must preserve its prefix when",
    "The research notebook records every measurement because",
    "A small compiler can improve memory efficiency when",
)
SUFFIXES = (
    " two branches share the same history.",
    " a speculative branch is rejected safely.",
)
FloatArray = NDArray[np.float32]


def pack_layers(layers: Sequence[tuple[FloatArray, FloatArray]]) -> tuple[FloatArray, FloatArray]:
    """Pack (batch=1, heads, tokens, dim) layers. / 打包单批次原生缓存。

    Returns independent (layers, tokens, heads, dim) float32 arrays.
    返回独立的 (layers, tokens, heads, dim) float32 数组。
    """
    if not layers:
        raise ValueError("At least one layer is required / 至少需要一层")
    shape = layers[0][0].shape
    if len(shape) != 4 or shape[0] != 1 or min(shape[1], shape[3]) < 1:
        raise ValueError("Expected batch=1, heads, tokens, dim / 需要单批次四维缓存")
    for pair in layers:
        for array in pair:
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError("Cache shapes and float32 dtype must match / 缓存形状和类型须一致")
            if not np.isfinite(array).all():
                raise ValueError("Cache must be finite / 缓存必须为有限数值")
    k = np.stack([pair[0][0].transpose(1, 0, 2) for pair in layers])
    v = np.stack([pair[1][0].transpose(1, 0, 2) for pair in layers])
    return np.ascontiguousarray(k), np.ascontiguousarray(v)


def unpack_layers(k: FloatArray, v: FloatArray) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Unpack stored arrays to independent native layers. / 解包为独立的原生层缓存。"""
    if k.ndim != 4 or k.shape != v.shape or min(k.shape[0], k.shape[2], k.shape[3]) < 1:
        raise ValueError("Expected matching four-dimensional K/V / 需要形状一致的四维 K/V")
    if k.dtype != np.float32 or v.dtype != np.float32:
        raise ValueError("Only float32 is supported / 仅支持 float32")
    if not np.isfinite(k).all() or not np.isfinite(v).all():
        raise ValueError("Cache must be finite / 缓存必须为有限数值")
    return tuple(
        (k[i].transpose(1, 0, 2)[None].copy(), v[i].transpose(1, 0, 2)[None].copy())
        for i in range(k.shape[0])
    )


def file_hash(path: Path) -> str:
    """Streaming SHA-256, including weights. / 流式计算 SHA-256，适用于权重。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(root: Path) -> dict[str, Any]:
    """Record revision and exact relevant bytes. / 记录提交与相关文件精确内容。"""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    files = sorted((root / "src" / "branchsafe").glob("*.py")) + [Path(__file__)]
    hashes = {str(path.relative_to(root)): file_hash(path) for path in files}
    return {
        "git_revision": result.stdout.strip() if result.returncode == 0 else None,
        "file_sha256": hashes,
        "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    }


def run_experiment(args: argparse.Namespace, report: dict[str, Any]) -> None:
    """Execute independent native/paged forwards and rollback. / 执行独立前向与回滚验证。"""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from branchsafe import CacheConfig, PagedCache

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    report["software"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "numpy", "safetensors", "huggingface-hub")
    }
    report["hardware"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "backend": "cpu",
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }
    download_start = time.perf_counter()
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            cache_dir=args.cache_dir,
            allow_patterns=list(MODEL_FILES),
            local_files_only=args.offline,
        )
    )
    report["snapshot_resolution_seconds"] = time.perf_counter() - download_start
    report["model"]["files"] = {
        name: {"sha256": file_hash(snapshot / name), "bytes": (snapshot / name).stat().st_size}
        for name in MODEL_FILES
    }
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    report["model_load_seconds"] = time.perf_counter() - load_start
    report["model"]["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    report["timing_scope"] = {
        "model_load": (
            "local safetensors + tokenizer load; download and SHA-256 excluded / "
            "本地权重和分词器加载；不含下载与 SHA-256"
        ),
        "steps": (
            "single observations, no warm-up; diagnostic breakdown, not a speedup benchmark / "
            "单次观测，无预热；仅诊断分项，不是加速基准"
        ),
        "adapted_seconds": (
            "materialize + NumPy/Torch conversion + CPU forward + delta extraction + append / "
            "物化、数组转换、CPU 前向、新增缓存提取及追加"
        ),
        "native_seconds": (
            "CPU forward only; constructing independent baseline cache excluded / "
            "仅 CPU 前向，不含独立基线缓存构造"
        ),
        "full_seconds": "CPU full-prefix forward only / 仅完整前缀 CPU 前向",
        "claim": (
            "timings are not comparable end-to-end inference performance / 不作为端到端性能对比"
        ),
    }
    config = CacheConfig(
        layers=model.config.n_layer,
        heads=model.config.n_head,
        head_dim=model.config.n_embd // model.config.n_head,
        page_size=16,
        max_pages=256,
    )
    store = PagedCache(config)
    comparisons: list[dict[str, Any]] = report["comparisons"]

    def legacy(cache: Any) -> tuple[tuple[Any, Any], ...]:
        return cache.to_legacy_cache()

    def clone_cache(past: tuple[tuple[Any, Any], ...]) -> Any:
        return DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in past))

    def as_numpy(past: tuple[tuple[Any, Any], ...]) -> tuple[FloatArray, FloatArray]:
        return pack_layers(tuple((k.detach().numpy(), v.detach().numpy()) for k, v in past))

    def as_cache(k: FloatArray, v: FloatArray) -> Any:
        return DynamicCache.from_legacy_cache(
            tuple((torch.from_numpy(a), torch.from_numpy(b)) for a, b in unpack_layers(k, v))
        )

    def check_step(
        seq: Any,
        history: list[int],
        incoming: list[int],
        native_past: tuple[tuple[Any, Any], ...],
        label: str,
    ) -> tuple[tuple[tuple[Any, Any], ...], int]:
        tokens = torch.tensor([incoming], dtype=torch.long)
        prefix_length = seq.length
        adapted_start = time.perf_counter()
        k, v = seq.materialize()
        adapted_past = as_cache(k, v)
        conversion_seconds = time.perf_counter() - adapted_start
        forward_start = time.perf_counter()
        adapted = model(tokens, past_key_values=adapted_past, use_cache=True)
        adapted_forward_seconds = time.perf_counter() - forward_start
        store_start = time.perf_counter()
        new_k, new_v = as_numpy(legacy(adapted.past_key_values))
        seq.append(new_k[:, prefix_length:], new_v[:, prefix_length:])
        storage_seconds = time.perf_counter() - store_start
        adapted_seconds = time.perf_counter() - adapted_start
        independent_past = clone_cache(native_past)
        native_start = time.perf_counter()
        native = model(tokens, past_key_values=independent_past, use_cache=True)
        native_seconds = time.perf_counter() - native_start
        full_input = torch.tensor([history + incoming], dtype=torch.long)
        full_start = time.perf_counter()
        full = model(full_input, use_cache=False)
        full_seconds = time.perf_counter() - full_start
        adapted_logits = adapted.logits[:, -1, :]
        native_logits = native.logits[:, -1, :]
        full_logits = full.logits[:, -1, :]
        native_error = float((adapted_logits - native_logits).abs().max())
        full_error = float((adapted_logits - full_logits).abs().max())
        native_full_error = float((native_logits - full_logits).abs().max())
        native_greedy = int(native_logits.argmax())
        adapted_greedy = int(adapted_logits.argmax())
        full_greedy = int(full_logits.argmax())
        expected_k, expected_v = as_numpy(legacy(native.past_key_values))
        stored_k, stored_v = seq.materialize()
        bytes_equal = np.array_equal(stored_k, expected_k) and np.array_equal(stored_v, expected_v)
        comparisons.append(
            {
                "case": label,
                "prefix_tokens": prefix_length,
                "incoming_token_ids": incoming,
                "native_max_abs_error": native_error,
                "native_logits_bitwise_equal": bool(torch.equal(adapted_logits, native_logits)),
                "full_recompute_max_abs_error": full_error,
                "native_vs_full_recompute_max_abs_error": native_full_error,
                "greedy_ids": {
                    "adapted": adapted_greedy,
                    "native": native_greedy,
                    "full": full_greedy,
                },
                "stored_kv_bitwise_equal": bool(bytes_equal),
                "native_check": "passed"
                if native_error <= THRESHOLD and adapted_greedy == native_greedy and bytes_equal
                else "failed",
                "full_recompute_check": "passed"
                if full_error <= THRESHOLD and adapted_greedy == full_greedy
                else "failed",
                "timing": {
                    "adapted_seconds": adapted_seconds,
                    "conversion_seconds": conversion_seconds,
                    "adapted_forward_seconds": adapted_forward_seconds,
                    "storage_seconds": storage_seconds,
                    "native_seconds": native_seconds,
                    "full_seconds": full_seconds,
                },
            }
        )
        return legacy(native.past_key_values), adapted_greedy

    with torch.inference_mode():
        for prompt_index, prompt in enumerate(PROMPTS):
            root_seq = store.create()
            root_ids = tokenizer.encode(prompt)
            prefix = model(torch.tensor([root_ids]), use_cache=True)
            prefix_past = legacy(prefix.past_key_values)
            prefix_k, prefix_v = as_numpy(prefix_past)
            root_seq.append(prefix_k, prefix_v)
            branches: list[Any] = []
            try:
                candidate_past: tuple[tuple[Any, Any], ...] | None = None
                candidate_ids: list[int] = []
                for branch_index, suffix in enumerate(SUFFIXES):
                    branch = root_seq.fork()
                    branches.append(branch)
                    suffix_ids = tokenizer.encode(suffix)
                    label = f"prompt-{prompt_index}/branch-{branch_index}"
                    native_past, next_id = check_step(
                        branch, root_ids, suffix_ids, prefix_past, label + "/suffix"
                    )
                    if branch_index == 0:
                        candidate_past, candidate_ids = native_past, suffix_ids
                    history = root_ids + suffix_ids
                    generated: list[int] = []
                    for step in range(args.steps):
                        generated.append(next_id)
                        native_past, next_id = check_step(
                            branch, history, [next_id], native_past, label + f"/decode-{step}"
                        )
                        history.append(generated[-1])
                    report["outputs"].append(
                        {
                            "case": label,
                            "prompt": prompt,
                            "suffix": suffix,
                            "generated_token_ids": generated,
                            "generated_text": tokenizer.decode(generated),
                        }
                    )
                assert candidate_past is not None
                candidate_k, candidate_v = as_numpy(candidate_past)
                original_length = root_seq.length
                with root_seq.transaction() as transaction:
                    transaction.append(
                        candidate_k[:, original_length:], candidate_v[:, original_length:]
                    )
                rolled_k, rolled_v = root_seq.materialize()
                rollback_ok = (
                    root_seq.length == original_length
                    and np.array_equal(prefix_k, rolled_k)
                    and np.array_equal(prefix_v, rolled_v)
                )
                accepted = 2
                with root_seq.transaction() as transaction:
                    transaction.append(
                        candidate_k[:, original_length:], candidate_v[:, original_length:]
                    )
                    transaction.commit(accepted_tokens=accepted)
                committed_k, committed_v = root_seq.materialize()
                retained_length = original_length + accepted
                partial_ok = (
                    root_seq.length == retained_length
                    and np.array_equal(committed_k, candidate_k[:, :retained_length])
                    and np.array_equal(committed_v, candidate_v[:, :retained_length])
                )
                native_past = tuple(
                    (k[:, :, :retained_length].clone(), v[:, :, :retained_length].clone())
                    for k, v in candidate_past
                )
                history = root_ids + candidate_ids[:accepted]
                replacement_ids = tokenizer.encode(" safely")
                native_past, next_id = check_step(
                    root_seq,
                    history,
                    replacement_ids,
                    native_past,
                    f"prompt-{prompt_index}/partial-accept/replacement",
                )
                history += replacement_ids
                for step in range(args.steps):
                    previous_id = next_id
                    native_past, next_id = check_step(
                        root_seq,
                        history,
                        [previous_id],
                        native_past,
                        f"prompt-{prompt_index}/partial-accept/decode-{step}",
                    )
                    history.append(previous_id)
                report["transactions"].append(
                    {
                        "prompt_index": prompt_index,
                        "rollback_parent_unchanged": bool(rollback_ok),
                        "partial_commit_matches_native": bool(partial_ok),
                        "proposed_tokens": len(candidate_ids),
                        "accepted_tokens": accepted,
                        "replacement_token_ids": replacement_ids,
                    }
                )
            finally:
                for branch in branches:
                    branch.close()
                root_seq.close()
    store.check_invariants()
    report["cleanup"] = store.stats()
    report["summary"] = {
        "comparisons": len(comparisons),
        "native_passed": sum(row["native_check"] == "passed" for row in comparisons),
        "native_failed": sum(row["native_check"] == "failed" for row in comparisons),
        "full_recompute_passed": sum(
            row["full_recompute_check"] == "passed" for row in comparisons
        ),
        "full_recompute_failed": sum(
            row["full_recompute_check"] == "failed" for row in comparisons
        ),
        "native_max_abs_error": max(row["native_max_abs_error"] for row in comparisons),
        "full_recompute_max_abs_error": max(
            row["full_recompute_max_abs_error"] for row in comparisons
        ),
        "all_greedy_equal": all(len(set(row["greedy_ids"].values())) == 1 for row in comparisons),
        "transactions_passed": all(
            row["rollback_parent_unchanged"] and row["partial_commit_matches_native"]
            for row in report["transactions"]
        ),
        "cleanup_passed": report["cleanup"]["live_pages"] == 0
        and report["cleanup"]["sequences"] == 0,
    }
    report["native_status"] = (
        "passed"
        if report["summary"]["native_failed"] == 0
        and report["summary"]["transactions_passed"]
        and report["summary"]["cleanup_passed"]
        else "failed"
    )
    report["full_recompute_status"] = (
        "passed" if report["summary"]["full_recompute_failed"] == 0 else "failed"
    )
    report["status"] = (
        "passed"
        if report["native_status"] == "passed" and report["full_recompute_status"] == "passed"
        else "failed"
    )
    report["interpretation"] = {
        "en": (
            "Native-cache equality isolates storage correctness. Full-recompute failures retain "
            "the original 1e-5 criterion; equal native/full discrepancies are consistent with "
            "float32 execution-shape roundoff. This is a correctness experiment, "
            "not an end-to-end speedup claim."
        ),
        "zh": (
            "原生缓存对照隔离存储正确性。完整重计算失败保留原定 1e-5 标准；"
            "原生与完整重计算的同等偏差符合 float32 计算形状舍入现象。"
            "这是正确性实验，不是端到端加速结论。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/model.json"), help="JSON report / JSON 报告"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/huggingface"), help="Weight cache / 权重缓存"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require cached pinned weights / 仅用已缓存的固定权重",
    )
    parser.add_argument("--threads", type=int, default=1, help="CPU threads / CPU 线程数")
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Greedy continuation steps per branch / 每个分支贪心续写步数",
    )
    parser.add_argument("--seed", type=int, default=20260927, help="Random seed / 随机种子")
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.steps <= 32:
        parser.error("threads >= 1 and 1 <= steps <= 32 required / 线程须为正，步数须在 1 到 32")
    root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "started_at": datetime.now(UTC).isoformat(),
        "source": source_identity(root),
        "command": [
            "python",
            "scripts/validate_model.py",
            "--threads",
            str(args.threads),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
        ]
        + (["--offline"] if args.offline else []),
        "seed": args.seed,
        "dtype": "float32",
        "threshold": THRESHOLD,
        "model": {
            "id": MODEL_ID,
            "revision": REVISION,
            "license": "MIT",
            "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
            "upstream_license": "https://github.com/openai/gpt-2/blob/master/LICENSE",
        },
        "workload": {
            "type": "pretrained model; original synthetic prompts / 预训练模型，原创合成提示词",
            "prompts": list(PROMPTS),
            "suffixes": list(SUFFIXES),
            "steps": args.steps,
        },
        "comparisons": [],
        "transactions": [],
        "outputs": [],
        "status": "running",
    }
    try:
        run_experiment(args, report)
    except Exception as error:
        # Preserve partial evidence; never turn an execution failure into a pass.
        # 保留已产生的证据；执行异常不可转化为成功状态。
        report["status"] = "error"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error)
            .replace(str(root), "<project>")
            .replace(str(Path.home()), "<home>"),
        }
    report["finished_at"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "summary": report.get("summary"),
                "error": report.get("error"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
