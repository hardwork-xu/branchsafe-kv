# 代码导读

[English](../en/WALKTHROUGH.md) · [API](ARCHITECTURE.md) · [研究](RESEARCH.md)

我用下面的阅读顺序，在修改代码时持续关注所有权规则。先理解一个序列和一个未满尾页，再考虑并发。这份导读解释实现，不声称读者已有任何相关经历。

## 1. 运行完整离线路径

在仓库根目录执行：

```sh
uv sync --frozen
uv run branchsafe demo
uv run pytest -q tests/test_core.py tests/test_concurrency.py
```

[cli.py](../../src/branchsafe/cli.py) 将 `demo` 分派给 `demo()`：创建 17-token 前缀、分叉子序列、在子序列追加五个 token 后接受其中三个，再丢弃父序列一个未提交事务。返回 JSON 前检查精确数组一致性和清理结果。默认示例全部采用合成张量，没有模型或网络请求。

文件接口为 `branchsafe branch --input INPUT.npz --output OUTPUT.npz --accept N`。输入键是 `prefix_k`、`prefix_v`、`suffix_k`、`suffix_v`，各自为 `[L,T,H,D]` float32 数组。输出键为 `k`、`v`，保存前缀和前 `N` 个后缀 token。页预算参数见 CLI `--help`；文件往返流程见 [集成测试](../../tests/test_integration.py)。

## 2. 在 Python 中跟踪一个事务

这个例子足够小，可以直接检查每个字节：

```sh
uv run python - <<'PY'
import numpy as np
from branchsafe import CacheConfig, PagedCache

cfg = CacheConfig(layers=2, heads=2, head_dim=4, page_size=4, max_pages=16)
prefix_k = np.zeros((2, 5, 2, 4), dtype=np.float32)
prefix_v = np.ones_like(prefix_k)
tail_k = np.full((2, 3, 2, 4), 2, dtype=np.float32)
tail_v = np.full_like(tail_k, 3)
with PagedCache(cfg) as cache:
    parent = cache.create()
    parent.append(prefix_k, prefix_v)
    sibling = parent.fork()
    with parent.transaction() as tx:
        tx.append(tail_k, tail_v)
        tx.commit(accepted_tokens=2)
    k, v = parent.materialize()
    np.testing.assert_array_equal(k, np.concatenate((prefix_k, tail_k[:, :2]), axis=1))
    np.testing.assert_array_equal(v, np.concatenate((prefix_v, tail_v[:, :2]), axis=1))
    np.testing.assert_array_equal(sibling.materialize()[0], prefix_k)
    cache.check_invariants()
    print({"parent_tokens": parent.length, "sibling_tokens": sibling.length})
PY
```

打印长度为 7 和 5。原尾页容量为四个 token，其中只有一个可见 token。事务追加时将共享尾部复制到私有页，兄弟序列保留原来的五-token 历史。接受数量相对事务起点，因此父序列变成 `5+2`，不是长度 2。

## 3. 阅读追加路径

在 [core.py](../../src/branchsafe/core.py) 依次阅读 `CacheConfig`、`_validate`、`Sequence.append`、`PagedCache._reserve`：

1. 在池锁内检查所有权、可选凭据、形状、精确 dtype 及有限数值。
2. 计算部分尾页长度、是否共享，以及完整预留需求。
3. 发布状态前预留容量；新数组先在局部构造完成，再加入页池。
4. 必要时复制共享尾部，然后按页复制输入各段。
5. 可恢复复制异常释放尚未发布的预留页并继续抛出；成功则释放旧共享尾页引用，发布新页表、长度、epoch 和计数。

活动序列长度是可见边界。失败复制写入未使用位置后，不必擦除这些隐藏字节；`materialize()` 不返回它们，之后追加会覆盖自己的新增可见区域。这保证所有权隔离，不代表安全擦除数据。

## 4. 跟踪生命周期

`fork()` 共享引用或完整复制已用前缀。`truncate()` 释放新可见尾部之后的页；`close()` 只释放一次所有权，重复清理无害。父序列关闭后子序列仍有效，因为每个活动序列独立持有引用。

`WriteTicket` 包含序列标识、epoch 和长度，是乐观一致性凭据，不是认证凭证。若另一个写入者在事务创建后改变父序列，`commit()` 会抛出 `StaleWriteError`，不能静默覆盖新状态。事务在**创建时**分叉，早于 `__enter__`；不使用上下文时必须显式提交或关闭。

[连续基线](../../src/branchsafe/baseline.py) 具有相同公开语义。阅读其 `fork()` 可确认只复制已用前缀；追加不会反复拼接整个序列。事务提交将子数组转交给父序列，不再复制一次完整前缀。

## 5. 将代码对应到证据

| 证据 | 检查内容 |
|---|---|
| [test_core.py](../../tests/test_core.py) | 精确 KV、页边界、参考状态机、资源上限、回滚、非法输入及失败恢复 |
| [test_concurrency.py](../../tests/test_concurrency.py) | 真实线程中的分支隔离、同凭据竞争与多个提交竞争 |
| [test_model_adapter.py](../../tests/test_model_adapter.py) | 离线缓存布局转换 |
| [validate_model.py](../../scripts/validate_model.py) | 可选固定 GPT-2 原生/适配/完整前缀前向，检查缓存及 logits |
| [benchmark.py](../../benchmark.py) | Dense/shared/eager 比较及完整逐次结果 |
| [benchmark.json](../../configs/benchmark.json) | 固定工作负载与重复配置 |

主要管理基准路径是 fork → append → truncate 接受后缀 → close。它模拟分支管理，不是事务 `commit()` 路径，也不是推理基准。另一个计时范围会物化每个分支。事务由正确性测试、示例及模型验证覆盖。模型验证的诊断计时范围不同，不得换算成加速结论。

## 6. 定位回归

```sh
uv run pytest -q tests/test_core.py -k 'capacity or failure or transaction'
uv run pytest -q tests/test_concurrency.py
uv run python benchmark.py
uv run python scripts/analyze.py
uv run ruff check .
uv run mypy
```

发现不一致时记录形状、种子和操作顺序；每次状态转移后运行 `check_invariants()`，并与新的独立数组比较。检查数值计算前先排查部分尾页共享。遇到意外内存增长时，区分 `live_bytes`、保留的 `allocated_bytes`、输出物化和实测 RSS。分析耗时时检查所有原始样本与方法顺序，保留失败记录，并在比较版本前核查源码哈希。真实状态和实测值以 [EXPERIMENTS.md](EXPERIMENTS.md) 为准，本篇只解释机制。
