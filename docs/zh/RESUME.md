# 简历材料与项目讲解

[English](../en/RESUME.md)

我把以下条目作为有证据对应的仓库描述。它们说明已实现的工作，不声称已经掌握全部知识、生产采用、论文录用或具体投入时长。

## 四条项目描述

- 实现有界NumPy KV存储，组合引用计数前缀共享、末页写时复制、版本化写凭证和部分接受事务；验证容量、分配、复制及并发失败时可见状态保持一致。
- 设计实验前固定的评测，对比预分配dense-copy与eager-paged基线；在Apple M1 Pro CPU float32、2048-token前缀及8分支下，把缓存管理中位耗时从22.587降至4.502 ms（5.02倍），存储K/V完全一致。
- 同场景将存活受管KV容量由152,174,592降至17,825,792字节（基线11.7%）；另测进程RSS并披露频繁读取降至0.83倍的不利结果，不将子系统收益表述为端到端推理加速。
- 集成固定版本124M参数预训练GPT-2，45/45原生缓存logits逐位等价，同时保留41项更严格完整重计算失败；交付离线测试、锁定依赖、双语文档与可复现构建/CI配置。

## 证据索引

| 结论 | 实现 | 测试/输入 | 原始证据 |
|---|---|---|---|
| 所有权和失败合同 | [核心](../../src/branchsafe/core.py)、[基线](../../src/branchsafe/baseline.py) | [核心测试](../../tests/test_core.py)、[线程测试](../../tests/test_concurrency.py) | [验收](../../results/acceptance/checks.json) |
| 延迟与公平消融 | [基准](../../benchmark.py)、[分析](../../scripts/analyze.py) | [固定配置](../../configs/benchmark.json)、[计划](PLAN.md)、[工具测试](../../tests/test_harness.py) | [原始](../../results/benchmark.json)、[统计](../../results/summary.json) |
| 内存与不利边界 | [计数器](../../src/branchsafe/core.py)、[RSS工作进程](../../benchmark.py) | 同一配置中的primary/read_heavy | [memory/samples](../../results/benchmark.json) |
| 预训练集成与交付 | [模型验证](../../scripts/validate_model.py)、[CI](../../.github/workflows/ci.yml) | [适配测试](../../tests/test_model_adapter.py)、报告中的prompt/revision | [模型](../../results/model.json)、[发布](RELEASE.md) |

## 简短讲解

为什么选择：分支续写可重复复制长KV前缀，而共享又带来生命周期风险。本周上游工程活动提供具体动机，底层COW是已有方法。

如何工作：逻辑序列拥有页引用；fork增加所有权；追加共享末页前先预留私有页。事务写隔离分支，父版本未变才发布接受token。容量与序列数量限制让失败明确可控。

关键取舍：减少前缀数据复制，代价是Python页元数据、锁和昂贵连续读取。全局锁简化正确性论证，也限制并行计算；dense预分配是有意选择的实用基线。

最重要实验：管理与管理加读取的成对比较。5.02倍管理加速在一次物化后只有1.60倍，频繁读取是0.83倍。存储优化不能自动推导模型推理更快。

值得解释的失败：完整重算float32 logits超过1e-5，但原生和恢复缓存logits逐位相同。保留失败假设并隔离实现贡献，而不是更换指标。

下一步研究问题：attention消费者能否直接读页？页大小和分支数的收益交叉点在哪里？更细锁能否保持同样失败合同？这些是问题，不是已实现或承诺交付的功能。项目不保证申博或求职结果。
