# 实验

[English](../en/EXPERIMENTS.md)

## 固定假设与实际条件

[PLAN](PLAN.md)中的目标在实现与计时前以`8ac7d75`提交。看到结果后没有降低目标、更换基线、改变负载或放宽容差。最终缓存运行ID为`bc2deb1c-429c-420a-82ac-b6e003fc2b77`，开始于`2026-09-26T16:55:18.930061+00:00`，结束于`2026-09-26T16:55:59.563741+00:00`。原始文件含实际命令、完整配置、revision、逐文件SHA-256、环境和全部样本。revision可早于证据提交：逐文件哈希绑定实际被测工作区源码。

硬件为Apple M1 Pro、10逻辑CPU、16 GiB统一内存；macOS arm64、Python3.12.2、NumPy2.3.3。基准在CPU运行，导入前把数值工作线程环境变量固定为1；这里不运行依赖大量BLAS计算的attention算子。发现Metal且单独PyTorch MPS求和smoke通过，不代表项目支持GPU后端。CUDA不可用，本机没有Docker。公开依赖和模型下载成功，没有使用付费API。

主机未与其他应用隔离，没有CPU绑核、温度锁定或功耗测量。固定种子的交错顺序与IQR描述波动，但不能消除共享主机干扰。两个开发期运行的耗时和评测工具版本有差异，均保留；所有结论只使用最终运行。

## 公平工作量与成本边界

每个场景预载一个父序列，保持父序列和兄弟存活时创建所有分支，分别追加不同值、截断到接受长度，最后释放子序列、父序列和页池。它们是独立接受的分支，**不是针对同一父epoch成功提交8次事务**。事务发布在单元测试、demo和模型集成中另行验证。

Dense基线为每个序列预分配prefix+append的连续容量，fork只复制可见前缀，追加不拼接整段历史。Eager-paged消融使用相同页池和生命周期但fork复制各页。数据、类型、输入校验、完成工作量、总字节预算和独立副本物化合同一致。预算是上限，不强迫消耗相同字节；dense按序列固定容量而分页按需分配是有意比较的架构差异。

初始化和前缀复制记为`preload_ns`，不在`elapsed_ns`内。管理计时包含分配、校验、fork、追加、截断、释放和页池销毁。所有方法的输入生成均在计时外。第二范围还把每个分支物化为独立连续输出，是完整**存储**生命周期，不是模型端到端推理。原生连续缓存消费者可以直接读取内部数据，但本API承诺独立副本；不能把这里的物化成本当作推理必然成本。

全部形状为4层×4个KV头×64维、float32、页16、追加16、接受8。[7个工作负载配置](../../configs/benchmark.json)覆盖0/32/256/2047/2048/4096前缀及1/4/8分支。read-heavy对每个分支读取4次。整页主场景有利于共享；partial-page明确触发共享末页复制。

每方法/场景/范围预热3次、测21次，共保存1008条：126预热、882正式观测。保留交错运行顺序、中位数和IQR；预热标记而不删除，不丢离群点，不声称p95/p99或统计显著性。[summary](../../results/summary.json)中的吞吐为该存储生命周期每秒完成分支数，不是token/s或模型请求/s。进程冷启动和导入不计入稳态延迟；预载与独立RSS进程另有清楚边界。

## 目标与实测

| Target / 目标 | Measured / 实测 | Status / 状态 |
|---|---|---|
| Primary management ≥2.0× | 5.02×; 22.587 → 4.502 ms | passed / 通过 |
| Managed live payload ≤30% | 11.7%; 152,174,592 → 17,825,792 bytes | passed / 通过 |
| Exact K/V, every benchmark case/method | 21/21 case-method checks | passed / 通过 |
| Native GPT-2 logits ≤1e-5 + matching greedy IDs | 45/45; maximum error 0; bitwise identical | passed / 通过 |
| Stronger full-recompute logits ≤1e-5 | 4 passed, 41 failed; max 0.000701904296875 | failed / 失败 |


主要管理场景中位数/IQR：dense 22.587 ms [20.199,24.428]，共享 4.502 ms [2.970,5.144]。加上物化变为40.748对25.481 ms，仅1.60倍；频繁读取降为0.83倍，在相关不利负载中变慢。短前缀物化也没有可靠改善。主要目标通过不等于普遍加速。

完整生成结果：[英文表](../../results/TABLE_en.md)、[中文表](../../results/TABLE_zh.md)、[原始JSON](../../results/benchmark.json)、[统计JSON](../../results/summary.json)、[SVG](../../results/benchmark.svg)、[PNG](../../results/benchmark.png)。Eager-paged比共享慢，通常也比dense慢，表明分页本身不是收益来源。共享减少前缀复制；Python页表和gather都是实际成本。

## 三种独立的内存口径

`managed_peak_bytes`是有存活所有者的K/V页容量最大和，包含预留瞬时峰值及父序列；dense为存活预分配容量。它来自实际数组分配，是机制计数，不是RSS采样。`allocated_bytes_at_branch_peak`报告观察点保留的所有数组，包含已空闲页，但不是进程峰值采样；空闲页可保留至pool.close。两者均不含Python元数据、输入、有限性校验临时数组及物化输出。

新工作进程通过`resource.getrusage`取得高水位RSS，macOS为字节、Linux KiB转换为字节；包括解释器/导入、输入、前缀构建、读取流程与分配器行为。每方法/场景只有一次，属于描述性数据，不是分布。主场景RSS：dense 225,574,912字节，eager 226,099,200，共享93,995,008。不能用11.7%存活数据比代替RSS比。copied_bytes包含预载及成功追加/fork/COW，不含物化和失败过程中部分复制，不是硬件内存流量。

## 预训练验证与保留的失败假设

[GPT-2原始报告](../../results/model.json)运行124,439,808参数预训练模型，`openai-community/gpt2`固定revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`、safetensors、CPU float32、Torch单线程、eager attention。报告逐项列出下载文件SHA-256、原创合成提示词、token ID、全部比较和清理。3个提示词分别分出2种后缀并继续缓存解码，另外3组验证回滚和部分提交；这是实际预训练前向，不是随机权重或伪造缓存输出。

原生Transformers缓存与存储适配路径45项K/V和logits逐位相同。更严格的完整重计算对照有41项超过固定1e-5，最大0.000701904296875。每项native-full与adapted-full差异均相同，所有贪心ID一致。这说明被测路径没有额外存储误差，不证明更长生成中的概率分布或整体质量相同。不同float32运算形状是合理解释，也与[PyTorch数值精度文档](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)一致，但不是放宽容差的理由。该更强诊断保持failed且脚本退出1，没有删除断言或放宽阈值。

下载、哈希和加载为独立诊断计时；前向与转换耗时记录但不用于加速结论。模型权重不进入Git。

## 开发运行、失败和复现

[首轮缓存结果](../../results/benchmark-initial.json)与[首轮模型结果](../../results/model-initial.json)原样保留。首轮后审阅发现页容量向上取整保护、失败记录与样本完整性检查需要加固；最终完整重跑未变更的原始负载，不混合新旧样本，也不选择性删除。模型已知失败独立于这些工具修改。单元测试注入分配/复制失败验证恢复，正常数组路径和线程测试都执行实际代码。

```bash
uv sync --frozen
uv run python benchmark.py --output results/benchmark.json
uv run python scripts/analyze.py
uv run python scripts/update_readme.py
uv run python scripts/validate.py
uv sync --frozen --group model
uv run --group model python scripts/validate_model.py --output results/model.json
```

最后一条命令当前因已知诊断失败返回1。重跑会产生新run_id和时间；替换之前保存旧结果。固定快照下载后可用`--offline`，不要与性能基准并行。[验收日志](../../results/acceptance/checks.json)和[补充记录](../../results/acceptance/supplemental.json)将实际命令和状态绑定到源码。缺失硬件不计作通过，也不当作后端支持。
