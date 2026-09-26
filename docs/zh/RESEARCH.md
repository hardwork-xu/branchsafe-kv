# 研究动机

[English](../en/RESEARCH.md) · [实验前计划](PLAN.md) · [实验](EXPERIMENTS.md) · [已核查来源](../sources.json)

我选择分支存储，是因为它同时暴露了可测量的数据搬运成本，以及能够在普通开发机上检验的正确性问题。多个续写分支可以复用同一前缀；放弃一个分支时，必须释放资源且不改变父序列。只有这些生命周期保证成立，减少复制才有价值。

## 与本周的关系

资料窗口为 **2026 年 9 月 21–27 日**。vLLM 于 9 月 21 日合并 [PR #56734](https://github.com/vllm-project/vllm/pull/56734)，处理 dummy run 中投机草稿通过陈旧 block table 写入缓存的问题，合并提交为 [d2983f2](https://github.com/vllm-project/vllm/commit/d2983f2)。这证明本周仍有缓存生命周期正确性方面的上游工程活动；不表示 BranchSafe 复现了该 GPU 故障或修复了 vLLM。[9 月 22 日 vLLM Metal 官方公告](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0) 也讨论了 Apple Silicon 上的页式 KV 和显式内存预算。这里的“当前热点”指有日期可查的上游活动，不声称具有人气排名依据。

曾比较三个方向：完整投机解码、有限状态 token masking、分支 KV 存储。完整解码需要处理草稿模型质量与推理成本；masking 已有成熟原生引擎，包括 [XGrammar 9 月 24 日发布版本](https://github.com/mlc-ai/xgrammar/releases/tag/v0.2.8)。分支存储可以在现有 CPU 与内存预算下完成更聚焦、可独立检验的贡献。选题及数值目标在性能测量前写入 [PLAN.md](PLAN.md)。

## 问题、假设与贡献

**研究问题：** 哪些分支形状下，页共享可以降低存储管理延迟和复制字节数，同时让回滚、拒绝写入与资源失败保留父序列的全部可见状态？

**假设：** 长共享前缀与多个短分支可以避免大量前缀复制；短前缀、空前缀、Python 页表开销及频繁连续物化则可能抵消收益。实验配置保留短前缀、空前缀、非整页及频繁读取场景，使假设能够被反驳。

项目贡献是独立实现的有界 NumPy 存储：带校验的序列写入凭据、隔离追加事务、可恢复分配路径、完整复制页式消融方案及预分配连续复制基线。研究价值在于明确合同与可核查的取舍证据；它**不是**新的 attention 算法、新的 COW 方法、推理服务器，也不宣称优于 vLLM。

## 相关工作与归属

- [PagedAttention（SOSP 2023）§4.4–5.2](https://arxiv.org/pdf/2309.06180) 已说明页共享、引用计数、COW 以及 fork/append/free。BranchSafe 在小型 CPU 库中重新实现这些已有存储思想，没有实现论文的 GPU attention kernel，也没有复现其吞吐结果。
- [Transformers 缓存指南](https://huggingface.co/docs/transformers/kv_cache#prefill-a-cache-prefix-caching) 展示预分配 `StaticCache` 及复制前缀的复用方式，为实用连续复制基线提供依据。本地基线是独立 NumPy 实现，并非 Transformers 内部代码的性能测量。
- [Generational arena 文档](https://docs.rs/generational-arena/0.2.9/generational_arena/) 说明已有的陈旧引用防护思路。BranchSafe 检查序列标识、epoch 与长度；不暴露原始页句柄，也不宣称提出新的 ABA 问题解法。

以上是概念参考；核心实现没有复制第三方实现代码。依赖归属见 [NOTICE](../../NOTICE_zh.md)。

## 存储模型

令 `L` 为层数、`H` 为 KV head 数、`D` 为 head dimension、`s` 为单标量字节数、`P` 为每页 token 数、`T` 为可见前缀长度、`A` 为追加长度、`B` 为子分支数。数组形状统一为 `[L,T,H,D]`。单个 K/V token 对的载荷为：

\[
u = 2LHDs, \qquad \text{page bytes} = Pu.
\]

序列保存有序页表与可见长度。逻辑 token `t` 对应页 `floor(t/P)` 和偏移 `t mod P`；页表恰有 `ceil(T/P)` 项。物理页引用计数等于包含该槽位的活动序列页表数量。可见长度之外的字节未定义，不能读取出来。

向共享的未满尾页追加时，先预留全部所需新页，再将有效尾部复制到私有页并写入新增 token，最后发布新页表和长度。完整前缀页继续共享。如果尾页独占，则直接使用剩余空间。

保留一个父序列及 `B` 个追加同长后缀 `A` 的子序列，忽略元数据，令 `r=T mod P`。下式为**预期载荷页数**，不是进程内存实测：

\[
\text{distinct pages} = \lceil T/P\rceil + B\lceil(r+A)/P\rceil
\]

条件是 `A>0`、每个子序列均已追加且父子都未关闭。`r=0` 时新页只放后缀；`r>0` 时各子序列另有复制的部分尾页。`A=0` 时分叉不新增载荷页。连续基线每个序列预留 `S` 个 token 槽位，载荷为 `(B+1)Su` 字节。基准对各方法设置相同的最终存储需求和总字节预算。

## 正确性论证与不变量

1. 新序列没有可见 token。分叉要么复制可见字节，要么增加相同页的引用。
2. 共享页的可见范围不被修改。共享尾页追加使用私有存储，因此子分支写入不能改变其他序列的可见前缀。
3. 截断修改可见范围并释放完整无用页。后续写入仍检查所有权，所以截短的尾部不会破坏兄弟序列。
4. 事务保存 `(sequence_id, epoch, length)` 快照并持有隔离子序列。仅父序列仍匹配快照时提交，接受前缀并增加 epoch；回滚释放子序列。
5. 修改前检查容量。已经测试的可恢复分配、复制异常保留可见数据并回收活动预留页。这不是针对任意解释器 OOM、异步终止或恶意修改私有字段的形式化证明。

存储要求逐位相等，不执行压缩或量化。下游模型适配有独立数值容差：预先固定 `max_abs_logit_error <= 1e-5` 且 greedy token IDs 相同。数值一致是正确性检查，不代表输出质量提高。

## 复杂度与边界

| 操作 | 共享页式存储 | 预分配连续基线 |
|---|---|---|
| Fork | `O(ceil(T/P))` 元数据，前缀载荷零复制 | `O(TLHD)` 前缀复制；预留 `S` 容量 |
| Append | 部分尾页 COW 最坏 `O(ALHD + PLHD)`，另加页表复制 | 向预留空间写入，`O(ALHD)` |
| Truncate | `O(释放页数)` | 修改可见长度，`O(1)` |
| Materialize | `O(TLHD)`，新建输出数组 | `O(TLHD)`，新建输出数组 |
| Commit | 转移页表所有权并释放引用 | 转移数组所有权 |

Python 页式 append 还会复制页表列表，成本为 `O(ceil(T/P))`，长上下文下并非常数时间。整个池的锁会串行化操作。`max_pages` 限制保留的 NumPy 载荷，`max_sequences` 限制活动序列数；二者都不是进程 RSS 总限制。本研究覆盖常规 full-attention K/V 张量，不覆盖滑窗或循环模型状态、设备 kernel、分布式租约或恶意调用者。直接消费页式布局的 attention、细粒度锁和调度器准入策略是后续研究问题，均非本版本承诺。
