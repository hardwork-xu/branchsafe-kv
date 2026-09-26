# 实际开发记录

[English](../en/DEVELOPMENT.md)

我记录本次真实阶段，便于以后复查实现和证据。本次为Asia/Shanghai的2026-09-27连续交付（原始时间用UTC）；不暗示已进行数周研究或生产部署。

1. 检查空的非Git工作区、CPU/内存/磁盘、编译器/Python、公开下载和Docker缺失。只创建隔离项目及局部依赖环境。核实已有公开提交所用GitHub身份；本仓库沿用该公开名称与隐私邮箱，没有修改全局身份，也没有发布全局私人邮箱。
2. 比较三个方向、核查带日期的上游资料，在实现前提交范围与数值目标。核心是已有COW与明确事务生命周期合同的组合，不是新的attention算法。
3. 实现有界缓存和实用dense基线。审阅发现预留后异常清理、直接/浅复制句柄所有权和无限空序列元数据问题；增加清理、注册身份校验和max_sequences。除实际数组/线程测试外，还加入有意义的分配/复制失败注入回归。
4. 增加CLI、NPZ、固定种子交错基准及独立新进程RSS。首轮基准完整结束并保留；随后发现页取整预算、worker失败记录和缺失样本校验的缺口，修复后完整重跑同一负载，没有改变数值目标。
5. 下载固定revision的safetensors并运行真实GPT-2。原生cache logits在45项比较中逐位一致；完整重算float32有41项超过严格阈值，保留原始失败与退出状态，没有放宽容差。
6. 直接从JSON生成表图，编写语义对应双语文档，检查安装、构建、示例、类型、格式、引用和公开内容。外部发布与托管CI另外留证，区别于本地完成。

## 文档整理时的真实提交

| 提交 | 摘要 |
|---|---|
| 8ac7d75 | docs: freeze research scope and benchmark targets |
| f10e76c | feat: implement bounded transactional copy-on-write KV storage |
| 8aa4a0e | test: verify cache invariants and expose bilingual workflows |
| 4265c9b | build: add reproducible benchmark and acceptance tooling |
| c7931f5 | perf: record cache benchmarks and pretrained equivalence evidence |

这些哈希来自git log，不是事后构造。后续文档、验收与发布提交可在实际历史中查看；提交正文含中文对应说明与已观察验证。

## 验证与结论边界

发布检查前完整离线测试151项通过，没有通过跳过GPU测试暗示支持。覆盖独立随机状态轨迹、NumPy参考、输入输出别名隔离、并发ticket/CAS竞争和可恢复分配/复制失败。缓存不是崩溃一致数据库，也不是对抗恶意进程的安全边界。

[最终基准](../../results/benchmark.json)与[首轮基准](../../results/benchmark-initial.json)保留有效数据；[模型结果](../../results/model.json)保留更强诊断失败。[验收](../../results/acceptance/checks.json)与[补充检查](../../results/acceptance/supplemental.json)记录实际命令、退出状态和源码哈希。本机仍无Docker；若完成托管容器验证，其证据独立记录于[发布](RELEASE.md)。

不声称节省特定费用或token，不虚构个人研发时长。后续变更须重验受影响部分、保留旧证据并同步双语，参见[AGENTS](../../AGENTS.md)。
