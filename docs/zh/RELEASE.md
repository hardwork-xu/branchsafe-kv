# 发布准备 — 0.1.0

[English](../en/RELEASE.md)

我将 BranchSafe KV 准备为一个规模聚焦、实现可审阅的系统项目。本文是发布草稿。准备时尚未创建公开仓库、包注册平台版本、托管服务或已部署容器。预定仓库名为 `branchsafe-kv`；相对链接在公开发布之前也可使用。

## 发布说明草稿

0.1.0 版本提供有界 CPU NumPy KV 存储，包括共享前缀页、共享尾部页写时复制、失败原子的追加预留、过期写入检测、回滚，以及分支事务的部分接受。项目还提供预分配稠密基线、完整复制分页消融方案、离线 CLI、单元和集成测试、锁定依赖、双语文档与机器可读实验证据。

分页与写时复制是已有技术。本项目的贡献是可审阅的生命周期及失败不变量实现，以及边界清晰的测量。本版本不宣称提出新的注意力算法、支持 GPU、已进行生产部署，或实现普遍的推理加速。

可选的固定版本 GPT-2 实验记录了 45/45 次原生缓存对照，logits 与 K/V 均逐位一致。包括完整重计算在内，所有贪心 token ID 相同，三个回滚及部分接受场景通过。但 float32 完整重计算的 45 次对照中有 41 次超过预注册的 `1e-5` 绝对误差上限，最大值为 `0.000701904296875`。原生缓存对完整重计算的偏差，与分页缓存对完整重计算的偏差完全相同。更严格的诊断仍为**失败**，验证脚本返回退出码 1。这些结果支持与原生缓存的存储等价性，不支持更强的完整重计算误差上限。见[最终结果](../../results/model.json)及[保留的首次结果](../../results/model-initial.json)。

## 安装与演示

声明的 Python 范围为 3.11–3.13。本地已验证路径为 macOS arm64、Python 3.12、CPU、float32。项目提供 Linux GitHub Actions 配置，但 YAML 的存在不等于远端执行成功。CUDA、MPS 执行与 Windows 均不是已验证后端。

在已提供 `uv` 的本地仓库根目录执行：

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

`make build` 在 `dist/` 中生成带版本号的 wheel 与源码分发包。它们的存在属于本地构建结果，不代表已上传注册平台。将 wheel 安装到独立环境可验证打包后的 API：

```sh
uv venv work/release-check
uv pip install --python work/release-check/bin/python dist/branchsafe_kv-0.1.0-py3-none-any.whl
work/release-check/bin/branchsafe demo
```

可选的真实模型正确性验证：

```sh
uv sync --frozen --group model
uv run --frozen --group model python scripts/validate_model.py
uv run --frozen --group model python scripts/validate_model.py --offline --output results/model-recheck.json
```

首次调用脚本时，可能下载约 548 MB 固定版本的 GPT-2 safetensors 权重及分词器文件。离线命令要求快照已经存在。两种命令均保留严格诊断；对于已记录的完整重计算失败，预期退出码为 1。不要与性能基准同时运行模型验证。默认演示和测试无需模型权重或商业凭据。

## 容器状态

[Dockerfile](../../Dockerfile) 使用 `python:3.12.11-slim-bookworm`，安装锁定的运行时依赖，并以非 root 用户执行演示。开发主机没有可用 Docker，因此容器构建和运行**未在本地验证**：

```sh
docker build -t branchsafe-kv:0.1.0 .
docker run --rm branchsafe-kv:0.1.0
```

镜像仅提供 CPU 演示路径。宿主 macOS 的结果不能证明 Linux 容器性能，也不能证明容器中已支持可选模型路径。

## 公开元数据草稿

一句话介绍 / GitHub About：

> 提供写时复制前缀共享的有界事务式 KV 缓存，附 CPU 基准和可复现的双语证据。

对应英文投放文案：

> Bounded transactional KV cache with copy-on-write prefix sharing, CPU benchmarks, and reproducible bilingual evidence.

建议 Topics：`kv-cache`、`copy-on-write`、`memory-management`、`transactions`、`llm-inference`、`numpy`、`benchmark`、`reproducible-research`。

使用 [MIT 许可证](../../LICENSE)、[第三方来源说明](../../NOTICE_zh.md)、[贡献指南](../../CONTRIBUTING_zh.md)、[安全边界](../../SECURITY_zh.md)和[软件引用信息](../../CITATION.cff)。引用日期表示本地版本快照，不声称存在 DOI、学校归属或未经核实的作者信息。

## 公开发布边界

发布前检查最终受版本控制的文件、源码与证据对应关系、文档链接和敏感信息扫描。仅发布经过审查的源码、测试、文档、配置和小型结果文件。排除虚拟环境、下载的权重、缓存、私人机器日志与未审查的本地工作文件。

创建公开仓库、推送提交、创建 release tag、上传 wheel 或源码归档，以及部署服务，属于各自独立的外部操作。本发布准备记录中，这些操作尚未执行。真正执行后，应将状态更新为核实后的结果和精确公开链接，不预先填入假定的仓库地址或通过的 CI badge。
