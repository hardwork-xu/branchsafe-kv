# 第三方来源与归属

[English](NOTICE.md)

BranchSafe KV 在仓库中编写的实现采用 [MIT 许可证](LICENSE)。以下依赖、模型文件与既有方法仍归属于各自作者；本说明不改变其许可。

## 方法

分页、引用计数、写时复制与乐观版本检查均为已有系统技术。Woosuk Kwon 等人于 2023 年发表的 [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) 是分页 KV 存储与共享的直接概念参考。BranchSafe KV 独立实现 CPU NumPy 页池及事务检查，不包含 vLLM attention kernel，也不宣称发明了分页注意力。实测比较对象是仓库内的稠密基线与完整复制分页消融方案，不是 vLLM 服务系统。

## 软件与模型文件

| 组件 | 版本或 revision | 用途 | 上游许可 |
|---|---|---|---|
| NumPy | 2.3.3 | 运行时数组、复制及数值检查 | [BSD-3-Clause](https://github.com/numpy/numpy/blob/v2.3.3/LICENSE.txt) |
| PyTorch | 2.8.0 | 可选 GPT-2 CPU 前向计算 | [BSD-3-Clause 及上游声明](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE) |
| Transformers | 4.56.2 | 可选 GPT-2 实现、分词器与原生缓存 | [Apache-2.0](https://github.com/huggingface/transformers/blob/v4.56.2/LICENSE) |
| safetensors | 0.6.2 | 可选模型权重加载 | [Apache-2.0](https://github.com/safetensors/safetensors/blob/v0.6.2/LICENSE) |
| GPT-2 | `openai-community/gpt2`，revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 可选预训练模型正确性验证 | [MIT 模型卡](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)、[OpenAI 许可证](https://github.com/openai/gpt-2/blob/master/LICENSE) |

模型权重单独下载到被忽略的缓存中，不随仓库或软件包分发。模型结果记录精确的文件哈希。验证脚本中的提示词是原创合成测试输入，不是已发布的训练或评测语料库。GPT-2 的原始说明为 Radford 等人于 2019 年发表的 [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)。

开发及传递依赖固定在 [uv.lock](uv.lock) 中，各自保留原有许可证。此列表说明项目直接使用的机制与可选模型路径，不是已安装二进制分发包的完整物料清单。若重新分发依赖二进制文件或构建后的容器，需要保留分发包提供的许可证和归属文件，包括捆绑的第三方组件。Python wheel 不内嵌这些依赖的实现。

## 引用

本软件的引用信息见 [CITATION.cff](CITATION.cff)；讨论已有方法时应另外引用原始来源。软件元数据采用已核实的公开维护者用户名 `hardwork-xu`，不声称真实姓名、学校归属、论文发表或 DOI。版本日期描述本地 0.1.0 软件快照，不是公开发布的证明。
