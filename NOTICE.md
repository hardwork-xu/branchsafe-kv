# Third-party sources and attribution

[简体中文](NOTICE_zh.md)

BranchSafe KV's repository-authored implementation is provided under the [MIT license](LICENSE). The following dependencies, model artifacts, and prior methods remain attributable to their respective authors. This notice does not relicense them.

## Methods

Paging, reference counting, copy-on-write, and optimistic version checks are established systems techniques. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) by Woosuk Kwon and colleagues (2023) is a direct conceptual reference for paged KV storage and sharing. BranchSafe KV independently implements a CPU NumPy store and its transaction checks; it does not incorporate the vLLM attention kernel or claim to invent paged attention. The measured comparison is with this repository's dense baseline and eager-page ablation, not with vLLM serving.

## Software and model artifacts

| Component | Version or revision | Role | Upstream license |
|---|---|---|---|
| NumPy | 2.3.3 | Runtime arrays, copies, and numerical checks | [BSD-3-Clause](https://github.com/numpy/numpy/blob/v2.3.3/LICENSE.txt) |
| PyTorch | 2.8.0 | Optional GPT-2 CPU forward execution | [BSD-3-Clause and upstream notices](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE) |
| Transformers | 4.56.2 | Optional GPT-2 implementation, tokenizer, native cache | [Apache-2.0](https://github.com/huggingface/transformers/blob/v4.56.2/LICENSE) |
| safetensors | 0.6.2 | Optional model-weight loading | [Apache-2.0](https://github.com/safetensors/safetensors/blob/v0.6.2/LICENSE) |
| GPT-2 | `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` | Optional pretrained correctness validation | [MIT model card](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e), [OpenAI license](https://github.com/openai/gpt-2/blob/master/LICENSE) |

Model weights are downloaded separately into an ignored cache and are not redistributed in the repository or package. The model result records exact file hashes. Prompts in the validator are original synthetic test inputs; they are not a released training or evaluation corpus. GPT-2's original model description is [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford and colleagues, 2019).

Development and transitive packages are pinned in [uv.lock](uv.lock); they retain their own licenses. This list explains the project's direct mechanisms and optional model path, and is not a complete bill of materials for installed binary distributions. If redistributing dependency binaries or a built container, retain the license and notice files supplied by those distributions, including any bundled third-party components. The Python wheel does not vendor these dependency implementations.

## Citation

Use [CITATION.cff](CITATION.cff) for this software and cite the original methods separately when discussing them. The software metadata uses the verified public maintainer handle `hardwork-xu`; it does not assert a real-name identity, university affiliation, publication, or DOI. Its version date describes the local 0.1.0 software snapshot, not proof of a public release.
