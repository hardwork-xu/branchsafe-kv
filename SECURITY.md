# Security and operating limits

[简体中文](SECURITY_zh.md)

I maintain version 0.1.0 as a local research library. It has no hosted service, authentication layer, tenant isolation guarantee, or published security response SLA. The repository has not yet been publicly released at the preparation date; no private vulnerability-reporting endpoint is claimed.

## Trust boundaries

- Treat input arrays, NPZ files, paths, and model artifacts as trusted local inputs. The NPZ CLI disables pickle, limits declared uncompressed input size to 256 MiB, rejects identical input/output paths, and caps configured pool payload at 2 GiB. These checks are not a sandbox for hostile archives or operating-system resource isolation.
- `max_pages` bounds allocator-owned page payload. It does not bound input arrays, Python metadata, temporary materializations, NumPy internals, model memory, or total process RSS. Choose limits with those additional allocations in mind.
- Callers must not mutate input arrays concurrently with append. The cache lock and epoch checks protect its own state; they do not establish safe sharing of arbitrary external arrays or synchronization across processes.
- Closed sequences release ownership, and the pool can reuse freed pages. Data is not securely erased. Do not use the allocator as a secrecy boundary between mutually untrusted tenants.
- The cache library and default demo perform no network requests. Dependency installation can require network access. Optional model validation downloads only the pinned GPT-2 snapshot and loads safetensors. It does not provide a general loader for untrusted model code. Downloaded artifacts and optional dependencies retain their own security and license requirements.

Model validation uses short synthetic prompts. It does not evaluate model truthfulness, bias, memorization, or suitability for deployment. The stored-cache/native-cache checks pass, while the strict float32 full-recomputation error bound has recorded failures; see [the raw model result](results/model.json). Do not turn this limited check into a blanket model-quality guarantee.

## Reporting and fixes

Do not place credentials, personal records, private prompts, or exploitable private-system details in public issues or logs. If a maintainer contact or private reporting channel has been independently verified, send a minimal reproduction through that channel. Until such a channel is established, retain sensitive details locally and share only a sanitized description.

I prioritize reproducible failures affecting prefix isolation, stale-write rejection, capacity accounting, unsafe file loading, and unintended data exposure. A useful report identifies the exact version or source hash, environment, minimal synthetic input, expected behavior, and observed result. Do not include model weights or unrestricted memory dumps. Fixes need a regression test, affected-path verification, and aligned bilingual documentation.

This policy does not establish a warranty or production support commitment. The [MIT license](LICENSE) governs the software.
