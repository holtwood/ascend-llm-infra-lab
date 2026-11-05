# 进度快照（持续更新）

最后更新：2026-11-05

## 已完成

- [x] 4 机盘点与角色分配（见 environment.md）
- [x] 修复 vllm editable import bug（PYTHONPATH 遮蔽）→ scripts/env.sh
- [x] M3 起 `vllm serve Qwen3.5-4B`（:8000, OpenAI API 验证通过, 推理模型带 thinking）
- [x] 4 机免密 SSH（M3 的 ed25519 公钥已推到全部节点 authorized_keys）
- [x] 方案仓库建仓 + docs + scripts

## 进行中

- [ ] E1 基线压测（M3, Qwen3.5-4B, 并发 1→64 扫）— `results/e1/`
- [ ] D→M3 模型搬运：Qwen3-0.6B ✓ / Qwen2.5-VL-7B / DeepSeek-R1-Distill-32B-W8A8 → `/data/models/`
- [ ] M3→M2 模型拷贝：Qwen3.5-4B（副本 3 预备）

## 待做

- [ ] E2 量化对比（M1：27B w8a8 / W8A16 / bf16 参照）
- [ ] E3 特性消融（prefix caching / chunked prefill / spec decode）
- [ ] E4 多副本 + LB 扩展性
- [ ] E5 MindIE vs vLLM-Ascend
- [ ] E6 Ascend C 算子（进阶）

## 环境备忘

- 服务端口约定：vLLM :8000 / LB :8080
- vllm serve 启动命令模板见 scripts/serve.sh（待补）
