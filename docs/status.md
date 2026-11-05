# 进度快照（持续更新）

最后更新：2026-11-05

## 已完成

- [x] 4 机盘点与角色分配（见 environment.md）
- [x] 修复 vllm editable import bug（PYTHONPATH 遮蔽）→ scripts/env.sh
- [x] M3 起 `vllm serve Qwen3.5-4B`（:8000, OpenAI API 验证通过, 推理模型带 thinking）
- [x] 4 机免密 SSH（M3 的 ed25519 公钥已推到全部节点 authorized_keys）
- [x] 方案仓库建仓 + docs + scripts

## 已完成（续）

- [x] **E1 基线压测完成**：6 档并发全数据 + 解读 → `results/e1/README.md`
  （1→64 并发输出吞吐 75→807 tok/s 未饱和，TTFT p99@64=3.3s）
- [x] Gitee 仓库 `shijiashuai/ascend-llm-infra-lab` 建仓+推送（PAT）
- [x] D 重建为 vLLM 镜像并接入（**验证 /model 为宿主共享缓存**，73 项模型无需搬运）
- [x] 4 机免密 SSH 全部打通（含新 D）

## 进行中（迁移交接点 05:30）

- [ ] E3 spec-ngram c=8 压测中（random 数据集对 ngram 不公平→改用 sonnet 重测，见 handoff.md）
- [ ] E2 W8A16 sweep（M1，driver 在 M3 跑）
- [x] M2 副本就绪 qwen35-4b:8000；D nginx 装好；e4_cluster.sh 就绪

## 待做（接手指引见 docs/handoff.md）

## 踩坑记录（已修）

- `vllm bench serve` 需 `--tokenizer <本地路径>`（模型名会去 HF 查 → 不通）
- 压测用 `/v1/completions`；指向 `/v1/chat/completions` 会全 400（backend 默认按 completions 格式发包）
- `pkill -f` 匹配模式会杀掉自身 shell——按 PID 杀
- env.sh 必须 source set_env.sh + 显式 LD_LIBRARY_PATH，否则非交互 SSH 下 torch_npu 加载失败
- 量化模型（w8a8）还需 ATB：`LD_LIBRARY_PATH` 加 `/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib`
- `/model` 是**宿主机共享缓存**：同宿主机重建镜像模型不丢；跨宿主机需拷贝

## 待做

- [ ] E2 量化对比（M1：27B w8a8 / W8A16 / bf16 参照）
- [ ] E3 特性消融（prefix caching / chunked prefill / spec decode）
- [ ] E4 多副本 + LB 扩展性
- [ ] E5 MindIE vs vLLM-Ascend
- [ ] E6 Ascend C 算子（进阶）

## 环境备忘

- 服务端口约定：vLLM :8000 / LB :8080
- vllm serve 启动命令模板见 scripts/serve.sh（待补）
- e2_driver 远端 kill 循环 `pgrep -f "vllm serve"` 会匹配远端 wrapper 自身 cmdline（self-match，本次发生于远端 bash）→ 统一改 `bin/vllm serve` 模式
