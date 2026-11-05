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
- E3 spec 完成：sonnet 对照接受率仅 8-11%/长度~1.5 → ngram 全负收益（TPOT 6×恶化）；两数据集结论一致已入库
- E5 侦察：M1 /model/mindie 是部署包（MindIE3.0.0 配 CANN8.5.1），mindie-service daemon 未安装、venv 已坏；路径=装 .run 包 vs 9.2 CANN 兼容性风险，E5 排最后处理
- e4_cluster.sh 重写：副本 M3/M2/D(8010)；D 已建 nginx+双隧道+免密；D 模型拷贝中
- self-match 坑三连：pgrep -f 的 pattern 若出现在发起者自身 cmdline 会匹配自己——本地 kill、远端 kill 循环（wrapper 含 pattern 文本）、隧道清理（wrapper 含隧道命令）三种形态；修法分别=按PID杀/模式加锚点^ssh/或bin[/]vllm 正则断开字面量
- E2 定论：27B bf16(71G) 塞不进 64G 单卡；W8A16 模型被 vllm_ascend compressed-tensors 拒绝(NotImplementedError)；可跑变体=w8a8+Qwen3-8B bf16 参照
- E5 定论：MindIE3.0.0 全部二进制需 glibc2.38(Ubuntu24.04)，4 节点均 22.04/2.35 → 物理不可行；8 步排障链已录 results/e5/README.md（含 py311/apt 绕行、Python API 尝试、_libatb_torch.so 同墙）
- E4 第二轮发现双 driver 并发互踩（各自重建隧道+nginx upstream 互相打架）→ 全杀重开单实例；教训：后台 driver 必须先验证无同类进程再起
- E4 完成：干净轮 c128 吞吐 1027→1674→2160 tok/s（效率1.63×/2.10×），c32 N=3无增益（未饱和）已写 results/e4/README.md
- nginx daemon 挂 ssh：远端起 daemon 必须 setsid+重定向三个 fd，否则会话不返回

## 第二轮（E3b/E4b/E6/E7/issues）状态与踩坑

### 已完成

- **E3b chunked-prefill 消融**（commit b82828e）：mnb 256~8192 六档扫描。
  长 prefill 臂 TTFT 13.1s→1.06s 单调降，mnb≈2×prompt_len 进平台；
  TPOT 非单调（512 档最差 141.8ms——chunk 大但仍需多步时 decode 被持续挤压）。
  对照臂方向反转（小 mnb 保护 decode，TPOT 25.7 vs 32.6）
- **E4b 四副本**（f54a3da/1a58f9f）：M1 加入 upstream（D:8003 隧道）。
  c128 吞吐平台 ~2061（N=3 2160 平），但 TTFT/TPOT 改善——瓶颈迁到
  负载注入层（D 单机上 bench+nginx+隧道）；c32 增益反而最大 +23%
- **E6 精度评测**（进行中→见 results/e6/README.md）：
  QA 臂完成 w8a8-27b 94.3%/96.2%(MCQ score)、bf16-8b 84.8%/92.5%、bf16-4b 94.3%；
  PPL 臂 w8a8 nll 1.609/ppl 5.00，bf16-27b CPU 参照 ~13min/chunk 慢跑中，
  前 4 chunk Δnll∈±0.13 方向不一（实现噪声≥量化效应）
- **E7 PD 分离**（dbbc36d）：单卡共置 1P1D（各 0.35 显存）链路打通，
  ~50% 请求 hixl CS 握手 10s 超时 → TTFT 灾难 22-87s；成功请求
  TPOT 95ms < mono 108ms（decode 隔离收益 ~12% 存在但被淹没）
- **issue 草稿 ×3**（a9bcc0f）：W8A16 compressed-tensors 拒绝 /
  cudagraph 207005 SQ-CQ 耗尽 / causal_conv1d_update fallback
- clef-flash joint_head 判定**非 spec-decode 资产**（Cloudflare 决策模型
  schema head，非 next-token 草稿头）；EAGLE 路线作废。另发现
  Qwen3.8-27B-W8A16 目录含 model_mtp.safetensors（真 MTP 头，但模型本体
  暂不可加载）——备查

### 第二轮踩坑（新增）

- `max_num_batched_tokens` 合法性：`< max_model_len` 只在 chunked-prefill
  开启时合法；且必须 `≥ max_num_seqs`（默认 128）
- PD 共置两实例 `gpu_memory_utilization=0.35` 时装不下 max_model_len 8192
  的 KV（可用 0.29G < 需 0.31G）→ 降到 4096 过；D 侧单独再降一档
- e4_cluster.sh step1 的 ssh 远端 setsid 起隧道后**会话 channel 仍不释放**
  → driver 卡死；杀该 ssh PID 解决（远端 daemon 化≠会话必释放）
- 思考型模型（Qwen3.8-27B 吐 `<think>`）QA 评测 `max_tokens=8` 全截在
  think 里 → 76.2% 伪低分；需 ≥768 + 去 think 块再匹配
- `prompt_logprobs` 做选项打分：必须按字母 token id 查**该 token 自己**的
  logprob；取 max(topk) 会让 4 变体得分趋同 → 曾测出 8.8% 伪随机
- CPU 跑 qwen3_5 transformers 模型：即便 TORCH_DEVICE_BACKEND_AUTOLOAD=0
  也会链式 import torch_npu → 缺 libhccl.so/libascend_hal.so；
  必须 source env.sh 拿全 CANN LD_LIBRARY_PATH
- Qwen3.8-27B CPU forward ~13min/100tok chunk：hybrid 架构 conv/linear-attn
  算子走 torch 串行 fallback（"The fast path is not available" 警告）——
  CPU 评测只适合 NLL 小语料，生成式 QA 不可行
- M3 git remote 名为 gitee 且无 PAT → 同步凭据后 pull 成功
- D→M1 直连最初 Permission denied：补推 D 公钥到 M1 authorized_keys

### 在飞

- M1 `e6_ppl_cpu.py`（PID 25027）bf16-27B CPU NLL，~13min/chunk × 27，
  增量写盘 results/e6/ppl_bf16-27b.json —— 取部分 chunk 即可配对
- M1 `serve_e4b_m1.log` qwen35-4b:8000 在跑（E4 集群第 4 副本角色）
- spec-decode 两条可选路线均否决：clef-flash joint_head 是 schema 决策头
  非 next-token 草稿头；Qwen3-0.6B draft 与 qwen35-4b 词表不一致
  (151643 vs 248044)——draft_model 方法要求同 vocab

### 第三轮踩坑（E8/S级连跑）

- M3 `results/` 下未提交文件曾被清（疑外部 `git clean`/reset；lab 目录本身
  是 git 仓库停在旧 commit）：pressure_c64/c384 JSON + 脚本一度消失——
  **结果要随测随拷回本地仓库**，远端只当过夜缓存
- 同卡双 vLLM 实例：默认 0.92 util 下第二个 4B 实例引擎初始化直接失败
  （权重 8.6G 放不进剩余 ~10%）→ E8 复用现成副本，需小 KV 池时重启降 util
- python3 环境 torch 自动加载 torch_npu → 缺 libhccl.so：
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 可纯 CPU 跑 torch 脚本
- Qwen3.5-4B 顶层 AutoConfig 全 None：架构字段嵌在 `text_config`；
  且是 hybrid 模型——KV 按 8×full_attention 层算不是 32 层
- hybrid 统一分页：block 128→attention block 640 token，mamba 页 +22.14%
  填充对齐 → 有效 48.7KB/token（>纯 attention 理论 32.8KB）
- **容量边界 ≠ 首发瓶颈**：0.92 池 c384 峰值 running=142/KV 73%/0 抢占
  ——chunked-prefill 准入速率让 running 稳态停在容量界下；要打满 KV
  得缩池（util→0.5）或关 chunked-prefill
- KV 打满（99.1%）→ `num_preemptions_total` 计数真实抢占（recompute），
  请求不失败，代价在 p99 尾部（+28%）

### 第四轮踩坑（A级：E10/E11/E13）

- `/sleep` 404：sleep/wake/is_sleeping 路由在 `serve/dev/` 下，需
  `VLLM_SERVER_DEV_MODE=1`；`--enable-sleep-mode` 只管模型侧
- wake_up 500：`FRACTAL_NZ mode is enabled`——Ascend 权重 NZ 布局
  无法回灌；`--additional-config '{"rl_config":{"enabled":true}}'` 自动
  关 NZ 后正常。sleep2 释放 31.2G→1.0G（97%），wake 仅 84ms
- vllm bench serve 需要 `--tokenizer <本地路径>`（model 名是 served 名
  不是 HF id）；且 client 侧同样要 source env.sh（torch_npu 链）
- Ascend 默认 `async_scheduling` 开启（BatchJobAware 自动选 Async 变体
  佐证）；`--no-async-scheduling` 关闭做消融 → decode 重负载 TPOT
  +65%（48.4 vs 29.4ms），吞吐 −38%
- batch_job_aware 经 additional-config JSON 启用；同构负载无收益
  （LPT/EWMA 无素材），混合 job 负载两类均 ~6% 改善
- npu-smi 显存看「Process memory(MB)」表，芯片行是温度/利用率
