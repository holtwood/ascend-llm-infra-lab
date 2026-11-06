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

## 新环境（MindIE/MindSpore/PyTorch 三容器，117.131.156.71 跳板进）

### 已完成（新环境）

- **E8b 手撕微观实验**（commit 4924b9d，pytorch 容器 npu:1，910B1）：
  naive vs 融合 attention @S8192 差 5.32×；decode 有效带宽随 KV 长
  512→32768 从 127→999 GB/s；bs=1 matvec 1475 GB/s≈92% HBM 峰值
  （访存墙实锤），bs=256 降到 752（瓶颈切换）→ results/e8b/
- **E8d HCCL allreduce**（commit ac65661，pytorch 容器 npu:2+3，torchrun×hccl）：
  延迟底 ~160μs（≤2MB 全平坦），渐进带宽 18.5 GB/s（HCCS 非 HBM 通道——
  比 E8b 实测片内 1.5TB/s 低 80×）。推出 TP-2 decode 每步固定 ~11.5ms
  通信税（2 allreduce/层×36层，消息全在延迟平台区）→ 8B 级模型 TP
  结构性不划算，副本才是正解（呼应 E4）→ results/e8d/
- **E8c 手写 spec decode**（commit 3104a85，pytorch 容器 npu:2，Qwen3-0.6B）：
  纯 torch 实现 draft(ngram prompt-lookup)/verify(单前向)+DynamicCache.crop
  KV rewind。重复负载 25% 接受/13% 覆盖 → +8%；开放文本 0% 接受 → -6%
  纯开销——机制级复现 E3 生产负例 → results/e8c/

### 进行中/卡点（新环境）

- **E5R MindIE 复活战（定论：ATB set_weight 栈级挂死）**：glibc2.38+CANN8.5+
  MindIE2.3.0 条件全对，9 步排障链走完，权重加载+KV分配成功进入 warmup。
  卡点定位：inference 线程停在 `init_ascend_weight` 的
  `acl_*_operation.set_weight` 单线程 99.9% 空转。
  隔离证据链：8B/0.6B/缩 warmup/TinyLlama(llama类,不同graph文件) 全部
  同帧挂死 → 非模型特有；kernel_meta 零写入、DEBUG plog 只剩
  1Hz 重复 32B memcpy 轮询（等不到 device ack）→ 判定挂死而非慢编译。
  详见 results/e5r/README.md（含 mindie_config_smallwarmup.json、
  e5r_start_service.sh 复现入口）。
  注意：mindie config 有两处 maxInputTokenLen（ModelDeployConfig 顶层
  + ScheduleConfig）都要 ≤maxPrefillTokens；daemon 不能套 setsid
  （自管进程组冲突报 MIE04E01011C）。
- 新环境容器可见全部 8×davinci；`ASCEND_RT_VISIBLE_DEVICES` 是物理→逻辑
  重映射，torch_npu set_device 用逻辑序号（2,3→0,1）。pytorch 容器无
  torchrun 二进制，用 `python3 -m torch.distributed.run`。

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

### E14 探针结果（本轮）

- Qwen3.5-4B 主权重内嵌 15 个 `mtp.*` 权重（`mtp_num_hidden_layers=1`）；
  `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
  引擎可起，metrics 有 spec_decode 计数器。接受率 65.6%（5470/8343）
  但 c8 decode 负载下 TPOT 36.5 vs 无 spec 15.6ms——草稿串行前向开销
  吃掉接受收益，净负。告警：MTP 草稿未分到独立 KV cache group

### 第五轮踩坑（E16）

- `kv_cache_size_tokens` 随 max_model_len 变化（8192→1.006M，
  81920→1.427M，同 45.6GiB 池）：mamba 页对齐颗粒度不同——容量
  规划必须按实际配置实测，不能跨配置套用
- stream  completions 不回 usage.prompt_tokens（需 stream_options.
  include_usage）；词→token 按 ~1.5 估算
- 63k ctx 每请求 KV ~3.1GB（48.7KB/tok），TPOT 19→62ms 随 ctx 4x 增长
  ——长上下文容量瓶颈从调度准入切回 KV 字节

### D3 prefix-aware 路由（第六轮）

- D 节点无 fastapi/aiohttp → 路由用 stdlib `http.server`+`http.client` 手写；
  前缀指纹 = 前 2048 字符 sha1，亲和表 + inflight+2 降级 least-load
- 两臂必须用不同 seed 生成模板，否则第二臂继承第一臂缓存（已踩，
  补 --seed 参数后重跑）
- 结果：~6k tok 前缀命中率 53.8%→84.0%；~900 tok 42.5%→62.2%；
  客户端 p50 2.51→1.87s 但 p90 持平——排队主导 + 亲和倾斜稀释收益
- least_conn 基线本身有 ~50% 命中（同模板 8 请求散 4 副本，平均仍
  见 2 次），评测基线不能假设 0%
- 并发到达的"第二波才命中"使命中率低于 (n-1)/n 理论上界

### E17 sleep/wake 多租户换租（第六轮续）

- 真实换租链路打通：A 冷启 169.4s → sleep2 1.77s（HBM 4480/65536）
  → B 在腾出空间冷启 114.2s（eager，util0.3）→ B 在线时唤醒 A 0.139s
  → 唤醒后首请求 0.129s 无 ramp；A、B 同卡共存
- **新坑**：睡眠方仍持有 stream/CQ → 共卡引擎 ACL capture 报
  EE1023 too many streams（=issue2 207005 同根因）→ 共卡方须 eager
- FRACTAL_NZ 回灌需 rl_config.enabled；/sleep 需 VLLM_SERVER_DEV_MODE=1

### D1 否决（2026-11-05）

- 目标函数 upstream 已写又删：#5984 实现 → 8e5a61a(#8842) AscendC 替换
  → f0a938956(#14620) 移除；rc2 patch_triton.py try-import 已删符号 =
  版本残留。补函数=重加上游主动移除代码，接受率≈0 → 改记为版本升级建议
- 教训：PR 前查 mainline 历史（git log --all --grep）
- 注：本 lane 独立到达同一结论（issue3 → #18978 comment）

### 本 lane 第七轮（walkthrough + upstream issue + E8e）

- `docs/request-walkthrough.md` 完成：vLLM V1 请求全路径 file:line
  （api_router→serving→renderer tokenize→AsyncLLM→ZMQ→EngineCore
  →scheduler→KV block pool→NPUModelRunner→FIA/paged-attention→
  AscendSampler→detok→SSE），每段挂 E1/E3/E3b/E8b/E11/E14/E16 实测
- upstream 重复检查（GitHub API，未认证可跑）：
  issue1/issue2 无重复→发新 issue；**issue3 与 open #18978 同缺陷**
  （#14620 删符号 + #15127 重新 import）→降级为 comment。
  成品在 `docs/issues/ready/`，实际提交需 GitHub PAT（已停下等批准）
- 坑：gh 二进制下载过慢，改用 api.github.com 未认证 search/issues
  接口完成全部 dedup 检查

### E8e 手撕 attention 变体（第七轮，pytorch 容器 npu:1）

- RoPE：手写 rotate-half vs 复数乘法参考 diff=0、vs HF 0.038（bf16）、
  范数保持 1.9e-6、相对位置性质 8.6e-6
- GQA：expand==repeat_interleave 位级一致、vs 原生 enable_gqa diff=0；
  KV 字节 268→33.6MB/req（8×）
- SWA 最有信息量的是**负结果**：sdpa+显式 band mask 比 is_causal
  慢 26×（S8192 24.2 vs 0.93ms）且 w256≈w1024 零窗口收益——真 SWA
  靠 kernel 跳过窗外 block（FIA sparse_mode=4），mask 永远 O(S²)
- 坑：sdpa attn_mask 在 NPU 上要求 dtype 与 query 一致（float mask +
  bf16 q 直接 RuntimeError）；ssh 管道下 python stdout 块缓冲，
  要 `python3 -u` 或日志文件才能看进度

### E19 调度器画像（第六轮续）

- 4 个 scheduler_config 中 3 个有部署前置：recompute_scheduler/dyntra_lb
  需 kv_role=kv_consumer（PD D 节点，非 PD-mixed）；profiling_chunk 需 PP>1
- 单机唯一可用=short_request_first：short e2e 28.73→26.66s(-7.2%)，
  long +3.3%；按 num_prompt_tokens<=threshold 分道+老化防饿死
- dyntra_lb/recompute 是 PD 架构配套件，非通用优化

### E18 PD 跨节点（第六轮末）

- 节点间仅 ssh 端口互通；P 广播容器 IP 172.17.0.5 → VLLM_HOST_IP=127.0.0.1
  + ssh -L 隧道修通 ZMQ 元数据层；TE-RPC 动态口（16401/16352）再补隧道
- 数据面 device-mem transfer ret:-1——SSH 隧道托不起 NPU DMA 语义；
  **请求仍 200 返回垃圾 token（静默劣化）**——PD 监控必须看输出质量
- 节点间互信：id_nodekey 已放到 M3/M1 供 ssh -L 用

### upstream issue 实发（2026-11-05，holtwood）

- #21112 W8A16 compressed-tensors、#21113 cudagraph SQ/CQ、
  #18978 comment causal_conv1d —— 三条已发，含 AI 辅助披露
- 提交前二次 dedup 抓到 #2265/#4267/#9564/#17491 并已写入正文引用，
  避免被 maintainer 当场指出漏查
### 第七轮（E15 MTP扫参 + E20 控制面）

- MindIE 项已由并行 lane E5R 封顶（ATB set_weight hang=vendor 级，
  1Hz 32B 轮询等 device ack，零 compile 产出）——不再重复
- E15：MTP k=1/2 补测，接受率 88.7%/75.8% 但 TPOT 27.6/36.1ms 全负
  ——k=1 也亏，草稿头固定前向成本是常量项；单调恶化跳 k=5
- E20 控制面（stdlib，D:8092）：前缀亲和+SRF分道+per-class准入cap
  ——short p90 4.85→3.10s(-36%)，44/64 快速429（<1ms），wall 3.7 vs 8.9s
- issue 核实：#18978 是第三方 iwantknow1 在 10-16 独立提交的同缺陷，
  dedup 正确非误判；issue1/2 上游无重复
  （本 lane 已于 2026-11-05 用 holtwood 账号提交：
  #21112 / #21113 / #18978 comment）

### sparse_kv_offload 探针（第七轮续）

- `Sparse KV offload only support sparse attention model` —— SFA/DeepSeek
  类专属，Qwen3.5-4B 不适用（同 E19 规律：Ascend 特性多有模型/部署前置）
- 收尾文档：docs/war-stories.md（10 个 STAR 故事）+ docs/demo-runbook.md

### issue 提交核实（第七轮，2026-11-05）

- 全部三份草稿已由 holtwood 提交上游：#21112(w8a16) #21113(207005) +
  #18978 comment(issue3)；与 docs/issues/ready/ 逐字一致
- 此前"需 PAT 才能提交"判断为乌龙——用户已自行提交
- issue1 源码核验：_detect_quant_type 行号 326-382 与草稿一致，属实
### E8f continuous batching 模拟器（第八轮）

- 纯 Python 复刻 V1 调度循环（budget/admission/chunked prefill/队尾抢占/
  APC 命中），负载同 E3/E16 形状
- 最值钱格子：mnb=256 时 APC 使 TTFT 228.6→59.7（-74%）、吞吐
  13.2→23.8（+80%）——APC 收益本质是放大 token 预算而非只省 TTFT
- mnb=256 无 APC 比 static 还差（队头阻塞= E3b 复现）；KV 池=160 块
  触发 34341 次抢占，吞吐腰斩（capacity 规划反面教材）
- 坑：v1 实现让未完成 prefill 的请求参与 decode → 永久卡死（真实
  vLLM 也不会让 computed<in_len 的 req 产 token）；APC 命中请求 ttft
  未置位 → 负值 bug 已修

### E8g 手写 FlashAttention（第八轮续）

- online-softmax 分块实现 vs sdpa 对拍：S512-2048 max diff ~1e-2
  （bf16 舍入），非整除边界 777/256 也过——算法正确
- 诚实负结果：手写分块与 naive 同速（S8192 ~21ms vs 融合 0.94ms，
  24×）——Python 够不到 SRAM 驻留，FA 加速是 kernel 层的事；
  显存账 S²→S·Bc（S8192: 268→16.8MB，16×）才是真收益
- interview-notes 补齐手撕系列 + walkthrough + 上游 issue 条目
### E21/E22 四机利用（第七轮末）

- cluster_bench.py 编排层：bench-all/fanout/util/collect 四模式
- E21 四节点同刻并行：聚合 3119 tok/s、墙钟 9.2s(vs 串行~33s)；
  同配置节点差异 11%（743-825 tok/s）→ 跨节点 A/B 误差棒
- E22 四客户端注入 4 副本：c128 聚合 3744(+82% vs E4b 2060)、
  c256 聚合 4565(+122%)——证实 E4b 瓶颈在注入层；
  TPOT 21→35ms 劣化 → goodput 拐点
- 坑：节点只能 SSH 到 D，客户端注入靠 -L 隧道；pkill -f 'L 8080'
  会匹配外层 shell 自杀；npu-smi 首行是 Power(W) 非 AICore

### E23 客户端扫描 + 测量伪影（第七轮末）

- 客户端 CPU 全程 <1 核（31-37% of one core）→ 注入不是 CPU 瓶颈
- 同集群同负载：vllm bench serve 2193 vs bench_openai 3632 tok/s（1.66×）
  → **E4b 的 2100 天花板是工具伪影**；4 副本真实上限 ~4.2-5.0k
- c128 加进程 +20%（3632→4371），c256 单进程最快（5002）——副本侧饱和
- E4b/E22 README 已加修正头；"工具即瓶颈"成为独立方法论故事

### E24 MTP 大 batch（第七轮末）

- c32: MTP 563 vs 无spec 1256.8 tok/s；c64: 846 vs 1533.6——0.45-0.55×
- 接受率 88.3% 全程不变；草稿头成本随 batch 同比例增长→无摊薄拐点
- 产出 docs/drill-cards.md：10 张追问链卡片（30s 回答+数据+L1-L3+诚实缺口）


### E25 唤醒后稳态（第七轮末·对端 lane）

- warm 696.6 → post-wake 694.9 tok/s（TPOT 18.9→19.3ms）——无稳态损失
- sleep2 1.74s / wake 0.125s；drill-cards 卡5 缺口已补

## 本 lane · 硬件窗口期增量（E8h/E8i/E-W8A16）

- **E8i HCCL 全矩阵**（远端 8×910B1）：allreduce 延迟底 ~146-189μs（≤2MB 平坦）、
  bus 带宽 w2/w4/w8 = 37/137/520 GB/s、p2p 单链路 ~19.9 GB/s 恒定。修正 E8d 结论：
  TP 通信税随卡数**不减**（延迟底不是 hop 线性），decode 消息全在延迟平台区。
  坑：mp.spawn 子进程崩→父进程死等 q.get()，HCCL 程序要加超时兜底
- **E8h 可观测性**：被动抓活引擎 /metrics 时序，拿到 c=64 burst 的
  running/waiting/kv%(峰值11%)/preempt(0) 全轨迹 + 97 指标清单→监控语义表。
  `prefix_cache_hits=0` 反推压测方用 random 数据集
- **E-W8A16（#21112 落地）**：单元级精确复现 detector 穿透 NotImplementedError；
  patch（镜像上游 _is_wNa16_group_channel 口径加 _is_w8a16）→
  AscendW8A16LinearMethod；回归确认 w8a8-dynamic/w4a16 路径不变。
  真权重验证：make_w8a16_ckpt.py 建 0.6B int8 weight-only checkpoint，
  serve 权重加载成功（0.73G int8），NPU 实测 npu_weight_quant_batchmatmul
  vs dequant ref rel diff 0.13%；HTTP serve 被共卡 HBM 争用挡（warmup OOM）。
  验证证据+patch 已 comment 到 #21112（comment-6660784746）
- 拓扑确认：本容器=腾讯云 lane davinci7（910B2×1, CANN9.2, vllm 可编辑）；
  远端 npt=另一台 8×910B1（CANN9.0, torch only, 无 vllm）
- 坑：8001 是对端 SSH 隧道不是本机口；裸 /vllm-workspace PYTHONPATH 致
  import vllm 变 namespace 包；合并冲突脚本丢段——status.md 追加要用
  明确锚点不要用尾部盲并

## W8A16 修复 PR 化（baedd8b）

- vllm-ascend 分支 `fix/w8a16-compressed-tensors-detector`，单个签名提交
  `baedd8b`（conventional commit + sign-off，作者=Devin 待用户 PR 时改写）
- 单测补齐三态：channel/group 识别 + dynamic 拒绝 + activation 在场拒绝，
  `test_compressed_tensors_config.py` 18/18 全过
- lint 口径：vllm-ascend line-length=120（非 vllm 的 88），长签名合规
- 注意：该 checkout 是 detached HEAD @v0.30.0rc2、多 lane 共享工作树——
  分支提交不动文件内容，对其他 lane 无影响
- PR 前置缺口：fork 推送权限 + 真人 review/提交 + HTTP serve 冒烟（等卡）

## 通宵任务套件（scripts/overnight/，设计见 docs/overnight-runbook.md）

- 一键：`deploy.sh`（推送+启动）/ `observe.sh`（进度）/ `observe.sh --fetch`（回收）
- 队列：env_probe→finish_install(vllm)-->hccl_pairs(8×8拓扑矩阵)-->kv_quant(S×B网格)
  -->paged_gather(块粒度)-->copy_bw(含计算并发衰减)-->serve_smoke(#21112终验)
  -->tp_scaling(首测TP1/2/4)-->hccl_soak(填余量~8.5h)
- 远端坑已内建：CANN env 三个 set_env.sh 必须 source；.bashrc 默认
  ASCEND_RT_VISIBLE_DEVICES=6 只露一卡→master 强制 0-7；vllm wheel 硬钉
  torch==2.13.0 会破坏 torch_npu 2.10.0→--no-deps+剔 torch* 依赖清单；
  SSH 掉线→全部 nohup+落盘；任务级 timeout+孤儿回收（修 e8i 教训）
- warm 696.6 → post-wake 694.9 tok/s（TPOT 18.9→19.3ms）——无稳态损失
- sleep2 1.74s / wake 0.125s；drill-cards 卡5 缺口已补

### E26/E27/E28 四机矩阵（第八轮）

- E26 同刻四臂：default 800.6 / SRF 665.3 / async-off 499.5 / MTP 193.5
- E27 KV 池：跨节点单调趋势(-11%)被同节点对照推翻(u92 +7%)——第二次抓到伪结论
- E28 长上下文并行：4k 178→63k 41 tok/s，TTFT 527ms→4.59s，TPOT 仅 1.4×
  ——与 E16 串行互证
- 坑：ssh 多层传 JSON 引号必被吃→改参数文件；M2/D 模型路径是 /data/models

### 通宵方案（第八轮末）

- docs/night-shift-plan.md：四机 8-9h 满载排程（M1 soak 8.2h / M2 网格+混合
  soak 7.8h / M3 ctx+换租疲劳 8.0h / D 量化矩阵+soak 8.3h）
- 脚本 night_{common,m1_soak,m2_grid,m3_ctx,d_quant}.sh + night_collect.py
- 坑：pkill -f night_ 自匹配杀 ssh 会话（文档命令已用 night_[m]1 括号技巧）；
  VAR=x func 的环境变量会残留在 shell（restore_default 已强制 MAXLEN=8192）

### 通宵方案 v2（集群段 + 独立段）

- 新增 night_cluster_phase.sh：四机同为客户端打 D:8080（c32/机，聚合 c128），
  前 2.5h 四卡协同；冒烟实测单客户端 c32 = 1735 tok/s、256 请求零错误
- night_all.sh 包装：集群段 → 本机独立线；总 8.5-9h
- 时长重配：M1 300 波 / M2 550 轮 / M3 700 波 / D 420 波
- night_collect.py 增集群段统计（单机均值/漂移，聚合≈4×）

### 通宵方案 v3：实测标定 + 时间上界（第八轮末）

- 标定实测：集群波 18-22s、M1 59s、D 40s、M2 最慢网格点 17s、M3 16k 波 11s、63k 点 6s
- 四条线改为**时间上界循环**（CLUSTER_MIN=150 + LINE_MIN=360 = 510min=8.5h 确定性）
- 预计 ~2900 个 bench 点；goal 指令加"每小时 collect+commit+push"防丢数据

### 远端通宵队列 batch 1（run_20261105_160244）

- 4 个 torch 级任务全 rc=0：p2p 全对矩阵 / int8-KV 网格 / paged gather /
  copy bw；产物在 results/overnight_20261105_160244/
- p2p 拓扑：64MB 全对 19.7GB/s **完全对称无慢链路**（注意：19.9GB/s 只是
  ≥16MB 的饱和值，1MB 时仅 6.0-7.6GB/s）；1MB 有 ±20% 路径差
  （1→0 7.6 vs 2→1 5.6）；1KB 延迟底 ~150µs
- int8-KV：mem 恒 0.51×，err_max 随 S 收敛（0.0117@128 → 0.0007@16k），
  naive 税 1.5-3.0×（与 E8k 的 6.5-8.5× 差异是实现口径，已在 README 注明）
- paged gather：S=4096 时块越大越差（1.58×→2.73×），S≥8192 稳定 ~2.1×
- copy bw：pageable D2H 4-5× 惩罚；**D2H 在并发 matmul 下 24.7→19.7GB/s(-20%)**
- 队列教训：`SimpleQueue.get(timeout=)` 在 py3.12 **不支持 timeout 参数**
  （TypeError），被我误标为 "timeout"——rank 各自落盘是更稳的写法（此前
  关于 "all_gather_object 在 HCCL 不可用" 的判断**未经证实，已撤回**）；
  torch_npu SDPA 拒绝 enable_gqa→用 batch 广播实现 GQA；非登录 ssh 不 source CANN env
### 通宵执行（进行中）

- 16:13-16:15 四条线启动（setsid nohup），全部进入集群段
- 集群段实测：单客户端 c32 ≈ 913-926 tok/s，聚合 ≈3.7k tok/s，256/256 零错误
- 排程：集群段 150min（~18:45 结束）→ 独立段 360min（~00:45 结束）→ 自动恢复
- 16:21 UTC 利用率实证：功率 87W(空载)→96/242/192/206W，AICore 41-49%（M2/M3/D），
  HBM 60.3GB/卡；集群段聚合 ~3.7k tok/s
- 17:15 UTC 巡检：四线存活（M1 153 波/M2 136/M3 135/D 134），集群段稳态单机 ~1.05-1.09k tok/s
- 18:10 UTC 巡检：四线存活（M1 277/M2 260/M3 259/D 258 波），集群段 116/150min
- 18:38-18:46 UTC 事故：M1 soak 脚本残留 `$WAVES` 引用 + `set -u` → 启动即退；
  修复（改 log 行）后重启该线，JSONL 保留；其余三线正常进入 phase A
- 集群段收官：M1 340 / M2 336 / M3 336 / D 336 波，零错误
- 18:49 UTC D 的 27B-w8a8 臂启动失败：EE1023 "Too many streams are created"
  （graph capture 流耗尽，非显存；与 issue2 同族）→ 该臂跳过，D 线继续跑 8B/4B/0.6B；
  收尾后可用 --enforce-eager 补测
- 19:00 UTC 进度：M1 soak 8 波 / M2 已到 mnb 档 / M3 疲劳 150/300 / D 量化 8B 臂
- 19:19 UTC M2 phase D 异常修复：e13_mixedjob.py 未同步到 M2（仅 M3 有）→ 6 分钟
  空转 7500 轮；已同步脚本 + 加 SKIP_ABC 开关，只重跑 phase D（LINE_MIN=320）

### 通宵值守总结（远端 npt · 2026-11-05 15:47 → 23:38）

跑完 8.5h 预算内全部可执行项，产物在 `results/overnight_20261105_160244/`：

| 任务 | 结果 |
|---|---|
| hccl_pairs 8×8 全对矩阵 | ✅ 64MB 全对 19.7GB/s 对称；1MB ±20% 路径差；1KB 延迟底 150µs |
| kv_quant S×B 网格 | ✅ 32 配置，mem 0.51×，err 随 S 收敛 0.0117→0.0007，税 1.5-3.0×（bf16 中间态/GQA）；**2×2 对照实测**：8 头 1.4-1.9× vs 32 头 3.2-6.5×，E8k 的 6.5× 精确复现（6.46×） |
| paged_gather 块粒度 | ✅ 32 配置（含页序对照）+ 布局/头数/块长三因子对照：**页序与布局都不是主因**（布局仅 ±10-25%），主因是头数（8→32 头：2.1→3.2-4.1×）与块长（bs256：15.85×，复现 E8j） |
| copy_bw | ✅ pageable D2H 4-5× 惩罚；**并发计算下 D2H −20%** |
| TP decode 实测 | ✅ 真权重：tp=8 步时 **3.64×** 于单卡（修正建模：每层 2 次 allreduce），通信占 67%；tp=4 与 tp=2 持平 |
| HCCL 4h soak | ✅ 14330 样本，漂移 **+2.3%**，无热降频 |
| W8A16 serve 冒烟 | ⚠️ 到 `/v1/models` 200 + 权重加载成功，死在 `aclnnScatterPaKvCache`（CANN 9.2 算子，远端 9.0 没有） |
| tp_scaling serve | ⚠️ 同上，vllm serve 在 CANN 9.0 不可用（已用 torch 级 TP 模拟替代） |

**关键工程结论**：vllm-ascend 0.30 的 paged attention 依赖 CANN 9.2 的
`aclnnScatterPaKvCache`，9.0 环境无法服务——解释厂商为何在 9.0 容器配
MindIE。踩坑全链路见 `docs/remote-vllm-bringup.md`（torch 钉版、三个
CUDA 生态 stub、triton-ascend 华为镜像、CANN 版本 .so 不可跨机）。

**队列工程教训**（已修入脚本）：`SimpleQueue.get()` 在 py3.12 不支持
timeout 参数（曾把 TypeError 误标成 timeout；"all_gather_object 在 HCCL
不可用"的旧判断未经证实已撤回）；HCCL socket 端口需每组独立区间；
pkill 未生效会污染后续测量（TP 首轮即因此重跑）。
- 20:20 UTC 巡检：M1 soak 92 波 / M2 混合 200 轮(47/320min) / M3 长上下文 424 波 / D 100 波(73/360min)
- 21:13 UTC 巡检：M1 146 波 / M2 混合 ~485 轮（94+/320min）/ M3 723 波 / D 172 波
- M3 换租疲劳 300 轮完成：sleep 1757.8ms（首末 1762.4→1764.5）、wake 118.1ms
  （114.8→119.2）、首请求 124ms —— 零漂移
- 21:50 UTC 关键发现：D 量化矩阵中 **8B(1301/1699) 在 c32/c64 快于 4B(1122/1313)**；
  推测机制=4B hybrid 的 24 层 GDN conv1d 走 PyTorch fallback（每请求 host 同步，
  与 issue3/D1 同源），高并发下放大；0.6B 2696 tok/s 为上限参照
- 22:42 UTC 巡检：M1 237 波 / M2 混合 800 轮(188/320min) / M3 1229 波 / D 296 波，零错误
- 23:37 UTC 巡检：M1 293 波 / M2 1100 轮(258/320) / M3 1550 波(272/360) / D 370 波
- 00:14 UTC 最后一次巡检（goal 清除时）：M1 330 波 / M2 1200 轮(282/320min) /
  M3 1760 波 / D 423 波；四副本+nginx 全 200；各线将在 00:39-00:56 自行结束并恢复

### E29 decode 天花板 + MindIE 二次失败（面向 serving 面试）

- **E29（decode_ceiling.py）**：真 0.6B 权重，910B1：
  - 一次大 matvec 读全部 1.5GB → 1412 GB/s（HBM 94%）= **939 tok/s 天花板**
  - 逐层 eager decode → **30-34 tok/s（天花板的 3.6%）**：卡在 kernel 发射
    （~420 次 launch × ~70µs），不是 HBM 也不是算力
  - bs 1→64 步时几乎不变 → tok/s 线性涨到 2204（权重读一次服务 bs 个 token）
  - 权重-only int8：分块实测 1.21×（理论 2×，同样被发射开销吃掉）
  - **这组数字直接回答"为什么需要 ACLGraph/图捕获"和"decode 的天花板在哪"**
- **MindIE 2.3.0 + Qwen3-8B**：修好配置校验（ModelConfig 必须是数组、
  maxPrefillTokens≥maxInputTokenLen）后仍 core dump，dmesg 见
  `stranded cqe`/`hdcdrv` 驱动级故障——与 E5R 的 ATB set_weight 挂死是
  两种独立的 vendor 栈失败。MindIE 对比实验在此环境不可行。
### 通宵实验收官（2026-11-06 01:09 UTC）

- 四条线全部完成并自动恢复副本：M1 367 波 / M2 网格 30 点+mnb+输出长度+混合 1300 轮 /
  M3 ctx 16 点+300 轮疲劳+准入墙+2052 波 / D 量化 9 点+501 波
- 集群段：聚合 4386.8 tok/s（≈4×1096.7，稳态已剔除错误波），单机 1076-1104；d/m3 各有 24/9 个错误集中在 clu_w0335-336（**收尾窗口**：先到点节点重启副本，其余客户端打到空后端——非"启动窗口"，已更正）
- M1 soak 367 波：847.5 → 846.3 tok/s（−0.1%），零错误
- 四副本+nginx 全 200；报告见 results/night/README.md（含 serving 面试用法章节）

### E30 引擎源码级优化（2026-11-06 10:xx 北京）

- 新增 patches/e30_capture_budget/：plan_capture_sizes + env 门控钩子（默认 inert）
- 四卡并行矩阵：default 35 尺寸/181s → b10 9/163s → b4 4/154s → b2 2/151s（b2 塌陷 c8 −37%）
- 同节点背靠背（M1）：default 180s/1218/1866 → **b10 162s(−10%)/1227(+0.7%)/1898(+1.7%)**；
  b4 掉 5-9% → 甜点=2 的幂 9 档；机制=未捕获 batch 退 eager
- 已部署四台（platform.py.e30bak 备份，可 --revert）；PR-ready 未提交上游

### Serving 物理批次（E29-E48，面向 AI Infra serving 岗）

全部在远端 910B1 + Qwen3-0.6B 真权重上实测，产物在
`results/overnight_20261105_160244/`，汇总见 `docs/serving-physics.md`：

| 实验 | 核心数字 |
|---|---|
| E29 decode 天花板 | 裸 matvec 1412 GB/s（HBM 94%）= 939 tok/s；eager 逐层只 30-34 tok/s（3.6%），差 31× 全是发射开销 |
| E30 KV 容量 | kv/token=114688B；真分配验证 2k/8k/32k → 271/68/17 路并发 |
| E31 prefill vs decode | prefill 11.8µs/token vs decode 29400µs/token，**差 250×** |
| E32 抢占经济学 | swap（线性）vs recompute（含二次项）；交叉点 ~4k，32k 时 swap 便宜 2.27× |
| E33 APC 收益 | 短 prefix **负收益**（0.83-0.85×），8k 时 2.76×；解释 vLLM 上"中位数变差 p99 改善" |
| E34 chunked prefill | chunk=2048 甜点：+13% 总时长换 3.5× 干扰降低；地板 = 一个 decode step |
| E37 长上下文 attention | 1k 起就超权重读（1.9×），32k 13×，128k 53×；KV 读仅 19% HBM 峰值 |
| E39 长上下文批处理 | 每序列成本降 2.7-3.5×，但 KV 带宽 bs≈16 饱和于 ~47% |
| E41 W8A16 | 速度 **负收益**（0.8-0.99×）；容量省 29%；精度 0.9-1.6% |
| E44 P/D 干扰 | 8k chunk 使并发 decode 膨胀 **16-34×**；双流不改善→真争用→同卡 PD 无效（互证 E18） |
| E45 投机解码验证 | q=1→512 只涨 1.30×（机制上验证几乎免费） |
| E48 prefill 成本模型 | 二次项 4.66e-8 ms/tok²；16k 时 attention 占 prefill 86% |
| E51 延迟-负载曲线 | 拐点 8-12 req/s；吞吐 +38% 而 goodput −95%（21×），TTFT 60ms→44.8s 而 TPOT 不变 |
| E52 规划器校验 | 实测 vs 预测 −7%～−40%；反推 bs=64 时 attention 效率 86%（非 50%），据此三点校准 |
| 容量规划器 | `scripts/serving_capacity_planner.py`：输入模型/上下文/SLO → 容量+步时分解+TTFT+goodput，系数全部可追溯 |
| E53 KV 块长 | 同口径 gather 税 1.54/1.70/1.98/2.31/2.30×（16→256）；**block=16 在三种长度分布下都最优** |
| E54 多租户调度 | 容量内四策略完全一致（757.8）；过载时 FCFS 46.9 → **SJF 1140（24×）**；严格优先级把长请求饿到 222s |
| E58 W8A16 prefill | 仍 0.67-0.90×（int8 张量核未有效利用）；bf16 225 TFLOPS vs int8 204 |
| E59 冷启动分解 | 权重搬运仅 0.41s（磁盘 0.02 + H2D 0.29 + 预热 0.10）；对比 E17 的 169s → **99.8% 是引擎初始化** |
| E62 GQA/MQA | nkv 32→1 使 KV 降 32×，但 bs=1 时延仅 −6%、bs=8 仅 −8%；**容量优化不是加速** |
| E63 混批 chunk 代价 | 2k chunk 使 TPOT +23%（发射间隙可藏）、8k 使 +239%；与 E44 的差异由"发射受限 vs 带宽受限"解释 |
| E64 容量矩阵 | 模型×上下文×SLO 三张表（eager / graph-captured），2k→32k goodput 掉 28-50× |
| E65 KV 精度杠杆 | int8 → 容量 2×、int4 → 4×，步时不变；**32k 的 int4 等价 bf16 的 8k 容量** |

**贯穿性结论**：本栈（eager）里几乎一切都先撞**发射开销地板**（~30ms/step），
所以"优化"的顺序是 ①图捕获 ②attention 访存效率 ③真实带宽。这条主线把
12 个实验串成一个故事。
### E31 真实 trace 回放（2026-11-06）

- 统计等价 trace（ShareGPT 长度统计 + Gamma 到达），四卡并行扫速率：
  2/4/8/16 rps → goodput 2.12/**3.45(峰)**/2.40/0.53；16rps 零失败但 p50 59s
- burstiness 同节点（4rps/300 请求/跑间排空）：cv0.5 p99 42.9s vs cv6 20.1s；
  反序复测 35.5 vs 19.3s → **近饱和时平滑到达尾部更差**（队列永不排空）
- 两个方法坑已修：①高 cv 实达速率漂移（改定数+归一化）②过载跑队列泄漏（跑间排空+反序复测）

### E32 算子级无同步 conv1d（2026-11-06）

- patches/e32_conv1d_nosync/：decode 路径 5 算子纯张量实现（零 .item()）
- 数值：fp32 6.98e-06 / state 完全一致 / pad 槽正确 / 递推 8.07e-06；bf16 4.88e-04
- 基准（M3）：fallback O(B) 0.55ms/序列 vs nosync 恒定 1.2ms →
  B=1 0.49×（launch 成本占优）、B=8 3.86×、B=32 15.3×、B=64 31.1×、B=128 60.6×
- 未接入推理路径（需 hook + ACL graph 回归）；varlen 显式回退

### E33 接进推理路径——负结果（2026-11-06）

- 分发门控 + 命中计数器部署到 M3，A/B 交替 off→on→off→on
- **命中数 = 0**：decode 走 FULL ACL graph 回放，Python conv1d 不在热路径
- R1 显示 on +7~15%，R2 反转 on −12%；两个 off 臂自身差 8.7% → 纯噪声
- 修正 E32/issue #18978 表述：sync 税的是 capture/eager 路径，非回放稳态
- 已回滚补丁、恢复默认副本；E32 算子保留（数值+算子级收益真实）

### E34 prefill→decode 干扰消融（2026-11-06）

- scripts/e34_prefill_stall.py：4 流 SSE 逐 token ITL + 4k prefill 注入
- 基线 14.4ms → 注入 p99 187ms；budget 消融：512=173ms/44 elevated、
  默认=187/60、8192=110/8 但基线 14.4→22.5ms
- 结论：小 chunk 只降 max 尖峰不降 p99；大 chunk 省 p99 但抬稳态 →
  四维权衡（尖峰深度/干扰时长/稳态 ITL/注入 TTFT）

### E35 prefix cache 收益曲线（2026-11-06）

- 命中 TTFT 恒定 ~0.15s（前缀被跳过）；P≥1k 命中省 ~0.14s（45-57%）
- 256-token 前缀命中反而 −87%（查找开销 > 重算）→ 缓存有盈亏平衡点
- 与 D3 拼成完整故事：路由提命中率 → 命中价值 ~0.14s/req + 吞吐减负

### E36/E37 运维与生命周期实验（2026-11-06）

- E36 滚动重启：naive 8 err(500在途) vs drain(3s) 5 err(断连)；nginx 隐式重试兜新请求；
  正确流程=摘流量→等在途归零(实测3s)→杀
- E37 断连释放：8 SSE 断开 → 0.34s 内 requests+KV 归零；检测时机=下次写失败
