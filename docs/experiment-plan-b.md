# 环境B实验计划：白盒推理实验线（3×910B2 单容器）

> 环境A（E 系列，vllm 0.30 四节点）负责**服务化黑盒**：压测、调度、多副本、PD 分离。
> 环境B（本计划，N 系列，裸 torch_npu）负责**白盒**：算子级剖析、手写实现、新架构实测。
> 两条线互补，面试叙事 = "服务层我压测过 + 算子层我 profile 过 + 核心机制我手写过"。

## 环境盘点（2026-11-06 验证）

| 项 | 值 |
|---|---|
| 接入 | `ssh -p 40045 root@223.111.149.177`（密码不入库） |
| 卡 | **3×910B2（npu 0/1/7）**，61GB HBM/卡 |
| 拓扑 | ⚠️ 两两 **SYS**（跨 PCIe/NUMA），无 HCCS 直连；CPU 亲和 NPU0:144-167 / NPU1:0-23 / NPU7:48-71 |
| 软件 | torch 2.9.0+cpu · torch_npu 2.9.0 · CANN 8.5.1 · transformers 5.16.1（**原生支持 qwen3_5**）· Python 3.10 |
| vllm 线 | CANN 8.5.x 配套 vllm-ascend 0.18/0.19（可选，独立 venv） |
| 模型 | `/model/Qwen3.5-4B`（hybrid GDN+GatedAttn, 8.8G）· `/model/Qwen3.8-27B-w8a8`（30G）· MiniMax-H3（仅 README）· mineru |
| 网络 | pypi / 华为云源直连 200，**无需代理**；HF 下载用 `HF_ENDPOINT=https://hf-mirror.com` |
| 资源 | 192 核鲲鹏 · 2TB 内存 · /data 600G |

冒烟已过：`device_count=3`，matmul on npu:0 OK，`AutoConfig.from_pretrained("/model/Qwen3.5-4B")` → `model_type=qwen3_5` ✅

## 实验矩阵

| # | 实验 | 依赖 | 预估工作量 | 面试锚点 |
|---|------|------|-----------|---------|
| N2 | msprof 白盒剖析：prefill vs decode 算子级证据 | 无 | 0.5 天 | "为什么 decode 是 memory-bound"——甩 profiler 数据 |
| N3 | Hybrid GDN 架构实测：KV 曲线 + cache 干预 | transformers | 1 天 | "GQA/MLA/DeltaNet 的 KV cache 区别"（2026 新题） |
| N4 | TP on 跨 NUMA PCIe：通信开销主导性实测 | torch.distributed/HCCL | 0.5 天 | "为什么 TP 大了 decode 反而慢"——有 SYS 拓扑实测 |
| N1 | 手写最小推理引擎（nano-vllm on NPU） | N2 结论 | 2–3 天主线 | "PagedAttention 原理"——"我从零手写过" |
| N5 | w8a8 权重解剖 | safetensors | 0.5 天 | "W8A8 scale 怎么存/精度损失在哪" |
| N6 | （可选）vllm-ascend 0.18 对照 | 独立 venv | 1 天 | 与 A 环境 vllm 0.30 形成版本代差对照 |

建议顺序：**N2 → N3 → N4 → N1（主线）→ N5 穿插**。N2/N3/N4 每个半天到一天出数据，先攒可讲解的结论；N1 是长线工程。

---

## N2 · msprof / torch_npu.profiler 白盒剖析

**目标**：给环境A的服务级结论补算子级证据。

**方法**：
1. `torch_npu.profiler.profile`（对标 PyTorch GPU profiler 用法）采 Qwen3.5-4B 的 prefill（一次 4k token）与 decode（batch=1, 200 步）
2. 读 `op_summary_*.csv`：算子耗时分布、kernel_type（AI_CORE / AI_VECTOR_CORE / AI_CPU / MIX_AIC）
3. `experimental_config` 开 `ProfilerLevel.Level1` + `AiCMetrics.PipeUtilization/Memory`，拿 Cube/Vector 利用率与 HBM 带宽占比
4. 对照三组数字：prefill 的 AIC 占比 vs decode 的 AIV/memcpy 占比、算术强度（FLOPs/字节）、单步耗时分解

**预期结论**：
- prefill：MatMul(FlashAttention) 主导，Cube 利用率高 → compute-bound
- decode：小算子（Mul/Cast/Add）高频调用 + 权重读取主导，Cube 利用率低 → memory-bound
- 已知坑位（MiniMind 910B4 案例）：ScatterElements/MaskedFill/Cumsum 落 AI_CPU、主 stream 利用率个位数、host 侧 launch 长尾

**产出**：`results/n2/`：算子耗时表、利用率对比、一页"decode 每毫秒花在哪"的分解图。

## N3 · Hybrid 架构（Gated DeltaNet）实测

**背景**：Qwen3.5-4B = 32 层，其中 **24 层 GDN（线性注意力，固定 recurrent state）+ 8 层 Gated Attention（KV cache 随长度增长）**，3:1 布局。2026 面试新热点（Qwen3.5/Kimi Linear/Falcon-H1 全走 hybrid），绝大多数候选人只会背公式。

**方法**：
1. **KV 账本实测**：跑不同上下文长度（1k/4k/16k/64k），记录每层 cache 实际字节数（KV cache 只在 8 层涨；GDN state 恒定：24 层 × 32 V头 × 128×128 × 2B ≈ 48MB/seq）
2. **公式 vs 实测**：hybrid 显存公式（对环境A E8 的理论推导做实测闭环）
3. **cache 干预实验**（复现 arXiv 2609.04434 思路）：
   - 单独清空 recurrent state（保留 KV cache）→ 模型失去"响应策略"（语言风格/指令遵循），保留检索
   - 单独清空 KV cache（保留 state）→ 失去精确检索，保留风格
   - 这是"两类记忆功能分离"的一手证据，面试绝杀故事
4. （加分）GDN chunked prefill vs recurrent decode 的耗时差：线性注意力 prefill 为什么快

**产出**：`results/n3/`：显存-长度曲线（hybrid vs 全注意力斜率对比）、干预实验输出样例。

## N4 · TP on 跨 NUMA PCIe：通信开销主导性

**背景**：本机三卡两两 SYS（无 HCCS），是"慢互联"的天然实验场。业界数据：H200 NVLink 上 all-reduce 占 TP=8 decode 的 30–50%，其中 >99% 是 barrier 握手而非数据搬运（16KB 消息走 900GB/s 理论只需 0.018µs，实测 6.9µs）。

**方法**：
1. HCCL 基线：torch.distributed all_reduce 微基准，消息 16KB–16MB 扫描，测 PCIe 路径的带宽/延迟曲线（对比 NVLink 公开数据）
2. TP=1 vs TP=2 vs TP=3 跑 Qwen3.5-4B（16 Q 头：TP=2 每卡 8 头 ✅；TP=3 需验证头数整除，不整除则记为约束发现）
3. 度量 TTFT/TPOT 变化 + `torch_npu.profiler` 里 HCCL 算子占比
4. NUMA 绑定对照：进程绑到卡所在 NUMA（NPU1→CPU 0-23）vs 错绑，看 host 侧开销差

**预期结论**：4B 小模型 decode 是 memory-bound，TP=2 每卡读的权重减半，但每层 2 次 all-reduce 走 PCIe 的 barrier 开销 ~几十 µs × 64 次/步 → 大概率净负。若实测为负，即验证"barrier 主导"结论在慢互联上被进一步放大；若为正，找到 crossover batch size。

**产出**：`results/n4/`：all-reduce 微基准曲线、TP 扫描表、HCCL 占比剖析。

## N1 · 手写最小推理引擎（nano-vllm on NPU）——主线

**目标**：在 torch_npu 上从零实现可度量替换的推理引擎，每一级优化都有 A/B 数据。

**参考生态**（调研结论）：nano-vllm 系（ovshake / Aparajith24 / Mihawii / AEM001）在 GPU/MPS 上已有成熟方法论与基准数字（paged KV 2.5×、prefix caching TTFT 5.8×、chunked prefill stall 3.3×）——**在 NPU 上复刻这套度量本身就是差异化**。

**递进阶梯**（每级对照上一级测吞吐/时延/显存）：
1. **L0 naive**：transformers `generate()` 基线（贪心、无任何优化）
2. **L1 手写解码循环**：手动 prefill→decode、显式 position_ids、on-device argmax；对照 L0 分离"框架开销"
3. **L2 KV cache**：on/off 对照（文献基准 2.6×，看 NPU 数字）
4. **L3 静态 batch → continuous batching**：迭代级调度，完成即退出、新请求即加入
5. **L4 paged KV cache**：block 池 + block table + prefix 共享（内容哈希 + 引用计数）
6. **L5 chunked prefill**：长 prompt 切片，解码不被阻塞
7. （可选 L6）hybrid 模型支持：GDN 层的 recurrent state 管理（接 N3）——**业界还没有开源 NPU 实现的 hybrid paged 引擎，做出来就是简历级差异点**

**实现要点（来自参考项目的已知深坑）**：
- decode attention 用 reshape 成 plain matmul 比 SDPA 快 ~10×（28 用户 × 512 ctx：4.1ms → 0.43ms，同结果 3e-7）——NPU 上待验证
- prefix caching 用于 prefill 时 GQA 广播会悄悄变慢（参考项目踩过：4.5ms vs 1.3ms）
- 每次改动同会话重测（同代码不同天速度可差，环境A E23 抓过同类伪影）

**产出**：`examples/engine_np/`（引擎代码）+ `results/n1/`（L0→L5 各级基准表）。

## N5 · w8a8 权重解剖

**方法**：直接读 `Qwen3.8-27B-w8a8` safetensors：
1. scale 的粒度（per-tensor/per-channel/per-group）与 dtype（环境A docs 里 fp8 scales 的 worked answer 在此实测验证）
2. scale 分布统计（直方图、离群层）
3. 反量化重建 → 与 bf16 参考（若 HF 有同模型）比 top-1 一致率 / PPL；没有对照则做自一致性检查
4. （可选）int8 权重加载推理跑通，测 W8A8 vs（4B 的）BF16 的 TPOT 差 → 量化加速比的一手数据

**产出**：`results/n5/`：scale 统计表、反量化精度数据。

## N6 · （可选）vllm-ascend 0.18/0.19 对照

CANN 8.5.1 配套行。独立 venv 装 `vllm==0.18.0 + vllm-ascend==0.18.0`（有 A3 上跑 Qwen3.5-27B-w8a8 + qwen3_5_mtp 的成功先例）。价值：与 A 环境 vllm 0.30 形成版本代差对照 + 给 N1 手写引擎一个"工业引擎"参照系（L5 之后对比离 vllm 还差多少）。优先级最低，N1 做到 L4 之后再考虑。

---

## 风险与备注

- transformers 5.16.1 加载 qwen3_5 已验证 config 层 OK；**完整 forward 未验证**，N3 第一步先跑通 generate（GDN 算子在 torch_npu 上可能有 fallback/不支持，若报错则转手写前向线，反而更硬核）
- torch_npu 2.9.0 + CANN 8.5.1 为官方配套行（TorchNPU 7.3.0），但跨 NUMA 三卡的 HCCL 行为需微基准先行确认（N4 第一步）
- 环境B无 HCCS 直连 → **不要**把此机 TP 数据当作 HCCS 机器的代表；讲解时明确"慢互联放大了通信项"这一前提
- 密码/订阅等敏感信息不入库
