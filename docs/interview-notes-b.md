# 面试知识点索引·环境B白盒线（数据 → 结论 → 原理）

环境A（`interview-notes.md`）从服务层看推理；本篇下钻到算子层/实现层。每条：结论 → 我们的实测 → 机制。原始数据 `results/n{N}/`。

## Decode 的真实瓶颈（N2，profiler 白盒）

**Q: 为什么说 decode 是 memory-bound？**
- 标准答案只对了一半。torch_npu.profiler 实测（Qwen3.5-4B bf16, 910B2, eager）：单 token decode 每步 wall 114ms，**device 算子合计只跑 24.1ms——79% 时间 NPU 在等 host**
- 每步下发 2935 个算子（32 层 ≈ 92 算子/层），host 调度/launch 是第一瓶颈；MatMul(GEMV) 34.7%（权重读取）才是第二位的 memory-bound 项
- 机制：eager 模式每个 op 一次下发；小 batch decode 的算子都极小（1 token），下发开销 >> 执行
- **这就是 vLLM graph capture（cudagraph/fullgraph）存在的理由**：把 2935 次下发折叠成 1 次图重放。反问面试官："你们的引擎 decode 路径开 graph 了吗？"

**Q: prefill 和 decode 的算子分布差在哪？**
- prefill 4096tk：AIV(向量核) 68.9%、AIC(矩阵核) 仅 20.4%——Top 算子是 `Mul`×8205、`Slice`×19946、`Tril`(单次 688µs) 等未融合小算子；`MatMulV3` 单次 551µs 只占 13%
- decode 1tk：MatMulV2 34.7% 居首，`TransData` 22.1%（bf16→Cube 计算格式的搬运税）
- 结论：eager 下两个阶段都先被"小算子未融合+host 开销"拖累；融合优化（RMSNorm 算子化、FusedQKV、FA）收益直接写在 profile 里

**Q: 什么算子会落到 AI CPU（性能坑）？**
- 实测本模型 AI_CPU 占比仅 0.4%（GDN 无回退，好）；已知高危名单：ScatterElements/MaskedFill/Cumsum（MiniMind 910B4 案例，单次耗时远超 AI Core 路径）
- 排查方法：op_statistic.csv 按 Core Type 聚合，AI_CPU 行非零即候选

## Hybrid 架构（N3 + N1，Qwen3.5 一手数据）

**Q: GQA/MLA/DeltaNet 的 KV cache 区别？**
- 我实测过 Qwen3.5-4B（24 层 GDN + 8 层 Gated Attention，3:1）：**KV 斜率 32 KB/token，六个长度点（512→131072）与公式逐字节吻合**（8 attn层×2×L×4 KV头×256×2B）；全注意力同规模对照是 128 KB/token——**hybrid 把 KV 增长压到 1/4**
- GDN 层不存 KV，存固定 recurrent state S∈R^(32头×128×128)：实测恒 49.5MB（fp32 存储=bf16 理论 2.06×，框架为数值稳定升精度——GDN 的 decay `g=-exp(A_log)·softplus(a+dt_bias)` 必须全程 fp32，bf16 下 A_log.exp() 会溢出）
- 131072 ctx 显存剖面：权重 7.93G + KV 4.1G + state 49.5MB；全注意力模型此时 KV 已 16G
- prefill 时间拐点：512→32768（64×长度）时间仅 15×（GDN chunked O(L) 主导）；65536→131072 涨 2.41×（8 层 attn 的 O(L²) 显形）——hybrid = 24 层线性 + 8 层二次的合成曲线

**Q: 手写 hybrid 前向最容易踩的坑？**（N1 实现+对齐 transformers 5.16.1，greedy 逐 token 一致，rel 8e-3∈bf16 噪声）
- GDN 更新顺序：先衰减 `S←S·exp(g_t)` 再算 delta `=(v−Sᵀk)·β`，输出用更新后的 S
- RMSNorm 是 zero-init delta 语义 `out=rms(x)·(1+w)`（GDN 的门控 norm 则是 one-init 直乘）——照抄 Llama 会整体偏移
- q_proj 输出 2×（q 与 sigmoid gate 融合投影）；attention 有 q_norm/k_norm（RoPE 前）；conv state 存最后 K=4 个原始输入（非 K−1）
- tied weight 陷阱：TextConfig 默认 tie=False 而 4B 是 True，load_state_dict 会经共享 Parameter 静默覆盖 embedding（missing/unexpected 都为空，极难察觉）

## TP 并行（N4，跨 NUMA PCIe 实测）

**Q: TP 为什么对 decode 收益差甚至为负？**
- 微基准（2×910B2 跨 NUMA PCIe，hccl）：**裸 all_reduce ~288µs（1 元素）**；4KB–256KB 延迟与消息大小无关（292–302µs 平台）→ **>97% 是 barrier/握手，不是带宽**。对比 NVLink H200 同类 6.9µs：慢互联放大 40×
- 带宽饱和 16.3 GB/s（PCIe Gen4 x16 理论 51%）；延迟模型 `latency ≈ 285µs + bytes/16.6GB/s` 拟合极好
- 推论：4B 模型 TP=2 decode 每层 2 次 all-reduce ≥576µs 固定开销，vs 单层计算 150–300µs——**通信是计算的 2–4 倍**，扩展比失效。小消息通信是"按调用次数交税"，不是按字节
- 工程含义：TP 放高速互联（HCCS/NVLink）内；慢互联优先单卡/PP/通信-计算重叠；MB 级消息要看 p99（实测 16MB p99 是 p50 的 4–5 倍）

## W8A8 量化解剖（N5，msModelSlim 格式一手分析）

**Q: W8A8 的 scale 怎么存、哪些层不能量化？**
- 实测 Qwen3.8-27B-w8a8（msModelSlim/ascendv1_saver）：**权重 per-channel int8 对称量化**——scale F32 shape (out_channel,1)（对齐 Cube 算子的轴布局），offset 全 0 是占位字段；group_size=0 无 per-group
- 激活 **per-token 动态量化**（在线 absmax，不落盘）——部署时即 vllm-ascend 的 dynamic quant cube 算子
- 覆盖 84.5% 权重参数；**排除项都有理由**：GDN out_proj（exclude `*linear_fc2*`，精度敏感，代价 3.0GiB bf16）、lm_head/embedding（各 2.37GiB，输出质量敏感）、全部 norm、A_log/dt_bias（存 BF16 运行时升 fp32）
- 压缩比 1.73×（vs 理想 2×，被排除项拖低）；理论字节 29.922GiB 与磁盘完全对账
- scale 分布：median 3.9e-4，离群集中在深层 full_attention 的 v_proj（layers 51/55/59/63，~3.5× 中位）；权重侧几乎无离群通道（>8×median 占比≈0）→ 量化难点在激活侧而非权重侧——这正是 per-token dynamic + exclude out_proj 的动机
- 该格式是 vllm-ascend 专用产物（verified_tags 锁 Atlas A2/A3），HF transformers 加载不了

## 白盒方法论（可迁移的面试谈资）

- torch_npu.profiler 完全对标 PyTorch GPU profiler：`ProfilerLevel.Level1 + AiCMetrics.PipeUtilization` 拿 Cube/Vector 利用率，导出 op_statistic.csv 按算子/核类型聚合
- 判据链：wall vs device 合计 → host gap 占比；算子次数/步 → 下发税；Core Type 分布 → 计算形态；TransData 占比 → 格式搬运税
- 长上下文实验必须 `logits_to_keep=1`：全序列 lm_head 在 65536 ctx 会申请 30GB（65536×248320×2B）直接 OOM——这本身就是"引擎只对最后位置算 logits"的原因的直观演示

## PD 分离（N7，账本分析）

**Q: PD 分离什么时候值得？KV 传输开销多大？**
- 核心比值 = **KV 体积 ÷ 互联带宽 vs 一个 decode 步时间**。我用 N3/N4 实测数据算过账（results/n7/n7_pd_ledger.md）：跨 NUMA PCIe 16.3GB/s 下，4B hybrid @32K 传 1GB KV 要 63ms ≈ 0.6 个 decode 步（勉强可行）；@128K 传 4GB 要 250ms ≈ 2.5 步（收益被吃掉大半）——DistServe "传输<一步"的结论依赖 NVLink/PCIe5，慢互联不成立
- 层级传输（Splitwise 式逐层搬运）在慢互联上有隐藏税：每层 KV 只有几十 KB，落在 N4 实测的 288µs 固定延迟平台，32 层握手税 9.2ms > 层传输本身
- 吞吐公式：T_PD = max(I/mP, O/nD) 只有 P/D 各自差异化调优后才可能赢融合式的 (a+b)/(m+n)，否则纯亏（多一份模型副本+传输）；SGLang 文档原话"不做差异化优化，PD 分离吞吐反而更低"
- hybrid 模型的新难点：GDN 的 S 矩阵（49.5MB/请求，fp32）**不可分块/部分回滚，必须原子交接**（TreeWY 论文的动机）——KV cache 可按页传，RNN state 不行

## DeepSeek 昇腾组件开源（2026-09-30，生态认知题）

**Q: 你知道 DeepSeek 给昇腾开源的那套组件吗？能直接用吗？**
- 六件套：TileLang 昇腾后端 + DeepGEMM-Ascend + DeepEP-Ascend + TileKernels + FlashMLA + DeepSelect，与英伟达版一一对应 API
- **五个性能组件全部绑定 Ascend 950（dav-3510）+ CANN 9.2/9.20 bisheng 编译器**：DeepGEMM 要 CANN 9.20、DeepEP 要 950 独有的 UBMEM/URMA 通道、FlashMLA 昇腾 decode 只收 FP8/FP4 KV、TileKernels Requirements 写死 950+CANN≥9.2——**910B/910C 存量卡（含我实验用的 910B2）无一能跑**，且宿主机 HDK 25.2 驱动也升不到 CANN 9.2
- 唯一例外：**tilelang-ascend 适配器仓**（tile-ai 社区线，非 DeepSeek 主线后端）覆盖 A2/A3（910B/C），CANN ≥8.3 即可——我在 CANN 8.5.1 上具备实验条件（用 TileLang DSL 手写融合算子，接 N2 的"eager 小算子未融合占 68.9%"做融合收益量化）
- 面试加分点：昇腾算子开发三条路线之争（TileLang DSL vs Ascend C 原生 vs CATLASS 模板）、DeepGEMM-Ascend 的 1 Cube+2 Vector 混合核、scale 格式差异（UE8M0 对打包 int16）——能讲出"为什么这批内核 910 存量卡用不了"的人极少

## Hybrid 两类记忆的功能分离（N3b，干预实验——负结果金矿）

**Q: GDN 的 recurrent state 和 KV cache 各自承载什么信息？**
- 我们做了六条件干预实验（1606 tok prompt：海盗 persona + 10 条事实 + 两问，prefill 后把对应 cache `.zero_()` 再 decode）：
  - **清空 GDN recurrent state（一次性）→ 完全无影响**：照样答出密钥 4F2C、海盗腔 10 词——state 不承载 prefill 的可检索副本
  - **清空 8 层 attn KV → 机制性塌缩**："本本本本…"单字重复（3-gram 重复率 0.86），密钥/persona/连贯全丢——KV cache 承载全部 prefill 内容
  - **每步都清 recurrent state → 塌缩成"俺老铁，俺老铁…"**（rep 0.58）——state 的真实作用是 **decode 期递推连续性**，不是存储器
- **轴向与 arXiv 2609.04434 的"recurrent=响应策略 / KV=精确检索"结论相反**（至少在本模型单轮短生成下）：实测 KV 是唯一的内容载体，GDN state 是递推器
- 机制解释（面试加分）：全零 K/V 不是"遗忘"——softmax(全零分数) 对全部历史位置**均匀加权**，等效有效上下文归零，所以连语言连贯都不保；这类干预实验的判读必须区分"内容丢失"与"计算结构破坏"
- 工程坑：Qwen3.5 是思考模型，短预算 decode 必须 `apply_chat_template(enable_thinking=False)`，否则 100 token 全被 think 段吃掉

**Q: KV cache 到底值多少加速？（手写引擎 A/B，非框架数字）**
- 自写引擎同权重对照（严格等价：30 greedy token 全一致）：no-cache vs cache = **4.4×@128tok → 9.6×@2048tok**，且 128 token 时就已 4.4×——no-cache 的代价大头是每层算子调度的固定开销而非重算本身
- cache 版 decode 对上下文长度脱敏：128→2078 ctx（16×）每步只 +2.2%（GDN 层 O(1)/步是架构红利，attention 层 q=1 时 KV 读占比极小）
- **手写引擎 88-90ms/步 vs transformers 手动循环 96-98ms（快 8-10%）**：省的是 HF DynamicCache 封装、mask/position 构造的 Python 层开销——框架税的实测大小
- 手撕陷阱：裸调 `model(seq)` 忘 `no_grad()` → autograd 保留全序列中间量，542 token 峰值 +20.4 GiB 直接 OOM（CPU 小模型完全感知不到）——NPU 裸循环验证必须显式 no_grad

（N6 vllm-ascend 部署+MTP+TP：进行中，数据回来后补条目。）
