# 面试知识点索引（数据 → 结论 → 原理）

每个条目：一句话结论 + 我们的实测数据 + 背后机制。面试时先抛数据再讲机制。
全部原始数据与解读见 `results/e{N}/README.md`。

## 推理服务化

**Q: vLLM 为什么快？核心机制是什么？**
- Continuous batching：请求随到随插入 decode 批次（而非等整批结束）→ 吞吐近线性扩到几十并发
- PagedAttention：KV cache 按页管理（类 OS 虚拟内存），碎片率低，batch 才能做大
- 实测：Qwen3.5-4B on 910B2，并发 1→64，输出吞吐 75→807 tok/s（E1）

**Q: TTFT 和 TPOT 分别受什么影响？怎么优化？**
- TTFT = 排队 + prefill 算力。并发越高排队越久：实测 c=1→64 TTFT 161ms→797ms（p99 3.3s）
  - 优化：prefix caching（共享前缀免 prefill）、chunked prefill、扩副本分担
- TPOT = decode 单步时延。c=64 时 71ms vs c=1 的 12ms
  - 优化：量化（减带宽）、spec decode、扩副本

**Q: 单流吞吐受什么限制？**
- 小 batch 的 decode 是 memory-bandwidth bound：每 token 读全部权重
  - 4B bf16 ≈ 8.8G/步；910B2 HBM 带宽 ~1.6TB/s → 理论 ~180 tok/s；实测单流 82 tok/s
  - 差距来自 KV cache IO、采样、调度开销 → 这就是量化和 spec decode 的价值点

## 量化与服务变体（E2）

**Q: 27B 模型怎么上 64G 单卡？**
- bf16 27B = 71G > 64G HBM：量化不是优化项而是**入场券**，必须 W8A8（30G）
- 实测 27B-W8A8：c=64 输出吞吐 396 tok/s / TPOT 144ms（vs 8B bf16 1318 tok/s / 42.6ms）
- 代价：batch 一大 decode 就吃力——W8A8 省显存但 KV cache 仍挤占 HBM

**Q: W8A8 / W8A16 / bf16 怎么选？**
- 我们的可行性矩阵：bf16-27B 塞不下；W8A16（仅权重量化）被
  vllm_ascend 0.30rc2 的 compressed-tensors 路径拒绝（不支持无 input-quant 配置）；
  W8A8 可跑但激活量化有精度代价 → "能跑/精度/吞吐"三角权衡
- **先查引擎支持的量化格式清单再谈精度对比**——部署侧的真实约束

## 调度特性消融（E3）

**Q: prefix caching 值多少？**
- 4×2048-token 共享前缀 ×128 请求：关 PC 后 TTFT +28%、TPOT +56%、吞吐 −34%
- 收益 ∝ 负载前缀共享率（system prompt/多轮历史/RAG 模板天然共享）→ 免费午餐；
  前缀全不同则零收益

**Q: speculative decoding 上线前要验证什么？**
- ngram 实测**全负收益**：random 数据集接受率 0-8%、sonnet 真实文本也只有 8-11%
  （接受长度 ~1.5），TPOT 恶化 3.7-6 倍、吞吐 −80%
- 公式：收益 = 接受率 × 每接受省一步 − draft+verify 开销；接受率 <30-40% 直接关
- ngram 的 lookup 只匹配当前请求自身上下文 → 只对"复读机"式输出有效；
  通用路线要 eagle/medusa 类 draft 模型
- **教训：消融要用对方法公平的负载**（random 对 ngram 先天不利）

## 系统扩展性（E4）

**Q: 多副本 vs 张量并行怎么选？**
- 实测 3×单卡副本 + nginx least_conn（SSH 隧道组网）：
  c=128 吞吐 1027→1674→2160 tok/s，扩展效率 1.63×/2.10×
- 亚线性来源：隧道带宽、连接级均衡、NPU 时钟差异
- **扩容只在单副本饱和后兑现**：c=32 时 N=3 无增益（837→837）
- TTFT 改善先于吞吐改善（2751→1262ms）——低延迟 SLA 场景扩容收益更大

## 昇腾生态与工程化（差异化点）

- torch_npu = PyTorch NPU 后端；vllm-ascend = vLLM platform plugin；
  CANN 对标 CUDA；ATB = 推理算子库；HCCS 对标 NVLink
- **MindIE 部署实测**（E5）：3.0.0 全部二进制绑死 glibc2.38/Ubuntu24.04 +
  CANN 8.5.1 + py311，22.04 容器上 8 步排障后物理不可行——
  闭源栈的平台锁定 vs vLLM 源码可移植性，这是引擎选型的隐性成本
- 运维坑位速查（docs/status.md）：editable 包被 namespace 遮蔽、
  pgrep self-match 三形态、ssh 起 daemon 必须 setsid+断 fd、客户端 CPU 污染压测

## 第六轮增量（E16–E19, D3, E17）

**Q: 长上下文下 TTFT/TPOT 怎么变？KV 真的撑得住吗？**
- E16 实测 4k→64k ctx：TTFT 1.39→12.2s（近线性于 prompt 长度），
  TPOT 19.1→61.8ms（prefill/decode 争抢调度），KV 峰值仅 1.9→13.9%（c≤4）
- 反直觉点：`kv_cache_size_tokens` 随 max_model_len 变化（8192→1.006M，
  81920→1.427M）——容量规划必须按实际配置实测
- 长上下文容量瓶颈从调度准入切回 KV 字节：64k 单请求 ~3.1GB

**Q: 多租户/弹性伸缩怎么做？sleep mode 真实成本？**
- E17 完整换租链：A 冷启 169.4s → sleep2 1.77s（HBM 4480/65536MB）
  → B 冷启 114.2s 进腾出空间 → B 在线时 A wake **0.139s**，首请求 0.129s
- 机制：CaMem 按 tag 搬显存，allocator/graph 元数据留着→wake=O(权重拷贝)
- 边界坑：睡眠方仍持 stream/CQ → 共卡引擎只能 eager（EE1023 同根因）

**Q: 多副本路由怎么让 prefix cache 生效？**
- D3 实测：自写 stdlib LPR 路由（前缀 sha1 亲和 + inflight+2 降级）
  命中率 53.8%→84.0%（~6k 前缀）/42.5%→62.2%（~900 前缀）
- 但 p90 持平：并发到达时"第二波才命中"是命中率上界；
  亲和倾斜要靠负载降级兜底——cache-aware 路由的真实权衡

**Q: Ascend 有哪些调度器，单机该用哪个？**
- E19 画像：4 选项 3 个有前置（recompute/dyntra_lb 要 kv_consumer；
  profiling_chunk 要 PP>1）——它们是 PD 配套件不是通用优化
- 单机唯一可用 SRF：short 28.73→26.66s（−7.2%），按 prompt 长度分道
  + 老化防饿死——输出预算未知时最现实的 SRPT 近似

**Q: 跨节点 PD 为什么更难？失败是什么样子？**
- E18 三层失败画像：ZMQ 元数据（pod IP 广播错→VLLM_HOST_IP+隧道修通）
  → TE-RPC 动态口（再补隧道）→ 数据面 device-mem 传输物理不可行
- **最危险发现：传输失败请求仍 200，返回垃圾 token**——PD 监控要盯
  输出质量不是状态码
- 面试加分提问点：遇到 PR 先查 mainline 历史（D1 否决案例：目标符号
  上游已删，补了=重加被移除代码）

## 手撕系列（E8b–E8g，910B1 实测）

**Q: 手写 attention，融合算子快多少？**
- naive MHA vs FIA 融合：S8192 差 5.32×，差距随序列长放大（E8b）
- 单请求 decode matvec 实测 1475 GB/s ≈ 910B1 HBM 峰值 92%——
  访存墙实锤，decode 快不快看带宽不看看 FLOPs

**Q: RoPE/GQA/SWA 实现细节？**
- RoPE：rotate-half 与复数乘法 diff=0；范数保持 1.9e-6、相对位置
  性质 8.6e-6（E8e）——`R_m^T R_n = R_{n-m}`，插值外推的地基
- GQA：数学上就是 KV 头广播（vs 原生 enable_gqa diff=0）；收益是
  KV 字节 268→33.6MB/req（8×），不是 FLOPs（E8e）
- SWA：显式 band mask 在 NPU 比 is_causal 慢 26× 且窗口零收益——
  真 SWA 必须 kernel 跳过窗外块（FIA sparse_mode=4），mask 永远 O(S²)（E8e）

**Q: 手写 FlashAttention 怎么答？**
- online-softmax 递推式能写、S²→S·Bc 显存账能算（S8192: 268→16.8MB）、
  "为什么 Python 版不快"能讲——加速来自片上驻留+消物化，不是少算（E8g）
- 手写分块与 naive 同速（~21ms @ S8192），比融合慢 24×

**Q: continuous batching / preemption / APC 怎么联动？**
- 手写调度模拟器：static→continuous +15% 吞吐（死槽浪费消失）；
  mnb=256 队头阻塞比 static 还差（=E3b 实测复现）（E8f）
- 最值钱机制：mnb=256 时 APC 让 TTFT -74%、吞吐 +80%——APC 收益
  本质是放大 token 预算，不是只省 TTFT
- KV 池耗尽→队尾抢占→34341 次抢占吞吐腰斩：容量规划的反面教材

**Q: spec decode 什么时候赚什么时候亏？**
- 手写 ngram draft-verify（DynamicCache.crop 回滚）：重复负载 +8%、
  开放文本 -6%——净收益 = 覆盖率×接受率 vs verify 开销（E8c）
- 对照 E14 原生 MTP：接受率 65.6% 仍亏（串行草稿+验证前向固定开销）；
  E15 k=1/2 也亏——Ascend 上 spec 的固定成本项是常量

## 系统走读与上游贡献

**Q: 一个请求在 vLLM 里怎么走？**
- docs/request-walkthrough.md：API 进程 tokenize→ZMQ→EngineCore→
  scheduler 双预算/prefix 命中/队尾抢占→NPUModelRunner 五态 attn→
  FIA/paged-attention→ACLGraph→sampler（绕开同步 multinomial）→
  detok→SSE，全程 file:line + 实测挂点

**Q: 上游有什么实质贡献？**
- vllm-ascend #21112：W8A16 compressed-tensors 掉穿 _detect_quant_type
  （bare NotImplementedError）——附 detector 缺 weight-only 分支的
  代码级根因 + wna16 scheme 已存在的复用建议
- #21113：默认 cudagraph_capture_sizes 耗尽 SQ/CQ（207005，plog
  签名 NormalSqCqAllocate）——与 #9564/#2629 划清边界，呼应 #9691
- #18978 comment：causal_conv1d_update_npu 缺失在 Qwen3.5-4B 的复现
  + ITL p99 160-250ms vs median 16ms 实测，自荐测 patch
