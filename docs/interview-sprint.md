# 面试冲刺包（Mock Drill）

> 用法：每个题目按"首答 → 追问树 → 陷阱"三层演练。所有数字均已回查原始 JSON。
> 配套：`docs/interview-notes.md`（知识点索引）、`docs/request-walkthrough.md`（源码走读）、
> `docs/war-stories.md`（STAR 故事）、`docs/demo-runbook.md`（现场演示）。

## 0. 开场叙事（90 秒）

> "我做过两条互补的线。CUDA 侧我手写过一个 paged serving 引擎——continuous
> batching 调度 + Paged KV 分配器 + OpenAI API（open-infra-ai/paged-serving），
> 配套手写过 FlashAttention 风格的 CUDA kernel（cuflash）。昇腾侧我在 910B 上
> 把 vLLM-Ascend 生产栈测穿：25+ 个实验覆盖调度/KV/量化/扩容/长上下文，
> 给 vllm-ascend 报了 3 个上游 issue（两个独立 bug + 一个带实测数据的 comment），
> 还把 MindIE 一个 warmup 挂死定案到 ATB 算子库层面。所以我既知道每一层
> **应该**怎么工作（因为写过），也知道真实部署里它**实际**怎么坏（因为测过）。"

简历 bullets（每条带可验证链接）：

1. 手写 paged-serving 引擎（continuous batching + PagedAttention + OpenAI API，
   真实权重 decode）＋ FlashAttention 式 kernel 深挖 —— open-infra-ai
2. 昇腾 910B 上 vLLM-Ascend 全栈实测 25+ 实验：并发扩展、量化可行性矩阵、
   prefix caching、spec decode 负收益边界、goodput@SLO、长上下文容量 —
   gitee/ascend-llm-infra-lab
3. 上游贡献：vllm-ascend #21112（W8A16 compressed-tensors detector 穿透）、
   #21113（cudagraph 默认尺寸耗尽 SQ/CQ）、#18978 comment（GDN 模型实测）

## 1. 深挖题一：一个请求在 vLLM 里怎么走（必考题）

**首答（30s 版）**：API 进程做 chat template + tokenize → ZMQ/msgpack 发给
独立 EngineCore 进程 → scheduler 按"token 预算 + 最大并发数"双约束准入，
查 prefix 命中分配 KV block → model runner 组 batch 前向 → sampler 出 token
→ output processor 增量 detok → SSE 流回。

**追问树**：

| 追问 | 答案骨架 |
|---|---|
| 为什么 token 预算不是请求数预算？ | 请求长度方差大；按 token 预算 prefill/decode 公平分算力。实测 mnb=256→4096 TTFT 13.1s→1.06s（E3b） |
| KV 不够怎么办？ | 队尾抢占（preempt 最新请求，KV 释放后可重算/换出）。模拟器实测 KV 池 160 blocks 时 128 请求产生 34341 次抢占、吞吐腰斩（E8f） |
| prefix cache 存在哪？ | KV block pool 按 block 内容 hash 索引；命中=整块免算。走读过 `kv_cache_manager` 的 computed-block 查找 |
| 为什么 API 和 EngineCore 分进程？ | tokenize/detok 是 CPU 活，放同进程会抢 decode 的 GIL/调度节拍；V1 设计让 EngineCore 只碰 token id 不碰文本 |
| decode 一步的延迟下限？ | 权重读一遍的访存时间。实测 4B bf16 单流 matvec 1475 GB/s ≈ 910B1 HBM 峰值 92%（E8b）——访存 bound 实锤 |

## 2. 深挖题二：spec decode 值不值（讲负结果，反而加分）

**首答**：我做了三层验证，结论是该场景下全线负收益——
① vLLM 原生 ngram：真实文本接受率 8-11%、接受长度 ~1.5，TPOT 恶化 3.7-6×、
吞吐 −80%（E3）；② 原生 MTP draft 模型：接受率 65.6% 了仍然亏，TPOT
36.5 vs 15.6ms（E14）；k=1/2 也亏（E15）；③ 我自己手写 draft-verify
复现机制：重复负载 +8%、开放文本 −6%（E8c）。

**追问树**：

| 追问 | 答案骨架 |
|---|---|
| 盈亏平衡点公式？ | 净收益 = f(覆盖率 × 接受率, verify 固定开销)。接受率 <30-40% 时 draft+verify 的固定前向开销吃光收益 |
| 为什么接受率 65% 还亏？ | Ascend 上 draft/verify 的固定开销是常量（串行草稿 + 验证前向），只有接受长度足够长才摊得平；k=1 接受 88.7% 也只省一步 |
| 什么场景会赚？ | 高可预测负载（代码补全、模板化输出、复读式多轮）+ 轻量 draft（eagle/medusa 头）+ 大 batch 下 verify 摊薄 |
| 你的手写实现验证了什么？ | ngram lookup + 单前向 verify + `DynamicCache.crop` KV 回滚；第一版我多算一次前向做 rewind，直接把 57% 接受率的收益吃没了——这本身就是"开销敏感"的活教材 |

## 3. 深挖题三：prefix caching 到底省多少（分布口径杀招）

**首答**：4×2048-token 共享前缀、128 请求、c32 实测：关 PC 后 mean TTFT
+28%、TPOT +56%、吞吐 −34%（E3）。收益 ∝ 负载前缀共享率。

**追问树**：

| 追问 | 答案骨架 |
|---|---|
| 中位数也降了吗？ | **没有**——median 反而 621.9→695.9ms（+12%）。但 p99 5908→2675ms（−55%）、std 1432→667 砍半。收益集中在尾部：cache 命中让长队消散，"典型请求"只省后缀 prefill 的一小段 |
| APC 只在省 TTFT 吗？ | 不是。调度模拟器显示 mnb=256 时 APC 让 TTFT −74%、吞吐 **+80%**——命中请求的 prefill 不占 token 预算，等效于放大预算（E8f） |
| 多副本下怎么让 PC 生效？ | cache-aware 路由。自写前缀亲和路由实测命中率 53.8→84.0%，但 p90 持平——并发到达时"第二波才命中"是上界，亲和倾斜要靠负载降级兜底（D3） |
| 命中率上界怎么算？ | 前缀共享率 × 到达时序。同时到达的同前缀请求第一波必 miss |

**陷阱**：只背"−28%"会被统计口径问死。主动讲 median/p99 分叉 = 真的看过分布。

## 4. 深挖题四：讲一个最难的排障（MindIE 定案）

**首答**：MindIE warmup 在 `init_ascend_weight→set_weight` 挂死 67 分钟。
9 步排障：权重加载 ✓、KV 分配 ✓、显存 41G ✓，最后用 py-spy+strace 定位到
单线程 99.9% 在 `do_mem_abort+sigreturn` 死循环。

**追问树**：

| 追问 | 答案骨架 |
|---|---|
| 怎么确定是挂死不是慢编译？ | 三个证据：kernel_meta 编译缓存零写入；DEBUG plog 只剩 1Hz 重复 32B memcpy 轮询；单线程 fault-retry 内核栈形态 |
| 怎么排除是模型特有？ | 隔离变量：Qwen3-8B、0.6B、缩 warmup shape、**TinyLlama（不同模型族不同 graph 文件）**全部同帧挂死 → ATB 算子库层问题 |
| 为什么值得写进简历？ | vendor 级 bug 定案（atb-models 2.3.0 + CANN 8.5.0 + 910B1），排障链本身就是 solid vendor ticket 材料；也暴露了闭源栈平台锁定的隐性成本（E5：绑死 glibc2.38/Ubuntu24.04） |

## 5. 深挖题五：PD 分离为什么难（E18 失败案例）

**首答**：跨节点 PD 我做了，三层全踩穿：ZMQ 元数据广播 pod IP 错误
（VLLM_HOST_IP+隧道修通）→ TE-RPC 动态端口（再补隧道）→ 数据面
device-mem 传输在当前网络形态物理不可行。

**追问树**：

| 追问 | 答案骨架 |
|---|---|
| 最危险的发现？ | 传输失败的请求仍返回 **HTTP 200**，输出是垃圾 token——PD 监控必须盯输出质量（perplexity/采样校验），不能只看状态码 |
| PD 分离的收益前提？ | prefill/decode 用不同机型/不同并行度才赚；通信税 > 拆开省的排队时才值。我们 HCCL 实测延迟底 160μs、带宽 18.5 GB/s（E8d），跨节点只会更差 |
| 重来会怎么做？ | 先单机内 PD（同卡 NIXL/共享内存路径）验证收益，再上跨节点 |

## 6. 速查数字卡（全部已核实）

| 场景 | 数字 |
|---|---|
| 并发扩展 | 4B 单卡 c1→64：75→807 tok/s；TTFT 161→797ms；TPOT 12→71ms |
| 访存墙 | 4B bf16 matvec 1475 GB/s ≈ HBM 峰值 92%；理论单流 ~180 tok/s vs 实测 82 |
| 量化入场券 | 27B bf16=71G>64G；W8A8=30G 可跑，c64 396 tok/s / TPOT 144ms |
| chunked prefill | mnb 256→4096：TTFT 13.1s→1.06s；goodput 达标率 2%→84%（E9） |
| goodput 杀招 | w8a8-27B c64：396 tok/s 里 93% 请求违反 2000x100 SLO，TTFT p99 7.2s |
| async 调度 | TPOT mean 48.4→29.4ms（−39.3%） |
| spec 负收益 | ngram 接受 8-11% TPOT×3.7-6；MTP 接受 65.6% 仍 TPOT 36.5 vs 15.6 |
| 长上下文 | 3.9k→63k：TTFT 1.39→12.2s、TPOT 19.1→61.8ms、KV 峰值 1.9→13.9% |
| 副本扩容 | 3 副本 c128：1027→2160 tok/s；TTFT 先于吞吐改善（2751→1262ms） |
| sleep/wake | 冷启 169s vs 唤醒 0.139s；共卡 stream/CQ 耗尽要 eager 兜底 |
| HCCL | 延迟底 ~160μs、渐进带宽 18.5 GB/s → TP-2 decode 每步 ~11.5ms 通信税 |
| 融合算子 | naive vs FIA @S8192 差 5.32×；手写 FA 与 naive 同速、比融合慢 24× |
| SWA 真相 | 显式 band mask 比 is_causal 慢 26× 且窗口无关——真 SWA 靠 kernel 跳块 |
| 准入控制 | 短请求 p90 4.85→3.10s（−36%），代价 44/64 快速 429 |

## 7. 陷阱题预案

**"你的数字是模拟器出的还是真实测的？"**
→ 明确区分：E8f 是 Python 模拟器（机制演示），其余全部是 910B 真实硬件 +
vLLM 0.30.0rc2 实测。模拟器数字我都标注了 simulation，绝不混用。

**"NPU 上做的，迁到 GPU 还算数吗？"**
→ 机制层全通用：访存墙、token 预算、prefix 块命中、抢占、goodput 口径
都是 vLLM V1 通用概念；硬件层数字不同（HBM 峰值、互联拓扑），我在
CUDA 侧还手写过一个 paged 引擎做对照（open-infra-ai）。

**"你行号引用的源码是什么版本？"**
→ file:line 引自 dev checkout，实测跑的是 0.30.0rc2，行号可能漂移几十行，
符号名不变可搜——文档里有显式声明。

**"负结果也拿出来讲？"**
→ spec decode 负收益是我最能讲的实验：三层验证（原生 ngram/MTP + 手写
机制复现），给出盈亏平衡公式和上线检查清单。知道"什么时候不该用"
比知道"怎么用"值钱。

**"遇到 PR/issue 先做什么？"**
→ 先搜 mainline 历史再动手。我有现成案例：一个草稿差点发成新 issue，
dedup 时发现 #18978 已覆盖同缺陷，改成带实测数据的 comment——对 maintainer
更有价值（补了 GDN 模型的 ITL 实测），也避免公开 duplicate。

## 8. 反问环节（挑 1-2 个）

- "你们 serving 的 SLO 是按 TTFT/TPOT 分还是端到端 E2E？goodput 怎么报给上层？"
- "调度器侧你们用的是 upstream scheduler 还是自研准入/路由层？"
- "对多租户/弹性，你们在用 sleep-wake 还是冷启+预热池？"
