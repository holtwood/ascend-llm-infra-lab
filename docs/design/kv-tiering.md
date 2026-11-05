# 设计题：KV Cache 分级存储怎么设计？

> 面试设计题形态：不给标准答案，给**实测常数 → 架构推导 → 决策边界**。
> 所有常数来自本仓库实测（标注出处），不是估的。

## 一、问题约束（实测常数表）

| 层 | 实测带宽 | 延迟底 | 容量 | 出处 |
|---|---|---|---|---|
| NPU HBM | ~1475 GB/s（910B1 decode matvec 实测，910B2 理论 1.6TB/s） | ~ns 级 | 64G/卡 | E8b |
| HCCS（卡间） | ~18.5-20 GB/s | ~160μs | 同机 8 卡 | E8d/E8i |
| Host DRAM↔HBM | H2D ~24 GB/s；D2H pinned ~21 / pageable ~7-11 GB/s（实测） | ~μs | TB 级 | 本文件末节 |
| 跨节点网络 | 未测 | ≥ms 级 | — | E18 证伪过 device-mem 传输 |

KV 每 token 字节（4B GQA 模型，Hkv=4, D=128, bf16, 36 层估算）：
`2(KV) × 4头 × 128 × 2B × 36层 ≈ 73.7 KB/token`（E8e 账本同口径）。

64k 上下文单请求 KV ≈ **4.7 GB**（实测 63k KV 峰值时 E16 数字吻合）。

## 二、推导：每层能装什么、什么时候换层

### L1 HBM：只装"热 KV"

- 64G 卡刨去权重（4B bf16 ~9G / 27B-W8A8 ~30G），KV 池按 gpu_utilization
  分配——E16 实测 63k ctx 时 KV 占用 13.9%
- **决策**：当前活跃请求的工作 KV 必须在 HBM——decode 每步全量读，
  带宽墙实测 1475GB/s 是吞吐上限的来源，换到下层直接掉一个数量级

### L2 HCCS 邻居卡 / DRAM：温 KV（被抢占/闲置会话）

- HCCS 18.5GB/s → 读回 4.7G 需 ~250ms；DRAM↔HBM 实测 ~21-24 GB/s →
  单向 ~200ms（pageable D2H 慢至 ~7GB/s，必须 pinned buffer）
- **触发条件**：抢占驱逐（E8f 模拟显示 KV 耗尽→34341 次抢占吞吐腰斩）、
  会话空闲（E17 sleep/wake 模型：权重 offload 类比，KV 同理）
- **设计**：preempt 时 swap-out 到 DRAM 而非重算——实测带宽下换入 ~200ms：
  长 prompt（8k+，重算 ~秒级）选 swap 划算；短 prompt（<1-2k，重算
  ~百毫秒级）选 recompute——这正好解释了 vLLM 默认 recompute 的合理性
  边界：短负载下 swap 是纯开销，而 KV-offload 系统的价值全在长前缀负载

### L3 远端/跨节点：冷 KV（prefix 池）

- E18 实测：跨节点 device-mem 传输物理不可行（我们环境）→ 冷 KV 只能
  走 host 侧网络拷贝，延迟 ms 级
- **适用**：prefix cache 跨实例共享（D3 路由命中率 84% 的另一条路——
  不亲和调度而是 KV 池共享），llm-d/Mooncake 的 KV pool 就是这个
- **决策边界**：TTFT 收益 = 省 prefill 算力(E3: +28%) − 换入延迟；
  前缀 >4k token 且命中率 >50% 时才值得（我们的 E3+D3 数据联合推）

## 三、面试速答（30s）

> "KV 分级按温度分层：热 KV 在 HBM（decode 每步读，1475GB/s 实测带宽墙），
> 温 KV（被抢占/空闲会话）换到 DRAM（60ms 回换 vs 重算按 prompt 长选），
> 冷 KV（prefix 池）跨实例共享只在高共享率时划算（实测命中 84% 但受限于
> 网络延迟）。换层的经济学：收益 = 省的算力 − 搬运延迟×IO 频率。"

## 四、和真实系统对照

| 系统 | 对应层 | 我们验证过的 |
|---|---|---|
| vLLM swap/recompute preempt | L1→L2 | E8f 模拟抢占抖动机理 + E8h metrics 零抢占观测 |
| vllm-ascend sparse_kv_offload | L1→L2(DRAM) | 源码走读过，SFA/DeepSeek 专属（不适用 Qwen3.5） |
| llm-d precise-prefix routing | L3 | D3 亲和路由实测（命中率上界=到达时序） |
| Mooncake KV pool | L3 | E18 跨节点传输在本环境证伪 → 单机内才有戏 |

## 补充实测：DRAM↔HBM 拷贝（910B1, torch copy_）

| 大小 | pinned H2D | pinned D2H | pageable D2H |
|---|---|---|---|
| 16MB | 21.0 | 19.0 | 11.0 GB/s |
| 256MB | 23.7 | 21.0 | 6.9 GB/s |
| 1GB | 23.7 | 21.0 | 7.2 GB/s |

- pageable D2H 只有 pinned 的 1/3——做 KV offload 必须 pin_memory buffer 池
- 24 GB/s ≈ HBM 实测带宽的 1/60 → 换层是 2 个数量级的带宽落差

## 遗留测量

- [ ] B2 独占时 decode 带宽干净值（E8-B1B2 共卡修正）
