# E8f — continuous batching + paged-KV 机制模拟器

纯 Python 离散步模拟，复刻 vLLM V1 调度循环（walkthrough §4-5）：
**不是**性能基准，是把机制变成可运行、可改参、可面试演示的模型。
脚本：`scripts/e8f_batching_sim.py`（~200 行 stdlib）；数据：`e8f.json`。

## 负载与模型

- 128 请求，Poisson 到达；75% 共享 4 个 ~2000-tok system-prompt 模板
  + 25% 独立 300-tok 短请求；输出 96±32 tok（E3/E16 同款负载形状）
- 步进语义：每步 budget = `mnb` token——running decoder 各吃 1，
  余量给等待请求的 chunked prefill（FCFS）；`max_seqs=64` 准入上限
- KV 池：block=128 token/块，用满时**队尾抢占**、computed 归零重算
  ——对应 vLLM recompute-preemption（`scheduler.py:1487-1518`）
- APC：模板前缀命中 → `computed` 直接置满，跳过一次完整 prefill

## 结果（step 为离散时间单位）

| 场景 | makespan | 吞吐 tok/step | TTFT mean | TTFT p99 | 备注 |
|---|---|---|---|---|---|
| static 基线 | 596 | 20.6 | 129.5 | 213 | decode 死槽浪费 |
| continuous mnb=256 | **933** | **13.2** | **228.6** | 531 | 预算饿死：prefill 排队 |
| continuous mnb=1024 | 520 | 23.6 | 1.9 | 6 | 甜点 |
| continuous mnb=4096 | 518 | 23.7 | ~0 | ~0 | 离散模型下 TTFT 归零 |
| mnb=4096 + APC | 518 | 23.7 | ~0 | ~0 | 68/128 命中，预算已不紧张 |
| **mnb=256 + APC** | **517** | **23.8** | **59.7** | 142 | **本实验最值钱的格子** |
| KV 池=160 块 | 1136 | 10.8 | 279.2 | 697 | **34341 次抢占**（抖动） |

## 四个机制结论（每条可对应真实数据）

1. **continuous vs static：+15% 吞吐，-13% makespan**
   static 的浪费 = 死槽 × (max_out - 实际 out_len)。本负载 out_len 方差
   小（96±32）所以差距温和——**方差越大 CB 收益越大**，这是 E1 c1→c64
   吞吐翻 10 倍的微观解释。

2. **mnb 的双刃剑=E3b 的仿真复现**
   mnb=256 反而比 static 还差：64 decoder 吃掉 1/4 预算，2000-tok
   prefill 要 ~10+ 步才能凑齐，排队深度爆炸。对应 E3b 实测
   `mnb=256 → TTFT 13.1s`。甜点在 ~1-2× 典型 prompt 长——E3b 的
   `mnb=4096 → 1.06s` 同位置。

3. **APC 的真正收益是放大 token 预算，不只是省 TTFT**
   mnb=256 单独跑 TTFT 228.6；**+APC 后 59.7（-74%）、吞吐 13.2→23.8
   （+80%）**。机制：62/128 命中请求的 prefill 成本=0，等于把被
   decoder 挤占的预算又拿了回来。真实对应物：E3 APC（mean TTFT
   -22%）、E16 prefix 路由（命中 53.8%→84%）。
   **面试话术**：prefix caching 的收益 ∝ 共享率 × prefill 占预算比——
   预算越紧，APC 收益越大（本表 mnb256 格子 vs mnb4096 格子直接可见）。

4. **KV 池满=抢占抖动，吞吐腰斩**
   池子压到 160 块（≈20 个请求的 KV）触发 34341 次抢占，
   吞吐 23.7→10.8。抢占-重算-再被抢的恶性循环=capacity 规划的反面：
   `max_model_len × max_num_seqs` 必须落在 KV 池内，否则 SLO 崩坏——
   对应 E8 容量账本与 E16 长上下文 KV 曲线的实测。

## 简化声明（边界诚实）

- 步为离散单位，不含真实 kernel 耗时——mnb 大时 TTFT→0 是模型假象，
  真实世界大 prefill 会抬 TPOT（E3b 实测 TPOT 非单调即此）
- 抢占只重算 prompt 部分，未模拟 output-token 重算细节
- static 基线未实现"整批 prefill 一次算"的算子级合并，吞吐对比偏保守
- 单副本 FCFS——优先级/priority 调度、spec-decode、PD 分离未建
