# E20 — SLO 感知控制面：准入控制 + 短/长分道 + 前缀亲和

> 面试题：**"高峰期是该让所有请求排队，还是拒绝一部分？"**
> 数据回答：自写控制面（stdlib）实测——准入 cap 让被接纳请求的
> **short p90 从 4.85s 压到 3.10s（−36%）**；代价是 44/64 请求拿到
> <1ms 的快速 429。**排队把成本均摊给所有人，拒绝把成本集中给后来者。**

## Artifact：`scripts/e20_control_plane.py`

把仓库四个实验的机制合成一个可运行网关（纯 stdlib，D 节点）：

| 机制 | 来源 | 实现 |
|---|---|---|
| 前缀亲和路由 | D3 | prompt 前 256 词 sha256 → 固定后端 |
| 短/长请求分道 | E19/SRF | prompt ≤192 词 → short 通道 |
| 准入控制 | E9 goodput | per-class inflight cap → 429+Retry-After |
| 可观测性 | E8 metrics | `/stats`：by_class/rej/inflight/lat p50p90 |

## 数据（混合负载 32 short + 32 long，c32，四副本拓扑）

| 臂 | short ok | short p90 | long ok | long p90 | 429 |
|---|---|---|---|---|---|
| nginx（无控制） | 30/32 | **4.85s** | 34/34 | 5.39s | 0 |
| 控制面（l8/s12 cap） | 12/32 | **3.10s (−36%)** | 8/32 | 3.67s (−32%) | 44 |
| 控制面 wall time | — | **3.7s** vs 基线 8.9s | — | — | — |

## 机制解释

1. **为什么排队更糟**：64 请求同时到达，后端每步准入 ~1-2 个——
   基线把 60+ 请求塞在 nginx/客户端连接里干等，short 被排在 long
   后面（FCFS），p90 被队尾拖住。控制面立即 429 → 客户端可立刻
   转投其他端点或退避，**把"等"变成了"换"**。
2. **为什么被接纳的更快**：inflight ≤cap ⇒ 引擎侧 running 集不超过
   吞吐拐点，E9 的 goodput 曲线不掉入"排队死亡区"。
3. **429 的语义价值**：HTTP 429 + `retry_after_ms` 是可重试信号，
   对 SDK/调用方来说比连接超时/静默挂起友好一个量级——这是
   "fail fast"原则在 LLM 服务层的落地。
4. **分道的作用**：short-cap 12 与 long-cap 8 独立——long 突发不
   会挤占 short 名额（类 SRF，但做在网关而非调度器内）。

## 面试口径

- "过载时排队不是免费的——E8 证明 vLLM 的 waiting 队列吸收溢出，
  但代价是 TTFT 里全是排队时间。准入控制把 429 变成显式背压信号"
- "这个 demo 把四个实验串成一个系统：命中亲和(D3)、SRF 分道(E19)、
  goodput 准入(E9)、指标观测(E8)——每个机制都有独立实验背书"
- "生产版该加：令牌桶限速、租户权重、动态 cap（跟 p99 反馈调节）"

## 复现

```bash
# D 节点（后端=4 副本隧道口）
python3 e20_control_plane.py --port 8092 \
  --backends 127.0.0.1:8010,127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003 \
  --short-threshold 192 --long-cap 8 --short-cap 12

python3 e20_mixedload.py --entry http://127.0.0.1:8092/v1/completions \
  --tag arm_cp --n 64 --concurrency 32 --frac-short 0.5
curl -s http://127.0.0.1:8093/   # 控制面统计
```
