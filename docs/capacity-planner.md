# Serving 容量规划器（基于实测常数）

> 工具：`scripts/serving_capacity_planner.py`
> 常数来源：本仓 `results/overnight_20261105_160244/`（910B1 + Qwen3-0.6B 真机）
> 用法：给定模型规模/上下文/SLO，输出**容量、延迟、goodput**三件套。

## 模型（每一项都可追溯到实测）

```
KV/token        = 2 · 层数 · n_kv · d_head · 2B                    (E30，真分配验证)
最大并发         = (可用HBM − 权重 − 开销) / (KV/token · ctx)        (E30)
decode 步时      = 权重读 + attention读 + 发射地板
  权重读        = 权重GB / 1412 GB/s                               (E29 裸 matvec)
  attention读   = 并发 · KV/token · ctx / (1412 · 效率)
  效率          = 19%（bs=1）→ 50%（bs=16）→ 86%（bs≥64）           (E37/E39/E52)
  发射地板      = 29 ms（eager）；图捕获后 ≈ 0                       (E29)
prefill         = 层数 · (0.10ms + 0.09µs·L + 4.66e-8·L²)          (E48)
TTFT            = prefill + 一步 decode
goodput         = 满足 TPOT SLO 的最大并发 × 该并发下的吞吐
```

## 实测标定的三个输出样例

### ① 0.6B / 8k 上下文 / TPOT ≤ 50ms

```
KV/token            114,688 B
最大并发            61 seqs
decode 步时         111.2 ms（权重 1.1 + attention 81.2 + 地板 29.0）
TPOT SLO            在最大并发下 FAIL
SLO 运行点          13 seqs（步时 49.8ms）→ goodput 261 tok/s
```

**这就是"容量 ≠ goodput"**：能装 61 路，但只有 13 路能满足 50ms 的 TPOT。
把并发堆到 61 会让每用户只剩 9 tok/s——**长上下文高并发是幻觉**。

### ② 0.6B / 2k 上下文 / TPOT ≤ 50ms

```
最大并发            244 seqs
SLO 运行点          59 seqs（步时 49.7ms）→ goodput 1,187 tok/s
```

上下文从 8k 降到 2k：最大并发 4×（61→244），goodput 4.5×（261→1187）。
**上下文长度是容量的第一性参数**，不是"次要配置"。

### ③ 8B bf16 / 8k / TPOT ≤ 100ms / 已图捕获

```
最大并发            35 seqs（权重 16GB 挤掉 14GB 的 KV 空间）
decode 步时         71.2 ms（权重 11.3 + attention 59.9 + 地板 0）
TPOT SLO            PASS
goodput             492 tok/s
```

8B 与 0.6B 在 8k 下的对比很有意思：8B 的权重读贵 10×（11.3 vs 1.1ms），
但 attention 反而便宜（36 层但 KV 头相同 → 每 token KV 只多 28%）。
**结论：模型越大，decode 越"权重读受限"；上下文越长，越"KV 读受限"。**

## 面试用法

被问"这张卡能服务多少用户"，标准答法分四步：

1. **算容量**：`可用HBM / (KV/token × ctx)` → 能装多少
2. **算步时**：权重读 + KV 读（用实测的 attention 效率，不是理论带宽）
3. **对 SLO**：步时是否满足 TPOT 目标；不满足就往下扫 batch
4. **给 goodput**：满足 SLO 的最大并发 × 吞吐

这四步本工具都实现了，且每个系数都有实测出处——被追问"这个 19% 效率哪来的"
可以直接指到 E37。

## 实测校验（E52）

| bs | ctx | 实测 | 规划器 | 误差 |
|---|---|---|---|---|
| 1 | 4k | 29.5 ms | 31.8 ms | −7% |
| 8 | 8k | 30.6 ms | 46.0 ms | −34%（该区间被 29ms 地板支配，误差不敏感） |
| 16 | 8k | 31.0 ms | 51.4 ms | −40% |

高 batch 的 attention 效率由实测反推校准（bs=64 时 86%）。
**注意：bs≤16 时步时恒为 ~30ms，地板掩盖一切——规划器在低并发区的 attention
项基本不可观测。**


## 容量矩阵（可直接读表）

生成：`python3 scripts/overnight/tasks/capacity_matrix.py [--graph-captured]`
单位 = goodput tok/s（括号内为满足 SLO 的并发数）。

### 图捕获后（真实引擎的常态，如 vLLM ACLGraph）

# 容量矩阵（goodput tok/s @ SLO，graph-captured）


## TPOT SLO = 50 ms

| model | ctx 2048 | ctx 8192 | ctx 32768 |
|---|---|---|---|
| 0.6B (28L, 8kv, bf16) | 5,056 (244seq) | 1,232 (61seq) | 100 (5seq) |
| 4B (36L, 8kv, bf16) | 3,541 (168seq) | 642 (32seq) | 52 (2seq) |
| 8B (36L, 8kv, bf16) | 3,044 (142seq) | 510 (25seq) | 46 (2seq) |
| 8B w8a16 (36L) | 3,356 (158seq) | 588 (29seq) | 50 (2seq) |
| 32B (64L, 8kv, bf16) | OOM | OOM | OOM |

## TPOT SLO = 100 ms

| model | ctx 2048 | ctx 8192 | ctx 32768 |
|---|---|---|---|
| 0.6B (28L, 8kv, bf16) | 5,056 (244seq) | 1,232 (61seq) | 178 (15seq) |
| 4B (36L, 8kv, bf16) | 3,541 (168seq) | 732 (42seq) | 103 (10seq) |
| 8B (36L, 8kv, bf16) | 3,044 (142seq) | 604 (35seq) | 86 (8seq) |
| 8B w8a16 (36L) | 3,356 (158seq) | 679 (39seq) | 95 (9seq) |
| 32B (64L, 8kv, bf16) | OOM | OOM | OOM |

## TPOT SLO = 200 ms

| model | ctx 2048 | ctx 8192 | ctx 32768 |
|---|---|---|---|
| 0.6B (28L, 8kv, bf16) | 5,056 (244seq) | 1,232 (61seq) | 178 (15seq) |
| 4B (36L, 8kv, bf16) | 3,541 (168seq) | 732 (42seq) | 103 (10seq) |
| 8B (36L, 8kv, bf16) | 3,044 (142seq) | 604 (35seq) | 86 (8seq) |
| 8B w8a16 (36L) | 3,356 (158seq) | 679 (39seq) | 95 (9seq) |
| 32B (64L, 8kv, bf16) | OOM | OOM | OOM |

### eager（无图捕获，29ms 发射地板）

# 容量矩阵（goodput tok/s @ SLO，eager (29ms floor)）


## TPOT SLO = 50 ms

| model | ctx 2048 | ctx 8192 | ctx 32768 |
|---|---|---|---|
| 0.6B (28L, 8kv, bf16) | 2,061 (103seq) | 261 (13seq) | 23 (1seq) |
| 4B (36L, 8kv, bf16) | 1,163 (58seq) | 83 (4seq) | SLO✗ |
| 8B (36L, 8kv, bf16) | 502 (25seq) | 41 (2seq) | SLO✗ |
| 8B w8a16 (36L) | 862 (43seq) | 63 (3seq) | SLO✗ |
| 32B (64L, 8kv, bf16) | OOM | OOM | OOM |

## TPOT SLO = 100 ms

| model | ctx 2048 | ctx 8192 | ctx 32768 |
|---|---|---|---|
| 0.6B (28L, 8kv, bf16) | 3,158 (244seq) | 777 (61seq) | 92 (9seq) |
| 4B (36L, 8kv, bf16) | 2,198 (168seq) | 486 (42seq) | 51 (5seq) |
| 8B (36L, 8kv, bf16) | 1,877 (142seq) | 403 (35seq) | 42 (4seq) |
| 8B w8a16 (36L) | 2,077 (158seq) | 451 (39seq) | 50 (5seq) |
| 32B (64L, 8kv, bf16) | OOM | OOM | OOM |

## TPOT SLO = 200 ms

| model | ctx 2048 | ctx 8192 | ctx 32768 |
|---|---|---|---|
| 0.6B (28L, 8kv, bf16) | 3,158 (244seq) | 777 (61seq) | 132 (15seq) |
| 4B (36L, 8kv, bf16) | 2,198 (168seq) | 486 (42seq) | 80 (10seq) |
| 8B (36L, 8kv, bf16) | 1,877 (142seq) | 403 (35seq) | 66 (8seq) |
| 8B w8a16 (36L) | 2,077 (158seq) | 451 (39seq) | 73 (9seq) |
| 32B (64L, 8kv, bf16) | OOM | OOM | OOM |

**读表要点**：
- 同一模型，上下文 2k→32k，goodput 掉 **28-50×**（0.6B：5056 → 100 @50ms）
- SLO 从 200ms 收紧到 50ms，8B@8k 的 goodput 从 604 掉到 510（图捕获下影响小），
  但 eager 下从 403 掉到 41（10×）——**地板会放大 SLO 收紧的代价**
- 32B bf16 在 64GB 卡上直接 OOM（权重就 64GB）→ 必须量化或换卡，
  这是"模型选型先于调优"的定量提醒
- 8B w8a16 相比 bf16：goodput +10%（图捕获 588 vs 510 @8k/50ms），
  与 E41 的"容量收益为主"一致

## 已知局限

- attention 效率曲线（19%→50%）是 0.6B/GQA-8 头实测，其他形状需重测
- 未建模：批内 prefill/decode 混跑的干扰（E44 显示 8k chunk 可致 16-34×
  膨胀）、抢占、prefix 复用
- eager 地板 29ms 是"无图捕获"的实测值；真实 vLLM-Ascend 有 ACLGraph，
  应按 `--graph-captured` 计算
