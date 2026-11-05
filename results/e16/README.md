# E16 — 长上下文扫描：ctx 长度 → TTFT/TPOT/KV 占用曲线

> 面试原题："长上下文 serving 的成本怎么涨？瓶颈是什么？"
> 串起 E8（KV 容量账本）+ E8b（访存墙）+ E3b（chunked prefill）的闭环。

## 设置

- serve：`qwen35-4b, --max-model-len 81920`（模型上限 262144，KV 池固定）
- 负载：`scripts/e16_ctx_sweep.py`，唯一 nonce+词表 prompt（防 prefix 复用），
  out=128，低并发（c3–4）隔离单请求成本
- 注：`usage.prompt_tokens` 在 stream 模式未回（需 `stream_options.include_usage`），
  tokens 数按 1.5 tok/word 估算（与 E8 实测 5387/3600 一致）

## 结果

| 目标 ctx | ≈prompt tok | TTFT mean | TPOT mean | 峰值 KV 占用 |
|---|---|---|---|---|
| ~3.9k | 2600w | 1.39 s | **19.1 ms** | 1.9% |
| ~15.8k | 10500w | 3.61 s | 31.1 ms | 5.1% |
| ~31.5k | 21000w | 5.55 s | 36.0 ms | 7.2% |
| ~63k | 42000w | 12.2 s | **61.8 ms** | 13.9% |

## 机制解读（面试口径）

1. **TTFT 随 ctx 近似线性略超**：4k→63k = 16.2× ctx，TTFT 1.39→12.2s
   = 8.8×——chunked prefill 摊销 + 固定开销被长 ctx 稀释；prefill 本体
   是 O(ctx) 计算（hybrid 线性层也是 O(ctx)）。

2. **TPOT 随 ctx 显著恶化（19→62ms，3.2×）**：decode 每步要重读全部
   KV——63k ctx 的 KV 是 63k×48.7KB ≈ **3.1GB/请求**，每步访存量
   16× 于 4k ctx。与 E8b 微观实测"decode 单步 ∝ KV 长度、逼近 HBM
   带宽上限"完全吻合。**长上下文成本在 decode，不在 prefill**。

3. **KV 占用低但不可忽略**：3 个 63k 请求用掉 13.9% 池（~140k tokens
   ÷ 1.43M）；按此换算 63k-ctx 满载并发上限 ~20 会话——长上下文场景
   容量瓶颈从"调度准入"（E8 发现）切回"KV 容量"。

4. **`kv_cache_size_tokens` 不是纯字节换算**：同一池 45.6GiB 在
   max_len=8192 下报 1,006,322 tokens、在 81920 下报 **1,426,875
   tokens**——报告的 token 预算随 per-request 上限配置变化（mamba
   页对齐颗粒度不同），容量规划要用实测值不是公式直推。

## 容量规划速查

| 场景 | 单卡 4B 上限 | 约束 |
|---|---|---|
| 4k ctx 短对话 | ~185 会话（E8） | KV token 预算 |
| 63k ctx 长文档 | ~20 会话 | KV 字节 3.1GB/req |
| 任何 ctx | decode TPOT ∝ ctx_len | HBM 带宽 |

数据：`ctx_w2600c4.json`、`ctx_w10500c4.json`、`ctx_w21000c3.json`、
`ctx_w42000c3.json`、`serve.log`、`drv.log`。
