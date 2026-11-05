# E8 — KV cache 容量账本：公式 vs 实测 + 耗尽边界压测

> 面试原题："KV cache 显存怎么算？单卡能并发多少请求？满了会发生什么？"
> 本实验给出**混合架构模型**下的非标答案 + 实测验证。

## 1. 模型架构前提：Qwen3.5-4B 是 hybrid，naive 公式错 4 倍

`/model/Qwen3.5-4B/config.json`（text_config）：

| 字段 | 值 |
|---|---|
| num_hidden_layers | 32 |
| layer_types | **24× linear_attention + 8× full_attention**（interval=4） |
| num_attention_heads / num_key_value_heads | 16 / 4（GQA，仅 full_attn 层用） |
| head_dim | 256 |
| linear 层 | 16 kv_heads×128 + 32 v_heads×128，conv kernel=4，state dtype fp32 |
| mtp_num_hidden_layers | **1**（自带 MTP 层——spec decode 资产，记录备查） |

**KV cache 只在 8 个 full_attention 层上随上下文增长**；24 个 linear
（gated-deltanet）层每请求只占一份**定长循环状态**（与上下文长度无关）。

- naive 全注意力公式（错）：`2 × 32层 × 4kv头 × 256d × 2B` = **128 KiB/token**
- hybrid 正确公式：`2 × 8层 × 4kv头 × 256d × 2B` = **32 KiB/token**

## 2. 实测账本（vLLM 0.30.0 + vllm-ascend 0.30.0rc2，910B2 64G）

来自 `/metrics` 的 `cache_config_info` 与启动日志：

| gpu_memory_util | KV 池 tokens | num_gpu_blocks | block | 有效 B/token | 标称最大并发@8192 |
|---|---|---|---|---|---|
| 0.92（默认） | **1,006,322** | 2334 | 640 | 45.60 GiB / 1.006M = **48.7 KB** | 122.84 |
| 0.50 | **441,074** | 1023 | 640 | ≈20 GiB 等效 | 53.84 |

**为什么实测 48.7KB > 理论 32.8KB（+48%）？** 统一页管理开销：

```
Setting kv cache block size to 128 for CUSTOM backend
Setting attention block size to 640 tokens (保证 attention 页 ≥ mamba 页)
Padding mamba page size by 22.14% → mamba 页与 attention 页等大
kv lcm block sizes 640
```

vLLM 对 hybrid 模型用**统一分页**：每个 640-token block 同时承载
attention KV 页和 mamba 状态页（mamba 页按 +22% 填充到与 attention
页等大）。mamba 状态按请求摊销进每 block 成本 → 有效 48.7KB/token。
即便如此仍比纯注意力模型的 131KB/token 好 **2.7 倍**。

## 3. 容量边界压测（prefix caching 开启 → prompt 必须唯一化）

压测脚本 `scripts/e8_kv_pressure.py`：每请求 16 字符 nonce + 3600 随机词
（`/tokenize` 实测 **5387 tokens**），footprint ≈ 5451 tok ≈ 9 个 640-block。

| 档位 | 并发×输出 | KV池 | ok | 峰值KV | 峰值running | 抢占 | 结论 |
|---|---|---|---|---|---|---|---|
| c64 | 64×64tok | 0.92 | 64 | 12.0% | 23 | 0 | 裕量内 |
| c384o256 | 384×256tok | 0.92 | 384 | **73.0%** | 142 | 0 | **准入受限**：调度器每步准入 ~1 个长 prompt（chunked-prefill 预算），running 稳态 < 容量界 |
| c256o256u50 | 256×256tok | 0.50 | 256 | **99.1%** | 84 | **3** | **容量受限**：KV 打满→抢占发生 |

## 4. 三个机制结论（面试可直接讲）

1. **KV 公式要先看架构**。hybrid 模型（Qwen3.5/Qwen3-Next 类）的 KV
   只随 full_attention 层数增长；naive 套 32 层公式高估 4 倍。容量
   换算：`tokens_budget × B/token` 对 `num_gpu_blocks × block` 双向核算。

2. **默认调度下，长 prefill 负载先撞的是"准入墙"不是"容量墙"**。
   c384 时峰值 running=142 < 理论容量 ~175，KV 只用 73%、0 抢占——
   chunked-prefill 的 token 预算使准入速率 ~1 长prompt/调度步，
   running 集合被完成速率平衡在容量界之下。**队列里的等待
   （peak waiting=309）表现为 TTFT 膨胀而非失败**。

3. **打满 KV 后不是立刻报错，而是抢占（preemption→recompute）**。
   util=0.5 下 running 封顶 84（≈441k tokens ÷ ~5.2k/req ✓ 与
   token 预算模型吻合），KV 99.1% 时 `num_preemptions_total` 计 3 次；
   256/256 请求全部成功，代价体现在尾部延迟（p50=111.8s vs
   p99=143.6s）。被抢占请求重新排队做 recompute prefill。

## 5. 复现

```bash
# 账本：serve 起来后看 metrics
vllm serve /model/Qwen3.5-4B --port 8000 --max-model-len 8192 \
  [--gpu-memory-utilization 0.5]
curl -s localhost:8000/metrics | grep cache_config_info

# 压测（唯一化长 prompt，防 prefix cache 合并）
python3 scripts/e8_kv_pressure.py --tag c256u50 \
  --concurrency 256 --prompt-words 3600 --max-tokens 256
```

数据：`pressure_c64.json`（0.92池·裕量内）、`pressure_c384o256.json`
（0.92池·准入受限）、`pressure_c256o256u50.json`（0.50池·容量受限+抢占）、
`serve_util50.log` / `serve_restore.log`。

> 注：c64/c384 两个 JSON 为事后重建（M3 结果目录文件曾被外部清理，
> 数值为原始驱动输出，见 docs/status.md）。
