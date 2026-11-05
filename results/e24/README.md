# E24 — MTP 大 batch 验证：不存在"摊薄拐点"

> 追问链：**"spec decode 接受率 89% 都亏，什么场景能赚？"** →
> "batch 大了会不会摊薄草稿头开销？"
> 数据回答：**不会**。c8→c64 四档实测，MTP k=1 吞吐始终约为无 spec 的
> 45-55%、TPOT 2.1-2.2×；草稿头是完整 1 层 transformer，其成本随 batch
> 同比例增长，**摊薄逻辑不成立**。

## 数据（同负载 in256/out96，bench_openai，M3 910B2）

| 并发 | 无 spec 吞吐 | 无 spec TPOT | MTP k=1 吞吐 | MTP TPOT | 吞吐比 |
|---|---|---|---|---|---|
| c8（E15） | 479.3 tok/s | 15.6ms | 239.1 | 27.6ms | 0.50× |
| c32 | **1256.8** | 20.5ms | 563.0 | 44.2ms | 0.45× |
| c64 | **1533.6** | 33.1ms | 846.2 | 57.9ms | 0.55× |

MTP 全程接受率 88.3%（11494/13013 draft tokens），与 c8 的 88.7% 一致。

## 机制解释

1. **为什么摊薄不成立**：draft 头（`mtp_num_hidden_layers=1`）是一次完整的
   transformer 层前向，其 FLOPs 随 batch 线性增长；target 步也随 batch 线性
   增长——两者比值是常数，**batch 不改变成本比**。
2. **为什么 c64 反而略好（0.55 vs 0.45）**：大 batch 下 target 步的
   memory-bound 特性更强（算力利用率更低），draft 头相对"更便宜"一点，
   但远不足以翻正。
3. **要翻正需要什么**（面试延伸）：更小的 draft 结构（单层线性/浅 MLP）、
   草稿独立 KV 组（当前日志警告 `no KV cache group configured for spec`）、
   或 draft 与 target 共享权重的早期退出——都不是本模型的 MTP 形态。

## 面试口径

- "MTP 在 910B2 上 c8-c64 全负：接受率 89% 也没用，因为草稿头是整层
  transformer，成本随 batch 同比例涨——**接受率不是收益，成本比才是**"
- "什么时候能赚？草稿成本 < target 单步成本的 1/(k+1) 量级时；MTP 的
  1 层头在 4B 模型上就满足不了这个条件"（给出可判定的条件）
- 方法论：负收益实验要做**参数扫描**证明不是配置问题——k∈{1,2,3}×c∈{8,32,64}
  全覆盖后，"这个硬件+模型上不可行"才站得住

## 复现

```bash
# 无 spec 基线
vllm serve /model/Qwen3.5-4B --served-model-name qwen35-4b --port 8000 --max-model-len 8192
# MTP k=1
vllm serve ... --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
python3 scripts/bench_openai.py --base-url http://127.0.0.1:8000 \
  --concurrency 64 --num-requests 128 --input-tokens 256 --output-tokens 96
```
