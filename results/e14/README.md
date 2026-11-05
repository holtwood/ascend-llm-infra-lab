# E14 — MTP speculative decoding on Ascend：接受率真实，净收益为负

> 面试原题："spec decode 什么时候有用？怎么衡量？"
> 结论：Qwen3.5-4B 自带 MTP 头（`mtp_num_hidden_layers=1`，15 个 `mtp.*`
> 权重嵌在主 checkpoint）在 vLLM 0.30 + vllm-ascend 0.30.0rc2 上**能跑通、
> 接受率真实，但当前平台下净收益为负**。

## 启用方式

```bash
vllm serve /model/Qwen3.5-4B --port 8000 --max-model-len 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

引擎日志确认：`Resolved architecture: Qwen3_5MTP`、`Loading drafter model`、
`Sharing target model embedding weights with the draft model`、
`Wrapping draft model with ACLGraphWrapper`。

已知告警（记录在案）：
- `num_speculative_tokens > 1` 会对同一 MTP 层跑多次前向 → 接受率下降
- `no KV cache group could be identified as the draft model's`（hybrid
  KV group 下草稿模型未分到独立 KV 组，机制影响待查）
- spec 开启后 `max_num_scheduled_tokens` 被压到 2048

## 结果（`bench_openai.py`，in256/out512，c8×32req，temperature=0）

| 配置 | TPOT mean | 输出吞吐 | TTFT mean | spec 指标 |
|---|---|---|---|---|
| MTP k=3 | 36.5 ms | 195.5 tok/s | 299 ms | 2781 drafts / 8343 draft tok / **5470 接受 = 65.6%** |
| 无 spec | **15.6 ms** | **479.3 tok/s** | 297 ms | — |

首请求探针：39 draft tokens / 36 接受 = 92.3%（重复性文本上限）。

## 机制解释

- 接受率不是瓶颈（65.6% 远高于 E3 ngram 的 ≤11%）。瓶颈是**每步成本**：
  k=3 时草稿头串行跑 3 次前向 + 目标模型一次 4 位置验证前向；在 910B2
  上这些串行小前向的固定开销（调度 + kernel launch + conv1d fallback
  同步）超过了平均 ~2 token/步 的接受收益（5470/2781 ≈ 1.97）。
- 净效果 = 接受 token/步 × 目标步成本 − 草稿+验证开销 < 0。
  E8c 手写实验的 `coverage × acceptance vs verify_overhead` 公式在
  生产引擎上得到验证：**接受率高 ≠ 净收益**。
- 平台侧放大器：`causal_conv1d_update` 缺 NPU triton 实现（issue3），
  每请求同步阻塞；混合架构 MTP 未获独立 KV 组。这两个是上游改进点。

## 复现

```bash
# MTP 臂
vllm serve ... --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
python3 scripts/bench_openai.py --endpoint /v1/chat/completions \
  --concurrency 8 --num-requests 32 --input-tokens 256 --output-tokens 256 \
  --out mtp_bench.json
curl -s localhost:8000/metrics | grep spec_decode   # 接受率
```

数据：`mtp_bench.json`、`nospec_bench.json`、`serve_mtp.log`、`serve_nospec.log`。
