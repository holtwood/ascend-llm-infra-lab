# E15 — MTP spec-decode 参数扫描：接受率随 k 降，净收益全负

> 面试题：**"spec decode 接受率 60%+，为什么还是亏的？"**
> 数据回答：910B2 上 MTP k∈{1,2,3} 三档实测——
> **接受率 88.7%→75.8%→65.6% 递减，TPOT 27.6→36.1→36.5ms 递增，
> 但即使最优的 k=1 也比无 spec 差 1.8 倍**。接受率≠收益。

## 数据（sonnet 类负载，c8，in256/out96，同 bench_openai.py）

| 配置 | TTFT mean | TPOT mean | 输出吞吐 | 接受率 |
|---|---|---|---|---|
| 无 spec（E14 基线） | 297.5ms | **15.6ms** | **479.3 tok/s** | — |
| mtp k=1 | 330.8ms | 27.6ms | 239.1 | 88.7% (1455/1640) |
| mtp k=2 | 455.7ms | 36.1ms | 170.3 | 75.8% (1850/2442) |
| mtp k=3（E14） | 299.1ms | 36.5ms | 195.5 | 65.6% (5470/8343) |

接受率口径：accepted_tokens / draft_tokens（k=1 时每 draft 1 token；
k=3 时 3 token/draft，接受率按 token 计）。

## 机制解释

1. **k 越大接受率越低是定义使然**：第 i 个草稿 token 的正确率随外推
   步长衰减（误差累积），token 口径的接受率被尾部拖低。
2. **收益公式**：净增益 ≈ 接受率×(k−1)/k 省下的步数 − draft 头 k 次
   前向成本。910B2 上 MTP 头是完整 1 层 transformer，k 次串行前向
   的固定开销 >> 单次 decode 步——**草稿头本身的 FLOP 成本在小
   batch/小模型上不可摊薄**。
3. **为什么 k=1 都亏**：k=1 时 verify 只省 0~1 步，却固定多付 1 次
   draft 前向——分子接近 0，分母非零。
4. **架构细节**：日志警告 `no KV cache group configured for spec`——
   MTP 草稿与 target 未分 KV 组，调度器按普通请求管理，进一步稀释收益。

## 面试口径

- "spec decode 上线先算收益公式：接受率×节省步数 vs draft×k 开销——
  在 NPU 上单层 draft 头的固定前向成本让 89% 接受率都是负收益"
- "k 要扫参：单调恶化说明草稿头成本是常量项，不是摊薄项"
- 对照 E3 ngram（≤11% 接受率）：**接受率从 11% 提到 89% 仍然亏**——
  瓶颈根本不在接受率，在草稿前向成本

## 复现

```bash
vllm serve /model/Qwen3.5-4B --served-model-name qwen35-4b --port 8000 \
  --max-model-len 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'  # k=1
python3 scripts/bench_openai.py --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions --model qwen35-4b \
  --concurrency 8 --num-requests 32 --input-tokens 256 --output-tokens 96
curl -s :8000/metrics | grep spec_decode  # drafts/draft_tokens/accepted_tokens
```
