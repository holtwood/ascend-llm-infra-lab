# N3: Hybrid 架构（Gated DeltaNet + Gated Attention）显存曲线实测

环境：Qwen3.5-4B bf16，910B2 单卡，transformers 5.16.1 eager，`logits_to_keep=1`。
方法：各长度整段 prefill 后遍历 `DynamicCache` 内全部 tensor，按名字分类统计（key/value → attn KV；recurrent/conv/state → GDN state）。脚本 `n3_kv_curve.py`。

## 主表：cache 字节 vs 上下文长度

| ctx_tokens | attn_KV(MB) | GDN_state(MB) | KV_theory(MB) | prefill wall(ms) |
|---:|---:|---:|---:|---:|
| 512 | 16.0 | 49.5 | 16.0 | 500 |
| 2048 | 64.0 | 49.5 | 64.0 | 821 |
| 8192 | 256.0 | 49.5 | 256.0 | 2 102 |
| 32768 | 1 024.0 | 49.5 | 1 024.0 | 7 394 |
| 65536 | 2 048.0 | 49.5 | 2 048.0 | 31 928 |
| 131072 | 4 096.0 | 49.5 | 4 096.0 | 77 021 |

理论公式（4B 模型）：`KV = 8 attn层 × 2(K,V) × L × 4 KV头 × 256 dim × 2B = 32 KB/token`；
`GDN state = 24 层 × 32 V头 × 128 × 128 × 2B(bf16) = 24 MiB 恒定`。

## 结论

1. **KV 六个长度点实测与公式逐字节吻合（16.0→4096.0 MB，斜率恒 32 KB/token）**——只有 8/32 层产生随长度增长的 KV cache。
2. **GDN state 恒定 49.5 MB，与上下文长度完全无关**（512→131072 一字不变）。为 bf16 理论值 24 MiB 的 2.06× → 框架按 fp32 存储 recurrent state（GDN 数学 fp32 计算，数值稳定的实现细节——调研与实现一致）。
3. **对照**：同规模全注意力模型（32 层全 attn、同 KV 配置）斜率应为 128 KB/token，131072 上下文时 KV = 16 GB vs hybrid 4 GB——**3:1 hybrid 布局把 KV 增长斜率压到 1/4**。
4. **显存剖面（131072 ctx）**：权重 7.93 GB + KV 4.1 GB + GDN state 49.5 MB ≈ 12.1 GB——61 GB 卡还剩 ~49 GB 给 batch/激活。全注意力模型此时 KV 已 16 GB，剩 ~37 GB。
5. **prefill 时间拐点**：512→32768（64×）时间涨 ~15×（近线性，GDN chunk 主导）；65536→131072（2×）时间涨 2.41×（超线性）——8 层 attn 的 O(L²) 项开始显现。hybrid = 24 层 O(L) + 8 层 O(L²)。
6. 实现细节（面试加分）：DynamicCache 每层携带 `conv_states`（CausalConv1d k=4 的滚动窗口）+ `recurrent_states`（S 矩阵），两者都属于"decode 状态"，attention 层则是标准 key/value。

## 坑记录

- 全序列 lm_head 会在 65536 ctx 时申请 65536×248320×2B ≈ 30 GB logits 直接 OOM——必须 `logits_to_keep=1`。这也是 vLLM 类引擎只对最后位置算 logits 的原因的直观体现。

## 面试讲法

"Qwen3.5-4B 的 KV cache 为什么省？" → 我实测过：3:1 hybrid 布局下 32 KB/token vs 全注意力 128 KB/token，131k 上下文时 4 GB vs 16 GB；GDN 的固定 state（fp32 存储实测 49.5 MB）不随长度变。prefill 时间曲线还能看到 24 层线性 + 8 层二次的合成拐点。
