# N2b: batch 维度扫描 — Qwen3.5-4B hybrid（Ascend 910B2 卡1）

- 统一 prompt 324 tok（海盗简报），greedy；decode 30 步取稳态 p50（跳过前 5 步）；prefill 5 次 p50
- 参照：b1 decode 110.3 ms/步（与 N1b 的 89ms/步同量级）；prefill 单条 504.0 ms

| B | decode ms/步 | 每步涨 vs b1 | 聚合 tok/s | 吞吐 vs b1 | 均摊 ms/tok | prefill ms | prefill vs b1 | 等长输出一致 | 左pad输出一致 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 110.3 | +0% | 9 | 1.00× | 110.34 | 504.0 | 1.00× | YES | — |
| 2 | 107.8 | +-2% | 19 | 2.11× | 53.9 | 495.8 | 0.98× | YES | NO [0, 1] |
| 4 | 110.1 | +-0% | 36 | 4.00× | 27.51 | 513.8 | 1.02× | YES | NO [0, 1, 2] |
| 8 | 109.0 | +-1% | 73 | 8.11× | 13.63 | 505.0 | 1.00× | YES | NO [0, 1, 2, 3, 4, 5, 6, 7] |
| 16 | 108.3 | +-2% | 148 | 16.44× | 6.77 | 832.2 | 1.65× | NO | NO [0, 1, 2, 3, 4, 5, 6, 7] |
| 32 | 108.7 | +-1% | 294 | 32.67× | 3.4 | 1583.3 | 3.14× | NO | NO [0, 1, 2, 3, 4, 5, 6, 7] |

## 结论要点

- decode：b1→b32 每步 110.3→108.7 ms（+-1%），聚合吞吐 9→294 tok/s（32.7×）——接近免费午餐的实锤。
- prefill：b1→b32 504.0→1583.3 ms（3.1×，理想线性 32×）——compute-bound，batch 无白嫖。
- **left-padding 语义检查失败**（最小 B=2 即出现，首个分叉槽 0 第 0 步）：hybrid 的 GDN 层没有 attention-mask 概念，左 pad 进入 conv 滑窗与 recurrent 状态递推（prefill 阶段右侧补齐 pad 同样污染状态），attention_mask 只挡住全注意力 8 层。对纯 Transformer 这是位置对齐小坑，对 hybrid 是语义性破坏——batch 化 serving 必须 varlen/无 pad 路径（vLLM-ascend 混合模型走 cu_seqlens 即为此因）。
- 等长组一致性也有失败（见上表），batch 数值等价本身破裂，需查 attention mask 路径。
