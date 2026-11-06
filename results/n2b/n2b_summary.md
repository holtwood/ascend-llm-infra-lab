# N2b: batch 维度扫描 — decode 白嫖 vs prefill 的实证（Qwen3.5-4B hybrid，Ascend 910B2 卡1）

- 环境：ASCEND_RT_VISIBLE_DEVICES=1，torch 2.9.0 / torch_npu / transformers 5.16.1，bf16 eager
- 统一 prompt 324 tok（海盗简报 + 一问），greedy；decode 30 步取稳态 p50（跳过前 5 步）；prefill 5 次 p50
- 脚本：`scripts/n2b_batch_scan.py` + `scripts/n2b_divergence_probe.py`；原始 log/json 同目录

## 主表

| B | decode ms/步 | vs b1 | 聚合 tok/s | 吞吐 vs b1 | 均摊 ms/tok | prefill ms | prefill vs b1 | 等长输出一致 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 110.3 | — | 9 | 1× | 110.3 | 504.0 | 1× | YES |
| 2 | 107.8 | −2% | 19 | 2.1× | 53.9 | 495.8 | 0.98× | YES |
| 4 | 110.1 | 0% | 36 | 4.0× | 27.5 | 513.8 | 1.02× | YES |
| 8 | 109.0 | −1% | 73 | 8.1× | 13.6 | 505.0 | 1.00× | YES |
| 16 | 108.3 | −2% | 148 | 16.4× | 6.8 | 832.2 | 1.65× | NO（见下） |
| 32 | 108.7 | −1% | 294 | 32.5× | 3.4 | 1583.3 | 3.14× | NO（见下） |

## 结论

1. **decode 白嫖实锤**：b1→b32 每步 110.3→108.7 ms（**batch 增大反而略快**——GEMV→GEM 后权重仍只读一次，算子下发次数不变而矩阵形状变胖），聚合吞吐 9→294 tok/s = **32.5×，白嫖系数 1.015（>100% 线性）**。与 N2 的"每步 2935 次算子下发、79% host 固定开销、device 仅忙 24ms"完全自洽：固定 host 开销不随 B 增长，device 有大量空闲算力吸收 B×计算。
2. **prefill 并非"不能白嫖"，而是"饱和点之后不能"**：324-tok × 32 = 10K tok 的 batch prefill 只从 504ms 涨到 1583ms（3.1×，远低于线性 32×），因为短 prompt prefill 也处于 host-bound 段，固定下发开销被摊薄。"prefill 不能白嫖"的准确表述是：**prefill 一旦 compute-bound（N2：8K ctx 2765 tok/s 饱和），batch 只能线性涨时间**；未饱和段（本实验的短 prompt）同样有摊薄红利。白嫖的本质是"固定开销/空闲算力可摊"，不是 decode 的专利。
3. **等长 batch 数值等价在 B≥16 破裂（确定性漂移，非随机）**：相同 prompt、零 padding 下 B≤8 全槽与 b1 逐 token 一致；B=16/32 时 logits 漂移 0.14-0.16（bf16 归约顺序随 GEMM tile 形状变化），漂移在贪心解码中累积，**B=16 全部 16 槽在第 10 步同步翻边**，其后 67.7% token 分叉。判别实验：同 B 独立重跑 bit 一致、batch 内各槽 bit 一致 → kernels 确定，差异纯由 batch 形状（tile/归约顺序）决定。**修正 E83 的边界**：E83 证"同 batch 内 identical concat bit-identical"，本实验补上"跨 batch 大小不 bit-identical"——面试题"batch 会改变输出吗"的完整答案：不会随机变，但 bf16 下大 batch 的 greedy 轨迹可能在中途翻边（噪声量级，非 bug；需要逐位复现时应固定 batch 组合）。
4. **left-padding 语义性失效（hybrid 特有坑）**：B≥2 的左 pad 组全部失败且**连无 pad 的槽 0 都从第 0 步分叉**——机制有二：①GDN 层没有 attention-mask 概念，pad token 直接进入 conv 滑窗与 recurrent 状态递推（槽 0 右侧的补齐 pad 在 prefill 末位污染 state；左 pad 污染所有槽的滑窗起点）；②batch 序列长度变化触发不同内核分派路径。纯 Transformer 的 left-padding 只是 position_ids 对齐问题，hybrid 模型上是语义破坏——batch 化 serving 必须 varlen 路径（vLLM 混合模型走 cu_seqlens 的原因）。

## 附：吞吐/延迟曲线要点

- decode 均摊 ms/tok：110.3 → 3.4（32.5×），几乎完美的 B 倒数曲线——host 固定开销是唯一"不免费"的部分。
- prefill 均摊 ms/tok：504/324=1.56 → 1583/10208=0.155 ms/tok（b32），短 prompt 场景 prefill 也有 10× 摊薄红利。
