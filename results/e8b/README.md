# E8 · 手撕微观实验：attention / KV cache / decode 访存墙（910B1 实测）

> 环境：pytorch 容器，torch 2.10 + torch_npu 2.10 + CANN 9.0，物理卡 npu:1（910B1）
> 脚本：`scripts/e8_micro.py`，原始数据：`results_e8/e8_micro.json`

这组实验全部手写 PyTorch 实现 + 实测，对应面试"手撕"层的三件事：
手推 attention → KV cache 增长 → decode 为什么是访存瓶颈。

## A. naive MHA vs 融合算子（`F.scaled_dot_product_attention`）

手写的三步 attention（matmul → softmax → matmul）vs 融合算子，bf16, H=32, D=128, B=1：

| seq_len | naive ms | fused ms | speedup |
|---|---|---|---|
| 512 | 0.12 | 0.077 | 1.55× |
| 2048 | 1.74 | 0.45 | 3.86× |
| 8192 | 35.19 | 6.61 | **5.32×** |

**机制**：naive 版本把 S×S 注意力矩阵物化到 HBM（两次写读），融合算子在片上 SRAM 内完成
softmax 不做大矩阵落盘。序列越长，naive 的访存量以 O(S²) 增长而计算只以 O(S²) 增长但
访存占比更高——融合收益随 S 单调放大。**这正是 FlashAttention 的核心思想，实测复现**。

## B. decode 单步成本 vs KV 长度

decode 一步 q=[1,H,1,D]，KV 长度增长时的单步耗时与有效带宽：

| KV len | 单步 ms | KV 显存 | 有效带宽 GB/s |
|---|---|---|---|
| 512 | 0.066 | 0.008G | 127 |
| 2048 | 0.066 | 0.034G | 507 |
| 8192 | 0.147 | 0.134G | 912 |
| 32768 | 0.538 | 0.537G | **999** |

**机制**：decode 阶段每步要重读全部 KV——单步耗时 ~ 线性于 KV 长度；长上下文时有效带宽
逼近 1TB/s（910B1 HBM ~1.6TB/s 的 62%）。**这就是长上下文 TTFT 不变但 TPOT 恶化的物理根因**，
也是 prefix caching / KV 压缩的价值所在。

## C. decode 的访存墙实测

4096×4096 权重矩阵，单请求 matvec（decode 形态）vs 256 请求 matmul（batch 形态）：

| | 耗时 | 权重读带宽 |
|---|---|---|
| decode bs=1 | 0.0228ms | **1475 GB/s**（≈HBM 峰值 92%）|
| batch 256 | 0.0446ms | 752 GB/s |

**机制**：bs=1 的 matvec 把权重完整读一遍只为 1 个 token——吞吐完全被 HBM 带宽钉死；
bs=256 同样一遍权重摊给 256 个 token，硬件转为计算瓶颈、权重读带宽反而下降。
**算术强度决定瓶颈属性——decode 天然访存 bound，batching 把它变成 compute bound**。
这直接解释了 E1 里"并发越高吞吐越高但 TPOT 恶化"的基线数据。

## 一句话总结

三个微观测量分别对应：融合算子收益来源（访存量）、KV 重读成本（线性带宽）、
decode 瓶颈本质（算术强度）——全部是面试白板推导题的实测验证。
