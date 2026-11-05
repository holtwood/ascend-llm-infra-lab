# E8g — 手写 FlashAttention：online-softmax 分块算法

910B1（pytorch 容器 npu:1，torch 2.10 + torch_npu）。
脚本：`scripts/e8g_flash_attention.py`；数据：`e8g.json`。
"手撕 transformer"系列收尾件（E8b 融合 vs naive → E8e RoPE/GQA/SWA
→ E8f 调度器 → 本篇 attention 核心算法）。

## 实现

`flash_manual(q,k,v,Bc)`：K/V 按 Bc 分块顺序处理，逐块维护
running max `m_i`、归一化分母 `l_i`、输出累积 `O_i`——真
online-softmax，不是"切小块再 softmax"的伪 FA：

```
m_new = max(m_i, rowmax(Q K_jᵀ/√d))
P     = exp(S_ij - m_new)
l_i   = l_i·exp(m_i - m_new) + rowsum(P)
O_i   = O_i·exp(m_i - m_new) + P V_j
```

## 结果

### 正确性（vs `sdpa(is_causal)`，bf16 输入 fp32 累加）

| 形状 | max diff |
|---|---|
| S512 Bc256 | 1.56e-2 |
| S1024 Bc128 | 7.81e-3 |
| S777 Bc256（非整除边界） | 1.56e-2 |
| S2048 Bc512 | 1.56e-2 |
| 非因果 S1024 | 1.95e-3 |

bf16 量级一致（~1e-2），online 重缩放与单次 softmax 的舍入差——
算法正确。非整除边界（777/256）也过。

### 性能（ms/iter，B1 H8 D64）

| S | naive 物化 | sdpa 融合 | 手写分块 Bc512 |
|---|---|---|---|
| 2048 | 0.99 | 0.172 | 2.62 |
| 4096 | 5.51 | 0.302 | 5.10 |
| 8192 | 22.98 | 0.937 | 21.02 |

### 显存账本（score 矩阵峰值，fp32）

| S | naive S² | 分块 S·Bc | 压缩比 |
|---|---|---|---|
| 2048 | 16.8 MB | 4.2 MB | 4× |
| 8192 | 268.4 MB | 16.8 MB | **16×** |

## 三个诚实的结论

1. **算法对、数学对**——分块 online-softmax 与全量 softmax 在 bf16
   噪声内逐位等价；这就是"手写 flash attention"白题的正确答案形状。

2. **Python 层分块省显存不省时**——手写分块与 naive 同速（~21ms），
   因为两者的 matmul 都落在 HBM 上；真 FA 的加速来自 **kernel 把
   分块驻留在 SRAM/UB、消除 S² 物化**，这个层 Python 够不到。
   和 E8e 的 SWA 负结果是同一条规律：**"省什么"取决于资源在哪一级
   被消掉**——FA 省的是片上缓冲占用+kernel 融合，不是 Python 层少算。

3. **面试话术**：被问"实现 flash attention"，答三层——
   online-softmax 递推式（能写）、S²→S·Bc 显存账（能算）、以及
   "为什么 Python 写出来不快"（片上驻留+消除物化是 kernel 层的事，
   本实验 24× 差距即证据）。三层都答到的人不多。

## 复现

```bash
python3 -u e8g_flash_attention.py --device npu:1 --json-out e8g.json
```
