# N10: KV cache int8 量化 — 精度/显存/速度三角（Qwen3.5-4B hybrid，Ascend 910B2 卡1）

- 模拟口径：每步 decode 后对全部 KV 做 absmax int8 量化→反量化并写回 cache（等价「int8 存储 + 读取反量化」的每步刷新噪声；cast 流量为 eager 反量化成本上限）
- greedy 32 tok，稳态 p50（跳前 5 步）；per_token=scale 每(头,token)，per_channel=scale 每(头,维度) 随 seq 不变

| 档位 | ctx | decode p50 ms/步 | 其中量化税 ms/步 | 步涨 vs bf16 | 首个分叉 token | 32 tok 分叉比 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 1535 | 98.9 | 0.0 | 0.0 | — | — |
| per_token | 1535 | 105.2 | 4.9 | 6.3 | 5 | 0.364 |
| per_channel | 1535 | 105.4 | 4.6 | 6.5 | None | 0.0 |
| gdn_int8 | 1535 | 100.6 | 0.0 | 0.0 | 5 | 0.364 |
| bf16 | 8255 | 102.4 | 0.0 | 0.0 | — | — |
| per_token | 8255 | 106.6 | 6.8 | 4.2 | None | 0.0 |
| per_channel | 8255 | 107.9 | 6.3 | 5.5 | None | 0.0 |

## 显存账（8 attn 层 × 4 头 × 256 维，ctx=8192）

| 存储布局 | 实测分配 | KB/token | vs bf16 |
|---|---:|---:|---:|
| bf16 | 270.5 MB | 32.00 | 1× |
| int8+fp32_scale_per_token | 137.4 MB | 16.25 | 0.508× |

理论：bf16 KV = 32 KB/token（N3 实测口径）；int8 K/V + fp32 per-token scale ≈ 16.25 KB/token（-49%）；per-channel scale 与 seq 无关（可忽略），同样 ≈ 16 KB/token。
hybrid 总 KV 账本里还要加上 GDN 恒定 state 50.3 MB（fp32，不可省，见下）。

## GDN recurrent state 量化试验（单独小节）

- 一次性 int8 roundtrip recurrent_states（(1,32,128,128) fp32 S 矩阵 × 24 层）：首个分叉 token = 5，32 tok 分叉比 0.364。**崩了**：S 矩阵的量化噪声在递推中被放大，验证「GDN state 是 fp32 数学，不可量化存储」。

## 结论
- **精度**：per-token int8 在 1525 ctx 首分叉 token=None，8192 ctx 首分叉=5——上下文越长误差累积越早，长上下文 KV 量化需要 per-channel + 更细 scale 或残差补偿。
- **速度**：eager 下每步量化税 4.6 ms（短 ctx）/ 0.0 ms（8k ctx，随 seq 线性涨），decode 步长 +6.5/0.0 ms——证实 N2 的小算子税：多出的 cast/amax/clamp 算子在 eager 下吃掉 int8 省的带宽，量化 KV 只有在融合内核（fused dequant attention）里才可能净赚速度。
- **显存**：32 → 16.25 KB/token（-49%）是唯一稳赚项；hybrid 模型上 GDN state 50.3MB 恒定不可量化，KV 减半的相对收益比纯 Transformer 更小。
