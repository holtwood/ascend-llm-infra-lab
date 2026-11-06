# N10: KV cache int8 量化 — 精度/显存/速度三角（Qwen3.5-4B hybrid，Ascend 910B2 卡1）

- 环境：ASCEND_RT_VISIBLE_DEVICES=1，torch 2.9.0 / torch_npu / transformers 5.16.1，bf16 eager
- 口径：每步 decode 后对 8 个 attn 层全部 KV 做 absmax int8 量化→反量化并写回 cache（模拟"int8 存储 + 读取反量化"，每步刷新噪声；cast 流量是 eager 反量化成本的诚实上限）。per_token=scale 每(头,token)（fp32，随 seq 线性），per_channel=scale 每(头,维度)（与 seq 无关）
- prompt：海盗 persona + 事实清单（含密钥 4F2C），short=1535 tok / long=8255 tok，greedy 32 步（主表）+ 128 步交叉验证（n10b）
- 脚本：`scripts/n10_kv_int8.py` + `scripts/n10b_long_gen_check.py`

## 主表（32 步，N10）+ 128 步交叉验证（N10b）

| 档位 | ctx | decode p50 ms/步 | 量化税 ms/步 | 步涨 vs bf16 | 32 步首分叉 | 128 步首分叉 | 128 步分叉比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 1535 | 98.9 | — | — | — | — | — |
| per_token int8 | 1535 | 105.2 | ~4 | +6.3 | **5**（分叉比 0.36） | — | — |
| per_channel int8 | 1535 | 105.4 | ~4 | +6.5 | none | — | — |
| bf16 | 8255 | 102.4 | — | — | — | — | — |
| per_token int8 | 8255 | 106.6 | ~4 | +4.2 | none | **37** | 0.705 |
| per_channel int8 | 8255 | 107.9 | ~4 | +5.5 | none | **40** | 0.690 |
| gdn_state int8（一次性） | 1535 | — | — | — | **5**（分叉比 0.36） | — | — |

## 显存账（8 attn 层 × 4 KV 头 × 256 维，实测分配）

| 存储布局 | 实测 | KB/token |
|---|---:|---:|
| bf16 K/V | 270.5 MB | 32.00 |
| int8 K/V + fp32 per-token scale | 137.4 MB | **16.25（−49.2%）** |

与理论完全对齐（N3 的 32 KB/token 口径 ÷2 + 256 B/token scale）。hybrid 总账：GDN state 50.3 MB（fp32）恒定不可省，KV 减半对总显存的相对收益比纯 Transformer 更小（4B 模型 128k ctx：KV 4.1G→2.05G，占大头的是权重 7.9G）。

## 结论（"KV int8 换显存还是换速度"）

1. **显存是唯一稳赚项**：32→16.25 KB/token（−49.2%），理论=实测。
2. **速度：NPU eager 下意外地接近免费，但也赚不到**。每步全量量化税仅 +4~6 ms（cast/amax/clamp 是纯带宽操作，910B2 HBM 带宽充裕；32 步窗口内短 ctx +6.4%、长 ctx +4~5%）。但同样赚不到速度：4B decode 每步的带宽大头是权重读取（7.9 GB），8k KV 全读也只有 270 MB——**KV int8 省的是容量，不是 decode 带宽**；要变成速度收益需要大 KV 占比场景（大模型/超长 ctx/大 batch）+ 融合反量化内核（eager 多算子只会加 N2 说的小算子税）。
3. **精度的真实机制是"打平步级联翻边"，不是均匀退化**：
   - bf16 基线自身在 128 步轨迹上有多个 top1=top2 **完全打平的步**（gap=0.0，第 5/7/39/50/85 步）——量化噪声只需在这些步轻轻一推。
   - 分叉时机由轨迹上最小 gap 的步决定，与 ctx 长度**不是单调关系**：1535 ctx per_token 第 5 步就翻，8255 ctx 反而撑到第 37/40 步（N10 的 32 步窗口曾误显示"长 ctx 零分叉"，128 步验证推翻——**32/64 步的精度窗口对量化评估是不够的**）。
   - 翻边是级联的：一步翻边后 70% 后续 token 走上不同轨迹（贪心解码没有自愈）。
   - per_channel 与 per_token 在 32 步窗口差异明显（per_channel 全程未翻），128 步下两者趋同（40 vs 37 步）——**per-channel 更稳但只是延迟，不是免疫**。
   - 工程含义：温度采样下该噪声无害（采样本就随机）；greedy 评测/复现场景下 KV int8 会在几十 token 尺度改变轨迹，需要逐位复现时必须固定量化方案。
4. **GDN recurrent state 不可量化（实证）**：一次性 int8 roundtrip（(1,32,128,128) fp32 S 矩阵 ×24 层）→ 第 5 token 翻边、32 步分叉比 0.36，与 KV per-token 量化同期的破坏力度。S 矩阵是累积算子，量化噪声在递推中逐层放大——印证 N5 的 w8a8 排除 GDN 相关层、N3 的"state 必须 fp32"观察。KV int8 的 −49% 不适用于 state（它本就恒定 50 MB）。

## 附：原始产物

- `n10_kv_int8.md` / `.json` / `.log` — 主实验（脚本自动报告，其"长上下文误差累积更早"的结论句被 n10b 推翻，以本 summary 为准）
- `n10b_long_gen.json` / `.log` — 128 步交叉验证 + 每 step top1-top2 gap
- 运维备注：n10b 首跑遇卡 1 瞬态占满 OOM（容器内 npu-smi 无进程可见、几分钟后自愈，疑宿主侧其他租户）——共享集群的容器可见性局限，重跑即成
