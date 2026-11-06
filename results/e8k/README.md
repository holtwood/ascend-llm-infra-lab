# E8k · INT8 KV Cache：省一半显存是真，但要 kernel 融合才划算

> 单步 decode attention，KV 存 int8（per-channel / per-64-token group 对称量化）
> vs bf16。910B1 torch_npu。脚本 `scripts/e8k_kvquant.py`，数据 `../e8k_kvquant.json`。

## 结果

| KV 长 | bf16 ms | int8 per-channel ms(税) | int8 g64 ms(税) | KV 字节减半 | max_diff |
|---|---|---|---|---|---|
| 8k | 0.152 | 0.991 (**6.5×**) | 0.988 (6.5×) | 134→67MB | 0.0012 |
| 32k | 0.537 | 4.364 (**8.1×**) | 4.344 (8.1×) | 537→268MB | 0.0006 |
| 64k | 1.046 | 8.807 (**8.4×**) | 8.851 (8.5×) | 1074→537MB | 0.0004 |

## 结论

1. **显存减半 + 精度几乎无损**（max_diff ≤ 0.0012，bf16 舍入量级）——
   KV int8 的"省"是真的；per-channel 与 per-64-token 粒度差异可忽略
2. **naive 反量化路径慢 6.5-8.5×**——省下的读取带宽被"先物化回 bf16 再算"
   的写读放大吃光还倒贴；量化收益必须在 attention kernel 内部完成
   （int8 读入 + kernel 内反量化）才能兑现
3. 这解释了为什么 vLLM 的 C8 INT8 KV（上游 #7474）是独立 kernel 特性
   而不是开关——**KV 量化、分页间接、稀疏注意力是同族问题**：
   收益在访存层，只能 kernel 内兑现
4. 与 E8j/E8e-SWA 构成"kernel 层三诫"：
   - 分页不物化（E8j: 3-16× 税）
   - 量化不反物化（E8k: 6.5-8.5× 税）
   - 窗口不用 mask（E8e: 26× 税）

## 扩展（通宵队列，含中间态精度对照）

`results/overnight_20261105_160244/kv_quant_matrix.json`（S 128-16k × B 1-64，
32 配置）：

- mem_ratio 恒 0.51（int8 + bf16 scale）
- err_max 随 S 收敛：0.0117@S=128 → 0.0007@S=16384（token 越多量化噪声越被平均）
- **naive 税 1.5–3.0×，而非本文的 6.5–8.5×**——原因已定位到代码：
  本文 `(q.float() * scale.float()).to(bf16)` 用 **float32 中间态**，
  扩展实验用 `q.to(bf16) * scale`（bf16 中间态）

**差异已用 2×2 对照实测定死**（`results/overnight_20261105_160244/kv_dequant_ab.md`，
S=8192 B=1 同硬件）：

| KV 头数 | fp32 中间态 | bf16 中间态 |
|---|---|---|
| 8（GQA） | 1.87× | 1.40× |
| 32（MHA，本文口径） | **6.46×** | 3.16× |

两个因子叠加：中间态精度（≈2× 流量）+ KV 体积/头数（8→32 头使反量化
从延迟受限转为带宽受限）。本文的 6.5× 被精确复现（6.46×）。

**引用规则：报告 int8-KV naive 税必须同时给 (a) 头数/总字节 (b) 中间态 dtype。**

局限：扩展实验的 scale 是 per-token-per-head，非真实 C8 的 per-channel 方案。
