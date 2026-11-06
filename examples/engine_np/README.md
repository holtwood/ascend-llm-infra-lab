# engine_np — Qwen3.5-4B 手写推理引擎（实验 N1）

在 torch_npu 上从零实现 Qwen3.5-4B（hybrid 架构）的最小推理引擎。本目录是
`docs/experiment-plan-b.md` N1 实验线的代码部分：L1 手写解码循环 + L2 KV cache
已经可用且与 transformers 数值对齐，L3–L5 在此骨架上迭代。

架构：32 层 3:1 混合 —— 24 层 GatedDeltaNet（线性注意力）+ 8 层 GQA gated
attention（每 4 层 1 个 full attention），hidden 2560，vocab 248320（tied）。
纯 torch 算子（`F.scaled_dot_product_attention` + 自写 delta rule kernel），
不依赖 flash-attn / triton / fla / causal_conv1d，CPU 与 NPU 皆可跑。

## 文件

| 文件 | 内容 |
|---|---|
| `model_qwen35.py` | 模型定义（RMSNorm / partial RoPE / GDN / GQA gated attn / MLP / DecoderLayer / 顶层），`load_weights()` 直接吃 released safetensors |
| `gdn_kernels.py` | Gated Delta Rule 两种数值实现：`recurrent_*`（逐 token，decode 用）与 `chunked_*`（分块 64，prefill 用），数学等价 |
| `cache.py` | `KVCache`（attention 层，append/read）+ `RecurrentState`（GDN 层 conv 尾巴 + S 矩阵）+ `HybridCaches` 按 layer_types 组装 |
| `engine_naive.py` | L1 引擎：`prefill()` → `decode_step()` → `generate()`，greedy + temperature 采样；附带 `generate_no_cache()` 作 L0 对照 |
| `verify_logits.py` | 数值对齐验证：mini 模型（本地 fp32）与真 4B checkpoint（`--real`，远端跑） |

## 快速开始

```bash
cd examples/engine_np

# 1) 各模块自检（conv 语义、delta rule 两实现一致性、cache 前向等价、引擎冒烟）
python3 cache.py && python3 gdn_kernels.py && python3 model_qwen35.py && python3 engine_naive.py

# 2) 数值对齐：2 层迷你模型（1 GDN + 1 attn），同权重下 vs transformers
#    需要 transformers >= 5.16（本地 pip install transformers 即可，纯 CPU）
python3 verify_logits.py
#    → prefill logits max|diff| ~1e-6，GDN final state ~1e-10，greedy 逐 token 一致

# 3) 真 4B 权重验证（在 NPU 宿主机上跑；CPU 也能跑但很慢）
python3 verify_logits.py --real --model-dir /model/Qwen3.5-4B --device npu
```

## 阶梯计划（N1 主线）

| 级别 | 内容 | 状态 |
|---|---|---|
| L0 naive | transformers `generate()` 基线 | 计划内（benchmark 侧） |
| L1 手写解码循环 | `engine_naive.py`：显式 prefill/decode/position/argmax | **完成**（本目录） |
| L2 KV cache | on/off 对照（`generate` vs `generate_no_cache`） | **完成**（跑分待 NPU） |
| L3 continuous batching | 迭代级调度：完成即退出、新请求即加入 | 待做（caches 已支持 B>1） |
| L4 paged KV | block 池 + block table + prefix 共享 | 待做（`KVCache` 的 allocate/append/read 接口已按可替换设计） |
| L5 chunked prefill | 长 prompt 切片 | 待做（GDN 层 conv/delta rule 已支持带状态的任意长度前向） |

接口为 L4 预留的约定：模型代码只调用 `KVCache.append/read` 与
`RecurrentState.conv_input/read_state/write_state`，换 paged 实现时只动 `cache.py`。

## 从 transformers 5.x 源码确认的架构事实

以下全部以 `transformers/models/qwen3_5/modeling_qwen3_5.py`（5.16.1）为准，
其中带 ⚠ 的是与常见直觉（或本实验最初假设）不同的点：

1. **RMSNorm 是 zero-init delta 形式** ⚠：`out = rmsnorm(x) * (1 + w)`，
   `w` 初始为 0；checkpoint 里存的就是 delta。与 Llama/Qwen2 的
   `rmsnorm(x) * w`（w≈1）不同，照抄 Llama 语义会整体偏移。
   但 GDN 的输出门控 `RMSNormGated` 是 **one-init** `w`（直接乘），两者不要混。
2. **q_proj 输出维度是 2×** ⚠：q 和 sigmoid 门控融合在一个投影里
   （`attn_output_gate: true`），前半 q、后半 gate；gate 无任何 norm，
   直接 `attn_out * sigmoid(gate)`，在 o_proj 之前。
3. **attention 有 q_norm/k_norm**（per-head RMSNorm，作用在 RoPE **之前**）。
   这是 Qwen3 系常规，但最初假设里漏了。
4. **partial RoPE**：`partial_rotary_factor=0.25` → 256 维头只转前 64 维
   （theta=1e7），rotate_half（`cat(-x2,x1)`）语义，其余 192 维 passthrough。
5. **RoPE 是 interleaved mrope 的文本退化** ⚠：HF 建的是 3-grid (T/H/W)
   mrope 并交错频率槽（mrope_section=[11,11,10]）；纯文本三 grid 相同，
   交错后等于 plain rope —— 引擎只实现文本一路，数学等价。
6. **GDN 的 delta 用的是 decay 后的 S** ⚠：正确顺序是
   `S ← S·exp(g_t)`，然后 `delta = (v_t − Sᵀk_t)·β_t`，`S += k⊗delta`，
   输出 `o_t = Sᵀ q_t` 用的是**更新后**的 S（q 已预乘 1/√d_k）。
7. **QK 头扩展是 repeat_interleave** ⚠：32 个 v 头两两共享 16 个 k 头
   （[k0,k0,k1,k1,...]），不是 GQA 式跨头分组。g/β 本来就在 32 个 v 头上。
8. **GDN 输入链**：`in_proj_qkv`（8192 维融合投影）→ depthwise CausalConv1d
   (kernel=4，无 bias) → SiLU → split 成 q/k/v → q/k l2norm（在 kernel 内，
   FLA 对齐：`x·rsqrt(Σx²+1e-6)`）→ delta rule → RMSNormGated(z 门控) → out_proj。
9. **conv state 存最后 K=4 个** ⚠（不是 K−1=3）：存的是 SiLU 之前的原始
   mixed qkv。首次 prefill 时短序列左 pad 零到 K；decode 时
   `cat(state(4), x)` → padding=0 conv → 取尾部。与全序列 conv 逐位一致
   （`cache.py` 自检验证 bit-exact）。
10. **prefill 用 chunked kernel、单 token 用 recurrent kernel**；两者在 fp32
    下输出差 ~1e-7（`gdn_kernels.py` 自检）。chunk_size=64。
11. **GDN 全程 fp32**（config `mamba_ssm_dtype: float32`）：`g = −exp(A_log)·
    softplus(a+dt_bias)` 中 `A_log.exp()` 在 bf16 下可能 −inf/溢出，
    g/β/S 状态必须 fp32。S 形状 `(B, 32, 128, 128)` fp32。
12. **checkpoint 键前缀是 `model.language_model.`** ⚠（VL 包装壳
    Qwen3_5ForConditionalGeneration），不是 `model.`；`model.visual.*` 与
    `mtp.*`（MTP draft 头，`mtp_num_hidden_layers=1`）整段忽略；
    **`lm_head.weight` 不在文件里**（tied embedding）。
13. **权重文件名是连字符** ⚠：`model.safetensors-0000{1,2}-of-00002.safetensors`
    （HF 惯例是 `model-0000X-of-00002.safetensors`）。加载一律走
    `model.safetensors.index.json` 的 weight_map，不要猜文件名。
14. 每 4 层 1 个 full attention（`full_attention_interval=4`，层 3,7,11,...31）；
    因此 **KV cache 只占 8 层**：bf16 下 2(K,V)×4 头×256 维×2B = **32 KiB/token**
    （与 `results/e8` 结论一致）。GDN 层的缓存是**定长**的（S 约 4 MB/层/请求
    fp32 + conv 64 KiB），与序列长度无关 —— hybrid 架构的显存优势来源。

## 内存账（单请求，bf16）

```
权重               ≈ 8.0 GiB（4.75B 参数，tied）
KV  (8 层)         32 KiB/token → 262k ctx ≈ 8 GiB
GDN S (24 层)      24 × 2 MiB ≈ 48 MiB（定长，与 ctx 无关）
GDN conv (24 层)   24 × 8192 × 4 × 2B ≈ 1.5 MiB（定长）
logits 1 token     248320 × 2B ≈ 0.5 MiB（只算最后位置）
```

## 已知坑 / 调试记录

- **tied weight 的 load 覆盖**：`state_dict()` 里 `model.embed_tokens.weight`
  与 `lm_head.weight` 是同一存储；若对 tied 模型 `load_state_dict` 一个
  lm_head **没有** tie 的 state_dict（例如 `Qwen3_5TextConfig` 默认
  `tie_word_embeddings=False`，而 4B checkpoint 是 True），第二个键会把
  embedding 覆盖成随机 lm_head —— 表面 missing/unexpected 都是空，
  实际 embedding 已被污染。两边 tie 配置必须显式一致（verify_logits.py 里有注释）。
- **GDN 数值敏感**：g 的 exp 在 fp32 才安全；自写模型与 HF 的逐位对齐
  依赖同样的 fp32 计算顺序（`gdn_kernels.py` 照抄 HF fallback 实现）。
- **decode 与 no-cache 的阈值**：GDN 状态每步重递推，fp32 舍入随步数累积，
  引擎自检用 1e-4（纯 attention 模型可以到 1e-5）；greedy token 一致性
  是行为等价的硬标准，随 max|diff| 一起断言。
- HF 的 conv/delta rule 有优化 kernel（`causal_conv1d` / `flash-linear-attention`），
  未安装时 fallback 到与本文相同的纯 torch 参考实现 —— 意味着 verify 时
  两边都是参考实现，对齐结论对 NPU 上的 kernel 替换依然成立（数学相同）。
- MTP（`mtp.*`）加载时跳过；引擎当前 ignore MTP draft 头，与
  vllm-ascend 的 `qwen3_5_mtp` 对照实验留给后续。

## 远端验证命令（NPU 宿主机 223.111.149.177:40045）

```bash
# transformers 5.16.1 + torch_npu 已装好
python3 /path/to/engine_np/verify_logits.py --real --model-dir /model/Qwen3.5-4B --device npu
```

真模型 bf16 下重点看三件事：last-position logits 的 max|diff|（bf16 量化噪声
量级，~1e-2 量级属正常）、top-1 token 是否一致、greedy 8 token 是否逐个一致。
