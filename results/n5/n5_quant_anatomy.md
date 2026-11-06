# N5: Qwen3.8-27B-w8a8 量化权重解剖

对象: `/model/Qwen3.8-27B-w8a8`（Ascend msModelSlim 导出，description version=1.0.0, optional keys=[]，共 2080 条状态标记）

## 1. 量化格式（quant_model_description.json）

- 状态标记只有两种：`FLOAT`（759 条）与 `W8A8_DYNAMIC`（1321 条），无 W4/W8A16 等混合格式。
- `group_size = 0` → 无 group-wise 量化，粒度到 channel 为止。
- 每个被量化模块携带 `weight_scale` 与 `weight_offset` 两个伴生 tensor；description 中**没有任何** `deq_scale` / `input_scale` / `activation_scale` 条目。

## 2. Tensor 命名与 dtype（safetensors 实测）

| tensor | dtype | shape |
|---|---|---|
| `mtp.layers.0.self_attn.q_proj.weight` | `I8` | (12288, 5120) |
| `model.language_model.layers.0.mlp.down_proj.weight` | `I8` | (5120, 17408) |
| `model.language_model.layers.0.linear_attn.in_proj_qkv.weight` | `I8` | (10240, 5120) |
| `model.language_model.layers.3.self_attn.q_proj.weight` | `I8` | (12288, 5120) |

### 粒度判定

- 全部 440 个 weight_scale 的 shape 与对应 weight 的 dim0（out_channel）一致（存成 `(out, 1)` 二维，对齐 Ascend cube 算子）：投票明细 {'dim0(out,1)': 440, 'dim1': 0, 'scalar': 0, 'other': 0} → **True**。
- 结合 `group_size=0` → **per-channel（out_channel 维）int8 权重量化，无 per-group**。每行独立 absmax：`w_int8[i,:] = round(w[i,:] / scale[i])`，反量化 `w ≈ w_int8 × scale[i]`。
- 无 input/activation scale tensor，且格式名含 `DYNAMIC` → **activation 在线动态量化**（best_practice `act.scope: per_token`：逐 token 求 absmax 得 scale，运行时计算，不落盘）。

### weight_offset 抽样

| 模块 | dtype | shape | min | max | #=0 | numel |
|---|---|---|---|---|---|---|
| `model.language_model.layers.0.linear_attn.in_proj_z.weight_offset` | `F32` | (6144, 1) | 0.0 | 0.0 | 6144/6144 | 6144 |
| `model.language_model.layers.1.linear_attn.in_proj_qkv.weight_offset` | `F32` | (10240, 1) | 0.0 | 0.0 | 10240/10240 | 10240 |
| `model.language_model.layers.10.mlp.down_proj.weight_offset` | `F32` | (5120, 1) | 0.0 | 0.0 | 5120/5120 | 5120 |
| `model.language_model.layers.11.mlp.up_proj.weight_offset` | `F32` | (17408, 1) | 0.0 | 0.0 | 17408/17408 | 17408 |
| `model.language_model.layers.12.linear_attn.in_proj_qkv.weight_offset` | `F32` | (10240, 1) | 0.0 | 0.0 | 10240/10240 | 10240 |
| `model.language_model.layers.13.linear_attn.in_proj_qkv.weight_offset` | `F32` | (10240, 1) | 0.0 | 0.0 | 10240/10240 | 10240 |

- 抽样 offset 全为 0：**True** → offset 是 ascendv1_saver 的占位/兼容字段，实际仍是对称量化（与 best_practice `weight.symmetric: true` 一致）。

## 3. 覆盖范围：哪些模块被量化

| 模块类别 | 量化 weight 数 | FLOAT weight 数 | 参数量 | 占用 |
|---|---|---|---|---|
| mlp | 192 | 0 | 17117.9M | 15.96 GiB |
| gdn | 96 | 240 | 5563.6M | 6.62 GiB |
| attn | 64 | 32 | 1678.4M | 1.56 GiB |
| lm_head | 0 | 1 | 1271.4M | 2.37 GiB |
| embed | 0 | 1 | 1271.4M | 2.37 GiB |
| visual | 81 | 86 | 461.2M | 0.60 GiB |
| mtp | 7 | 8 | 424.8M | 0.44 GiB |
| norm | 0 | 129 | 0.7M | 0.00 GiB |

各类别内部构成（对照 description）：

- **attn**（16 层 full_attention，每 4 层 1 个）：`q/k/v/o_proj.weight` 量化（I8）；`q_norm/k_norm.weight`（BF16）不量化。注意 `q_proj` 输出维 12288 = 24 头 × 256 × **2（attn_output_gate）**。
- **gdn**（48 层 linear_attention）：`in_proj_qkv` (10240, 5120) 与 `in_proj_z` (6144, 5120) 量化；`in_proj_a/in_proj_b/out_proj/conv1d/A_log/dt_bias/norm` 保持高精度。**`out_proj` (6144, 5120) bf16 ≈ 63 MB/层 × 48 = 3.0 GiB**，是最大的未量化部分（对应 best_practice `exclude: *linear_fc2*`）。
- **mlp**：`gate/up/down_proj` 全量化，是参数量与字节数大头。
- **lm_head / embed_tokens**：各 248320×5120 ≈ 1.27B 参数、2.37 GiB bf16，**不量化**。
- **mtp**（1 层 MTP 投机解码头）：attn+mlp 与主干同样量化。
- **visual**（ViT 27 blocks）：按 best_practice `*visual.blocks*` 一并量化。

**dtype 普查**：`I8` × 440 tensors = 21.85 GiB；`BF16` × 678 tensors = 8.04 GiB；`F32` × 961 tensors = 0.03 GiB
（F32 共 961 个 tensor：440 个 weight_scale + 440 个 weight_offset（量化元数据，合计 30 MiB）；其余 81 个 F32 为高精度参数（如 `model.visual.blocks.0.attn.proj.bias`；GDN 的 A_log/dt_bias 存为 BF16，运行时才按 `mamba_ssm_dtype: float32` 升精度）。）

- 量化 weight 参数量 **23.47B** / FLOAT weight 参数量 4.31B / 全部 tensor 合计 27.79B 参数 → **量化覆盖 weight 参数的 84.5%**。

## 4. weight_scale 分布统计

- 共 440 个 scale tensor、合计 3.95M 个 scale 值，dtype 全部 `F32`，shape `(out_channel, 1)`。
- 全局分位数：min=1.192e-07 / p1=2.804e-04 / p5=3.033e-04 / p25=3.433e-04 / median=3.891e-04 / p75=4.787e-04 / p95=6.523e-04 / p99=9.651e-04 / max=1.129e-02；mean=4.286e-04, std=1.440e-04。

| 模块类别 | scale tensor 数 | per-tensor 均值 (min / median / max) |
|---|---|---|
| attn | 64 | 4.617e-04 / 5.827e-04 / 1.360e-03 |
| gdn | 96 | 4.633e-04 / 5.241e-04 / 6.226e-04 |
| mlp | 192 | 3.036e-04 / 3.762e-04 / 4.873e-04 |
| mtp | 7 | 4.079e-04 / 5.461e-04 / 1.060e-03 |
| visual | 81 | 3.736e-04 / 4.656e-04 / 9.909e-04 |

### 离群层（主干按 scale 均值排序，Top-8 / Bottom-3）

| rank | 模块 | scale mean | scale max | #channels |
|---|---|---|---|---|
| 1 | `model.language_model.layers.59.self_attn.v_proj.weight` | 1.360e-03 | 7.324e-03 | 1024 |
| 2 | `model.language_model.layers.63.self_attn.v_proj.weight` | 1.259e-03 | 7.996e-03 | 1024 |
| 3 | `model.language_model.layers.55.self_attn.v_proj.weight` | 9.591e-04 | 2.945e-03 | 1024 |
| 4 | `model.language_model.layers.51.self_attn.v_proj.weight` | 9.243e-04 | 3.479e-03 | 1024 |
| 5 | `model.language_model.layers.63.self_attn.o_proj.weight` | 8.192e-04 | 6.256e-03 | 5120 |
| 6 | `model.language_model.layers.63.self_attn.k_proj.weight` | 8.057e-04 | 2.136e-03 | 1024 |
| 7 | `model.language_model.layers.59.self_attn.k_proj.weight` | 7.582e-04 | 2.060e-03 | 1024 |
| 8 | `model.language_model.layers.59.self_attn.o_proj.weight` | 7.180e-04 | 8.728e-03 | 5120 |
| … | … | | | |
| 350 | `model.language_model.layers.1.mlp.up_proj.weight` | 3.244e-04 | 1.686e-03 | 17408 |
| 351 | `model.language_model.layers.0.mlp.gate_proj.weight` | 3.166e-04 | 3.326e-03 | 17408 |
| 352 | `model.language_model.layers.0.mlp.up_proj.weight` | 3.036e-04 | 1.282e-03 | 17408 |

### 离群通道（tensor 内 scale 超过 median 4×/8× 的通道占比）

| 类别 | tensor 数 | >4×median 占比 median/max | >8×median 占比 median/max |
|---|---|---|---|
| attn | 64 | 0.08% / 1.27% | 0.00% / 0.02% |
| gdn | 96 | 0.02% / 0.17% | 0.00% / 0.12% |
| mlp | 192 | 0.02% / 0.78% | 0.00% / 0.09% |
| mtp | 7 | 0.20% / 0.42% | 0.00% / 0.02% |
| visual | 81 | 0.00% / 1.13% | 0.00% / 0.02% |

Top-8 张量（按 >8×median 占比）：

- `model.language_model.layers.18.mlp.down_proj.weight`: 0.12%
- `model.language_model.layers.63.self_attn.v_proj.weight`: 0.10%
- `model.visual.blocks.1.attn.proj.weight`: 0.09%
- `model.language_model.layers.1.mlp.down_proj.weight`: 0.06%
- `model.language_model.layers.7.self_attn.o_proj.weight`: 0.04%
- `model.language_model.layers.62.mlp.down_proj.weight`: 0.04%
- `model.language_model.layers.6.mlp.down_proj.weight`: 0.04%
- `model.language_model.layers.27.self_attn.o_proj.weight`: 0.02%

- 结论：离群通道占比中位数 0.00%（>1% 的 tensor 0 个）——**per-channel minmax 本身已吸收通道间幅度差异，量化难点不在权重离群通道，而在激活侧离群（这才是 per-token dynamic act quant + 排除 out_proj 的动机）。**

### 直方图（log10(scale)，20 bins）

```
10^-6.92~10^-6.67 |        12 
10^-6.67~10^-6.43 |         2 
10^-6.43~10^-6.18 |         0 
10^-6.18~10^-5.93 |         2 
10^-5.93~10^-5.68 |         3 
10^-5.68~10^-5.43 |         8 
10^-5.43~10^-5.18 |         2 
10^-5.18~10^-4.93 |        13 
10^-4.93~10^-4.68 |        10 
10^-4.68~10^-4.44 |        91 
10^-4.44~10^-4.19 |       106 
10^-4.19~10^-3.94 |       144 
10^-3.94~10^-3.69 |       726 
10^-3.69~10^-3.44 |   1496309 ########################################
10^-3.44~10^-3.19 |   2242442 ############################################################
10^-3.19~10^-2.94 |    193718 #####
10^-2.94~10^-2.69 |     18268 
10^-2.69~10^-2.44 |      1654 
10^-2.44~10^-2.20 |       118 
10^-2.20~10^-1.95 |        20 
```

## 5. 大小对账

- index `metadata.total_size`（safetensors 净荷） = **29.922 GiB**
- 逐 tensor numel×dtype_bytes 重建             = **29.922 GiB**（与 index 完全一致）
- 磁盘 10 分片实际字节                          = **29.922 GiB**（header/对齐 ≈ 0.2 MiB）
- 目录标称 30 GB：磁盘 32.13 GB（十进制）/29.9 GiB（二进制），口径吻合，无冗余文件。

| 组成 | 大小 | 占比 | 说明 |
|---|---|---|---|
| int8 量化 weight | 21.85 GiB | 73.0% | 23.47B 参数 × 1B |
| 未量化 weight（bf16/f32） | 8.04 GiB | 26.9% | embed_tokens + lm_head 各 ~2.4 GiB，GDN out_proj ~3.0 GiB，norm/MTP 余项 |
| weight_scale（F32） | 0.015 GiB | 0.05% | out_channel 维，每通道 4B |
| weight_offset（F32，全 0） | 0.015 GiB | 0.05% | 占位字段，同 scale 形状 |

对照：全部 27.79B 参数若存 bf16 约需 51.8 GiB → W8A8 实测 29.9 GiB，压缩比 ≈ 1.73×。scale+offset 总开销不足 1%。注意 embed/lm_head/GDN-out_proj 未量化，使整体压缩比达不到理想 2×。

## 6. Qwen3.8-27B_best_practice.yaml 关键配置

```yaml
apiversion: multimodal_vlm_modelslim_v1
spec:
  process:
  - type: linear_quant
    qconfig:
      act:
        scope: per_token
        dtype: int8
        symmetric: true
        method: minmax
      weight:
        scope: per_channel
        dtype: int8
        symmetric: true
        method: minmax
    include:
    - '*self_attn*'
    - '*mlp*'
    - '*visual.blocks*'
    - '*linear_attn.in_proj_qkv*'
    - '*linear_attn.in_proj_z*'
    exclude:
    - '*linear_fc2*'
  save:
  - type: ascendv1_saver
    part_file_size: 4
  dataset: calibImages
  default_text: Describe this image in detail.
metadata:
  config_id: qwen3_8_27B_dense_w8a8
  score: 100.0
  label:
    w_bit: 8
    a_bit: 8
    is_sparse: false
    kv_cache: false
  verified_model_types: []
  verified_tags:
    Qwen3.8-27B:
    - - vLLM_Ascend
      - Atlas_A2_Inference
    - - vLLM_Ascend
      - Atlas_A3_Inference
default_w8a8_dynamic:
  act:
    scope: per_token
    dtype: int8
    symmetric: true
    method: minmax
  weight:
    scope: per_channel
    dtype: int8
    symmetric: true
    method: minmax
```

值得注意的 3 条：

1. `exclude: '*linear_fc2*'` → **Gated DeltaNet 的输出投影（本模型命名 `linear_attn.out_proj`）不量化**，保持 bf16。实测 48 层 GDN 的 `out_proj` 全为 FLOAT（~3.0 GiB bf16）——delta-rule 输出路径对量化误差敏感，官方精度优先。同理 `in_proj_a/in_proj_b/conv1d/A_log/dt_bias`（SSM 状态参数）也未量化。
2. `act.scope: per_token + method: minmax + symmetric: true`，且 activation scale 不落盘 → **W8A8 dynamic**：部署时逐 token 在线算激活 absmax（对应 vLLM-Ascend per-token dynamic quant cube 算子），无校准数据依赖的静态 act scale。产物中的 `weight_offset` 虽与 yaml 的 `symmetric: true` 表面矛盾，实测全 0，为 ascendv1_saver 占位输出。
3. `save.type: ascendv1_saver, part_file_size: 4`（GB/分片）+ `config_id: qwen3_8_27B_dense_w8a8`；`verified_tags` 锁定 **vLLM_Ascend + Atlas_A2/A3**。该产物是 msModelSlim→vLLM-Ascend 专用链路，不能直接用 HF transformers 加载（需 Ascend 侧 custom quant 方案）。

## 7. generation_config.json / config.json 要点

- generation：`top_k=20, top_p=0.95, temperature=1.0`（Qwen 默认采样）；`eos=[248046,248044]`，`bos=pad=248044`。
- config：64 层（48 linear_attention + 16 full_attention，`full_attention_interval=4`），hidden 5120，intermediate 17408，vocab 248320，head_dim 256（24 Q 头 / 4 KV 头，`attn_output_gate`→q_proj 输出翻倍 12288），`partial_rotary_factor=0.25`、mrope interleaved；`mtp_num_hidden_layers=1`（带投机解码头）。

## 8. 结论

- **格式一句话**：Ascend msModelSlim `W8A8_DYNAMIC`（ascendv1_saver 导出）= 权重 per-channel（out 维）int8 静态对称量化（group_size=0，scale 为 F32 `(out,1)`，offset 占位全 0）+ 激活 per-token 在线动态 int8 量化，activation scale 不存文件。
- **覆盖**：主干 self_attn / mlp / GDN 的 in_proj_qkv+in_proj_z / MTP 层 / ViT blocks 全部量化（量化 weight 23.5B，占 weight 参数 84%）；embed_tokens、lm_head、全部 norm、GDN 的 out_proj/in_proj_a/in_proj_b/conv1d/A_log/dt_bias 保持 bf16/f32。
- **对账**：理论 29.92 GiB = 实际 29.92 GiB（差 0 MiB header），vs 全 bf16 51.8 GiB（1.73×）。

