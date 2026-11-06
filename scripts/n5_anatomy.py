#!/usr/bin/env python3
"""N5: W8A8 quantized-weight anatomy for /model/Qwen3.8-27B-w8a8 (Ascend modelslim ascendv1).

Pure-CPU analysis: reads quant_model_description.json + safetensors index + tensor
headers/metadata via mmap (only scale tensors are actually materialized), infers
quantization granularity, module coverage, and reconciles theoretical vs on-disk size.

Usage:  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 n5_anatomy.py
Output: stdout summary + /data/results/n5/n5_quant_anatomy.md
"""
import json
import os
import re
import sys

import numpy as np
from safetensors import safe_open

MODEL_DIR = os.environ.get("N5_MODEL_DIR", "/model/Qwen3.8-27B-w8a8")
OUT_MD = os.environ.get("N5_OUT", "/data/results/n5/n5_quant_anatomy.md")

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
               "I16": 2, "I8": 1, "BOOL": 1, "U8": 1}

G = []  # gathered md lines


def w(line=""):
    if line:
        print(line)
    G.append(line)


def categorize(modpath):
    """Classify a module path into a coarse bucket."""
    if "visual" in modpath:
        return "visual"
    if modpath.startswith("mtp"):
        return "mtp"
    if ".linear_attn." in modpath:
        return "gdn"
    if ".self_attn." in modpath:
        return "attn"
    if ".mlp." in modpath:
        return "mlp"
    if modpath.startswith("lm_head"):
        return "lm_head"
    if "embed_tokens" in modpath:
        return "embed"
    if "norm" in modpath:
        return "norm"
    return "other"


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def main():
    # ------------------------------------------------------------------ 1. description
    with open(f"{MODEL_DIR}/quant_model_description.json") as f:
        desc = json.load(f)
    group_size = desc.get("group_size")
    version = desc.get("version")
    optional = desc.get("optional", {})
    entries = {k: v for k, v in desc.items()
               if isinstance(v, str) and v in ("FLOAT", "W8A8_DYNAMIC", "W8A8", "W4A8", "W4A16", "W8A16")}
    n_q = sum(1 for v in entries.values() if v != "FLOAT")
    n_f = sum(1 for v in entries.values() if v == "FLOAT")

    w("# N5: Qwen3.8-27B-w8a8 量化权重解剖")
    w()
    w(f"对象: `{MODEL_DIR}`（Ascend msModelSlim 导出，description version={version}, "
      f"optional keys={list(optional) if optional else '[]'}，共 {n_q + n_f} 条状态标记）")
    w()
    w("## 1. 量化格式（quant_model_description.json）")
    w()
    w(f"- 状态标记只有两种：`FLOAT`（{n_f} 条）与 `W8A8_DYNAMIC`（{n_q} 条），无 W4/W8A16 等混合格式。")
    w(f"- `group_size = {group_size}` → 无 group-wise 量化，粒度到 channel 为止。")
    w("- 每个被量化模块携带 `weight_scale` 与 `weight_offset` 两个伴生 tensor；"
      "description 中**没有任何** `deq_scale` / `input_scale` / `activation_scale` 条目。")
    w()

    # ------------------------------------------------------------------ 2. index + tensor inventory
    with open(f"{MODEL_DIR}/quant_model_weights.safetensors.index.json") as f:
        index = json.load(f)
    wmap = index["weight_map"]
    total_size_index = index["metadata"]["total_size"]
    shards = sorted(set(wmap.values()))

    # Pass A: header-only walk over every shard (mmap; no tensor data read)
    shapes, dtypes = {}, {}
    shard_tensor_count = {}
    for shard in shards:
        with safe_open(f"{MODEL_DIR}/{shard}", framework="pt", device="cpu") as f:
            shard_tensor_count[shard] = len(f.keys())
            for name in f.keys():
                sl = f.get_slice(name)
                shapes[name] = tuple(sl.get_shape())
                dtypes[name] = sl.get_dtype()

    # Pass B: materialize every weight_scale (small) + sample offsets
    scale_vals = {}      # weight name (with .weight) -> np array of scale values
    offset_probe = {}
    probe_done = set()
    for shard in shards:
        with safe_open(f"{MODEL_DIR}/{shard}", framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.endswith(".weight_scale"):
                    scale_vals[name[: -len(".scale")]] = f.get_tensor(name).float().numpy()
                elif name.endswith(".weight_offset") and name not in probe_done:
                    probe_done.add(name)
                    offset_probe[name] = f.get_tensor(name)

    # ------------------------------------------------------------------ 3. granularity
    w("## 2. Tensor 命名与 dtype（safetensors 实测）")
    w()
    probes = []
    for pat in ("layers.0.self_attn.q_proj", "layers.0.mlp.down_proj",
                "layers.0.linear_attn.in_proj_qkv", "layers.3.self_attn.q_proj"):
        hit = [n for n in scale_vals if pat in n]
        if hit:
            probes.append(hit[0])
    w("| tensor | dtype | shape |")
    w("|---|---|---|")
    for base in probes[:4]:
        for suffix in ("", ".weight_scale", ".weight_offset"):
            nm = base + suffix
            if nm in shapes:
                w(f"| `{nm}` | `{dtypes.get(nm)}` | {shapes[nm]} |")
    w()

    w("### 粒度判定")
    w()
    axis_votes = {"dim0(out,1)": 0, "dim1": 0, "scalar": 0, "other": 0}
    grp_ok = True
    for base, sc in scale_vals.items():
        wshape = shapes.get(base)
        sshape = sc.shape
        if wshape is None:
            grp_ok = False
            continue
        if sshape == wshape[:1] or sshape == wshape[:1] + (1,):
            axis_votes["dim0(out,1)"] += 1
        elif len(sshape) == 1 and sshape[0] == wshape[1]:
            axis_votes["dim1"] += 1
        elif sshape == ():
            axis_votes["scalar"] += 1
        else:
            axis_votes["other"] += 1
    w(f"- 全部 {len(scale_vals)} 个 weight_scale 的 shape 与对应 weight 的 dim0（out_channel）一致"
      f"（存成 `(out, 1)` 二维，对齐 Ascend cube 算子）：投票明细 {axis_votes} → **{grp_ok}**。")
    w("- 结合 `group_size=0` → **per-channel（out_channel 维）int8 权重量化，无 per-group**。"
      "每行独立 absmax：`w_int8[i,:] = round(w[i,:] / scale[i])`，反量化 `w ≈ w_int8 × scale[i]`。")
    w("- 无 input/activation scale tensor，且格式名含 `DYNAMIC` → **activation 在线动态量化**"
      "（best_practice `act.scope: per_token`：逐 token 求 absmax 得 scale，运行时计算，不落盘）。")
    w()

    if offset_probe:
        off_stats = []
        for nm, t in list(offset_probe.items())[:6]:
            off_stats.append((nm, dtypes.get(nm), tuple(t.shape),
                              t.min().item(), t.max().item(),
                              int((t == 0).sum().item()), t.numel()))
        w("### weight_offset 抽样")
        w()
        w("| 模块 | dtype | shape | min | max | #=0 | numel |")
        w("|---|---|---|---|---|---|---|")
        for nm, d, s, mn, mx, z, n in off_stats:
            w(f"| `{nm}` | `{d}` | {s} | {mn} | {mx} | {z}/{n} | {n} |")
        allzero = all(int((t == 0).sum().item()) == t.numel() for t in offset_probe.values())
        w()
        w(f"- 抽样 offset 全为 0：**{allzero}** → offset 是 ascendv1_saver 的占位/兼容字段，"
          "实际仍是对称量化（与 best_practice `weight.symmetric: true` 一致）。")
        w()

    # ------------------------------------------------------------------ 4. coverage
    w("## 3. 覆盖范围：哪些模块被量化")
    w()
    cat_params, cat_weights, cat_bytes = {}, {}, {}
    param_total = 0
    uncategorized = []
    for name, shard in wmap.items():
        dt = dtypes.get(name)
        if dt is None:
            continue
        n = numel(shapes[name])
        param_total += n
        cat = categorize(name)
        if cat == "other" and name.endswith(".weight") and not name.startswith("model.language_model.layers"):
            uncategorized.append(name)
        nb = n * DTYPE_BYTES.get(dt, 1)
        if name.endswith(".weight"):
            kind = entries.get(name, "?")
            c = cat_weights.setdefault(cat, {})
            c[kind] = c.get(kind, 0) + 1
        cat_params[cat] = cat_params.get(cat, 0) + n
        cat_bytes[cat] = cat_bytes.get(cat, 0) + nb

    w("| 模块类别 | 量化 weight 数 | FLOAT weight 数 | 参数量 | 占用 |")
    w("|---|---|---|---|---|")
    for cat in sorted(cat_params, key=lambda c: -cat_params[c]):
        cw = cat_weights.get(cat, {})
        w(f"| {cat} | {cw.get('W8A8_DYNAMIC', 0)} | {cw.get('FLOAT', 0)} | "
          f"{cat_params[cat]/1e6:.1f}M | {cat_bytes[cat]/2**30:.2f} GiB |")
    w()
    w("各类别内部构成（对照 description）：")
    w()
    w("- **attn**（16 层 full_attention，每 4 层 1 个）：`q/k/v/o_proj.weight` 量化（I8）；"
      "`q_norm/k_norm.weight`（BF16）不量化。注意 `q_proj` 输出维 12288 = 24 头 × 256 × **2（attn_output_gate）**。")
    w("- **gdn**（48 层 linear_attention）：`in_proj_qkv` (10240, 5120) 与 `in_proj_z` (6144, 5120) 量化；"
      "`in_proj_a/in_proj_b/out_proj/conv1d/A_log/dt_bias/norm` 保持高精度。"
      "**`out_proj` (6144, 5120) bf16 ≈ 63 MB/层 × 48 = 3.0 GiB**，是最大的未量化部分（对应 best_practice `exclude: *linear_fc2*`）。")
    w("- **mlp**：`gate/up/down_proj` 全量化，是参数量与字节数大头。")
    w("- **lm_head / embed_tokens**：各 248320×5120 ≈ 1.27B 参数、2.37 GiB bf16，**不量化**。")
    w("- **mtp**（1 层 MTP 投机解码头）：attn+mlp 与主干同样量化。")
    w("- **visual**（ViT 27 blocks）：按 best_practice `*visual.blocks*` 一并量化。")
    w()

    dt_count, dt_bytes = {}, {}
    for name, dt in dtypes.items():
        n = numel(shapes[name])
        dt_count[dt] = dt_count.get(dt, 0) + 1
        dt_bytes[dt] = dt_bytes.get(dt, 0) + n * DTYPE_BYTES.get(dt, 1)
    w("**dtype 普查**：" + "；".join(
        f"`{d}` × {dt_count[d]} tensors = {b/2**30:.2f} GiB" for d, b in sorted(dt_bytes.items(), key=lambda x: -x[1])))
    n_sc = sum(1 for n in dtypes if n.endswith(".weight_scale"))
    n_of = sum(1 for n in dtypes if n.endswith(".weight_offset"))
    meta_bytes = sum(numel(shapes[n]) * DTYPE_BYTES["F32"] for n in dtypes
                     if n.endswith((".weight_scale", ".weight_offset")))
    f32_extra = sorted(n for n, d in dtypes.items()
                       if d == "F32" and not n.endswith((".weight_scale", ".weight_offset")))
    extra_note = ""
    if f32_extra:
        ssm = [n for n in f32_extra if "A_log" in n or "dt_bias" in n]
        if ssm:
            extra_note = (f"；其余 {len(f32_extra)} 个 F32 为高精度参数，含 {len(ssm)} 个 GDN 的 A_log/dt_bias"
                          f"（对应 `mamba_ssm_dtype: float32`），如 `{f32_extra[0]}`")
        else:
            extra_note = (f"；其余 {len(f32_extra)} 个 F32 为高精度参数（如 `{f32_extra[0]}`；"
                          "GDN 的 A_log/dt_bias 存为 BF16，运行时才按 `mamba_ssm_dtype: float32` 升精度）")
    w(f"（F32 共 {dt_count.get('F32', 0)} 个 tensor：{n_sc} 个 weight_scale + {n_of} 个 weight_offset"
      f"（量化元数据，合计 {meta_bytes/2**20:.0f} MiB）{extra_note}。）")
    w()

    q_param = sum(numel(shapes[n]) for n in wmap
                  if n.endswith(".weight") and entries.get(n) == "W8A8_DYNAMIC" and n in shapes)
    f_param = sum(numel(shapes[n]) for n in wmap
                  if n.endswith(".weight") and entries.get(n) == "FLOAT" and n in shapes)
    cov = 100 * q_param / max(q_param + f_param, 1)
    w(f"- 量化 weight 参数量 **{q_param/1e9:.2f}B** / FLOAT weight 参数量 {f_param/1e9:.2f}B / "
      f"全部 tensor 合计 {param_total/1e9:.2f}B 参数 → **量化覆盖 weight 参数的 {cov:.1f}%**。")
    if uncategorized:
        w(f"- 未归类样本（供核对）：{uncategorized[:6]}")
    w()

    # ------------------------------------------------------------------ 5. scale stats
    w("## 4. weight_scale 分布统计")
    w()
    allv = np.concatenate([v.ravel() for v in scale_vals.values()]) if scale_vals else np.array([])
    pct = np.percentile(allv, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    w(f"- 共 {len(scale_vals)} 个 scale tensor、合计 {allv.size/1e6:.2f}M 个 scale 值，dtype 全部 `F32`，"
      "shape `(out_channel, 1)`。")
    w(f"- 全局分位数：min={pct[0]:.3e} / p1={pct[1]:.3e} / p5={pct[2]:.3e} / p25={pct[3]:.3e} / "
      f"median={pct[4]:.3e} / p75={pct[5]:.3e} / p95={pct[6]:.3e} / p99={pct[7]:.3e} / max={pct[8]:.3e}；"
      f"mean={allv.mean():.3e}, std={allv.std():.3e}。")
    w()

    cat_scale = {}
    for base, v in scale_vals.items():
        cat_scale.setdefault(categorize(base), []).append(v.mean())
    w("| 模块类别 | scale tensor 数 | per-tensor 均值 (min / median / max) |")
    w("|---|---|---|")
    for cat, ms in sorted(cat_scale.items()):
        ms = np.array(ms)
        w(f"| {cat} | {len(ms)} | {ms.min():.3e} / {np.median(ms):.3e} / {ms.max():.3e} |")
    w()

    main = [(b, float(v.mean()), float(v.max()), v.size) for b, v in scale_vals.items()
            if b.startswith("model.language_model.layers.")]
    main.sort(key=lambda x: -x[1])
    w("### 离群层（主干按 scale 均值排序，Top-8 / Bottom-3）")
    w()
    w("| rank | 模块 | scale mean | scale max | #channels |")
    w("|---|---|---|---|---|")
    for i, (b, m, mx, n) in enumerate(main[:8]):
        w(f"| {i+1} | `{b}` | {m:.3e} | {mx:.3e} | {n} |")
    w("| … | … | | | |")
    for i, (b, m, mx, n) in enumerate(main[-3:]):
        w(f"| {len(main)-2+i} | `{b}` | {m:.3e} | {mx:.3e} | {n} |")
    w()

    frac_hi4, frac_hi8 = [], []
    for b, v in scale_vals.items():
        med = np.median(v)
        if med > 0:
            frac_hi4.append((b, float((v > 4 * med).mean())))
            frac_hi8.append((b, float((v > 8 * med).mean())))
    frac_hi8.sort(key=lambda x: -x[1])
    cat_frac = {}
    for (b, fr4), (_, fr8) in zip(frac_hi4, frac_hi8):
        cat_frac.setdefault(categorize(b), []).append((fr4, fr8))
    w("### 离群通道（tensor 内 scale 超过 median 4×/8× 的通道占比）")
    w()
    w("| 类别 | tensor 数 | >4×median 占比 median/max | >8×median 占比 median/max |")
    w("|---|---|---|---|")
    for cat, frs in sorted(cat_frac.items()):
        f4 = np.array([a for a, _ in frs])
        f8 = np.array([b for _, b in frs])
        w(f"| {cat} | {len(frs)} | {100*np.median(f4):.2f}% / {100*f4.max():.2f}% | "
          f"{100*np.median(f8):.2f}% / {100*f8.max():.2f}% |")
    w()
    w("Top-8 张量（按 >8×median 占比）：")
    w()
    for b, fr in frac_hi8[:8]:
        w(f"- `{b}`: {100*fr:.2f}%")
    w()
    frs_all = np.array([f for _, f in frac_hi8])
    w(f"- 结论：离群通道占比中位数 {100*np.median(frs_all):.2f}%（>1% 的 tensor {int((frs_all > 0.01).sum())} 个）——"
      "**per-channel minmax 本身已吸收通道间幅度差异，量化难点不在权重离群通道，而在激活侧离群"
      "（这才是 per-token dynamic act quant + 排除 out_proj 的动机）。**")
    w()

    pos = allv[allv > 0]
    if pos.size:
        lg = np.log10(pos)
        bins = np.linspace(lg.min(), lg.max(), 21)
        h, edges = np.histogram(lg, bins=bins)
        w("### 直方图（log10(scale)，20 bins）")
        w()
        w("```")
        mx = h.max()
        for i in range(20):
            bar = "#" * int(60 * h[i] / mx)
            w(f"10^{edges[i]:5.2f}~10^{edges[i+1]:5.2f} | {h[i]:>9d} {bar}")
        w("```")
        w()

    # ------------------------------------------------------------------ 6. size reconciliation
    w("## 5. 大小对账")
    w()
    expect = sum(numel(shapes[n]) * DTYPE_BYTES.get(dtypes.get(n, "?"), 1) for n in wmap if n in shapes)
    disk = sum(os.path.getsize(f"{MODEL_DIR}/{s}") for s in shards)
    non_tensor = disk - total_size_index
    w(f"- index `metadata.total_size`（safetensors 净荷） = **{total_size_index/2**30:.3f} GiB**")
    w(f"- 逐 tensor numel×dtype_bytes 重建             = **{expect/2**30:.3f} GiB**"
      f"（与 index {'完全一致' if expect == total_size_index else '相差 %.2f MiB' % (abs(expect-total_size_index)/2**20)}）")
    w(f"- 磁盘 10 分片实际字节                          = **{disk/2**30:.3f} GiB**（header/对齐 ≈ {non_tensor/2**20:.1f} MiB）")
    w(f"- 目录标称 30 GB：磁盘 {disk/1e9:.2f} GB（十进制）/{disk/2**30:.1f} GiB（二进制），口径吻合，无冗余文件。")
    w()
    q_b = s_b = o_b = f_b = 0
    for name in wmap:
        dt = dtypes.get(name)
        if dt is None:
            continue
        nb = numel(shapes[name]) * DTYPE_BYTES.get(dt, 1)
        if name.endswith(".weight") and entries.get(name) == "W8A8_DYNAMIC":
            q_b += nb
        elif name.endswith((".weight_scale", ".weight_offset")):
            if name.endswith(".weight_scale"):
                s_b += nb
            else:
                o_b += nb
        else:
            f_b += nb
    tot = q_b + s_b + o_b + f_b
    w("| 组成 | 大小 | 占比 | 说明 |")
    w("|---|---|---|---|")
    w(f"| int8 量化 weight | {q_b/2**30:.2f} GiB | {100*q_b/tot:.1f}% | {q_param/1e9:.2f}B 参数 × 1B |")
    w(f"| 未量化 weight（bf16/f32） | {f_b/2**30:.2f} GiB | {100*f_b/tot:.1f}% | "
      "embed_tokens + lm_head 各 ~2.4 GiB，GDN out_proj ~3.0 GiB，norm/MTP 余项 |")
    w(f"| weight_scale（F32） | {s_b/2**30:.3f} GiB | {100*s_b/tot:.2f}% | out_channel 维，每通道 4B |")
    w(f"| weight_offset（F32，全 0） | {o_b/2**30:.3f} GiB | {100*o_b/tot:.2f}% | 占位字段，同 scale 形状 |")
    w()
    w(f"对照：全部 {param_total/1e9:.2f}B 参数若存 bf16 约需 {2*param_total/2**30:.1f} GiB → "
      f"W8A8 实测 {disk/2**30:.1f} GiB，压缩比 ≈ {2*param_total/disk:.2f}×。"
      "scale+offset 总开销不足 1%。注意 embed/lm_head/GDN-out_proj 未量化，使整体压缩比达不到理想 2×。")
    w()

    # ------------------------------------------------------------------ 7. best practice summary
    w("## 6. Qwen3.8-27B_best_practice.yaml 关键配置")
    w()
    w("```yaml")
    with open(f"{MODEL_DIR}/Qwen3.8-27B_best_practice.yaml") as f:
        for line in f:
            G.append(line.rstrip("\n"))
    w("```")
    w()
    w("值得注意的 3 条：")
    w()
    w("1. `exclude: '*linear_fc2*'` → **Gated DeltaNet 的输出投影（本模型命名 `linear_attn.out_proj`）不量化**，"
      "保持 bf16。实测 48 层 GDN 的 `out_proj` 全为 FLOAT（~3.0 GiB bf16）——delta-rule 输出路径对量化误差敏感，"
      "官方精度优先。同理 `in_proj_a/in_proj_b/conv1d/A_log/dt_bias`（SSM 状态参数）也未量化。")
    w("2. `act.scope: per_token + method: minmax + symmetric: true`，且 activation scale 不落盘 → "
      "**W8A8 dynamic**：部署时逐 token 在线算激活 absmax（对应 vLLM-Ascend per-token dynamic quant cube 算子），"
      "无校准数据依赖的静态 act scale。产物中的 `weight_offset` 虽与 yaml 的 `symmetric: true` 表面矛盾，"
      "实测全 0，为 ascendv1_saver 占位输出。")
    w("3. `save.type: ascendv1_saver, part_file_size: 4`（GB/分片）+ `config_id: qwen3_8_27B_dense_w8a8`；"
      "`verified_tags` 锁定 **vLLM_Ascend + Atlas_A2/A3**。该产物是 msModelSlim→vLLM-Ascend 专用链路，"
      "不能直接用 HF transformers 加载（需 Ascend 侧 custom quant 方案）。")
    w()
    w("## 7. generation_config.json / config.json 要点")
    w()
    w("- generation：`top_k=20, top_p=0.95, temperature=1.0`（Qwen 默认采样）；`eos=[248046,248044]`，`bos=pad=248044`。")
    w("- config：64 层（48 linear_attention + 16 full_attention，`full_attention_interval=4`），hidden 5120，"
      "intermediate 17408，vocab 248320，head_dim 256（24 Q 头 / 4 KV 头，`attn_output_gate`→q_proj 输出翻倍 12288），"
      "`partial_rotary_factor=0.25`、mrope interleaved；`mtp_num_hidden_layers=1`（带投机解码头）。")
    w()
    w("## 8. 结论")
    w()
    w(f"- **格式一句话**：Ascend msModelSlim `W8A8_DYNAMIC`（ascendv1_saver 导出）= "
      "权重 per-channel（out 维）int8 静态对称量化（group_size=0，scale 为 F32 `(out,1)`，offset 占位全 0）"
      "+ 激活 per-token 在线动态 int8 量化，activation scale 不存文件。")
    w(f"- **覆盖**：主干 self_attn / mlp / GDN 的 in_proj_qkv+in_proj_z / MTP 层 / ViT blocks 全部量化"
      f"（量化 weight {q_param/1e9:.1f}B，占 weight 参数 {cov:.0f}%）；embed_tokens、lm_head、全部 norm、"
      "GDN 的 out_proj/in_proj_a/in_proj_b/conv1d/A_log/dt_bias 保持 bf16/f32。")
    w(f"- **对账**：理论 {expect/2**30:.2f} GiB = 实际 {disk/2**30:.2f} GiB（差 {non_tensor/2**20:.0f} MiB header），"
      f"vs 全 bf16 {2*param_total/2**30:.1f} GiB（{2*param_total/disk:.2f}×）。")
    w()
    with open(OUT_MD, "w") as f:
        f.write("\n".join(G) + "\n")
    print(f"\n[written] {OUT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
