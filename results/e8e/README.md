# E8e — 手撕 attention 变体：RoPE / GQA / Sliding-Window

910B1（pytorch 容器 npu:1，torch 2.10 + torch_npu）纯手写实现 + 独立参考对拍。
脚本：`scripts/e8e_attention_variants.py`；原始数据：`e8e.json`。
定位：补齐"手撕 transformer"系列的 attention 三件套，与 E8b（融合 vs
naive attention）同系列。

## 结果速览

| 实验 | 正确性 | 关键发现 |
|---|---|---|
| RoPE | vs 复数乘法参考 **diff=0**；vs HF `apply_rotary_pos_emb` 0.038（bf16 噪声）；范数保持 1.9e-6；相对位置性质 8.6e-6 | rotate-half 与复数旋转数学等价，NeoX 约定实现正确 |
| GQA | `expand` view == `repeat_interleave`（精确）；vs 原生 `sdpa(enable_gqa=True)` **diff=0** | 每请求 KV 268→33.6 MB（Hq32/Hkv4，8×）——GQA 的收益是 KV 字节不是 FLOPs |
| SWA | vs sdpa-masked 参考 ≤3.6e-7 | **显式 attn_mask 在 NPU 上比 is_causal 慢 26×** 且窗口大小零收益 |

## RoPE（theta=1e6, D=128, S=4096）

手写 `rotate_half` 版本：

```python
x * cat(cos,cos) + rotate_half(x) * cat(sin,sin)
```

对拍口径：复数乘法 `(a+bi)·e^{ipw}` 独立实现（同一 NeoX 配对）+ HF
transformers `apply_rotary_pos_emb` + 两条代数性质：

- **diff=0**（vs 复数参考）——两实现逐位相同
- 范数保持误差 1.9e-6：旋转是正交变换的实锤
- `<Rope_m(q), Rope_n(k)> = <Rope_{m-n}(q), Rope_0(k)>` 误差 8.6e-6：
  **注意力只看相对位置**——这是 RoPE 存在的原因，也是"位置外推/插值
  为什么有效"的底层性质
- vs HF 0.038：bf16 输入下 rotate-half 与 HF 的 sin/cos 融合路径的舍入差，
  fp32 下应归零（正确性结论不变）

**面试点**：RoPE 把绝对位置编入相对角度——`R_m^T R_n = R_{n-m}`；长上下文
插值（position interpolation / YaRN）就是重标定 `inv_freq` 里的 `p` 或
`theta`。

## GQA（Hq=32, Hkv=4, D=128, S=8192）

- `unsqueeze→expand→reshape` 零拷贝 view 与 `repeat_interleave` **位级一致**
- 与原生 `sdpa(enable_gqa=True)` 输出 **diff=0**——重复 KV 头在数学上
  就是 GQA 语义
- **KV 字节账本**：bf16 下每请求 K+V 从 268.4 MB 压到 33.6 MB（8×）——
  这是 serving 侧 GQA 存在的真正原因：decode 是访存 bound（E8b 实测
  matvec 1475 GB/s ≈ HBM 峰值 92%），KV 读量决定 TPOT；同时 KV 池容量
  也按字节数线性放大（E8/E16 的容量账本直接受益）

## SWA（B=1, H=8, D=64）——本实验最有信息的负结果

正确性：手写 `band_mask` + softmax 与 `sdpa(attn_mask=band)` 逐窗口一致
（≤3.6e-7）。

**但性能反直觉**（910B1 实测，ms/iter）：

| S | `sdpa(is_causal)` | `sdpa(attn_mask=band)` w256 | w1024 | 手写 fp32 masked w256 |
|---|---|---|---|---|
| 2048 | 0.224 | 1.069 | 1.073 | 4.108 |
| 4096 | 0.302 | 5.699 | 5.723 | 11.17 |
| 8192 | 0.928 | **24.19** | 24.13 | 33.52 |

两个独立结论：

1. **attn_mask 路径 ~26× 慢于 is_causal**（S8192: 24.2 vs 0.93ms）——
   显式 mask 使 sdpa 落入通用 masked kernel，丢掉 fused causal 快路径。
   vllm-ascend 源码印证：`attention_v1.py:956-972` 滑窗走 FIA
   `sparse_mode=4` + `pre_tokens=sliding_window`——**窗口是 kernel 参数，
   不是 mask**。
2. **窗口缩小零收益**（w256 ≈ w1024）：masked kernel 仍算满 S² 分数矩阵，
   band 只改 softmax 输入不改计算量——真 SWA 的收益依赖 kernel **跳过**
   窗外 block（flash-attn `window_size` 参数的分块跳过），mask 永远拿不到
   这个收益。

**面试点**：
- "SWA 怎么实现"——正确答案是 kernel 级窗口（FIA sparse_mode=4 /
  flash-attn window_size），不是构造 mask；mask 版复杂度仍是 O(S²·d)。
- "KV cache 呢"——SWA 的另一收益在 cache 侧：只保留 w 个 token 的 KV
  （vllm-ascend `sliding_window_manager`/RingBuffer），容量 O(w) 与 S 无关；
  本实验的 mask 路径不涉及 cache。
- 负结果本身就是证据：**"我实测过，mask 版 SWA 在 910B 上比全因果还慢
  26×，因为……"** 比背概念有说服力得多。

## 复现

```bash
# pytorch 容器（117.131.156.71:2307，经跳板 42.194.206.190）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -u scripts/e8e_attention_variants.py \
  --device npu:1 --json-out results/e8e/e8e.json
```

## 局限

- sdpa 的 mask 路径耗时含 mask 广播/加法，未分离 mask 加法本身的固定开销；
  但 `is_causal` vs `attn_mask` 26× 的量级差异不依赖该分解
- 未测 FIA `sparse_mode=4` 真窗口 kernel（需 vllm-ascend 运行时，属旧环境）
- RoPE 只验证了 NeoX 约定；GPT-J interleaved 约定未单独实现（两约定差
  一个固定 channel 置换，性质相同）
