# N1b：手写引擎 L0 vs L2 阶梯基准（Ascend 910B2，卡 0）

日期：2026-11-06。对象：`examples/engine_np/`（commit 02601e6 数值对齐版），
`bench_ladder.py`（本目录留有副本，远端跑于 `/data/engine_np`）。
bf16，greedy，每档 prompt 128/512/2048 + 30 decode 步；
每档先做同形状 warmup（L0 两次全序列 forward，L2 一次 prefill + 两次 decode），
计时全部以 `torch.npu.synchronize()` 包夹；显存用 `torch.npu.max_memory_allocated`
对 load 后权重基线（8118 MiB）取增量。每步计时脚本是 `generate_no_cache()` /
`generate()` 的逐算子复刻（temperature=0 → argmax），**三档 L0 与 L2 的 30 个
greedy token 全部逐个一致** —— L2 的加速是严格等价的计算，不是近似。

## 核心表（单请求，batch=1）

| prompt | seq 末态 | L0 p50 ms/step | L0 last10 | L2 decode p50 ms | L2 last10 | L0/L2 加速 | L2 prefill ms | 峰值显存增量 L0 / L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 158 | 387.9 | 403.6 | 87.7 | 87.4 | **4.43×** | 378.1 | +58 / +109 MiB |
| 512 | 542 | 454.2 | 456.3 | 88.9 | 89.8 | **5.11×** | 445.5 | +144 / +220 MiB |
| 2048 | 2078 | 856.5 | 861.8 | 89.6 | 89.5 | **9.56×** | 704.5 | +513 / +616 MiB |

首步 vs 稳态：warmup 后首步不再有编译尖峰（L2 首步 88.0/88.0/88.8 ms ≈ p50），
编译开销全部落在 warmup 区，符合预期。原始逐步数据见 `bench_ladder_out.json`。

## 结论

1. **L2 decode 对上下文长度脱钩**：128 → 2078 ctx（16×），每步只 +2.2%
   （87.7 → 89.6 ms）。来源是架构本身：24 层 GDN 是定长状态 O(1)/步，
   8 层 attention 的 KV 读在 2078 token 也只有 66 MiB、q=1 的 SDPA FLOPs
   占比太小——每步 ~89 ms 被 host 侧固定开销主导（32 层 × 每层十余个算子
   的 dispatch/launch + 单 token GEMV）。
2. **L0 随长度线性恶化，斜率 ~0.23 ms/token**：388 → 454 → 857 ms
   （128→512 为 0.173 ms/tok，512→2048 为 0.262 ms/tok，略超线性来自
   chunked GDN 每 64 token 一块的 attn 中间量）。但 L0 在 **128 token 时
   就已是 L2 的 4.4×** —— no-cache 路径的代价大头在固定开销而非重算，
   KV cache 的收益在短上下文就已兑现，随后随长度放大到 9.6×。
3. **显存两者都便宜，L2 的 cache 账面兑现**：L0 峰值增量 58→513 MiB
   （全序列激活，随 T 线性）；L2 109→616 MiB，其中 KV 本身 66 MiB@2078
   （32 KiB/token，与 README 内存账一致），其余为算子工作区。61 GiB 卡上
   两者都远未构成压力。
4. **对照 transformers（同机同卡昨日测量）**：
   - naive 无 cache 411 ms/step @seq16–36 ↔ 我们 L0@128–158 = 388–404 ms：
     量级一致。seq 更长反而略快，印证这些毫秒主要是**每层算子调度的固定
     开销**，不是序列上的计算量。
   - 带 KV cache 手动循环 96–98 ms/step ↔ 我们 L2 88–90 ms/step：**快
     ~8–10%**。推测来自去掉 HF 每步的 DynamicCache 封装、mask/position
     构造等 Python 层开销；算子路径两边相同（同一 SDPA + 参考 GDN kernel）。
   - prefill 4096tk = 1481 ms（0.362 ms/tok）↔ 我们 2048tk = 704.5 ms
     （0.344 ms/tok），线性外推 4096 ≈ 1.41 s，差 ~5%，基本持平。

## 坑记录（影响过第一轮数据）

`bench_l0` 初版直接调 `model(seq)` 而没有 `torch.no_grad()`（引擎方法自带
装饰器，裸循环不会自动有）：autograd 把全序列中间量全部保留，542 token 的
瞬时峰值从 +144 MiB 涨到 **+20.4 GiB**，prompt 2048 直接 OOM 打满 61 GiB
卡；计时也被虚高 ~35%（388→540 ms）。NPU 上验证/压测裸调 forward 时必须
显式 no_grad —— CPU 侧小模型感知不到，属
[[npu-strict-dtype-ops]] 同类"本地过、真机炸"陷阱。
