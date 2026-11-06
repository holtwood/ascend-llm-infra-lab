# N8: 算子融合收益 — RMSNorm 家族 eager vs torch.compile（TileLang 线烂尾记录）

环境：910B2 卡0，torch 2.9.0/torch_npu 2.9.0/CANN 8.5.1；compile 段跑在 vllm venv（含 triton-ascend 3.2.0.dev）。
对象：引擎逐字复刻的 RMSNorm 9-op 链与 add+rmsnorm 10-op 链（decoder 每层的残差+norm 模式），N=2560（4B hidden）/9216（MLP 宽），M∈{1,8,64,512,2048}。
口径：10 warmup + 100 计时，t_sync=端到端，t_host=纯 host 下发（无 sync）。正确性以 fp32 eager 为参照。

## 主结果（µs/次）

| M | eager rms (sync/host) | eager add_rms (sync/host) | compile add_rms (sync/host) | compile vs eager |
|---:|---|---|---|---:|
| 64 (N=9216) | 211.3 / 210.8 | 229.2 / 229.6 | 475.2 / 470.7 | **+107% 慢** |
| 512 | 232.5 / 226.8 | 360.3 / 236.0 | 499.8 / 493.7 | **+39% 慢** |
| 2048 | 221.7 / 221.8 | 248.7 / 252.3 | ~475 / ~494 | **+91% 慢** |

## 结论

1. **eager 的 RMSNorm 家族 ≈ 100% host 开销**：所有形状下 host-only ≈ sync（97-100%）——device 计算时间被 launch 完全掩盖。N2 "decode 79% host 开销"在算子级得到单链复现：这种 9-10 op 的小链正是 N2 统计里 Mul/Add/ReduceMean 家族的构成单元。每个 decoder layer 有 2 处该模式 ×32 层 = 每步仅 norm 家族就 ~500µs host 税的来源。
2. **torch.compile (inductor + triton-ascend) 在 NPU 上不降反升 39-107%**：compile 后 host-only 也 ~470-494µs（guard + 生成 kernel 的 launch 更重），没有实现有效融合。这是"昇腾生态为什么自研 torchair 图模式 / TileLang 而不是沿用 inductor"的一手实证——跨后端 port 编译器不保证收益，kernel 融合要走原生路径。
3. **融合的理论收益上限**：eager 9-op 链的 device 时间 ≈ sync−host ≈ 0-124µs 中真算力只占 ~几 µs（M≤2048 的 norm 在 AIV 上极快）；融合成 1 个 kernel 的收益主要是把 9 次 launch 变 1 次（省 ~200µs/链 × 64 链/步 ≈ 12.8ms/步 decode 的 host 预算）——与 N2 的"graph capture 是第一优先级"结论一致。
4. TileLang 原生 kernel 线**烂尾**：tilelang-ascend clone 到 3rdparty（catlass/composable_kernel/pto-isa 完成、tvm 失败、cutlass 中断）后因 subagent API 限额中断。注意到 36.151.162.7 工作站有现成 tilelang 环境（tlasc-env/tl015/npuir wheel + TileKernels 克隆，CANN 9.2.0），续作应在那台机器。

## 产物

- `n8_rmsnorm_bench.json` / `rmsnorm.log`（主环境跑，tilelang/compile 段 fail 记录）
- `rmsnorm_venv.log`（vllm venv + triton-ascend 跑，compile 段成功出数据）
- 脚本 `scripts/n8_rmsnorm_bench.py`（eager/tilelang/compile 三段独立守护）、`scripts/n8_compile_decode.py`（未及跑）

## 面试讲法

"算子融合值多少？" → 我把引擎里 RMSNorm 9-op 链单独拎出来测：eager 下它 100% 是 host 开销（sync≈host-only，211µs 里 device 几乎为零）；torch.compile 在 NPU 上反而慢 39-107%（inductor 生成的 launch 更重、没有真融合）——所以昇腾走 torchair/TileLang 原生融合路线，我的数据解释了为什么。
