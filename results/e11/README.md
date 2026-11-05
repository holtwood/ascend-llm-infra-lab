# E11 — async_scheduling 消融：调度器/模型步流水线重叠值多少钱

> 面试原题："vLLM V1 的 async scheduling 解决什么问题？"
> 前提：vllm-ascend 0.30 **默认开启** async scheduling（E13 中 BatchJobAware
> 自动选用 Async 变体为证）。消融方向 = `--no-async-scheduling` 关 vs 默认开。

## 负载

decode 重型：`random` 数据集 in=256 / out=512，192 prompts，c64，
qwen35-4b @ util0.5，单卡 910B2。

## 结果

| 指标 | async ON（默认） | async OFF（sync） | Δ |
|---|---|---|---|
| 输出吞吐 | **2067.7 tok/s** | 1291.2 tok/s | **+60.1%** |
| TPOT mean | **29.4 ms** | 48.4 ms | **−39.3%** |
| TPOT median | 29.5 | 48.2 | −38.8% |
| ITL mean | 29.4 | 48.4 | −39.3% |
| e2e mean | 15.8 s | 25.3 s | −37.6% |
| 总耗时 | 47.5 s | 76.1 s | |
| TTFT mean | 816.9 ms | 608.0 ms | −25.6%（sync 反优） |

## 机制解释

- **async**：调度器在 NPU 执行本步时**并行准备下一步调度决策**
  （CPU 侧：采样、KV 分配、block 表更新），隐藏调度开销。
  decode 每步只有 ~30ms，调度开销占比高 → 收益大（TPOT −39%）。
- **sync**：每个 decode 步串行 调度→执行→采样→调度，CPU 调度时间
  全额串进 token 间隔。在 910B2 这种主机端相对慢的平台，调度步
  ~19ms/token 直接可见。
- **TTFT 反向**：sync 模式 prefill 步调度决策更少（无重叠流水线
  抢占 prefill 间隙）→ TTFT 略优（608 vs 817ms）。
  但 decode 期差距碾压这一点。
- 推论：batch size 越大、单步越久，async 收益越小；TP1 小步长
  decode 场景收益最大——本次 TP1+64并发正是其主场。

数据：`bench_async_on.json` / `bench_async_off.json`（vllm bench serve,
`--percentile-metrics ttft,tpot,itl,e2el`）、`serve_sync.log`、
`drv_async_on/off.log`。
