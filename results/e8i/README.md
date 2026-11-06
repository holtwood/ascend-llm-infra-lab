# E8i · HCCL 集合通信全参数扫描（8×910B1）

> 扩展 E8d（2 卡 allreduce 单点）到 **2/4/8 卡 × 4 种集合原语 × 256B-256MB** 全矩阵。
> 远端 npt 容器实测，`torchrun`-less `mp.spawn` + `init_process_group("hccl")`。
> 脚本 `scripts/e8i_hccl_sweep.py`，原始数据 `e8i_hccl.json`（上一级目录）。

## 核心结果

### AllReduce 总线带宽（payload×world/t，近似 bus bw）

| world | 1MB | 4MB | 16MB | 64MB | 256MB |
|---|---|---|---|---|---|
| 2 | 15.5 | 31.8 | 35.6 | 36.7 | **37.0** |
| 4 | 19.8 | 78.6 | 122.0 | 137.2 | **122.0** |
| 8 | 32.8 | 136.5 | 383.7 | 519.8 | **480.1** GB/s |

w8 相对 w2 **~13×** 提升——多 rank 摊薄固定开销 + 环/mesh 链路并行兑现，
TP 的通信收益只在 ≥8 卡时才开始显现。

### 延迟底与单链路

| 指标 | 值 |
|---|---|
| allreduce 延迟底 | w2 **146μs** / w4 147μs / w8 189μs（≤2MB 全平坦） |
| p2p send 单向带宽 | **~19.9 GB/s**（三个 world 完全一致 = 单链路天花板，与 E8d 18.5 吻合） |
| allgather 峰值 | 36.3 / 67.8 / 116.1 GB/s（w2/w4/w8） |
| reduce_scatter 峰值 | 35.3 / 66.5 / 121.7 GB/s |
| broadcast 峰值 | 19.4 / 36.8 / 65.7 GB/s（≈allgather 一半，单向扇出特征） |

## 对 TP 成本模型的修正（接 E8d）

E8d 用 2 卡推出"TP-2 decode 每步 ~11.5ms 通信税"。E8i 全矩阵验证并修正：

- decode 每步消息量小（hidden 4096 × bf16 × bs：bs32 ≈ 256KB）→ 全落在
  **延迟平台区**（146-189μs/次），带宽再大也用不上
- 8 卡 TP：2 allreduce/层 × 36 层 × ~160μs ≈ **~11.5ms/step**——rank 数翻倍
  几乎不省（延迟底不是 hop-count 线性），TP 不随卡数改善 decode 延迟
- 结论重申且强化：**中小模型 decode 用副本不用 TP；TP 只在显存装不下
  或 batch 大到消息量出延迟平台区（≥16MB）时才值得**

## 坑位记录

- mp.spawn 子进程崩溃变 zombie 时父进程死在 `q.get()`——调试 HCCL 要
  给 worker 加 try/except + 超时，别裸等队列
- 小消息（≤64KB）有效带宽 <4 GB/s，全被 ~150μs 延迟底吃掉——
  **延迟底才是 TP 的真实成本单位**

环境：8×910B1，CANN 9.0，torch 2.10+torch_npu 2.10（npt 容器）。

## 扩展（通宵队列 run_20261105_160244，8×910B1 全对矩阵）

`results/overnight_20261105_160244/hccl_pairs.json`：56 个有向 pair × 5 尺寸。

- **64MB：全部 pair 19.7 GB/s，完全对称**——无慢链路，TP 不需要
  topology-aware 放置
- **1MB：6.0–7.6 GB/s，pair 间 ±20%**（1→0 = 7.6，2→1 = 5.6）——未饱和区
  才暴露路径差异。**注意：上面"p2p ~19.9 GB/s 恒定"仅指 ≥16MB 的饱和值**
- 1KB：141–215 µs，延迟底与集合通信一致

另见 `tp_decode.json`：真 Qwen3-0.6B 权重下 tp=8 decode 步时 = 单卡 4.47×，
通信占 ~70%，是"小模型用副本不用 TP"的第三条独立证据。
