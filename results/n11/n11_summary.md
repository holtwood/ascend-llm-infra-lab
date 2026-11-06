# N11: GDN chunked kernel — chunk size 扫描

日期：2026-11-06 · 环境 B（223.111.149.177:40045），Ascend910B2 卡 0，纯 torch kernel（`examples/engine_np/gdn_kernels.py`）。

## 设置

- 零 diff：`chunked_gated_delta_rule` 本就接受 `chunk_size` 关键字参数（gdn_kernels.py:79，默认 `CHUNK_SIZE=64`），无需参数化改动。
- 独立微基准，不加载模型：随机张量直接调 kernel，q/k/v (1, L, 32, 128) bf16（kernel 内部自行升 fp32），g/beta (1, L, 32) bf16。
- 每档 10 warmup + 50 计时（`time.perf_counter` + `torch.npu.synchronize` 包夹），脚本 `scripts/n11_chunk_scan.py`，原始数据 `results/n11/results_n11.json`、日志 `results/n11/n11.log`。
- 全部 8 档（2 L × 4 chunk）均可行，无块结构限制——kernel 对任意 chunk 做显式 pad + reshape，纯实现层面约束。

## 正确性前置检查

T=300（走 pad 分支）下各 chunk 对 chunk=32 参考的偏差：max|out| ≈ 1.22e-04（bf16 噪声水平），max|state| ≤ 1.79e-07。chunk size 只改结合顺序，不改数学结果。

## 结果（ms / call，50 次均值）

| chunk | L=1024 | L=4096 | n_chunks (L=4096) |
|------:|-------:|-------:|------------------:|
| 32    | 24.00  | 75.84  | 128               |
| **64**| **21.01** | 46.68 | 64             |
| 128   | 28.86  | **41.14** | 32            |
| 256   | 50.19  | 119.67 | 16                |

最优点：**L=1024 → chunk 64**（与 FLA 默认一致）；**L=4096 → chunk 128**（比默认 64 快 12%，比最差的 32/256 快 1.8–2.9x）。

## 机制：为什么不是"chunk 越大越好"

该纯 torch 实现有两个对 chunk 反向的代价项，扫描落在两者之间：

1. **主循环 launch 数** ∝ n_chunks = L/chunk：chunk 32 时 L=4096 要 128 次 Python 循环迭代（每次 5 个左右 matmul/einsum launch），chunk 32 的 75.8ms 主要耗在这。
2. **前向替换循环**（gdn_kernels.py:118 `for i in range(1, chunk_size)`，三角矩阵求逆）迭代数 = chunk 本身，且每次迭代操作 (B,H,n_chunks,chunk,chunk) 张量——总置换代价 ∝ chunk³ × n_chunks = chunk² × L。chunk=256 时 attn 中间矩阵 (1,32,16,256,256) fp32 = 134MB，255 次串行小迭代，119.7ms 全耗在这。

两者之积在 64–128 间取最小；序列越长，主循环 launch 项权重越大，最优点从 64 右移到 128。中间 attn 矩阵 (chunk²/L 项) 在 chunk≥128 后显存带宽与串行替换双重惩罚，chunk=256 双 L 下均劣于默认。

## 结论

- 仓库默认 chunk=64 对短 prefill（≤1k）已最优；长 prefill（≥4k）值得把 chunk 提到 128，预填充吞吐 +12%。
- chunk=256 在此实现下任何长度都不建议：前向替换的 O(chunk²·L) 串行迭代吞掉全部 launch 收益。
- 注：这是纯 torch 逐算子实现的成绩，launch 开销远高于 FLA Triton kernel；若换 fused kernel，最优点可能进一步右移（launch 项坍缩后只剩 chunk² 项）。

## 未完成部分

无——8 档全部完成（任务在启动后约 60s 内跑完）。
