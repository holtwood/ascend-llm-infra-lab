# E2 · 量化与服务变体对比

硬件：1× Ascend 910B2（64GB HBM）。负载：random 数据集 512in/128out，
`/v1/completions`，客户端在 M3 经 SSH 隧道打 M1:8000。

## 变体可行性矩阵（先于性能数据的结论）

| 模型变体 | 权重体积 | 能否上卡 | 结论 |
|---|---|---|---|
| Qwen3.8-27B **bf16** | ~71G | ❌ | 超 64G HBM，单卡无法部署——量化不是优化项而是**入场券** |
| Qwen3.8-27B **W8A16** | ~30G | ❌ | vllm_ascend 0.30.0rc2 的 compressed-tensors 路径拒绝：<br>`NotImplementedError: No compressed-tensors compatible quantization type was found`（weight-only 无 input-quant 的配置不受支持） |
| Qwen3.8-27B **W8A8** | ~30G | ✅ | `quantization=ascend` 正常加载，可服务 |
| Qwen3-8B bf16 | ~16G | ✅ | 精度参照组 |
| Qwen3.5-4B bf16 | ~8.8G | ✅ | E1 基线（见 results/e1） |

**面试点**：910B2 单卡 64G → 27B 级模型必须 W8A8（w8a8 同时量化 weight+activation）；
W8A16（仅权重量化）在 ascend 插件的 compressed-tensors 探测里不属于已实现的
quant type——版本支持的量化格式清单决定了模型选型，这是部署侧的真实约束。

## 吞吐/时延实测

### Qwen3.8-27B-W8A8（30G 量化模型，max-model-len 8192）

| 并发 | TTFT mean | TPOT mean | 输出吞吐 tok/s |
|---|---|---|---|
| 1 | 892ms | 31.9ms | 26 |
| 4 | 471ms | 38.9ms | 91 |
| 8 | 557ms | 46.4ms | 149 |
| 16 | 758ms | 63.6ms | 222 |
| 32 | 1023ms | 89.8ms | 312 |
| 64 | 1646ms | 144.0ms | 396 |

原始数据：`e2_w8a8_m1_c{1,4,8,16,32,64}.json`，serve 日志 `serve_w8a8.log`

### Qwen3-8B bf16（16G，同工况）

| 并发 | TTFT mean | TPOT mean | 输出吞吐 tok/s |
|---|---|---|---|
| 8 | 898ms | 20.0ms | 281 |
| 32 | 399ms | 33.8ms | 819 |
| 64 | 477ms | 42.6ms | 1318 |

原始数据：`e2_bf16-8b_c{8,32,64}.json`

### 跨模型同档对比（解读上限）

| c=64 | 输出吞吐 | TPOT |
|---|---|---|
| 4B bf16 (E1) | ~396-807 tok/s* | ~15-30ms |
| **8B bf16** | **1318 tok/s** | 42.6ms |
| 27B W8A8 | 396 tok/s | 144ms |

*E1 不同轮次 c64 数值有波动，量级参考。8B>4B 的反常说明单卡 4B 并未打满
（调度/CPU 侧先到瓶颈），8B 的算术强度更匹配 NPU 算力。
27B-w8a8 吞吐为 8B 的 30%，但承载的是 3.4× 参数量。

## 解读

- **W8A8 让不可能的部署变为可能**：27B 在单卡上跑出 396 tok/s@64，
  代价是 TPOT 随并发快速恶化（c64 时 144ms vs 4B 模型的 ~15-30ms）——
  大模型+小显存场景，batch 一大 decode 就吃紧，KV cache 可用空间小是硬伤
- **量化精度换容量的边界**：W8A8 的 activation 量化有精度风险（评测集上需另测），
  W8A16 更安全但本版本插件不支持 → 选型要在"能跑/精度/吞吐"三角里权衡
- 跨模型对比注意口径：27B-w8a8 与 8B-bf16 吞吐不可直接除出"量化加速比"，
  参数量差 3.4×，算术强度完全不同；本实验的价值点在可行性矩阵+同模型横向 scaling
