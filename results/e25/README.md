# E25 — 唤醒后稳态性能：无损失（补 E17 追问缺口）

> 追问链："换租快是快，**唤醒后性能有下降吗？**"
> 数据回答：warm 696.6 tok/s / TPOT 18.9ms → sleep2 1.74s → wake 0.125s →
> post-wake **694.9 tok/s / TPOT 19.3ms**——差异 −0.24%，在噪声内。

## 数据（同负载 c16/n64/in256/out96，M3 910B2，util0.5）

| 阶段 | 吞吐 | TPOT mean | TTFT mean |
|---|---|---|---|
| warm（服务就绪后） | 696.6 tok/s | 18.9ms | 409.8ms |
| sleep level=2 | 1.74s | — | — |
| wake_up | 0.125s | — | — |
| **post-wake（立即）** | **694.9 tok/s** | 19.3ms | 370.3ms |

## 机制解释

1. **为什么无损失**：CaMem 把权重按原地址回灌、KV 池指针恢复，
   ACL graph 与编译产物都保留——唤醒后执行路径与 warm 完全一致，
   不触发重新编译或 graph capture。
2. **TTFT 反而略低**（370 vs 410ms）：负载轻 + 唤醒后权重驻留局部性更好，
   属噪声范围（单次 64 请求，±10% 波动）。
3. **与冷启动的对照**（E17）：冷启 169.4s 且首请求有编译尾部；
   wake 0.125s 且首请求 0.129s 即可全速——差异来自"元数据保留"。

## 面试口径

- "sleep/wake 不是把进程冻住再解冻——它保留 allocator/graph 元数据，
  所以唤醒后吞吐与 warm 一致（696.6→694.9，噪声内）"（有数据）
- "真正的成本只有 sleep 的 1.74s PCIe 拷贝；wake 端 0.125s 且无稳态损失"

## 复现

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve /model/Qwen3.5-4B --port 8000 \
  --max-model-len 8192 --enable-sleep-mode --gpu-memory-utilization 0.5 \
  --additional-config '{"rl_config":{"enabled":true}}'
# warm bench → POST /sleep?level=2 → POST /wake_up → 同 bench 再跑
```
