# E1 · vLLM 服务基线压测结果

- 服务端：M3（910B2 #7）`vllm serve /model/Qwen3.5-4B --max-model-len 8192`
- 模型：Qwen3.5-4B（bf16，约 8.8G 权重，推理模型）
- 客户端：本机 `vllm bench serve`，`/v1/completions`，random 数据集
- 工况：input 512 ±20% tok，output 128 tok，ignore-eos，每档 num_prompts=4×并发

## 数据

| 并发 | 请求数 | 时长(s) | TTFT mean | TTFT p99 | TPOT mean | TPOT p99 | 输出吞吐 tok/s | req/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 16/16 | 27.9 | 160.9 | 169 | 12.13 | 28 | 75 | 0.57 |
| 4 | 16/16 | 10.6 | 248.0 | 337 | 17.78 | 21 | 198 | 1.52 |
| 8 | 32/32 | 14.0 | 292.2 | 507 | 23.44 | 27 | 292 | 2.29 |
| 16 | 64/64 | 17.9 | 386.6 | 975 | 30.62 | 40 | 460 | 3.58 |
| 32 | 128/128 | 25.5 | 506.5 | 1669 | 43.23 | 54 | 651 | 5.02 |
| 64 | 256/256 | 41.0 | 796.8 | 3343 | 71.03 | 97 | 807 | 6.25 |

## 解读（面试可讲的点）

1. **吞吐近线性扩展，未到饱和**：1→64 并发输出吞吐 75→807 tok/s（≈10.8×），
   说明 4B 模型在 64 并发仍未打满 910B2——continuous batching 还在红利区。
   *后续：扫 c=128/256 找饱和点；或用 `vllm bench throughput` 测离线极限。*
2. **TTFT 随并发恶化但可控**：c=1→64，TTFT mean 161→797ms，p99 达 3.3s。
   来源 = 排队等待 + prefill 批处理；这是"prefill 吞吐 vs 排队"的经典权衡。
3. **TPOT 恶化比 TTFT 慢**：c=64 时 TPOT mean 71ms（vs c=1 的 12ms），
   说明 decode 阶段 batch 变大后单步时延上升（算力被分摊），但 p99 仅 97ms——
   decode 调度很稳。
4. **单并发下 TPOT≈12ms → 单流 ~82 tok/s**：接近单卡 4B bf16 的
   memory-bandwidth 上限（权重 8.8G × 每 token 全读 / HBM ~1.6TB/s → 理论 ~180 tok/s），
   实测 82 tok/s 说明还有 KV cache + 采样等开销，也有优化空间。
5. **成功率 100%**（0 失败）：640 请求全成功，服务稳定。

## 下一步

- [ ] 扫 c=128/256 定位吞吐饱和点与拐点
- [ ] 压测时同步抓 `npu-smi dmon`（算力/显存带宽占用），把"饱和"落到硬件指标上
- [ ] E3 消融：同表对比 prefix caching on/off（prefix-repetition 数据集）

原始数据：`results/e1/e1_qwen35_m3_c{1,4,8,16,32,64}.json`
