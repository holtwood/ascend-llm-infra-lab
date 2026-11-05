# 实验路线图 v2（深度调研版）

调研方法：对照 inferenceinterviews.com 五大考察域（从零实现 / 推理算术 /
系统设计 / kernel&profiling / 指标排障）× 现有 E1–E7 覆盖面做差集，
再用 vllm-ascend 0.30 源码逐项验证可行性（不是"想当然能做"）。

## 现有覆盖面盘点

| 考察域 | 已覆盖 | 缺口 |
|---|---|---|
| 从零实现 | — | **手写 MHA+KV cache（硬缺口）** |
| 推理算术 | E4 副本扩、E7 KV 账本雏型 | **KV 容量公式实测、cost/token** |
| 系统设计 | E4 LB、E7 PD | **goodput@SLO、sleep/wake 多租户** |
| kernel/profiling | issue×3（机理） | MFU 数据、算子级（原 E6 进阶） |
| 指标排障 | E1/E3b/E4b 曲线 | 请求路径源码级 walkthrough |

---

## S 级——面试原题直接命中，且便宜

### E8 · KV cache 容量账本（并发上限的边界条件）
**面试原题**："KV cache 怎么算显存？并发上限被什么挤占？"（inherent.sh 十题 + inferenceinterviews memory-model 题）
- 方法：serve 固定 max_model_len=8192，读 `num_gpu_blocks` 实测值 →
  对比公式 `2×layers×kv_heads×head_dim×2B×block_size` 推算每 token KV；
  再用 `max_num_seqs` 与长 prompt 压到 KV 耗尽，观察排队/抢占/recompute
  （serve 日志有 `Preempted`/`recompute` 计数）
- 指标：每 token KV 实测 vs 公式、可容纳最大并发数、抢占率 vs 时延
- 成本：纯 serve+bench，一晚跑完

### E9 · Goodput@SLO 曲线（吞吐≠有效吞吐）
**面试原题**："goodput 和 throughput 区别？"（2026 面经定义题，多数人答不出）
- 方法：c=4→256 扫描已有工况，按 SLO（TTFT<500ms && TPOT<50ms）
  过滤达标请求算 goodput；画 throughput vs goodput 分叉点
- 指标：goodput 曲线 + "饱和点"定义 + 每 M token 的卡时成本
- 成本：可复用 E1/E3b/E4 数据 + 补 2-3 档新压测

### 手写 MHA+KV cache（examples/mha_from_scratch.py）
**面试原题**：入场券题（Blind 多家环线共识，"table stakes"）
- 产出：单文件 PyTorch 实现 MHA+RoPE+causal mask+KV cache 增量 decode，
  配自测（naive vs cached 输出一致、显存占用打印）
- 成本：半天，无 GPU 依赖

### SLO/成本换算节（analysis，可挂进 results/ 顶层 README）
**面试原题**："p99 TTFT<1s 下能撑多少 QPS？每 M token 多少钱？"
- 用 E1/E4b 数据换算：单卡 4B bf16 的 $/1M tok（按卡时假设价）、
  c128 时的 p99 分解（queue/prefill/decode/net）
- 成本：纯分析，写 markdown

---

## A 级——差异化素材，有真实增量

### E10 · Sleep/Wake 模式（serverless/多租户故事）
**面试原题**："4 分钟加载的模型怎么做 autoscaling？冷启动怎么降？"
- 依据：vllm_ascend worker.sleep/wake_up 已实现，还有官方
  `offline_inference_sleep_mode_npu.py` 示例
- 方法：`--enable-sleep-mode` + `VLLM_SERVER_DEV_MODE=1` 起 4B，
  测 level1（权重留 CPU）/level2（释放）唤醒时延 vs 冷启动 ~60s；
  进阶：4B↔8B 双模型单卡热轮换（doradusresearch 在 NV 上 4.5s）
- 指标：sleep/wake 时延、唤醒后首个请求 TTFT、双模型轮换可用性
- **差异化点**：全网无 Ascend 侧 sleep-mode 数据

### E11 · async_scheduling 消融
**面试原题**："调度器本身有开销吗？decode 两个 step 之间发生什么？"
- 方法：on/off 对照 c=8/32/128，看 TPOT/吞吐差（前租户配置用了它）
- 低成本：一次 serve 重启 + 三档 bench

### E12 · block-size 扫描（PagedAttention 内部）
**面试原题**："PagedAttention 解决什么？"（陷阱题：多数人答'让 attn 更快'）
- 方法：16/32/64/128 四档 × in2048 负载，看 num_blocks 分配、
  KV 碎片率（allocated vs used）、吞吐/TPOT
- 低成本：四次 serve 重启

---

## B 级——进阶/长线（按时间预算选）

### E13 · batch_job_aware_scheduler（Ascend 特有调度器）
- vllm_ascend/core/ 独有：长任务优先+短任务填缝，`#job_name[x]#` 协议
- 混合长度负载（in256+in2048 混流）测尾时延——没人有这种数据
- 风险：调度器可能只走 offline 路径，需读源码确认 serve 可用

### E14 · CPU offload 跑超显存模型（二合一）
- `--cpu-offload-gb` 让 27B bf16 在 64G 卡上跑起来（牺牲速度换可行）
- 附加收益：给 E6 补同实现路径的 bf16-27b QA/PPL 对照，
  消除"CPU-transformers vs NPU-vLLM"的实现噪声
- 风险：Ascend 上 offload 支持未验证（先跑通 serve 再谈 bench）

### E15 · Ascend C 算子（原 E6 进阶，未动）
- RMSNorm/RoPE 融合 kernel + 正确性验证 + 性能对比
- 最高成本（1-3 天），但"写过算子"是稀缺信号

### 非实验项（高性价比，随时可做）
1. **提 3 个 issue 到 vllm-ascend GitHub**——草稿现成，
   "开源贡献"是招聘方点名的差异化信号（issue 链接写进简历）
2. **请求路径 walkthrough 文档**——一次 completion 从 API 层 →
   engine → scheduler → model_runner → NPU kernel 的文件:行号索引，
   证明读源码深度（editable 源码在手，半天）
3. **hero 项目叙事稿**——选 E7-PD（有失败、有机制、有官方选型表对照）
   或 E3b（教科书 tradeoff 完整证据链）写六段式

## 建议执行序

```
E8 KV账本 → E9 goodput → 手写MHA → E10 sleep/wake →
issue提交(1h) → E11 async → E12 block-size → walkthrough →
（预算内）E13/E14/E15
```

理由：S 级全是"原题命中率×单位时间"最高；E10 是独有素材；
E13–15 依剩余时间弹性选择。
