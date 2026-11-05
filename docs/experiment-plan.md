# 实验方案（AI Infra 面试导向）

设计原则：每个实验都回答一个面试高频问题，产出"数据 + 机制解释"。

---

## E1 · vLLM 服务基线压测
**面试问题**：LLM 服务的时延/吞吐随并发怎么变化？TTFT 和 TPOT 分别受什么影响？

- 模型：Qwen3.5-4B（本机）、Qwen3-8B（M1）
- 方法：`vllm serve` + `vllm bench serve`（内置 benchmark_serving），随机数据集
- 扫描维度：并发数 1→2→4→8→16→32→64；输入 512/输出 256 tokens 为主工况，补 2048 长输入工况
- 指标：TTFT P50/P99、TPOT、端到端 QPS、每请求时延分解
- 预期结论：TTFT 随排队近似线性恶化；TPOT 在 compute-bound 区间恶化；吞吐在 KV cache 或算力饱和处见顶 → 引出 continuous batching 的收益曲线

## E2 · 量化对比（W8A8 / W8A16 / BF16）
**面试问题**：量化到底换来了什么？不同量化粒度差别在哪？

- 模型：Qwen3.8-27B 的 w8a8 vs W8A16（M1 本地齐全）；加 Qwen3-8B bf16 做跨尺寸参照
- 方法：同一 benchmark 工况跑三套权重，另跑 lm-eval 或人工评测集看质量损失
- 指标：显存占用、TPOT、吞吐、质量分（如可跑）
- 预期结论：W8A8 显存减半、吞吐提升但依赖算子支持；W8A16 更省带宽场景；量化是"用算力换显存带宽"的典型案例

## E3 · 调度特性消融
**面试问题**：vLLM 的 PagedAttention / prefix caching / chunked prefill / spec decode 各解决什么问题？

- 方法：editable 源码 + serve 参数开关，逐项消融
  - `--enable-prefix-caching` on/off × 重复前缀负载
  - `--enable-chunked-prefill` + `--max-num-batched-tokens` 扫描（prefill/decode 干扰）
  - spec decoding：若 clef-flash joint_head 可用则做 MTP/EAGLE 式 draft（M1 有资源）
- 指标：TTFT、TPOT、吞吐、接受率（spec）
- 预期结论：prefix caching 对共享前缀负载 TTFT 近零；chunked prefill 牺牲少量吞吐换 TPOT 平稳；spec decode 对低并发收益大、高并发可能负收益（draft 占算力）

## E4 · 多副本扩展性
**面试问题**：单卡容量不够时怎么扩？副本扩容的线性度如何？

- 拓扑：D 机 LB（nginx round-robin 或自写路由）→ M3/M1/M2 各一副本
- 方法：同模型（Qwen3.5-4B，scp 到 M2）1→2→3 副本，固定并发压测总吞吐
- 指标：聚合吞吐/副本数、尾时延变化
- 预期结论：无共享状态下近线性扩展；引出"何时需要 TP 而非副本"（模型放不下的情形）

## E5 · MindIE vs vLLM-Ascend
**面试问题**：官方推理引擎和开源栈差在哪？

- 方法：M1 用 `/model/mindie` 部署包起 MindIE 服务，同工况对比 vLLM
- 指标：吞吐/时延/功能面（量化格式、特性支持）
- 价值：国产化生态的差异化认知

## E6 · Ascend C 算子（进阶）
**面试问题**：算子级优化怎么做？

- 方法：用 ascend-toolkit + cannsim 写一个融合算子（RMSNorm 或 RoPE），对比基线
- 产出：kernel 代码 + 正确性验证 + 性能数据

---

## 复跑约定

- 脚本在 `scripts/`，命名 `eN_*.sh`；结果写 `results/eN/`
- 服务日志统一 `results/logs/`
- 每个实验完成后更新本文件状态与 `interview-notes.md`
