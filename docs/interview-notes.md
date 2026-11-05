# 面试知识点索引（数据 → 结论 → 原理）

每个条目：一句话结论 + 我们的实测数据 + 背后机制。面试时先抛数据再讲机制。

## 推理服务化

**Q: vLLM 为什么快？核心机制是什么？**
- Continuous batching：请求随到随插入 decode 批次（而非等整批结束）→ 吞吐近线性扩到几十并发
- PagedAttention：KV cache 按页管理（类 OS 虚拟内存），碎片率低，batch 才能做大
- 实测：Qwen3.5-4B on 910B2，并发 1→64，输出吞吐 75→807 tok/s（E1）
- 源码位置：`/vllm-workspace/vllm/vllm/v1/core/scheduler.py`（调度）、`vllm/v1/engine/`（执行）

**Q: TTFT 和 TPOT 分别受什么影响？怎么优化？**
- TTFT = 排队 + prefill 算力。并发越高排队越久：实测 c=1→64 TTFT 161ms→797ms（p99 3.3s）
  - 优化：prefix caching（共享前缀免 prefill）、chunked prefill（prefill 切块混入 decode，降 TPOT 抖动）、扩副本分担
- TPOT = decode 单步时延 ≈ batch 内单请求摊到的算力。c=64 时 71ms vs c=1 的 12ms
  - 优化：量化（减带宽）、spec decode（一次出多 token）、减小 max batch

**Q: 单流吞吐受什么限制？**
- 小 batch 的 decode 是 memory-bandwidth bound：每 token 要读全部权重
  - 4B bf16 ≈ 8.8G/步；910B2 HBM 带宽 ~1.6TB/s → 理论 ~180 tok/s；实测单流 82 tok/s（E1 c=1）
  - 差距来自 KV cache IO、采样、调度开销 → 这就是量化和 spec decode 的价值点

## 量化（E2 数据待补）

- W8A8 vs W8A16 vs BF16：显存减半 vs 激活也量化；对 TPOT/吞吐/精度的影响——见 results/e2/

## 调度特性消融（E3 数据待补）

- prefix caching / chunked prefill / ngram spec decode 的实测收益——见 results/e3/

## 系统扩展性（E4 数据待补）

- 多副本 vs TP 的取舍：副本扩容吃网络无成本、无负载倾斜问题；TP 用于单卡放不下的场景
- 实测 1→N 副本吞吐线性度——见 results/e4/

## 昇腾生态（差异化点）

- torch_npu 是 PyTorch 的 NPU 后端插件；vllm-ascend 是 vLLM 的 Ascend platform plugin（editable 源码在 /vllm-workspace）
- CANN = 华为异构计算栈（对标 CUDA）；ATB = 推理算子库（量化模型必装）；HCCS = 片间互联（对标 NVLink）
- 实操坑位：aarch64 wheel 生态、LD_LIBRARY_PATH 驱动/toolkit/ATB 三段式、HF 不通走 hf-mirror
