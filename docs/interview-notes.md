# 面试知识点索引（数据 → 结论 → 原理）

每个条目：一句话结论 + 我们的实测数据 + 背后机制。面试时先抛数据再讲机制。
全部原始数据与解读见 `results/e{N}/README.md`。

## 推理服务化

**Q: vLLM 为什么快？核心机制是什么？**
- Continuous batching：请求随到随插入 decode 批次（而非等整批结束）→ 吞吐近线性扩到几十并发
- PagedAttention：KV cache 按页管理（类 OS 虚拟内存），碎片率低，batch 才能做大
- 实测：Qwen3.5-4B on 910B2，并发 1→64，输出吞吐 75→807 tok/s（E1）

**Q: TTFT 和 TPOT 分别受什么影响？怎么优化？**
- TTFT = 排队 + prefill 算力。并发越高排队越久：实测 c=1→64 TTFT 161ms→797ms（p99 3.3s）
  - 优化：prefix caching（共享前缀免 prefill）、chunked prefill、扩副本分担
- TPOT = decode 单步时延。c=64 时 71ms vs c=1 的 12ms
  - 优化：量化（减带宽）、spec decode、扩副本

**Q: 单流吞吐受什么限制？**
- 小 batch 的 decode 是 memory-bandwidth bound：每 token 读全部权重
  - 4B bf16 ≈ 8.8G/步；910B2 HBM 带宽 ~1.6TB/s → 理论 ~180 tok/s；实测单流 82 tok/s
  - 差距来自 KV cache IO、采样、调度开销 → 这就是量化和 spec decode 的价值点

## 量化与服务变体（E2）

**Q: 27B 模型怎么上 64G 单卡？**
- bf16 27B = 71G > 64G HBM：量化不是优化项而是**入场券**，必须 W8A8（30G）
- 实测 27B-W8A8：c=64 输出吞吐 396 tok/s / TPOT 144ms（vs 8B bf16 1318 tok/s / 42.6ms）
- 代价：batch 一大 decode 就吃力——W8A8 省显存但 KV cache 仍挤占 HBM

**Q: W8A8 / W8A16 / bf16 怎么选？**
- 我们的可行性矩阵：bf16-27B 塞不下；W8A16（仅权重量化）被
  vllm_ascend 0.30rc2 的 compressed-tensors 路径拒绝（不支持无 input-quant 配置）；
  W8A8 可跑但激活量化有精度代价 → "能跑/精度/吞吐"三角权衡
- **先查引擎支持的量化格式清单再谈精度对比**——部署侧的真实约束

## 调度特性消融（E3）

**Q: prefix caching 值多少？**
- 4×2048-token 共享前缀 ×128 请求：关 PC 后 TTFT +28%、TPOT +56%、吞吐 −34%
- 收益 ∝ 负载前缀共享率（system prompt/多轮历史/RAG 模板天然共享）→ 免费午餐；
  前缀全不同则零收益

**Q: speculative decoding 上线前要验证什么？**
- ngram 实测**全负收益**：random 数据集接受率 0-8%、sonnet 真实文本也只有 8-11%
  （接受长度 ~1.5），TPOT 恶化 3.7-6 倍、吞吐 −80%
- 公式：收益 = 接受率 × 每接受省一步 − draft+verify 开销；接受率 <30-40% 直接关
- ngram 的 lookup 只匹配当前请求自身上下文 → 只对"复读机"式输出有效；
  通用路线要 eagle/medusa 类 draft 模型
- **教训：消融要用对方法公平的负载**（random 对 ngram 先天不利）

## 系统扩展性（E4）

**Q: 多副本 vs 张量并行怎么选？**
- 实测 3×单卡副本 + nginx least_conn（SSH 隧道组网）：
  c=128 吞吐 1027→1674→2160 tok/s，扩展效率 1.63×/2.10×
- 亚线性来源：隧道带宽、连接级均衡、NPU 时钟差异
- **扩容只在单副本饱和后兑现**：c=32 时 N=3 无增益（837→837）
- TTFT 改善先于吞吐改善（2751→1262ms）——低延迟 SLA 场景扩容收益更大

## 昇腾生态与工程化（差异化点）

- torch_npu = PyTorch NPU 后端；vllm-ascend = vLLM platform plugin；
  CANN 对标 CUDA；ATB = 推理算子库；HCCS 对标 NVLink
- **MindIE 部署实测**（E5）：3.0.0 全部二进制绑死 glibc2.38/Ubuntu24.04 +
  CANN 8.5.1 + py311，22.04 容器上 8 步排障后物理不可行——
  闭源栈的平台锁定 vs vLLM 源码可移植性，这是引擎选型的隐性成本
- 运维坑位速查（docs/status.md）：editable 包被 namespace 遮蔽、
  pgrep self-match 三形态、ssh 起 daemon 必须 setsid+断 fd、客户端 CPU 污染压测
