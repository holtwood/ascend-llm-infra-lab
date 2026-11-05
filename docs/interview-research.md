# AI Infra 面试深度调研（2026Q4 面经+JD+招聘方视角交叉验证）

来源：Blind 6 个月面经实录（OpenAI/Anthropic/xAI/Meta-MSL/Databricks 环线）、
inherent.sh 推理岗十题、inferenceinterviews.com 题库、Baseten/Together AI 在招 JD、
vLLM 官博（PD 分离部署实录）、科大讯飞星火推理工程部面经、Atlas 国产算力面试指南。
每条结论标注来源类型。

## 一、面试现在到底考什么（按考察频率排）

### 1. 手撕 transformer 是入场券（Blind 实录，多家环线共识）
"Hand-rolled transformer knowledge is table stakes"——MHA、KV cache、RoPE、
GQA/MQA、causal mask、stable softmax **从零实现**，禁用 nn.MultiheadAttention。
Anthropic 特别在意推理效率、tensor 推理、内存/计算权衡、cache 更新正确性。
→ **行动：手写一遍带 KV cache 的 MHA + 能默写 KV 显存公式**

### 2. 两大物理约束（inherent.sh 十题 + Baseten JD 原文）
Prefill 是 compute-bound、decode 是 memory-bandwidth-bound——
"interviewers rarely quiz the tool list; they pick one question forcing
first-principles reasoning." 所有题收敛到五域：prefill 算力 / decode 带宽 /
KV 容量 / 成本质量 / 系统与故障。

高频原题及信号：
- "TTFT 太高排查什么" → 分离 TTFT vs TPOT，先查队列再查算力（陷阱：瞎加卡）
- "为什么 prefill/decode 需要不同硬件" → 这是 PD 分离的存在性证明
- "continuous batching 为什么提吞吐" → 摊薄权重读 + 承认 TPOT 变差（陷阱：说是免费午餐）
- "spec decode 何时失效" → **接受率**，我们的负收益数据直接命中
- "KV cache 怎么算" → 2×layers×kv_heads×head_dim×bytes×ctx_len，
  我们的 27B-w8a8 实测正好验证"并发上限被 KV 挤占"
- "PagedAttention 解决什么" → 内存管理不是加速（陷阱：说它让 attention 更快）

### 3. PD 分离是当下最热（vLLM 官博×2 + Meta llm-d + Baseten JD 点名）
Baseten inference-performance JD 原文职责列表含 "prefill/decode disaggregation"。
vLLM 官方指南的选型表（值得背）：
- ITL p99 不达标 / 长 prompt 高并发 / agent 循环长上下文 → 分离
- TTFT 是瓶颈 / KV 传输慢 / 低流量 → 留在单机
**我们有 4 节点 + mooncake 参考资料，这是最值得补的实验**

### 4. 异构算力适配（国内差异化爆点，讯飞面经原题）
讯飞推理工程部真题："聊聊你对原生和异构算力实践的理解"。
标准答案要素我全部踩过：算子覆盖度（libatb/_libatb_torch 缺算子→PyTorch fallback）、
动态 shape 分档编译（cudagraph_capture_sizes 修 too-many-streams 就是这题）、
通信库差异（HCCL vs NCCL）、同款 memory_utilization 参数两边行为不同、
插件式后端（vllm-ascend 就是参考答案本身）、逐层对拍排障。
**昇腾经验在国产化语境下是稀缺信号**——政企/金融/运营商私有化项目硬指标。

### 5. 分布式+生产素质（Blind：baseline 了）
TP/PP/DP、NCCL 瓶颈、FSDP/ZeRO 概念必背；但真正筛人的是
"production-quality code fast" + 数值稳定性（log-sum-exp、bf16 精度坑、
NaN 排查）+ backprop 推导（softmax+CE 反传仍淘汰大批人）。

### 6. 成本与 SLO 口径（DEV 七步法 + Baseten "inference economics"）
每步要报数：延迟预算分解（embed 15ms/search 20ms/TTFT 300ms）、
$ per 1M tokens、10× 流量时什么先崩。面经反复强调 "reward specific numbers:
HBM capacities, KV-cache footprints, bandwidth ceilings"。
→ **行动：给现有数据补一节 SLO/成本换算**

## 二、让候选人脱颖而出的信号（招聘方+面经交叉）

| 信号 | 证据 | 我们现状 |
|---|---|---|
| 可复现仓库+真实数字 | 多处："reproducible experiment repository separates ML engineers" | ✅ 已具备 |
| 一个能讲 20min 的 hero 项目 | interviewnode 六段式：问题/约束/决策/失败/数字/反思 | ⚠️ 素材够，需选定主角 |
| **开源贡献** | CSDN 面经："30 个被合并的 PR 让面试官对我感兴趣"；FDE JD 要 vLLM 专家级经验 | ❌ 我们有 3 个现成 issue 素材没提 |
| PD 分离实战 | 官博级知识 | ❌ 实验可做未做 |
| 异构算力经验 | 讯飞面经整场围绕它 | ✅ 已具备，需包装成话术 |
| 失败复盘意识 | rubric 明确 "failure awareness & recovery" 是评分维度 | ✅ spec 负收益+MindIE 阻断 |

## 三、差距分析→行动清单（按 ROI）

1. **E7 PD 分离**（最高 ROI）：1P1D 起步→KV 传输对 TTFT 的实测代价；
   回答"何时该分离"用我们数据验证官方选型表
2. **上游 issue/PR**：W8A16 unsupported、cudagraph 流耗尽、
   causal_conv1d fallback——写成英文 issue 提 vllm-ascend
3. **手写 MHA+KV cache**（1-2天）：放 repo examples/，面试手撕同源
4. **E6 精度**：W8A8 vs bf16 自建 QA 对比
5. **SLO/成本节**：README 加"p99 TTFT<500ms 能撑多少并发"+每 M token 卡时
6. **chunked prefill**：补全 E3 网格
7. walkthrough 文档：拿 serve 日志把请求路径画到源码函数级

## 四、叙事建议（跨行者的故事线）

"我在 4×910B2 昇腾集群上从零搭了 vLLM-Ascend 推理实验场：
单卡基线→量化可行性→特性消融→三副本扩展，顺带做了 MindIE 部署审计。
亮点不是数字是判断：spec decode 接受率 10% 判定负收益、
MindIE glibc 绑死判定不可部署、压测客户端 CPU 污染发现并重测。
昇腾经验让我同时覆盖国产化（信创）和通用推理两套话语。"

## 五、参考链接

- inherent.sh/blog/inference-engineering-interview-questions/（十题框架+陷阱）
- inferenceinterviews.com（vLLM/SGLang/TRT-LLM 分引擎题库，标注出题公司）
- vllm.ai/blog/2026-09-29-disaggregated-serving-guide（PD 分离官方选型表）
- vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd（24×B300 PD 实录：TPOT 40→17ms）
- teamblind 6 个月面经实录（手撕 transformer/数值稳定/backprop/分布式）
- meko1.github.io/llm-interview-guide（昇腾软件栈对照表，面试先背段）
- mj.mianlingai.com 讯飞星火面经（异构算力真题+解析）
