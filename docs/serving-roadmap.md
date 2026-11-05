# AI-Infra Serving 方向：环境盘点 × 面试考点 × 实验路线

调研来源：llm-d roadmap、InferenceInterviews、Blind 面经（Anthropic/Meta-MSL/xAI/DB）、
fahimfaisal LLM serving 面经、SGLang/vLLM 官方文档、Mooncake/FlowKV/PrisKV 论文与博客。
产出日期：2026-11-05。

## 1. Serving 面试考点分层（调研归纳）

按出现频率和区分度排序，面试官实际考的是四层：

```
L4 生产判断    SLO 工程、成本核算、容量规划、故障诊断、"什么时候用什么"
L3 系统架构    PD 分离、KV 分级存储、cache-aware 路由、调度器策略、多租户
L2 引擎内部    continuous batching、PagedAttention、prefix caching、
               preemption/admission、spec decode、async scheduler
L1 底层机制    KV cache 数学、访存墙、融合算子、TP/通信原语、量化
```

L1-L2 已被现有实验覆盖（E1/E8b/E8c/E8d/E3/E3b/E10/E11）。
**当前缺口集中在 L3-L4**：路由与调度智能、KV 分级复用、SLO 驱动的容量故事。

## 2. 环境资产盘点

| 资产 | 位置 | 能干什么 |
|---|---|---|
| 4×910B2 + vLLM-ascend（可跑） | 旧环境·腾讯云 lane | 多副本路由、PD、调度策略、量化 serving 全部可测 |
| `/model` 共享模型缓存（73 项） | 旧环境 | Qwen 全系 + mooncake 参考目录 |
| pytorch 容器（torch_npu+CANN9，可见 8 卡） | 新环境·本 lane | 手撕微观实验、HCCL、分布式原语 |
| MindIE 2.3.0 | 新环境 | **栈级挂死已结案**（vendor bug 素材） |
| MindSpore 2.10（裸） | 新环境 | 仅补 mindformers 后有价值，低优先级 |
| /vllm-workspace 源码（vllm + vllm-ascend） | 本容器 | 源码 walkthrough、上游 issue/PR |
| vllm-ascend 独有调度器 | 源码 | `batch_job_aware`、`dyntra_lb`、`recompute`、`short_request_first`、`profiling_chunk_predictor` |
| vllm-ascend KV transfer 栈 | 源码 | `kv_p2p`、`kv_pool`、**`sparse_kv_offload`**（KV 分级卸载！） |

## 3. 缺口 → 实验映射（按面试 ROI 排序）

### S 级：直接命中最高频考点，且环境可支撑

**E14 APC/prefix caching 定量**（旧环境，~半天）
- 内容：`--enable-prefix-caching` 开/关 × 共享 system prompt 长度
  (256/1024/4096) × 并发，测 TTFT 降幅 + hit rate。
- 面试价值：高频题"prefix caching 能省多少"——行业口径 40-60%，
  你能给 NPU 实测数。是 E15 的基线。
- 风险：低，vLLM V1 APC 默认可用。

**E15 prefix-cache-aware 路由**（旧环境，1-2 天，**hero 级**）
- 内容：E4 的 3-4 副本 + D 上写 sticky-prefix 路由器
  （对 prompt 前缀 hash → 固定副本），对照 round-robin：
  每副本 APC 命中率、TTFT p99、吞吐。就是 llm-d
  `precise-prefix-cache-routing` 的迷你复刻。
- 面试价值：这是当前 serving 最热的功能（llm-d/SGLang router 主打）。
  "我在 Ascend 上复刻了 cache-aware 路由并量化了收益"——
  单这一条就是一场 30 分钟深挖。
- 实现量：~200 行 Python 路由层替换 nginx。

**源码级请求 walkthrough**（本容器，~1 天，零硬件依赖）
- 内容：从 `vllm/entrypoints` → engine → scheduler → KV block manager →
  model_runner → attention backend → detok/stream，以 file:line 写穿一个
  请求的生命周期，配我们实测日志佐证每一段。
- 面试价值：Blind 面经里出现频率最高的题
  （"walk a request through vLLM"）。零成本，必须做。
- 注意：按 AGENTS.md，用 uv/.venv 读代码即可，不改代码。

**上游 issue 落地**（本容器，~半天）
- 三个草稿已有（W8A16 拒绝/cudagraph 流耗尽/conv1d fallback）。
  按 vllm-ascend AGENTS 流程：查重复 → 提到 vllm-ascend repo。
  外加第 4 个候选：**MindIE set_weight 栈级挂死**——虽是 MindIE 非
  vLLM，但可以以"环境兼容性"角度提给 atb-models 或作为面经素材。
- 面试价值："我给 vLLM 生态报过 3 个真实 bug"一句顶十个压测数。

### A 级：差异化强，成本中等

**E16 KV 分级卸载（sparse_kv_offload）**（旧环境，1-2 天）
- vllm-ascend 自带 `sparse_kv_offload` connector——Mooncake 式
  HBM→DRAM 分级 KV 在 Ascend 上的实测，目前几乎没人有这数据。
- 测：长 prompt 重问场景下 L1 DRAM 命中率、TTFT 变化、显存节省。
- 面试价值：KV-centric 架构（Mooncake/PrisKV/llm-d）是 2026 面试新贵。

**E17 Ascend 独有调度器矩阵**（旧环境，1 天）
- `short_request_first`（SRF/最短优先） vs `batch_job_aware`（E13 已测）
  vs `recompute` vs `dyntra_lb`，混合负载下 SLO 对比。
- 面试价值：调度策略是 L3 核心考点，而且这是 Ascend 独有资产，
  别人没测过——"我对比了四种调度器在混合负载下的 SLO 表现"。

**E18 multi-LoRA serving**（旧环境，1 天+找 LoRA）
- 单引擎挂 2-3 个 LoRA adapter，测 per-request adapter 路由、
  切换开销、LoRA-aware 公平性（llm-d roadmap 里明确列为新方向）。
- 风险：需 Qwen3 LoRA 权重（HF 现找或快速训一个小的）。

**E19 guided/structured decoding 开销**（旧环境，半天）
- guided JSON/regex vs freeform：mask 计算对 TPOT 的影响。
  SGLang xgrammar 宣称 10×——你在 Ascend 上量一版真实数。
- 面试价值：structured output 是 agentic serving 的显学，差异化强。

### B 级：看剩余时间

- **E20 aclgraph on/off** decode 延迟（Ascend 版 cudagraph capture 实测，
  呼应已写的 cudagraph issue）
- **级联路由 demo**：0.6B 快答 + 8B 兜底的质量路由原型
  （成本/token 故事，偏设计轮素材）
- **NPU msprof op 级画像**：单层算子 roofline，给"kernel 级"深度背书
- **MindSpore 冒烟**：装 mindformers 跑一次推理（国产化全栈话术的
  最后一块拼图，性价比一般）

### 不建议再投入

- MindIE 复活（已结案为栈级 bug，等 vendor/镜像更新）
- 更多并发档/更大模型的同类压测（边际递减）
- 跨节点 TP/PP（旧环境每容器单卡，做不了；E8d 已用 HCCL 数据
  补了理论侧）

## 4. 分工建议

| lane | 接什么 |
|---|---|
| **腾讯云 agent**（旧环境） | E14 APC → **E15 cache-aware 路由（hero）** → E16 KV 分级 → E17 调度器矩阵 → E18/E19 |
| **本 lane**（新环境+源码） | 源码 walkthrough 文档 → 上游 issue 落地 → （可选）E8e attention 变体手写 |
| MindSpore/其他 | 暂缓 |

## 5. 一条线的叙事闭环

如果只做三件事：**E15 + walkthrough + upstream issues**。
它们刚好合成一个故事：

> "我在 4×910B 上用 vLLM-ascend 做 serving 实验。把请求从 HTTP 到 token
> 的路径在源码里走通了一遍（walkthrough）；在此之上把多副本的轮询路由
> 升级成 prefix-cache-aware 路由，实测 APC 命中率和 TTFT 的变化（E15）；
> 过程中发现的三个引擎 bug 已整理成 issue 提给 vllm-ascend 上游（开源足迹）。
> 配套的微观数据：HCCL 延迟底 160μs 解释了为什么选副本不选 TP，
> 手写的 spec-decode 解释了 ngram 负收益的机制边界。"

这段覆盖了 L1-L4 每一层，且每个论断背后都有可复现的脚本和数据。
