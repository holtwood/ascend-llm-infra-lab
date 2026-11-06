# 面经题目 → 实验对照表

每条题目给出：出处、对应实验、可以直接说的实测数字、以及要注意的坑/边界。
所有数字来自 8×Ascend 910B1（CANN 9.0, torch_npu 2.10, eager）实测；
标注"本栈"的结论在不同栈上绝对值会变，相对规律不变。

---

## 一、PagedAttention / KV 管理

**Q：PagedAttention 和 Continuous Batching 分别解决什么瓶颈？**（科大讯飞推理工程一面）

- **PA 治显存碎片**：KV cache 分页管理，按需分配，块可散放 + 写时复制共享前缀。
  实测证据：**block=16 在 lognormal/uniform/bimodal 三种长度分布下都最优**（E53）——
  小块→内部碎片少但 gather 税高，大块反之，16 是两条曲线的交点，这正是 vLLM 默认值。
- **CB 治 GPU 空转**：调度粒度下沉到每个 decode step。
  实测证据：**E81 真机 forward 对比 static vs continuous = 1.44×**（封闭池
  32 请求）——低于传说 2-3× 是因为传说值假设开放到达流，那种场景的
  差距由调度准入决定（E54 下 FCFS vs 智能调度 goodput 差 24×）。
  churn 下 TPOT 是**分布**（p50 44.7 / p99 68.2ms，E71），SLO 要按 p99 定。

**Q：为什么选 block_size=16？** — E53 全套数据：gather 税 1.54→2.30×（16→256），
碎片成本随块长线性涨，16 是总成本最低点。

**Q：KV cache 多大？一张卡能装多少并发？**

- 0.6B GQA：**kv/token = 114,688 B**（E30 实分配验证）。
- 单卡实测并发：2k ctx → **271 路**，8k → **68 路**，32k → **17 路**（E30）。
- Little's law 版回答：N_inflight = λ ×（TTFT + n_out×TPOT），
  再乘 kv/token×ctx 对比剩余显存——容量矩阵见 E64/E65。

**Q：GQA/MQA 为什么用？**

- **E62 反直觉结论**：nkv 32→1 让 KV 显存降 **32×**，但 bs=1 时延只降 ~6%。
  **GQA/MQA 是容量优化不是加速优化**——省下的显存换并发/上下文，不换代步时。
  （MHA 反而是唯一真跑到带宽墙 91% 的配置。）

---

## 二、Prefill / Decode 不对称

**Q：推理瓶颈在哪？内存瓶颈 vs 计算瓶颈怎么定位？**

- 标准答案（compute-bound vs memory-bound）+ **本栈修正**：
  E68 roofline 实测峰值 ~321 TFLOPS，但**每个算子有 ~24µs 发射地板**——
  eager 栈上 batch≤512 token 的一切先撞发射开销，**"launch-bound 第一，带宽第二"**。
  E29：权重 matvec 单独跑 1412 GB/s（HBM 94%）≈ 939 tok/s 天花板，
  逐层 eager 只有 30-34 tok/s（3.6%）——**这就是图捕获/ACLGraph 存在的理由**。
- prefill vs decode 同 token 成本差 **~250×**（E31）。

**Q：调度器放进一个超长 prompt，64 个 decode 用户 TPOT 怎样？**（Anthropic/GDM/Perplexity 原题）

E78 直接答——bs=64 @ ctx=4k decode 基线 p50=80.9ms，16k prompt 到达后：
- **不分块（ungoverned）：下一步 TPOT 跳到 320.8ms（4×）**，单步尖峰；
- **chunked 2048：连续 8 步 ~106ms（+31%）**，总量相同但形状完全不同。
  → 回答模板：chunked prefill 不改变总功，**改变的是 TPOT 违约的形状**——
  一次 4× 尖峰 vs 若干步 +30%，SLO 视角后者可接受。

**Q：chunked prefill 的 token budget 怎么定？**

- E34：chunk=2048 是甜点（总时长仅 +13%，每 chunk 干扰 3.5× 低于单发）；
  chunk=512 总时长 3.59×。
- E63/E78：2048 chunk 对混批 decode 的步时影响 +14~23%——发射受限栈里
  能藏进 launch 间隙；**带宽受限栈藏不住**（E44：大 chunk 干扰 16-34×）。

**Q：为什么 PD 分离？同卡混跑行不行？**

- E44：同卡 8k prefill chunk 使并发 decode 膨胀 **16-34×**，双流并发不救——
  真·资源争用（HBM 带宽没有第二条路，E75 也证实 D2D 并发 −24%）。
- E75 的正面结论：**跨卡传输几乎免费**（D2H 99% 藏进计算）——
  PD 分离的本质是把"同卡资源争用"换成"跨链路传输"，而后者可隐藏。

**Q：attention 用什么 kernel？融合 kernel 一定快吗？**

- E76：**SDPA 是"prefill kernel"**——prefill 快 2-17×（L 越大越好），
  但 decode（q_len=1）**比两个朴素 matmul 慢 9×**。
  → 为什么生产引擎 prefill/decode 用不同 attention 后端；
  "统一融合 kernel"是伪命题。

---

## 三、抢占 / 换出 / 分层

**Q：抢占时 recompute 还是 swap？推导 break-even。**（Anyscale/RedHat/Databricks 原题）

- 推导：swap 成本 = 2·KV_bytes/BW_link ≈ 2·114688·L/23GB/s；
  recompute = prefill_ms(L) ≈ 0.0112·L + 4.66e-8·L²（E48 拟合，含 attention 二次项）。
- E32/E70 实测：**~4k 是交叉点**——2k 基本打平、8k swap 胜 1.2×、32k 胜 ~2.7×。
- 结论：**抢占策略应按上下文长度自适应**（短的重算、长的换出），
  vLLM v1 默认 recompute 就是吃"短上下文居多"的分布。
- 附赠事实：Ascend 上 **pinned≈pageable**（is_pinned 验证为真，E70）——
  别照搬 GPU 经验。
- 追问"换出会不会拖慢 running 请求"——**E80：不会**。边换出边 decode，
  TPOT p50 45.9→45.7 零变化（每步持续 15MB D2H）；异步换出免费，
  同步换出才亏——vLLM async swap worker 的正确性被实测证实。

**Q：KV offload 到 CPU/NVMe 值不值？**

- E70：20-26 GB/s 链路，逐 block（128tok/15MB）换出无惩罚。
- E72 分层 APC：**CPU reload = 重算的 21-37%**——前缀 KV 驱逐前
  值得先落 pinned CPU 二级缓存（HBM hit≈0 ≫ CPU hit ≫ recompute）。

---

## 四、Prefix Cache / RadixAttention

**Q：RadixAttention 和 vLLM prefix caching 什么关系？**（科大讯飞追问+SGLang 经典）

- 同一目标（共享前缀不重算）的两代实现：vLLM APC 按 **block-16 哈希**命中，
  SGLang RadixAttention 按 **token 级 radix tree** 命中 + cache-aware 调度。
- E79 实测"分叉点 → 复用收益"曲线（8k prompt）：saved ≈ 复用比例的
  **线性函数**（12%→12%、50%→47%、99.8%→82% 饱和于 attention 仍要付）：
  - **system prompt 里插时间戳/工具顺序变化 → 分叉点 p 掉到几十 → 收益归零**，
    这就是"前缀必须逐 token 严格相同"的量化含义；
  - block-16 粒度对长前缀几乎无损（尾差 ≤15 tok），
    但**短前缀** p=40 时 vLLM 只命中 32 vs radix 40——相对差 20%。
- E33：APC 收益非线性——短前缀**负收益**（0.83×，元数据开销），8k 才 2.76×；
  解释了 vLLM 上"开 APC 中位数变差、p99 变好"的实测现象。

---

## 五、吞吐 / 延迟 / 调度

**Q：什么是 goodput？为什么吞吐更高反而更糟？**（Anthropic/OpenAI/Fireworks 原题）

- E51 教科书曲线：λ 8→12 req/s，吞吐 +38%（1013→1166 tok/s）
  但 **goodput −95%**（1013→53.8）——TTFT 60ms→32.6s 越过 SLO 的请求全变废功。
- 拐点就是"队列崩溃点"：到达率超过"按时效服务"的能力后，
  吞吐还在涨（batch 更大）但全是过期工作。

**Q：p99 TTFT 为什么 ~70% util 后飙升？队列模型哪里错？**

- E51 给出实测拐点；理论部分：M/M/1 假设服务率恒定，
  而 batching 引擎**服务率随队长上升**（batch 大→摊薄权重读→单 token 服务时间降），
  所以崩溃比纯 M/M/1 预测的**更晚但更陡**——一旦 attention/KV 饱和立刻垂直上升。

**Q：交互流量和批量流量同引擎，怎么隔离？**

- E54：容量内（λ=6）四种策略**完全无差别**；过载（λ=12）时
  FCFS goodput 46.9 → **SJF 1140（24×）**，严格优先级保护短请求但饿死长的；
  → 隔离要么靠准入控制（SLO-admit 保交互，宁可拒绝），要么分池。

**Q：TP 值不值？通信开销多少？**

- 8×8 p2p 全对矩阵：64MB 全对 19.7GB/s 对称，1KB 延迟底 ~150µs（E8i）。
- TP decode 模拟（真权重）：tp=8 步时 = 单卡 **3.64×**，通信 ~71%；
  tp=4≈tp=2（小消息延迟不随 rank 变）。
- E77：**allreduce ~80-87% 可藏进 GEMM**（HCCL 有独立链引擎）——
  所以真实 TP 实现靠 comm/compute overlap 救命；naive eager TP 是串行，税收满。
- 面试金句："**小模型 decode 用副本不用 TP，TP 是显存方案不是吞吐方案**"。

---

## 六、量化 / 投机解码

**Q：量化对推理有什么影响？怎么验收？**

- 三轴分开答（这是大多数人漏掉的）：
  | | 结论 | 实验 |
  |---|---|---|
  | 容量 | W8A16 权重 −29%；int8 KV 2×、int4 KV 4× | E41/E65 |
  | 速度 | W8A16 本栈 decode/prefill 0.8-0.99×（int8 张量核没用上） | E41/E58 |
  | 质量 | W8A16 ppl +1.27%；**fp8 KV +0.23%/top-1 96.1%（同字节全面碾压 int8）**；int8 KV +1.52%/91.5%；**int4 必须 KIVI**（K per-channel：+0.27%；naive +15M%、group-32 +328%） | E66 |
- **验收要看 top-1 一致率不能只看 ppl**：int8 KV 的 ppl 增量和 W8A16 同级，
  但 top-1 扰动更深（91.5% vs 96.1%）——ppl 被简单 token 主导会掩盖预测翻转。
- **int4 KV 的死因是 K 的通道级 outlier**——KIVI 论文核心论点在本模型复现。

**Q：投机解码什么时候有用？验证成本多少？**

- E45：verify q≤512 个 draft token，单层成本只 +30%——**验证几乎免费**（GEMM 带宽冗余大）。
- E69 真实 draft/verify（w8a16 当 draft）：低熵文本接受率 ~95-100%、
  q=8 全接受 → **9 token/target 前向**；创意文本深位置掉到 75%。
- 速度公式：speedup = E[tokens/cycle]/(q·d+v)；α 决定最优 q，
  α<0.7 时 q>4 收益急剧递减。**draft/target 同尺寸时零收益**（本实验如实记录）。

**Q："一个 DP rank 收到长 prompt，所有 rank 的 decode 都变慢"为什么？**

- E82 实测（2-rank allreduce 循环）：rank0 插入 ~10× prefill，
  **rank1 步时 0.5→4.3ms（8.6×）、首步 478ms**——集合通信让最慢
  rank 成为全员下限。→ 路由要按 prompt 长度均衡 + chunked prefill
  切小 straggler 尖峰。

**Q：为什么 decode 需要 CUDA graph/ACLGraph？**

- E29+E68：24-30ms 发射地板 vs 0.86ms 理论步时——**31× 全是 launch 开销**；
  bs 1→64 步时几乎不变（摊薄的是启动不是带宽）。

---

## 七、系统/容量规划

**Q：弹性扩缩容、冷启动多久？**

- E59：1.5GB 权重搬运只 **0.41s**（磁盘 0.02+H2D 0.29+预热 0.10），
  引擎级冷启动 ~169s 里 **99.8% 是初始化/图捕获**——
  → 解法不是加速 IO 而是 sleep-wake 复用/减少捕获。

**Q：怎么规划容量（模型×上下文×SLO）？**

- `scripts/serving_capacity_planner.py`：输入参数直接出容量/步时分解/goodput，
  所有系数标实测出处；E64 三张矩阵（eager/graph-captured）；
  E52 验证误差 −7%~−40%。
- 结论：**容量 ≠ goodput**——KV 装得下不代表 TTFT/TPOT SLO 满足得了。

**Q：长上下文为什么贵？**

- E37：attention KV 读 ~265GB/s（19% HBM 峰值），32k 时 attention=13×权重读；
  E39：每序列成本随 batch 降 2.7-3.5×，但 KV 带宽 bs≈16 饱和 ~47%；
  E65：int8 KV 把 32k 并发 15→30 路——**长上下文的第一杠杆是 KV 精度**。

---

## 未覆盖的题（诚实清单）

- 引擎内部分模块源码级细节（vLLM V1 EngineCore/AsyncLLM 进程组织）——有
  docs/request-walkthrough.md 但没有运行中引擎可剖。
- 8B+ 模型的真实 serving 曲线——远端 CANN 9.0 缺 `aclnnScatterPaKvCache`，
  MindIE 容器驱动级故障；Qwen3-8B 就位但引擎起不来。
- NVMe 级 KV 三缓（HBM→DRAM→NVMe）——只测到 DRAM 层。
- MLA（DeepSeek 式）KV 压缩——无对应 checkpoint。

---

## 八、算术题（带实测常数的 worked answers）

**Q：Little's law——10 req/s、300 输出、TPOT 45ms，多少请求在飞？KV 装得下吗？**

N = λ·W = 10 × (0.05 + 300×0.045) ≈ **136 路在飞**。
KV = 136 × ctx × 114688B：2k→**31.8GB**（65GB 卡勉强）；4k→**63.7GB（爆）**。
→ 4k 场景要么 int8 KV（32GB 达标）、要么限并发到 ~64 路——
**Little's law 直接换算成显存可行性**。

**Q：1M 上下文一条请求，KV 多大？**

本模型 kv/token=114688B → **115GB，一张 65GB 卡装不下**（~2 张）。
真 70B GQA-8（kv/token≈327KB）→ **327GB/条**——
"max_context 标称"和"能不能服务"是两回事，先算 KV 再说。

**Q：滑窗+全局混合层，128k KV 省多少？**

24 层 W=4096 + 4 层全局 128k → 71.4GB vs 全全局 421GB，**省 83%**。
滑窗层 KV 只随 W 长不随 L 长——这就是 hybrid 模型的容量杠杆。

**Q：TP 超过 KV 头数会怎样？**

8-kv-head 模型 TP=16 → KV 头数不够切，**每个 rank 复制一份（浪费 50%）**。
→ TP 上限 ≈ kv 头数（GQA 模型常见 8，所以 TP 到 8 是 KV 友好的边界）；
TP=16 时要么复制 KV（浪费）要么换 KV 并行策略（sp）。

**Q：MoE 671B/37B 激活，为什么 bs=64 的 decode 读的不止 37B？**

每个 token 路由到自己的 top-k 专家，**batch 内 64 个 token 激活的专家并集
远大于 37B**——decode 权重读由"并集"决定，不是"每 token 参数"。
这就是 MoE 大 batch 下有效参数逼近全量的原因（也是 E62 的 GQA 类比：
省的是容量不是读量……的反面）。

**Q："decode 步时该是 4.8ms 实测 12ms，最可能三个原因？"**

用我的实测答：
1. **发射地板**——每算子 ~24µs × 每层 ~15 算子 × 28 层 ≈ 10ms（E68/E29，
   eager 栈最常见主因）；
2. **attention 后端选错**——q_len=1 走 SDPA 慢 **9×**（E76）；
3. **隐性税**——KV 反量化 fp32 中间态 6.5×（E8k）、
   top-p 采样 bs=128 时 78ms > 整个前向（E67）、
   torch.cat/expand 触发隐藏拷贝（E52 的教训）。

**Q："长上下文时 KV 读何时超过权重读？"**

权重读/step ≈ 模型权重字节；KV 读 = bs×ctx×kv/token。
本模型：权重 1.5GB → 交叉点 ctx = 1.5e9/(bs×114688)：
bs=1→**13.1k**、bs=16→**819**、bs=64→**205**。
E37 实测吻合（~1k 起 attention 超权重读）。
→ **batch 越大，越早进入 attention-bound**——长上下文服务的 batch 甜点被前移。

**Q：FP8 KV 需要什么 scale？误差在哪最大？**

E66 实测表：**fp8 e4m3 per-token-per-head scale → ppl +0.23%、top-1 96.1%**，
同字节全面优于 int8（+1.52%/91.5%）——
浮点指数保住了 int8 线性网格浪费在 outlier 上的动态范围。
误差最大处：**K 的通道级 outlier**（E66 int4 实验证明，
per-token 定标会被通道 outlier 带崩；KIVI 的 per-channel-K 是解药）。
验证方法：测 ppl + **top-1 一致率**（ppl 会被简单 token 稀释，
一致率才是逐 token 扰动的真相）。

**Q：多轮对话调工具时 KV 放哪？**

E72/E80 实测： pinned CPU 二级缓存——resume 时 reload = 重算的
21-37%（8k 前缀 36ms vs 98ms），且 D2H 换出对 running decode
完全透明（E80：TPOT 不变）。
→ **工具等待期间 park 到 CPU，回来 reload，比重算便宜 3-5×**。

**Q："temperature=0 的两个相同请求输出不同，为什么？"（batch nondeterminism）**

机制答案：greedy 下输出只依赖 argmax(logits)，但 logits 的数值随
**kernel 调度/批形状**变化——不同 batch 组成会选不同分块/归约顺序，
末位漂移在 logits 接近时翻转 argmax，之后逐 token 发散。
E83 实测注记：`batch_nondet.py`（`batch_nondet.json`）——同一 prompt
拼接 8 份跑 8L 序列，**bit-identical 输出（Δlogit=0，top-1=100%）**。
说明：简单堆叠同序列不会改变前 L 位的归约顺序；真实漂移来自
**变长 batch 的 padding + kernel dispatch 差异**，不是"batch 本身"。
答题要点：temp=0 ≠ 确定性系统；要稳定复现需固定 batch 组成或
用 batch-invariant kernel。
