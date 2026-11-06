# 通宵队列 run_20261105_160244 · 第一批（torch 级矩阵）

远端 npt 独占 8×910B1（CANN 9.0.0 / torch 2.10.0 / torch_npu 2.10.0），
队列 `scripts/overnight/master.sh` 串行执行。本批 4 任务全部 rc=0。

## E8i-ext：8×8 全对 p2p 矩阵（hccl_pairs.json）

56 个有向 pair × 5 种消息尺寸。结论：

| 尺寸 | 现象 | 数值 |
|---|---|---|
| 64MB | **全部 56 pair 均匀 19.7 GB/s** | 单链路饱和带宽 = 端口上限，与 pair 无关 |
| 1MB | 6.0–7.6 GB/s，pair 间 ±20% | 未饱和区暴露路径差异（如 1→0 = 7.6，2→1 = 5.6） |
| 1KB | 141–215 µs | 延迟底 ~150 µs，与 E8i 集合通信延迟底一致 |

**拓扑结论**：910B1 8 卡在饱和带宽下是**完全对称**的（无慢链路），
但中等消息有 ±20% 的路径不对称——TP 设计上**不需要 topology-aware 放置**，
但 1MB 级的通信（如 MoE all2all 小块）会吃到路径差异。

## E8k-ext：int8-KV 精度/成本网格（kv_quant_matrix.json）

32 配置：S∈{128..16384} × B∈{1..64}，GQA 广播实现（无 KV 展开）。

- **显存**：mem_ratio 恒 0.51（int8+bf16 scale ≈ 一半）——与 E8k 一致
- **精度随 S 收敛**：err_max 从 S=128 的 0.0117 降到 S=16384 的 0.0007
  （token 越多，量化噪声被平均得越充分）
- **naive 反量化税**：1.5–3.0×，且**不随 S 增大而下降**（长上下文里
  dequant 的元素级开销与 attention 计算同阶）——结论仍是必须融合 kernel

与 E8k（6.5–8.5×）的差异已用 **2×2 对照实测**定死（见 `kv_dequant_ab.md`）：

| KV 头数 | fp32 中间态 | bf16 中间态 |
|---|---|---|
| 8（GQA，本网格） | 1.87× | 1.40× |
| 32（MHA，E8k 口径） | **6.46×** | 3.16× |

即两个因子叠加：**中间态精度**（fp32 vs bf16 ≈ 2× 流量放大）与
**KV 体积/头数**（8→32 头使反量化从延迟受限转为带宽受限）。
E8k 的 6.5× 被精确复现（6.46×）——不是"实现口径不同"的含糊解释。

**引用规则：报告 int8-KV naive 税必须同时给 (a) 头数/总字节 (b) 中间态 dtype，
否则 1.4× 与 6.5× 都"对"却差 4.6×。真实 C8（32 头 + fp32 中间态）在 6× 量级。**

局限：本实验的 scale 是 per-token-per-head（`amax(dim=-1)`），不是真实 C8
KV 量化的 per-channel/per-block 方案；误差量级可作为上界参考。

## E8j-ext：paged gather 块粒度 + 页序对照（paged_gather.json / paged_gather_v2.json）

S∈{4096..32768} × block∈{16,32,64,128}，decode 形态（Sq=1，B=4）。
v2 增加**打乱页表（randperm）vs 顺序页表**对照：

| S | block | 顺序页 tax | 打乱页 tax |
|---|---|---|---|
| 4096 | 16 / 32 / 64 / 128 | 1.63 / 1.85 / 2.17 / 2.73 | 1.55 / 1.77 / 2.09 / 2.75 |
| 8192 | 16 / 32 / 64 / 128 | 2.15 / 2.11 / 2.22 / 2.42 | 2.15 / 2.11 / 2.29 / 2.42 |
| 16384 | 16 / 32 / 64 / 128 | 2.14 / 2.18 / 2.16 / 2.27 | 2.14 / 2.18 / 2.16 / 2.27 |
| 32768 | 16 / 32 / 64 / 128 | 2.13 / 2.15 / 2.14 / 2.12 | 2.13 / 2.11 / 2.13 / 2.13 |

**页序几乎无影响**（差 <0.1×）——advanced indexing 的拷贝量只由总字节决定。

与 E8j（3.3–15.6×）的差异做了**对照实测**（`paged_tax_ab.md`），
并推翻了本仓先前"页内布局是主因"的假设：

| 因子 | 对照结果 |
|---|---|
| 页内布局 `[page,H,bs,D]` vs `[page,bs,H,D]` | 仅差 10–25%，且 B 布局反而更差 |
| **KV 头数 8 vs 32**（同 S/块长） | **2.10× vs 3.19–4.07×**（主因） |
| 块长 16/64/256（H=32, S=8k） | 3.19 / 5.92 / **15.85×**（复现 E8j 的 3.27/5.76/15.6） |

**修正结论：分页税的主因是 KV 体积（头数）与块长，不是布局。**
报"分页税"必须同时给头数/总字节、块长、布局三个数，缺一个则 2× 与 16×
都能"对"。机制不变：分页间接必须在 attention kernel 内完成。

## DRAM↔HBM 拷贝扩展（copy_bw.json）

| 方向 | pinned | pageable |
|---|---|---|
| H2D | 18.6–23.8 GB/s | 23.3–24.4 GB/s |
| D2H | 19.5–25.1 GB/s | **3.8–6.3 GB/s（4–5× 惩罚）** |

**新数据**：D2H 在并发 matmul 负载下降 **24.7 → 19.7 GB/s（−20%）**——
KV offload / PD 传输带宽预算不能按空载峰值算。

## 环境存证

见 env.txt：8×NPU 可见、torch 2.10.0+cpu、torch_npu 2.10.0、CANN 9.0.0、
transformers 4.51.0；vllm 当时未装（finish_install 排队中）。

## E8i-ext2：TP decode 步时实测（tp_decode.json，真 Qwen3-0.6B 权重）

单卡 8 卡独占，按行并行切分真实权重，decode 步（bs=1,seq=1）。
**建模已修正**：真实 TP 每层 2 次 allreduce（o_proj 后、down_proj 后；
qkv 是列并行不需要），先前脚本多算了一次。

| TP | 步时 (µs) | 相对 tp=1 | 单次 allreduce | 通信占比 |
|---|---|---|---|---|
| 1 | 213 | 1.00× | — | 0% |
| 2 | 629 | 2.96× | 199 µs | 63% |
| 4 | 633 | 2.98× | 227 µs | 72% |
| 8 | 774 | **3.64×** | 258 µs | 67% |

**结论**：0.6B 级模型 decode 步，TP 通信税压过计算收益——8 卡比单卡慢 3.6×。
tp=4 与 tp=2 几乎持平（633 vs 629µs），说明 allreduce 延迟在小消息区
对 rank 数不敏感（与 E8i 的延迟底一致），而计算切片早已无收益（小张量
matvec 是延迟受限）。与 E8d/E8i 自洽：**小模型 decode 用副本不用 TP**。

数据卫生：第一版脚本每层算 3 次 allreduce（多算 1 次），得 4.47×；
修正为 2 次后 3.64×。两版都记录在案，方向结论一致。
另有独立两次运行（含/不含并发 soak）一致（差 <4%）。

复现：`scripts/overnight/tasks/tp_decode_sim.py`（torch 级，不需要 vllm）。

## 远端 vllm-ascend serve 结论：CANN 9.0 硬版本墙

vllm+vllm_ascend 在远端成功安装并激活 `PLATFORM npu`、W8A16 模型权重
加载成功、`/v1/models` 返回 200——但首次前向死在
`aclnnScatterPaKvCache`（CANN 9.2 才有的内置算子，9.0 缺失）。
这不是配置问题：paged KV scatter 是 9.2 新增能力，vllm-ascend 0.30
的 paged attention 依赖它。也解释了厂商为何在 9.0 容器里配 MindIE。

踩坑全记录：`docs/remote-vllm-bringup.md`。

## 4h HCCL soak：8 卡稳定性（soak.jsonl）

256MB allreduce × 8 卡，1 秒间隔连续 4 小时，14330 个样本：

| 指标 | 值 |
|---|---|
| 中位延迟 | 4600 µs（= 463 GB/s 总线带宽） |
| p10 / p90 | 4401 / 4955 µs |
| 首 10% → 末 10% 中位 | 4611 → 4717 µs（**漂移 +2.3%**） |
| 逐小时中位 | 4617 / 4546 / 4609 / 4659 µs（无单调劣化） |
| 偶发尖峰 | 10.9–16.6 ms（OS 调度噪声，占比 <0.1%） |

**结论：8×910B1 连续 4 小时满带宽集合通信无延迟漂移（+2.3%，运行间噪声量级）**
**注意：未直接采集芯片温度/频率**，"无热降频"是由延迟稳定反推的推断而非实测。
这是"长稳"性质的正面证据——
对训练/推理长任务的可预测性有直接意义。

复现：`scripts/overnight/tasks/hccl_soak.py` + `soak_drift.py`。

## E29 · Decode 吞吐天花板与发射开销墙（decode_ceiling.json，真 Qwen3-0.6B 权重）

面试必问："这个模型在这张卡上 decode tok/s 上限多少、为什么？"
用真权重做三层测量（910B1，bs=1..64，ctx=512/4096）：

| 层次 | 实测 | 说明 |
|---|---|---|
| **带宽天花板**（一次大 matvec 读全部 1.5GB 权重） | 0.86 ms → **1412 GB/s**（HBM 峰值 ~94%） | 等效 **939 tok/s @bs=1** |
| **逐层 decode**（28 层 × 6 matvec + attention，eager python） | 29-33 ms → **30-34 tok/s** | 仅为天花板的 **3.6%** |
| **12 分块 matvec**（65535 列一块，受算子限制） | 2.80 ms → 575 GB/s | 分块发射把带宽砍半 |

**核心结论：batch=1 的 decode 不是被 HBM 或算力卡住，而是被 kernel 发射开销卡住**
（28 层 × ~15 个算子 ≈ 420 次 launch × ~70µs ≈ 29ms）。这解释了为什么
vLLM/ACLGraph 必须做图捕获：同样的权重、同样的卡，eager 只有天花板 3.6%。

**批处理经济学**：步时在 bs=1→64 之间几乎不变（29ms），所以 tok/s 随 bs
线性增长（34 → 2204 tok/s）——decode 阶段"一次权重读服务 bs 个 token"，
直到算力或 KV 带宽成为新瓶颈（本实验 ctx=4096/bs=64 时步时升到 44ms，
KV 读放大开始显现）。

**权重-only int8 的收益**：分块对照（12×65535 列）bf16 575 GB/s vs
int8 349 GB/s，但 int8 只读一半字节 → 端到端 **1.21×**；若发射开销被
图捕获消除，理论上限是 2×（939 → 1878 tok/s）。**量化收益同样被发射开销吃掉**，
这与 E8k/E8j 的"必须融合进 kernel"是同一结论。

复现：`scripts/overnight/tasks/decode_ceiling.py`（torch 级，不需要 vllm）。

## MindIE 2.3.0 + Qwen3-8B：第二次 vendor 栈失败（驱动级）

在 nmindie 容器（CANN 8.5.0，mindie-service 2.3.0，Qwen3-8B 16GB 在位）尝试
生产服务，逐层排障后确认是**驱动/运行时级故障**：

| 阶段 | 现象 | 处置 |
|---|---|---|
| 配置校验 | `maxPrefillTokens < maxInputTokenLen` | 调平到 8192/2048 |
| 配置解析 | `JSON error ... cannot use operator[] with a numeric argument with object` | `ModelConfig` 必须是**数组**不是对象 |
| 启动 | `Failed to init infer backend manager` | 修正配置后通过 |
| 运行 | 子进程 core dump；dmesg 出现 `stranded cqe` / `hdcdrv_recv_peek_wait` | **驱动级 CQE 卡死**，非配置问题 |

与 E5R（ATB `set_weight` 挂死）合起来，MindIE 在本容器有两种独立失败模式，
均为 vendor 栈问题。**结论：MindIE 2.3.0 + CANN 8.5.0 在此环境不可用**，
MindIE 对比实验只能留给能拿到可用 MindIE 镜像的场景。

## E30 · KV 容量与并发规划模型（capacity.json，真分配验证）

公式：`max_seqs = (HBM可用 − 权重 − 开销) / (kv_per_token × ctx)`
Qwen3-0.6B：kv_per_token = **114,688 B**（2 × 28 层 × 8 KV头 × 128 × 2B）。

实测（910B1 61GB 可用，真分配到 OOM）：

| 上下文 | KV/序列 | 最大并发 | 实测占用 |
|---|---|---|---|
| 2k | 235 MB | **271** | 63.7 GB |
| 8k | 940 MB | **68** | 63.9 GB |
| 32k | 3758 MB | **17** | 63.9 GB |

**与公式完全吻合**（271×235MB≈63.7GB 等）。这就是"8k 上下文能服务多少并发"
的可验证答案：**68 路**（0.6B，64GB 卡，全给 KV）。

推论：并发数与上下文长度**成反比**；把 ctx 从 8k 降到 2k 换来 4× 并发。
面试可现场推导：`并发 = 可用HBM / (2·L层·n_kv·d_head·2B·ctx)`。

## E31 · Prefill 与 Decode 的不对称性（PD 分离的物理依据）

同一权重、同一张卡，prefill（整段前向）vs decode（逐 token）：

| 场景 | tok/s | 每 token 成本 |
|---|---|---|
| prefill L=512, bs=1 | 15.0k | 66.7 µs |
| prefill L=512, bs=4 | **60.1k**（4×） | 16.6 µs |
| prefill L=8192, bs=1 | 84.6k | 11.8 µs |
| decode bs=1（E29） | **34** | 29,400 µs |
| decode bs=64（E29） | 2,204 | 454 µs |

**核心不对称**：prefill 是**算力受限**——加大 batch 直接换吞吐（bs 1→4 得 4×）；
decode 是**权重读受限**——加 batch 只摊薄权重读，单请求延迟（TPOT）不变。
prefill 每 token 比 decode 快 **250×**（11.8µs vs 29ms）。

**这解释了 PD 分离为什么存在**：两者吃不同资源（算力 vs HBM 带宽），
混跑时 decode 的长尾会被 prefill 的计算洪峰打断；拆开后各自按最优 batch 跑。
也解释了 chunked prefill 的价值：把长 prefill 切碎插入 decode 间隙，
用 TPOT 的轻微抖动换 TTFT 的显著改善。

注：L≤2048 的 prefill 步时被 ~30ms 的 launch 开销地板支配（同 E29），
因此短 prompt 的 tok/s 是发射受限而非算力受限。

## E32 · 抢占经济学：recompute vs swap（preemption.jsonl）

vLLM 抢占序列时有两条路：重算（re-prefill）或把 KV 换出到主机内存。
用真权重 + 真 KV 张量两条路都实测（910B1，D2H/H2D 实测 17-23 GB/s）：

| 上下文 | KV | recompute | swap | 谁赢 | 倍数 |
|---|---|---|---|---|---|
| 1k | 117 MB | 31.9 ms | **11.4 ms** | swap | 2.8× |
| 4k | 470 MB | **43.0 ms** | 44.6 ms | recompute | 1.04× |
| 8k | 940 MB | 97.6 ms | **85.9 ms** | swap | 1.14× |
| 16k | 1.88 GB | 282.0 ms | **204.0 ms** | swap | 1.38× |
| 32k | 3.76 GB | 966.3 ms | **425.9 ms** | swap | 2.27× |

**机制**：swap 成本 = `2 × KV字节 / 20GB/s`（线性）；
recompute 成本 = 发射地板(~32ms) + 线性 matvec + **二次 attention（O(L²)）**。
所以 1k 时 swap 赢在"不用付发射地板"，4k 附近两者打平（唯一 recompute 微胜点），
16k 之后 recompute 的二次项主导，swap 优势重新拉开到 2.3×。

**这对引擎设计意味着什么**：vLLM 默认用 recompute，前提是它的 prefill 被
图捕获（地板≈0）且 attention 是融合 kernel（二次项常数小）。本实验给出的
正是这个前提的量化版本——**换出/重算的交叉点由三件事决定**：
① prefill 的发射地板 ② attention kernel 的二次项常数 ③ 主机链路带宽。
在没有图捕获的 eager 栈上（本实验），交叉点只有 4k；工程上必须把 ①②
压下去，recompute 才划算。

复现：`scripts/overnight/tasks/preemption_economics.py`（逐行落盘，防超时丢数）。

## E33 · Prefix cache（APC）的真实收益曲线（prefix.jsonl）

N=8 个请求共享长度 P 的 prefix，各自只有 128 token 的独有后缀。
对比：cold（每个请求重算 P+128）vs warm（prefix KV 算一次，后缀只算 128 但
attend over [prefix KV | suffix]，即 APC 的真实行为）。

| Prefix | 命中率 | cold | warm | 单请求加速 | 8 请求总加速 | prefix KV |
|---|---|---|---|---|---|---|
| 512 | 80% | 30.2 ms | 35.7 ms | **0.85×（更慢）** | 0.86× | 59 MB |
| 2048 | 94% | 29.4 ms | 35.6 ms | **0.83×（更慢）** | 0.84× | 235 MB |
| 8192 | 98% | 98.3 ms | 35.6 ms | **2.76×** | 2.26× | 940 MB |

**结论：APC 的收益不是"命中率"决定的，而是"省下的二次项"决定的。**
- P≤2k：两条路都被 ~30ms 发射地板支配，APC 还要多付 `cat(prefix KV)` 和
  显式 mask 构造 → 净亏 15-17%
- P=8k：cold 要跑 8320 token 的**二次** attention；warm 只有 128 个 query
  对 8320 个 key（线性）→ 2.76× 收益

**这条正好解释 E3 在真实 vLLM 上的反常现象**：APC 开着时**中位数 TTFT
反而 +12%**，但 p99 −55%。机制就是本实验：短/中等 prefix 命中时省不下二次项，
只多付了哈希和 cat 开销；而长 prefix 的请求（p99 那些）省下大量二次计算。
**所以"APC 有没有用"必须按 prefix 长度分布回答，不能只报命中率。**

复现：`scripts/overnight/tasks/prefix_cache_benefit.py`。

## E34 · Chunked prefill：chunk size 怎么选（chunked.jsonl）

L=8192 的 prompt，按不同 chunk 切分预填（每 chunk 是调度器的一步，
其它请求的 decode 在 chunk 之间插入）：

| chunk | 步数 | 总时长 | 相对单发 | 每 chunk 均值 | 每 chunk 最大 |
|---|---|---|---|---|---|
| 8192（单发） | 1 | 151.9 ms | 1.00× | 151.9 ms | 151.9 ms |
| 4096 | 2 | 177.3 ms | 1.17× | 88.6 ms | 132.0 ms |
| **2048** | 4 | **171.9 ms** | **1.13×** | **43.0 ms** | 53.8 ms |
| 1024 | 8 | 273.8 ms | 1.80× | 34.2 ms | 35.5 ms |
| 512 | 16 | 545.3 ms | 3.59× | 34.1 ms | 36.1 ms |

**结论：chunk=2048 是甜点**——总时长只多 13%，但单步阻塞时间从 152ms
降到 43ms（干扰降 3.5×）。继续减小 chunk 到 1024/512 **不再降低单步延迟**
（都卡在 ~34ms），却让总时长涨 1.8-3.6×。

**关键洞察：每 chunk 的延迟地板 ≈ 一个 decode step 的延迟（29-34ms，见 E29）**
——在 eager 栈上，你无法让"一个 prefill chunk"比"一个 decode step"更便宜，
因为两者都是 28 层 × ~15 个算子的发射开销。所以 chunk size 的可行区间是
「够小以降低干扰」到「不小于一个 decode step 的等量级」之间。

**测量修正记录**：首轮未做 warmup，第一行（单发 8192）含冷启动，测出 409ms，
导致"分块比单发快 2.3×"的错误结论；加 warmup 后单发 152ms，结论翻正为
"分块小幅增本、显著降干扰"。独立复测单发 8192 = 97.6ms（与 E31 一致），
说明 152ms 仍含少量残余预热成本，**故本表只用于相对比较**。

复现：`scripts/overnight/tasks/chunked_prefill.py`。

## E37 · 长上下文 decode：attention 何时压过权重读（attn_scale.jsonl）

单层 decode attention（bs=1，GQA 8 KV 头 × 128 维）随上下文的变化，
对比"读一层权重"的成本（53.7MB / 1400GB/s ≈ **38µs**）：

| ctx | 每层 KV | attention | vs 权重读 | KV 带宽 |
|---|---|---|---|---|
| 1k | 4.2 MB | 73 µs | **1.91×** | 57 GB/s |
| 4k | 16.8 MB | 81 µs | 2.12× | 206 GB/s |
| 16k | 67.1 MB | 264 µs | 6.89× | 254 GB/s |
| 32k | 134 MB | 508 µs | 13.25× | 264 GB/s |
| 64k | 268 MB | 1021 µs | 26.63× | 263 GB/s |
| 128k | 537 MB | 2025 µs | **52.80×** | 265 GB/s |

**两个结论**：

1. **"decode 是权重读受限"只在极短上下文成立**。从 1k 起 attention 就比
   权重读贵（1.9×），32k 时贵 13×，128k 时贵 53×。长上下文服务的瓶颈
   不是模型权重，而是 KV 读。
2. **attention 的 KV 读只有 265 GB/s = HBM 峰值的 19%**（E29 实测裸 matvec
   1412 GB/s）。也就是说长上下文 decode 既不是"带宽打满"也不是"算力打满"，
   而是**访存效率只有 1/5** —— 这正是 FlashAttention 类融合 kernel、
   paged-KV 内联寻址、KV 量化（E8k）要解决的问题：把 265 GB/s 往 1412 拉。
   若能达到峰值，32k 上下文每层 attention 可从 508µs 降到 ~95µs（5.3×）。

**与 E29 的关系**：E29 的整模型 29ms/step 是**发射开销**主导（420 次 launch）；
本实验的单算子测量揭示的是**算子内部的物理**。两者合起来才是完整图像：
先被发射开销卡住（图捕获解决），再被 attention 访存效率卡住（融合 kernel 解决）。

复现：`scripts/overnight/tasks/attention_scaling.py`。

## E39 · 长上下文 decode 的批处理饱和点（attn_batch.jsonl）

单层 decode attention，ctx × batch 网格（GQA 8 KV 头）：

| ctx | bs=1 | bs=4 | bs=16 | bs=64 |
|---|---|---|---|---|
| 8k | 215 GB/s (15%) | 485 (35%) | 669 (48%) | 756 (54%) |
| 32k | 263 GB/s (19%) | 510 (36%) | 662 (47%) | 702 (50%) |

每序列成本（8k）：156 → 69 → 50 → **44 µs**（bs=64 时降 3.5×）
每序列成本（32k）：511 → 263 → 203 → **191 µs**（降 2.7×）

**结论**：
1. **长上下文 decode 想要大 batch**——每序列成本可降 2.7-3.5×，这与短上下文
   "decode 加 batch 只摊薄权重读、不降单请求延迟"（E29）不同：长上下文时
   KV 读是主体，批处理能提升访存并行度，**单序列成本真的下降**。
2. **但 KV 带宽在 bs≈16 就饱和到 ~47%**，bs=64 也只到 50-54%（~750 GB/s）。
   attention kernel 的访存效率上限只有 HBM 峰值的一半，且 batch 救不了。
3. 结合 E30（8k 时最多 68 并发）与 E29（整步 29ms 发射开销主导）：
   8k 场景的理论 attention 成本（68 × 44µs × 28 层 ≈ 84ms/step）在真实
   eager 栈上会被 29ms 的发射地板掩盖，但一旦图捕获把地板压掉，
   **attention 访存效率（50% 峰值）就会成为下一个瓶颈**——这正是
   FlashAttention/paged-KV 内联寻址要解决的问题。

复现：`scripts/overnight/tasks/attn_batch_scaling.py`。

## E44 · Prefill/Decode 干扰实测（pd_interf.jsonl，同卡）

单层 attention，decode 已在跑（ctx=4096），同时插入一个 prefill chunk：

| decode bs | chunk | decode 单独 | 串行 mixed | decode 膨胀 | **双流并发** |
|---|---|---|---|---|---|
| 1 | 512 | 98 µs | 139 µs | 1.41× | **3.39×** |
| 1 | 2048 | 98 µs | 334 µs | 3.40× | 3.16× |
| 1 | 8192 | 98 µs | 3315 µs | **33.7×** | 33.0× |
| 8 | 512 | 220 µs | 274 µs | 1.25× | 1.40× |
| 8 | 2048 | 220 µs | 473 µs | 2.15× | 1.75× |
| 8 | 8192 | 220 µs | 3527 µs | **16.1×** | 15.0× |

**三个结论**：

1. **一个 8k 的 prefill chunk 能让并发 decode 的延迟膨胀 16-34×**——这就是
   队头阻塞的真实量级。chunk 从 8192 降到 512，膨胀从 33.7× 降到 1.41×，
   **这是 chunked prefill 的直接收益（E34 的甜点 2048 对应 3.4×）**。
2. **换成双流并发没有改善**（甚至 512 chunk 时更差：3.39× vs 1.41×）——
   说明干扰不只是"排队"，而是**真实的 SM/HBM 争用**。多流/异步解决不了。
3. **推论：同卡 PD 分离在物理上收益有限**——P 和 D 抢同一批 SM 和同一条 HBM，
   分离进程并不分离资源。这与 E18 的结论（单卡 PD 数据面不可行）互相印证：
   PD 只有在**物理上不同的卡**上才有意义。

复现：`scripts/overnight/tasks/pd_interference.py`。

## E41 · W8A16（int8 weight-only）到底买到什么（w8a16.json）

用真 bf16 与真 int8 checkpoint（`Qwen3-0.6B` vs `Qwen3-0.6B-w8a16`）三轴对照：

**① 显存**：权重 1.503 GB → **1.064 GB**（省 0.439 GB，仅 29% 而非 50%——
因为 embedding / lm_head 仍是 bf16）。按 E30 公式换算容量增益：
8k 上下文只多 **+0.47 路**并发（0.6B 太小，可忽略）；
换成 8B 模型则是省 ~8 GB → **+8.5 路 @8k**，这才是 W8A16 的主场。

**② 速度**：**没有收益**（0.80-0.99×，即略慢）

| 张量 | bs | bf16 | int8 | 加速比 |
|---|---|---|---|---|
| q_proj | 1 | 31.6 µs | 32.1 µs | 0.99× |
| q_proj | 64 | 30.1 µs | 33.3 µs | 0.90× |
| gate_proj | 1 | 30.9 µs | 33.2 µs | 0.93× |
| down_proj | 64 | 30.5 µs | 38.2 µs | 0.80× |

原因：单层 matvec 只有 ~31µs，**已经被发射开销地板支配**（E29），
换 dtype 不改变地板；int8 算子还要多做一次反量化。**与 E8k 结论一致：
量化的速度收益必须在融合 kernel 里兑现，张量级调用拿不到。**

**③ 精度**：max 相对误差 **0.9%-1.6%**（对比 bf16 同层输出）。

**serving 结论**：W8A16 是**容量工具不是速度工具**——大模型/显存受限场景
（8B 以上、长上下文、高并发）才有意义；在 eager 小 batch 下它的速度是负收益。

复现：`scripts/overnight/tasks/w8a16_economics.py`。

## E45 · 投机解码的验证成本（spec_verify.jsonl，单层，ctx=4096）

"draft k 个 token 后一次 forward 验证几乎免费"——实测验证：

| q（验证行数） | attention | MLP | 层总计 | 相对 q=1 | 理论加速上限 |
|---|---|---|---|---|---|
| 1 | 94 µs | 131 µs | 286 µs | 1.00× | 1.0× |
| 8 | 82 µs | 136 µs | 278 µs | **0.97×** | 8.2× |
| 32 | 83 µs | 136 µs | 278 µs | 0.97× | 33× |
| 128 | 155 µs | 136 µs | 350 µs | 1.22× | 105× |
| 512 | 176 µs | 135 µs | 372 µs | **1.30×** | 394× |

**结论：在本栈（eager，算子发射受限）里验证成本几乎是常数**——
从 1 行到 512 行只涨 30%，因为 attention 是 KV 读受限（不随 q 涨），
MLP 的算力需求远未触及（512×3072 matmul 仍在发射地板下）。

**这解释了投机解码"机制上很美"**：验证 8 个 draft token 的成本 ≈ 验证 1 个
（0.97×），理论上限 8.2×。

**但也要说清为什么实测往往不赚**（E24/MTP 实验的负结果）：
本实验量的是**纯验证算子成本**；端到端还有 ① draft 模型自身的推理成本
② 多候选的 KV 管理/调度开销 ③ 接受率不足导致的浪费。机制上限高 ≠ 端到端收益高，
**这个区分本身就是很好的面试答案**。

复现：`scripts/overnight/tasks/spec_verify_economics.py`。

## E48 · Prefill 成本模型：线性项与二次项的分离（prefill_model.jsonl）

单层分解（attention / MLP / projections），L 从 1k 到 16k：

| L | attention | 每 token | MLP | 每 token | proj |
|---|---|---|---|---|---|
| 1k | 0.14 ms | 0.13 µs | 0.14 ms | 0.13 µs | 0.07 ms |
| 4k | 0.85 ms | 0.21 µs | 0.32 ms | 0.08 µs | 0.13 ms |
| 8k | 3.24 ms | 0.40 µs | 0.60 ms | 0.07 µs | 0.24 ms |
| 16k | **12.63 ms** | 0.77 µs | 1.37 ms | 0.08 µs | 0.59 ms |

**拟合结果**：attention 的二次项系数 **c = 4.66×10⁻⁸ ms/token²**；
MLP/proj 是纯线性（每 token 恒定 0.08 / 0.04 µs）。

**结论**：
1. **16k 上下文的 prefill 有 86% 花在 attention 上**（1k 时只有 ~40%）。
   所以"优化 prefill"在不同上下文长度下是不同的事：
   短 prompt 优化 MLP/投影（权重读），长 prompt 优化 attention（二次项）。
2. 二次项意味着**上下文翻倍 → attention 时间 4×**。16k 时 attention 已是
   MLP 的 9.2×，这解释了 E31 里 prefill 在 8k 后 tok/s 不再上涨。
3. 容量规划公式（0.6B/910B1）：
   `T_prefill(L) ≈ 28 × (0.10 + 0.09·L·1e-3 + 4.66e-8·L²) ms`
   —— 16k 时约 **340 ms**，与 E31 实测（8k 97ms）同量级吻合。

复现：`scripts/overnight/tasks/prefill_cost_model.py`。

## E51 · 延迟-负载曲线与 goodput 崩塌（latency_vs_load.json）

**说明：这是仿真**（离散事件，连续批处理 + chunked prefill），但每个服务时间
常数都来自本仓实测（E29 步时/地板、E37/E39 attention 效率、E48 prefill 模型）。
0.6B、8k 上下文、prompt 512-2048、输出 128、max_batch 64、
SLO = TTFT ≤ 2s 且 TPOT ≤ 100ms。

| λ (req/s) | TTFT p50 | TTFT p99 | TPOT p50 | 吞吐 | **goodput** | SLO 达成 |
|---|---|---|---|---|---|---|
| 1 | 50.6 ms | 66.3 ms | 32.7 ms | 123 tok/s | 123 | 100% |
| 2 | 51.1 | 68.6 | 34.3 | 248 | 248 | 100% |
| 4 | 55.0 | 76.1 | 36.6 | 505 | 505 | 100% |
| **8** | 70.3 | 97.0 | 47.0 | 1013 | **1013** | 100% |
| **12** | **32,580 ms** | 68,914 | 51.0 | 1166 | **53.8** | 5% |
| 16 | 60,555 | 124,060 | 51.0 | 1171 | 29.9 | 3% |
| 32 | 102,266 | 206,124 | 51.1 | 1171 | 26.0 | 2% |

**结论（serving 面试的核心认知）**：
1. **拐点在 λ≈8-12 req/s**：过了拐点吞吐只从 1013 涨到 1171（+16%），
   但 **goodput 从 1013 崩到 53.8 tok/s（−95%，19×）**。
2. **TPOT 几乎不变（47→51ms），TTFT 爆炸（70ms → 102s）**——过载时
   延迟恶化全部体现在**排队**上，不在单步计算上。所以"看 TPOT 判断健康度"
   是错的，必须看 TTFT 和队列长度。
3. 只报吞吐的压测（1171 tok/s）会让容量结论**高估 22 倍**（1171 vs 实际
   可交付 53.8）。这正是 E9 在真实 vLLM 上观察到的现象，此处给出了
   可解释的机理与量化曲线。

复现：`scripts/overnight/tasks/latency_vs_load.py`（可调模型规模/上下文/SLO）。

## E52 · 容量规划器的实测校验（planner_validation.jsonl）

用真机全模型 decode 步时校验规划器公式（`weight_read + attention + floor`）：

| bs | ctx | 实测 | 预测（原模型） | 误差 |
|---|---|---|---|---|
| 1 | 4k | 29.5 ms | 31.8 ms | −7% |
| 8 | 4k | 30.4 ms | 38.0 ms | −20% |
| 8 | 8k | 30.6 ms | 46.0 ms | −34% |
| 16 | 8k | 31.0 ms | 51.4 ms | −40% |

**校验暴露了两件事**：

1. **bs≤16 时步时恒定 ~30ms**——发射地板（29ms）完全支配，attention 成本被掩盖。
   规划器在这区间"高估"attention，实际影响可忽略。
2. **高 batch 时 attention 效率远高于模型假设**：用 bs=64/ctx=4096 的实测步时
   （54.6ms）反推，KV 读达 **1220 GB/s ≈ HBM 峰值 86%**，而非模型假设的 50%。

**据此校准规划器**：attention 效率曲线改为 bs=1 → 19%（E37）、bs=16 → 50%（E39）、
bs≥64 → 86%（E52 反推）。校准后：
- 0.6B @8k：最大并发 61，SLO(TPOT 50ms) 运行点 13 路 → goodput 261 tok/s
- 0.6B @2k：最大并发 244，SLO 运行点 103 路 → goodput **2061 tok/s**

**方法学记录**：第一次校验时我用 `torch.cat` 每层拼 KV，测出 +290% 的假误差；
改为预分配原地写后恢复正常。另注意：带 `synchronize()` 的逐算子剖析会把队列
等待算进 attention 名下（曾误得 attention=26.5ms / 其余=5.6ms），
**逐算子剖析与整步测量不可混用**。

复现：`scripts/overnight/tasks/planner_validation.py` + `scripts/serving_capacity_planner.py`。
