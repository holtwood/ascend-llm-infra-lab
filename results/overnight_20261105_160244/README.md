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
常数都来自本仓实测（E29 步时/地板、E37/E39/E52 attention 效率 19→50→86%、
E48 prefill 模型）。0.6B、8k 上下文、prompt 512-2048、输出 128、max_batch 64、
SLO = TTFT ≤ 2s 且 TPOT ≤ 100ms。

| λ (req/s) | TTFT p50 | TTFT p99 | TPOT p50 | 吞吐 | **goodput** | SLO 达成 |
|---|---|---|---|---|---|---|
| 1 | 50.6 ms | 66.3 ms | 32.7 ms | 123 tok/s | 123 | 100% |
| 2 | 52.1 | 68.3 | 34.3 | 248 | 248 | 100% |
| 4 | 54.5 | 73.2 | 36.2 | 505 | 505 | 100% |
| **8** | 60.2 | 80.9 | 40.5 | 1016 | **1016** | 100% |
| **12** | **11,887 ms** | 24,461 | 42.5 | 1399 | **150** | 11% |
| 16 | 44,830 | 90,807 | 42.5 | 1402 | **48** | 3% |

**结论（serving 面试的核心认知）**：
1. **拐点在 λ≈8-12 req/s**：过拐点后吞吐只从 1016 涨到 1402（+38%），
   但 **goodput 从 1016 崩到 48 tok/s（−95%，21×）**。
2. **TPOT 几乎不变（40→42ms），TTFT 爆炸（60ms → 44.8s）**——过载时延迟
   恶化全部体现在**排队**上，不在单步计算上。所以"只看 TPOT 判断健康度"是错的，
   必须看 TTFT 和队列长度。
3. 只报吞吐的压测（1402 tok/s）会把容量结论**高估 29 倍**（1402 vs 可交付 48）。
   这正是 E9 在真实 vLLM 上观察到的现象，此处给出了机理与量化曲线。

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

## E53 · KV 块长：碎片 vs gather 税，为什么 vLLM 选 16（block_*.json）

两个成本反向移动：块越小尾部浪费越少，但 gather 税越高。用**同一口径**实测
S=4096 的 gather 税（B=4，NKV=8），叠加三种真实长度分布的碎片：

| 块长 | 尾部浪费（lognormal/uniform/bimodal） | gather 税 | 效率指数（容量×速度） |
|---|---|---|---|
| **16** | ~0.5% | **1.54×** | **0.646** |
| 32 | 1.1 / 1.5 / 1.3 % | 1.70× | 0.580 |
| 64 | 2.2 / 2.9 / 2.6 % | 1.98× | 0.492 |
| 128 | 4.4 / 5.7 / 4.7 % | 2.31× | 0.411 |
| 256 | 8.5 / 10.6 / 8.5 % | 2.30×（平台） | 0.396 |

**结论**：
1. **block=16 在三种分布下都是最优**——碎片省下的容量收益压过 gather 税，
   这定量解释了 vLLM 默认值的选择。
2. gather 税在 128 之后进入平台（2.3×），所以再增大块只增加碎片、不换速度。
3. 从 16 增大到 256：容量效率降 9-11%（碎片），速度降 33%（税），
   综合效率降 **39%**。

**方法学**：先前用两套 harness 测出的 128 块税（2.73×）与 256 块（2.35×）
不单调，改用统一 harness 重测后得到单调曲线（1.54→1.70→1.98→2.31→2.30）。
**跨 harness 的数字不可混用。**

复现：`scripts/overnight/tasks/block_size_analysis.py`。

## E54 · 多租户调度：FCFS / SJF / 准入控制 / 优先级（multitenant_*.json）

**仿真**（同一套实测服务时间常数）。两类租户：short（prompt 128-256，占 70%）、
long（2048-4096）。SLO = TTFT ≤ 2s 且 TPOT ≤ 100ms。

### λ=6 req/s（容量内）——策略完全无差别

| 策略 | short p99 | long p99 | goodput |
|---|---|---|---|
| 全部四种 | 101.7 ms | 152.0 ms | **757.8** |

**没到过载点时，调度策略一点都不重要**——这是最容易被忽略的事实。

### λ=12 req/s（过载）——策略决定 24× 的差距

| 策略 | short p99 | short SLO | long p99 | long SLO | 拒绝 | **goodput** |
|---|---|---|---|---|---|---|
| FCFS | 79.5 s | 4.3% | 79.6 s | 4.2% | 0 | **46.9** |
| **SJF** | 3.15 s | 97.1% | 78.4 s | 49.2% | 0 | **1140.1** |
| 准入控制 | 2.56 s | 69.5% | 2.65 s | 60.9% | 1003 | 732.6 |
| 严格优先级 | **1.67 s** | **100%** | 221.9 s | 6.5% | 0 | 1070.5 |

**三个结论**：
1. **FCFS 在过载下崩溃**（两类都 79s p99、goodput 47）；**SJF 把 goodput 拉回 24×**
   且两类都不算被饿死（long 仍有 49% 达标）。
2. **严格优先级把 short 做到 100% 达标，但 long 饿死到 222s**——这就是
   "保护一类必牺牲另一类"的定量版本；goodput 反而不如 SJF。
3. **准入控制是另一种权衡**：拒绝 1003 个请求换来两类都 2.6s 的均衡延迟，
   但 goodput 最差（733）——**拒绝浪费了容量**。

**面试用法**：被问"过载怎么办"，先问 SLO 归属（哪类请求必须保），
再给这三条曲线的取舍；不要一上来就说"加机器"或"限流"。

复现：`scripts/overnight/tasks/multi_tenant_sched.py --rate 12`。

## E58 · W8A16 在 prefill 负载下也没提速（w8a16_prefill.jsonl）

E41 说 decode 上 W8A16 是负收益（发射地板支配）。prefill 是相反区间
（大 M、算力受限），理论上量化应该有用——实测却仍然更慢：

| L | bf16 | int8 | 加速比 |
|---|---|---|---|
| 1024 | 0.141 ms | 0.212 ms | 0.67× |
| 4096 | 0.323 ms | 0.381 ms | 0.85× |
| 8192 | 0.601 ms | 0.684 ms | 0.88× |
| 16384 | 1.339 ms | 1.483 ms | 0.90× |

bf16 在 L=16384 时 MLP 达 **225 TFLOPS**（算力受限区间），int8 反而只有 204 TFLOPS。

**结论：`npu_weight_quant_batchmatmul` 在本环境（CANN 9.0 / torch_npu 2.10）
没有用上 int8 张量核的高吞吐**，所以 W8A16 在两个区间（decode 与 prefill）
都没有速度收益，**唯一确定收益是容量**（E41：权重 −29%）。

这是一个有价值的负结果：它说明"量化提速"不是免费的，取决于后端 kernel 是否
真正走了低精度计算路径。要拿到速度收益，需要更好的融合 kernel
（这也解释了为什么 vLLM 的量化收益高度依赖具体后端实现）。

复现：`scripts/overnight/tasks/w8a16_prefill.py`。

## E59 · 冷启动分解：169 秒里有多少是"搬数据"（cold_start.json）

E17 在真实 vLLM 上测得冷启动 **169.4s**（4B 模型）。本实验在裸 torch 栈上
分解"搬权重"这一段的真实成本（0.6B，1.5GB 权重）：

| 阶段 | 耗时 | 带宽/说明 |
|---|---|---|
| 磁盘读（page cache 命中） | 0.02 s | 92.9 GB/s（内存带宽速度，非真盘） |
| H2D 拷到 HBM（逐 tensor） | **0.29 s** | **5.1 GB/s** |
| 首次调用（kernel 预热） | 0.098 s | **817× 稳态**（稳态 0.12ms） |
| **合计（仅权重）** | **0.41 s** | |

**三个可操作的发现**：

1. **逐 tensor 拷贝只有 5.1 GB/s，比批量拷贝（21-24 GB/s，E8-B1B2）慢 4×**——
   300+ 次小传输被发射开销吃掉。**加载权重必须批量搬。**
2. **首次调用是稳态的 817×**（98ms vs 0.12ms）——这就是引擎必须做
   warmup 请求才能标记"就绪"的原因，也是 ACLGraph 捕获要花几十秒的同一类成本。
3. **对比 E17 的 169s**：本实验证明"搬 1.5GB 权重"只需 0.41s——
   **冷启动 99.8% 的时间花在引擎初始化（图捕获、内存池、编译），不是数据搬运。**
   所以优化冷启动的方向是"减少图捕获/编译"或"用 sleep-wake 复用进程"（E17：唤醒 0.139s），
   而不是"加快权重加载"。

复现：`scripts/overnight/tasks/cold_start.py`。

## E62 · GQA / MQA / MHA 的成本结构（gqa_ratio.jsonl）

固定 hq=32、d=128、ctx=8k，只改 KV 头数 nkv：

| nkv | 分组 G | bs=1 时延 | KV/token | bs=1 带宽 | bs=8 时延 | bs=8 带宽 |
|---|---|---|---|---|---|---|
| 32（MHA） | 1 | 159.5 µs | 16,384 B | 841 GB/s | 836 µs | **1284 GB/s（91% 峰值）** |
| 8（GQA-4） | 4 | 149.9 µs | 4,096 B | 224 GB/s | 780 µs | 344 GB/s |
| 4（GQA-8） | 8 | 149.0 µs | 2,048 B | 113 GB/s | 772 µs | 174 GB/s |
| 1（MQA） | 32 | 149.9 µs | **512 B** | 28 GB/s | 769 µs | 44 GB/s |

**两个反直觉但重要的结论**：

1. **bs=1 时，KV 内存降到 1/32，attention 时间几乎不变**（160→150 µs）。
   因为低 batch 下 attention 是延迟受限（28 GB/s 远未触及带宽），
   **GQA/MQA 在低并发下只省显存，不提速**。
2. **bs=8 时也只有 8% 的提升**（836→769 µs），尽管 KV 读少 32×。
   只有 MHA 真正跑到了带宽墙（91% 峰值）；MQA 的 KV 读量太小，
   根本喂不饱带宽。

**serving 结论**：**GQA/MQA 首先是容量优化**——同样的 HBM 能装 32× 的序列
（E30 的公式里 KV/token 直接除以 32）。速度收益只在"KV 读成为瓶颈"的
高并发+长上下文区间才兑现。面试里把 GQA 说成"为了加速"是不准确的，
**它是为了在固定显存下把并发/上下文做上去**。

复现：`scripts/overnight/tasks/gqa_ratio.py`。

## E63 · 全模型混批：chunked prefill 一步的真实代价（mixed_batch*.json）

一个调度步里同时跑 8 条 decode（ctx=4k）和一个 prefill chunk，实测整步时延：

| prefill chunk | 步时 | prefill 增量 | **TPOT** | prefill 吞吐 |
|---|---|---|---|---|
| 0（纯 decode） | 33.5 ms | — | 34 ms | — |
| 512 | 42.8 ms | +8.2 ms | 43 ms | 12k tok/s |
| **2048** | 42.6 ms | +8.0 ms | **43 ms（+23%）** | 48k tok/s |
| 4096 | 57.4 ms | +23.8 ms | 57 ms（+71%） | 71k tok/s |
| 8192 | 113.7 ms | +80.1 ms | **114 ms（+239%）** | 72k tok/s |

**结论**：
1. **chunk=2048 时，混批只让 TPOT 涨 23%**——因为整步仍是发射受限（34ms 地板），
   prefill 的计算被塞进已有的发射间隙里，**几乎免费**。
2. **chunk=8192 时 TPOT 涨 3.4×**，prefill 吞吐稳定在 72k tok/s。
   这与 E34（chunk=2048 是甜点）和 E44（8k chunk 造成 attention 层 16-34× 干扰）
   在量级上互相印证：**chunk 超过 ~4k 后干扰急剧上升**。
3. **与 E44 的差异要说清楚**：E44 只测 attention 算子（无发射间隙可藏），
   8k chunk 造成 33× 膨胀；E63 测整模型（发射受限），2k chunk 只 +23%。
   **"chunked prefill 贵不贵"取决于栈是发射受限还是带宽受限**——
   eager 栈里它便宜，图捕获后它会变贵（没有发射间隙可藏了）。

**方法学**：首版用 `expand`+`reshape` 做 chunk 内 causal attention，
触发了 O(P²) 物化（2048 chunk 虚高到 512ms）；改为纯广播后降到 42.6ms。

复现：`scripts/overnight/tasks/mixed_batch_step.py`。

## E65 · KV 精度对容量的杠杆（规划器，graph-captured，TPOT SLO=100ms）

给容量规划器加 `--kv-scale`（1.0=bf16 / 0.5=int8 / 0.25=int4）后扫描：

| 上下文 | bf16 KV | int8 KV | int4 KV |
|---|---|---|---|
| 2k | 244 路 / 5,056 tok/s | 489 / **10,113** | 979 / **20,226** |
| 8k | 61 / 1,232 | 122 / 2,528 | 244 / 5,056 |
| 32k | 15 / 178 | 30 / 447 | **61 / 1,232** |

**三个结论**：
1. **KV 量化对容量是线性杠杆**：int8 → 2×，int4 → 4×，而**步时几乎不变**
   （48-49ms）——因为 2× 序列 × 0.5× 字节 = 相同的 KV 读量。
   （int4 质量门槛见 E66：必须 KIVI 式 per-channel-K，naive/group 毁掉模型。）
2. **但兑现它需要融合 kernel**：E8k 证明张量级反量化要付 1.4-6.5× 的税
   （fp32 中间态更差）。**容量收益是确定的，速度收益取决于 kernel**。

> **⚠️ 事后修正（E66 质量验收后）**：naive per-token int4 KV 使 ppl 从
> 20.8 涨到 316 万（top-1 3.9%）；group-32 定标只救到 ppl 89（+328%）。
> **但 KIVI 式（K per-channel + V group-32）ppl 仅 +0.27%，top-1 84.3%**——
> int4 容量兑现的路径存在，前提是 KIVI 式定标；int8 KV 是零技巧保底方案。

复现：`python3 scripts/serving_capacity_planner.py --kv-scale 0.5 ...`。

## E66 · W8A16 的质量验收：困惑度 +1.27%（ppl.json）

速度与容量之外，量化必须过"质量关"。用同一段中英混合文本（153 token），
同一前向（含 Qwen3 的 q_norm/k_norm + RoPE），bf16 vs int8 权重：

| 指标 | bf16 | int8 (W8A16) | 差异 |
|---|---|---|---|
| 困惑度 ppl | **20.820** | **21.085** | **+1.27%** |
| 平均 NLL | 3.0359 | 3.0486 | +0.0127 |
| top-1 预测一致率 | — | — | **96.08%** |

**扩展到两个量化轴（权重 int8 × KV int8）**：

| 配置 | ppl | ppl 变化 | top-1 一致率 |
|---|---|---|---|
| bf16（基线） | 20.820 | — | 100% |
| W8A16 权重 | 21.085 | +1.27% | 96.08% |
| bf16 + int8 KV | 21.136 | **+1.52%** | 91.50% |
| W8A16 + int8 KV | 21.113 | +1.41% | **88.24%** |
| bf16 + int4 KV（naive per-token） | **3,163,652** | +15,195,152% | **3.92%** |
| bf16 + int4 KV（group-32 定标） | 89.20 | +328% | 43.79% |
| **bf16 + int4 KV（KIVI：K per-channel + V group-32）** | **20.876** | **+0.27%** | 84.31% |

**两个新发现**：
1. **int8 KV 的 ppl 代价（+1.52%）与 W8A16（+1.27%）同量级**，但
   **top-1 一致率掉得更多**（91.5% vs 96.1%）——KV 量化扰动预测的程度
   比权重量化更深，尽管困惑度看不出差别。**验收要看 top-1 一致率，不能只看 ppl。**
2. **两者叠加时 ppl 不叠加**（+1.41% < 1.27+1.52），但 top-1 一致率
   继续下滑到 88.2%——**误差在困惑度上会互相抵消，在逐 token 预测上不会**。
3. **int4 KV 的成败全在定标方式（E66 第三轮，ppl4.json）**：
   naive per-token 毁掉模型（ppl 316 万）、group-32 仍然 +328%，
   但 **KIVI 式（K 按 channel 定标 + V 按 group-32 定标）ppl 只 +0.27%**——
   比 int8 KV 的 +1.52% 还好（top-1 一致率 84.3% vs 91.5%，逐 token
   扰动仍比 int8 大）。**失败机制确认是 K 的通道级 outlier 把 per-token
   scale 带崩**——这正是 KIVI 论文的核心论点，在本模型上被复现。
   → E65 的 int4 4× 容量**有条件成立**：必须 KIVI 式定标，naive/group
   int4 不可交付。int8 KV 仍是"零技巧可交付"的保底方案。

**结论：W8A16 的质量损失很小**（+1.27% ppl、96% top-1 一致），
属于可接受的量化方案。结合 E41/E58：
- **质量**：+1.27% ppl（可接受）
- **容量**：权重 −29%
- **速度**：0.80-0.99×（负收益，本栈 kernel 未用上 int8 张量核）

→ **决策：显存受限时用 W8A16（换容量、付一点质量、速度不变差）；
追求速度则不该用它。**

**方法学教训（值得记住）**：首版前向漏了 Qwen3 的 **RoPE + q_norm/k_norm**，
ppl 测出 **2,890,863**（垃圾值）。补上后降到 20.82。
**"ppl 荒谬"永远是实现错了，不是模型坏了**——这个自查信号要记住。

复现：`scripts/overnight/tasks/perplexity_check.py`。

## E67 · 采样开销：top-p 在大 batch 下比整个前向还贵（sampling.json）

logits→token 是 TPOT 里很少被量化的隐形成本。vocab=151936（Qwen3）：

| bs | LM head | argmax | softmax+multinomial | **top-p（sort）** |
|---|---|---|---|---|
| 1 | 0.84ms | 0.03ms | 0.17ms | **1.51ms（1.8× LM head）** |
| 32 | 0.86ms | 0.03ms | 0.62ms | **21.5ms（25×）** |
| 64 | 0.88ms | 0.04ms | 1.17ms | **40.9ms（46×）** |
| 128 | 0.95ms | 0.06ms | 2.33ms | **78.4ms（83×）** |

**结论**：
- greedy/argmax 几乎免费（reduce 类 kernel）；
- 朴素 top-p = softmax + **全词表排序** + cumsum + mask + multinomial，
  O(vocab·log vocab)，bs=128 时 78ms——**超过本模型整个 decode step（30ms）**；
- 这就是 vLLM 有独立 sampler kernel、且 top_k 先行截断词表的原因：
  **先 top_k 截到 k 再排序，成本从 151936·log 降到 k·log**；
- 温度采样（无 sort）居中，bs=128 时 ~2.3ms/step，是 TPOT 的可见税。

注意这是朴素 torch 实现的量级证据；生产 kernel 会快很多，但
**"top-p 排序是词表大小的超线性成本"这个结构事实不变**。

## E68 · Roofline 扫描：910B1 的实测峰值与拐点（roofline.json）

GEMM [M,K]×[K,N]，bf16：

| 维度 | 结果 |
|---|---|
| 实测峰值 | **~321-324 TFLOPS**（4096³ / ffn m=8192），对标官方 ~320 |
| 每算子延迟地板 | **~0.024ms**——m≤512 的所有 GEMM 都被发射地板钉死 |
| 达到接近峰值的 M | **m≈2048-4096**（ffn_up 在 m=8192 到 321TF） |
| m=1（单序列 decode） | 0.08-0.27 TFLOPS，**只用掉峰值的 0.03%** |

**roofline 的 serving 解读**：
- 机器平衡点 320TF/1.6TB/s ≈ **200 flop/byte**；
- 但比理论更糟的是：**每步 m（batch token 数）≤512 时，连访存受限都谈不上，
  直接卡死在 24µs 发射地板上**——又一次印证 eager 栈是"launch-bound 第一"；
- 只有图捕获/大 batch 后才能谈真正的 roofline；
- decode（m=bs）在 bs≤64 时永远是延迟受限，**"加大 batch 提吞吐"的真实
  机理是摊薄发射延迟，而不是摊薄带宽**（带宽早就够摊了）。

复现：`scripts/overnight/tasks/sampling_overhead.py`、`roofline_scan.py`。

## E69 · 真实投机解码：w8a16 当 draft + bf16 当 target（specdec.json）

不是模拟——真的跑 draft/verify 循环：int8 模型贪心起草 q 个 token，
bf16 模型一次前向验证，取最长匹配前缀 + target 的下一 token：

| q | 每次 target 前向产出 | 各位置接受率 |
|---|---|---|
| 2 | 3.0 | [1.0, 1.0] |
| 4 | 4.78 | [1.0, 1.0, 0.89, 0.89] |
| 8 | **9.0（全接受）** | [1.0×8] |

**结论与注意点**：
- **接受率 ~95-100%**（与 E66 的 96% top-1 一致率吻合）。**第二个
  prompt（英文创意诗，specdec2.json）接受率稍低且随草稿深度递减**：
  q=8 时前 6 位 100% 接受、第 7-8 位掉到 75%，tokens/fwd 8.5——
  验证了"深位置更容易被拒（误差累积）+ 高熵文本接受率更低"两点，
  但整体仍在 85-100% 区间；
- **速度模型**（draft=target 同尺寸，本实验 wall-clock 无加速，符合预期）：
  speedup = E[tokens/cycle] / (q·d + v)。换成真实配置（0.6B draft +
  32B target，步时比 ~5×：d=6ms, v=30ms）→ q=4 时 4.78 tok / 54ms
  = **11.3ms/token vs 30ms → ~2.65×**；
- **q 的选择**：接受率高时 q 越大越好（本例 q=8 全接受→9 tok/前向）；
  α≈0.7 以下 q>4 收益急剧递减。生产系统按实测 α 在线调 q。

## E70 · KV swap 时延与 offload 决策边界（kvswap.json）

真实 KV block 粒度（kv/token=114688B）的换入换出：

| tokens | KV 体积 | D2H | H2D | 往返 | 对比 recompute |
|---|---|---|---|---|---|
| 128（1 block） | 15MB | 0.81ms | 0.72ms | 1.5ms | — |
| 2048 | 235MB | 11.2ms | 9.2ms | **~21ms** | ≈重算 ~30ms → 打平 |
| 8192 | 940MB | 40.1ms | 40.5ms | **~81ms** | < 重算 98ms → 胜 1.2× |
| 32768 | 3.76GB | 187ms | 143ms | **~350ms** | < 重算 ~1s → 胜 ~2.7× |

**三个工程结论**：
- **pinned ≈ pageable**（都 ~20-26 GB/s）——已验证 `is_pinned()` 确实返回
  True，锁页分配是真实生效的，**所以"无收益"是硬件/驱动层事实而非测量
  假象**——与 GPU 上 pinned 显著更快的经验不同，值得标注；
- **逐 block 换出无额外惩罚**：64 个 128-token block 顺序换出 45ms ≈
  一次性大传输（每 op 开销 ~0.7ms 相对 15MB 传输可忽略）；
- **与 E32 结论一致但更精确**：swap 在 >4k 上下文胜出，但 2k 处基本打平——
  抢占策略应**按上下文长度自适应**（短上下文直接重算，长的换出）。

## E71 · 连续批处理下的 TPOT 分布（tpot_churn.json）

真实 decode 28 层逐 step 跑，bs 在 [1,64] 随机游走（ctx=4096）：

| 场景 | p50 | p90 | p99/max |
|---|---|---|---|
| 纯 churn（bs 7-38） | 44.7ms | 60.3ms | 68.2ms |
| churn + 每 12 步插 2048 prefill（decode 步） | 29.2ms | 44.7ms | 52.4ms |
| 同上（含 prefill 的步） | 33.2ms | 40.7ms | 40.7ms |

**结论**：
- **TPOT 是分布不是常数**：仅"同租请求数"波动就让 TPOT 从 ~30ms（bs≤8）
  涨到 ~68ms（bs≈36），p99/p50 ≈ 1.5×——SLO 余量要按 p99 定，不是按均值；
- **chunked prefill 注入每步只加 ~4ms（+14%）**，与 E63 的 +23% 一致量级——
  发射受限栈里 prefill chunk 大部分藏进启动间隙；
- 又一次冷启动污染教训：**t=0 步 372ms**（首次分配/编译），若不排除会把
  p99 假报成 8×——所有 step-trace 都必须舍掉首步。

## E72 · 分层 APC：前缀恢复的三级路径（apc_tier.json）

共享前缀 KV 的恢复成本，三条路径同形状实测：

| L | KV 体积 | CPU reload（pinned H2D） | 重算 prefill | reload/recompute |
|---|---|---|---|---|
| 2048 | 235MB | 10.0ms | 34.7ms | **0.29×** |
| 8192 | 940MB | 36.1ms | 98.5ms | **0.37×** |
| 32768 | 3.76GB | 203.2ms | 960.3ms | **0.21×** |

**结论**：**HBM hit（≈0）≫ CPU tier reload ≫ recompute**。
CPU 层恢复只要重算的 21-37%——这意味着**前缀缓存不该因 HBM 不够就驱逐到底，
值得常驻 pinned CPU 池做二级缓存**：8k 前缀 CPU 命中 36ms vs 重算 98ms vs
HBM 命中 ≈0。这是 docs/design/kv-tiering.md 分级模型的实测常数。

复现：`scripts/overnight/tasks/tpot_churn.py`、`tiered_apc.py`。

## E75 · 传输与计算的流并发：谁能藏进 GEMM 间隙（overlap.json）

双流并发：一路 ~51ms GEMM 链（14×8192³）+ 一路传输，对比顺序执行：

| 传输类型 | 单独 | 顺序 | 并发 | 隐藏率 |
|---|---|---|---|---|
| **D2H 940MB**（KV offload / 权重加载） | 39.7ms | 90.6ms | **51.3ms** | **98.9%** |
| **D2D 235MB×80**（设备内搬运） | 29.8ms | 79.3ms | **86.3ms** | **−23.6%** |

**结论**——两条互补的物理事实：
- **主机↔设备传输走独立 DMA 引擎 → 几乎 100% 可被计算掩盖**。
  含义：KV offload 换出、异步权重加载、PD 场景下经 PCIe/网卡发 KV，
  只要提前一拍发起，对 decode 步时近乎零成本（前提是带宽不饱和）。
- **设备内拷贝与 GEMM 共享 HBM 带宽 → 并发反而 −24%**。
  含义：设备内的 KV gather/搬运/重组**藏不住**——它直接从 attention/MLP
  嘴里抢带宽。这与 E44"同卡 PD 干扰是物理竞争"自洽：同卡上的显存流量
  没有第二条路可走。

复现：`scripts/overnight/tasks/stream_overlap.py`。

## E76 · 融合 attention 不是永远更快：SDPA 是"prefill kernel"（attn_backend.json）

同形状下 SDPA vs 手写（matmul+softmax+matmul，物化 scores）：

| 场景 | SDPA | manual | 倍数 |
|---|---|---|---|
| prefill L=512 | 0.12ms | 0.26ms | **SDPA 2.1×** |
| prefill L=2048 | 0.23ms | 1.24ms | **5.3×** |
| prefill L=8192 | 1.82ms | 27.8ms | **15.3×** |
| prefill L=16384 | 6.84ms | 118.7ms | **17.4×** |
| decode bs=16 ctx=4k | 3.99ms | 0.43ms | **manual 9.3×** |
| decode bs=64 ctx=4k | 15.4ms | 1.65ms | **manual 9.3×** |
| decode bs=16 ctx=32k | 31.5ms | 3.41ms | **manual 9.2×** |

**结论——"融合 kernel 更快"只在 prefill 形状成立**：
- prefill 下 SDPA 避免物化 [H,L,S] scores，L 越大收益越大（2→17×），
  这是 FlashAttention 的经典收益；
- **decode（q_len=1）下 SDPA 比两个朴素 matmul 慢 9×**——torch_npu 的
  SDPA 路径显然按大 query tile 优化，1-token query 走了低效分支；
- 这解释了为什么真实引擎**给 prefill 和 decode 用不同 attention 后端**
  （vLLM-Ascend 的 paged/decode kernel，不是朴素 SDPA）——
  **backend 选择是形状相关的，"统一融合 kernel"是伪命题**；
- 注：decode 的 SDPA 慢也部分抬高了 E29-E39  eager decode 的步时读数，
  生产 paged-attention kernel 下绝对值会更低（相对结论不受影响）。

复现：`scripts/overnight/tasks/attn_backend.py`。

## E77 · HCCL allreduce 与计算的并发（hccl_ovl64/hccl_ovl4.json）

2-rank allreduce + 51ms GEMM 链，async_op 重叠 vs 串行：

| 消息 | AR 单独 | 串行 | 重叠 | 隐藏率 |
|---|---|---|---|---|
| 64MB | 3.53ms | 55.2ms | 52.4ms | **78.6%** |
| 4MB | 0.32ms | 51.8ms | 51.5ms | **86.5%** |

**结论**：HCCL 与 D2H 一样走独立链引擎（HCCS），**~80-87% 可藏进计算**——
这是真实 TP 实现做 comm/compute overlap 的物理基础；naive eager TP
把 AR 串在依赖链里所以税收满（呼应 tp_decode 的 3.64×）。
注意与 E75 对比：D2D 并发反而 −24%（抢 HBM），
**"能藏的是跨链路传输，不能藏的是设备内搬运"**。

## E78 · 面经原题：巨型 prompt 进 64 路 decode（giant_pf.json）

bs=64 decode @ ctx=4k，基线 p50=80.9ms；16k prompt 在 t=15 被调度进来：

| 策略 | TPOT 变化 |
|---|---|
| ungoverned（整段进一个 step） | **下一步 320.8ms（4×）**，单步尖峰 |
| chunked 2048 | 连续 8 步 ~106ms（**+31%**），无尖峰 |

**面试答法**：chunked prefill 不改变总功（8×25ms ≈ 200ms vs 一次 240ms），
**改变的是 TPOT 违约的形状**——4× 尖峰 vs 有界的 +30%；
对 SLO 而言后者可接受，这就是 max_num_batched_tokens 的物理意义。

## E79 · 面经原题：前缀分叉点对复用收益的影响（prefix_div.json）

8k prompt，共享前缀在位置 p 分叉后还能省多少 prefill：

| 共享 p | 复用比例 | 实测省 prefill |
|---|---|---|
| 1024 | 12.5% | 12% |
| 4096 | 50.0% | 47% |
| 7168 | 87.5% | 80% |
| 8176 | 99.8% | 82%（饱和：新 token 仍要全量 attention） |

**结论**：
- 省成本 ≈ 复用比例的**线性函数**——system prompt 头部插时间戳，
  p 掉到几十 → 收益归零（"前缀必须逐 token 相同"的量化含义）；
- vLLM block-16 对齐对长前缀几乎无损（尾差 ≤15 tok），
  但短前缀相对损失大（p=40 时命中 32 vs radix 40 = −20%）——
  RadixAttention 的 token 粒度优势集中在短共享前缀场景。

复现：`scripts/overnight/tasks/{hccl_overlap,giant_prefill_trace,prefix_divergence}.py`。
