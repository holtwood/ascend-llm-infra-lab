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
