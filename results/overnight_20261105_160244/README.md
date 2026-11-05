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

与 E8k（6.5–8.5×）差异说明：E8k 是小尺寸/不同实现口径；本网格用广播式
GQA 消除了展开开销，是更干净的税基。两者不矛盾——税基取决于实现。

## E8j-ext：paged gather 块粒度敏感度（paged_gather.json）

S∈{4096..32768} × block∈{16,32,64,128}，decode 形态（Sq=1）：

- S=4096：税 1.58×（block16）→ **2.73×（block128）**——短上下文下大块更差
- S≥8192：税稳定 **~2.1×**，块大小影响很小
- 对照 E8j（3–16×）：同样是实现口径差异，本网格给出的是
  "物化 gather 在长上下文 decode 的真实税 = 2× 量级"

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

单卡 8 卡独占，按行并行切分真实权重，decode 步（bs=1,seq=1）含每层 3 次
allreduce（qkv/o/down 后各一次，1KB 级消息）。**两次独立运行**（含/不含
并发 soak）结果一致（差 <4%），下表为干净运行：

| TP | 步时 (µs) | 相对 tp=1 | 单次 allreduce | 通信占比 |
|---|---|---|---|---|
| 1 | 205 | 1.00× | — | 0% |
| 2 | 793 | **3.88×** | 185 µs | 70% |
| 4 | 842 | 4.11× | 206 µs | 73% |
| 8 | 915 | **4.47×** | 215 µs | 71% |

**结论**：0.6B 级模型的 decode 步，TP 的通信税**压过计算收益**——
8 卡比单卡慢 4.5×。计算本身从 205µs 只降到 ~180µs（权重切片后 matvec
本就受带宽限制，切分不省时），而 3×215µs 的 allreduce 吃掉 71% 步时。

与 E8i（HCCL 原始延迟底 ~150µs）和 E8d（TP 通信税不随卡数降）完全自洽：
**小模型 decode 用副本不用 TP**，这是第三条独立证据链。

数据卫生：第一次运行时意外残留了一个 soak 进程（master 的 pkill 因 SSH
断连未生效），因此做了第二次干净运行；两者一致说明干扰 <4%，
但首轮数字不采用。

复现：`scripts/overnight/tasks/tp_decode_sim.py`（torch 级，不需要 vllm）。

## 远端 vllm-ascend serve 结论：CANN 9.0 硬版本墙

vllm+vllm_ascend 在远端成功安装并激活 `PLATFORM npu`、W8A16 模型权重
加载成功、`/v1/models` 返回 200——但首次前向死在
`aclnnScatterPaKvCache`（CANN 9.2 才有的内置算子，9.0 缺失）。
这不是配置问题：paged KV scatter 是 9.2 新增能力，vllm-ascend 0.30
的 paged attention 依赖它。也解释了厂商为何在 9.0 容器里配 MindIE。

踩坑全记录：`docs/remote-vllm-bringup.md`。
