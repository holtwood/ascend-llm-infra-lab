# Ascend NPU LLM Infra Lab

在 4× 华为昇腾 910B2（64GB HBM）单卡容器上开展的 LLM 推理基础设施实验。
目标：围绕**推理服务化、量化、调度特性、系统扩展性**建立可量化、可复现、可讲解的实验体系。

## 硬件平台

| 项 | 规格 |
|---|---|
| 加速卡 | Ascend 910B2 ×4（每容器 1 卡），64GB HBM/卡，HCCS 片间互联（仅同机多卡） |
| CPU | 鲲鹏 920（ARM64），21 核/容器 |
| 内存 | ~200GB/容器 |
| 存储 | /data 200GB SSD/容器；/model 为宿主机共享模型缓存 |
| 软件 | CANN 9.2.0 · torch 2.10 + torch_npu · vllm 0.30 + vllm_ascend 0.30rc2（editable 源码）· MindIE（备用） |

## 集群拓扑

```
                    ┌─────────────────────────┐
                    │ D: 223.111.149.165:40082 │  压测客户端 + 模型库
                    │ (裸镜像, /model×73项)    │  locust/vllm-bench + 负载均衡
                    └───────────┬─────────────┘
                                │ SSH 隧道 / HTTP
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐        ┌───────────────┐
│ 本机 :40019    │      │ M1 .244:40010  │        │ M2 .253:40019  │
│ 910B2 #7      │      │ 910B2 #4      │        │ 910B2 #6      │
│ vLLM 副本 #1   │      │ vLLM 副本 #2   │        │ vLLM 副本 #3   │
│ +源码实验      │      │ +量化/MindIE   │        │               │
└───────────────┘      └───────────────┘        └───────────────┘
```

## 实验矩阵

| # | 实验 | 状态 | 产出 |
|---|------|------|------|
| E1 | vLLM 服务基线压测（并发×时延×吞吐） | ✅ | results/e1 |
| E3/E3b | 调度消融：prefix caching / chunked-prefill 六档扫描 | ✅ | results/e3 |
| E4/E4b | 多副本扩展 1→4 副本 + least_conn，瓶颈迁移到注入层 | ✅ | results/e4 |
| E5 | MindIE 平台绑定失败分析（glibc/CANN 锁定） | ✅ | results/e5 |
| E6 | W8A8-27B 精度代价：QA 94.3% + PPL/top1 一致性 94.4% | ✅ | results/e6 |
| E7 | PD 分离同卡共置：建链失败定位 + 收益上限测算 | ✅ | results/e7 |
| E8 | KV 容量账本：hybrid 公式 vs 实测 + 准入墙/容量墙 | ✅ | results/e8 |
| E8b–e | 手撕算子/attention 变体（并行 lane） | ✅ | results/e8b（含 e8c-e8e 同系列） |
| E9 | goodput@SLO：达标率 vs 吞吐分叉 | ✅ | results/e9 |
| E10 | sleep/wake：释放 97% 显存，wake 0.58s | ✅ | results/e10 |
| E11 | async_scheduling 消融：吞吐 +60% TPOT −39% | ✅ | results/e11 |
| E13 | batch_job_aware 调度器（Ascend 独有） | ✅ | results/e13 |
| E14 | MTP spec decode：接受率 65.6% 但净负（草稿开销） | ✅ | results/e14 |
| E16 | 长上下文扫描 4k→64k：TTFT 线性，KV 占用曲线 | ✅ | results/e16 |
| E17 | 多租户换租：冷启 169s vs 唤醒 0.14s（两引擎同卡共存） | ✅ | results/e17 |
| E18 | PD 跨节点重试：三层失败画像 + 静默垃圾输出 | ✅ | results/e18 |
| E19 | Ascend 调度器画像：4 选 1（SRF 修复优先级倒转 −7.2%） | ✅ | results/e19 |
| D2 | 上游 issue×2 + #18978 comment（vllm-ascend #21112 / #21113） | ✅ | docs/issues |
| E21 | 四节点并行执行：聚合 3119 tok/s，节点间差异 11% | ✅ | results/e21 |
| E23 | 客户端数扫描：抓到 E4b"2100 天花板"是工具伪影（真实上限 4.2-5.0k） | ✅ | results/e23 |
| E25 | 唤醒后稳态：warm 696.6 → post-wake 694.9 tok/s（无损失） | ✅ | results/e25 |
| E26 | 调度器/配置矩阵（4 机同刻）：default 800.6 / SRF 665 / async-off 500 / MTP 194 | ✅ | results/e26 |
| E27 | KV 池扫描：跨节点"趋势"被同节点对照推翻（容量旋钮非性能旋钮） | ✅ | results/e27 |
| E28 | 长上下文并行扫描：4k 178 → 63k 41 tok/s，与 E16 串行互证 | ✅ | results/e28 |
| D3 | prefix-aware 路由 vs least_conn：命中率 54%→84% | ✅ | results/d3 |
| — | 手写 MHA+KV cache（0 diff 验证） | ✅ | examples/mha_from_scratch.py |
| — | SLO/成本换算模型 | ✅ | docs/slo-cost-model.md |

## 通宵实验

- `docs/night-shift-plan.md` — 8-9 小时四机满载方案（启动/监控/收集/恢复）
- `scripts/night_*.sh` — 四条任务线；`scripts/night_collect.py` — 早晨汇总

## 环境B：白盒实验线（3×910B2 单容器，torch 2.9/torch_npu 2.9/CANN 8.5.1）

与环境A（vllm 服务化黑盒）互补的白盒线：算子级剖析、手写实现、新架构实测。规划见 `docs/experiment-plan-b.md`。

| # | 实验 | 状态 | 产出 |
|---|------|------|------|
| N2 | profiler 白盒剖析：eager 下 decode 79% 是 host 开销（device 仅忙 24ms/步，2935 次算子下发） | ✅ | results/n2 |
| N3 | Hybrid GDN 显存曲线：KV 32KB/token 实测=理论（6 点全吻合），GDN state 恒 49.5MB(fp32)，1/4 斜率 vs 全注意力 | ✅ | results/n3 |
| N4 | 跨 NUMA PCIe all-reduce：裸延迟 288µs barrier 主导（4KB-256KB 与大小无关），饱和 16.3GB/s，TP decode 通信=计算 2-4× | ✅ | results/n4 |
| N5 | w8a8 解剖：msModelSlim per-channel int8 对称 + per-token 动态激活量化，覆盖 84.5%，GDN out_proj/lm_head 排除 | ✅ | results/n5 |
| N1 | 手写推理引擎（L0-L5 阶梯）：Qwen3.5-4B hybrid 前向 + KV/RNN state cache + naive 引擎；**真机 4B 权重 greedy 逐 token 一致**（rel diff 8e-3 ∈ bf16 噪声） | ✅ 骨架+验证 | examples/engine_np, results/n1 |
| N7 | PD 分离慢互联账本：32K ctx 传 1GB KV@16.3GB/s=63ms≈0.6 decode 步；层级传输 288µs×32 层握手税 > 传输本身；DistServe 前提在本机不成立 | ✅ | results/n7 |
| N3b | cache 干预六条件：清 GDN state 无影响（它是 decode 递推器），清 KV 机制性塌缩（均匀注意力=有效上下文归零）；**轴向与 arXiv 2609.04434 相反** | ✅ | results/n3b |
| N1b | 引擎阶梯基准：L0/L2 = 4.4×→9.6×（随长度放大），L2 对 ctx 脱敏（+2.2%/16×），**手写比 transformers 快 8-10%**（88-90ms vs 96-98ms/步），30 token 严格等价 | ✅ | results/n1 |
| N6 | vllm-ascend 0.18 部署 27B-w8a8（CANN 8.5.1 兼容组合）+ MTP 扫描：c=1 TPOT **−39%**（73.6→45.1ms）但 c=16 吞吐**净负 15%**——TPOT 改善≠吞吐改善 + TP=2 跨NUMA：全面净负（c=16 TPOT +54%/吞吐 −35%），验证 N4 推论 | ✅ | results/n6 |
| N8 | 算子融合收益：eager RMSNorm 9-op 链 **100% host 开销**（sync≈host-only 211µs）；**torch.compile(inductor) 在 NPU 反慢 39-107%**（launch 更重无真融合）——昇腾走 torchair/TileLang 原生路线的实证；TileLang 原生 kernel 线烂尾（clone 中断，续作在 36.151 工作站的现成环境） | ✅ 主结论 | results/n8 |
| N1-L3 | continuous batching 引擎 NPU 收官：**4.99×（5.3s vs 26.6s，吞吐 5×）**；根因=共享 seq_lens 被 8 个 attn 层各 +T（257=32×8+1 破案），selftest 加第 2 个 attn 层防回归；1/8 token 一致 = N2b 的 bf16 batch 形状敏感 | ✅ | examples/engine_np/batch_engine.py, results/n1/l3_demo.log |
| N2b | batch 扫描：decode **白嫖 32.5×**（b32 每步 108.7ms ≈ b1 的 110.3ms，host-bound 自洽）；两个发现：bf16 大 batch 形状敏感确定性翻边（B=16 全槽第 10 步同步翻，修正 E83 边界）+ **left-padding 在 hybrid GDN 上语义性失效**（pad 进 conv/递推 → 必须 varlen） | ✅ | results/n2b |
| N11 | GDN chunk 扫描：U 形曲线——L=1024 最优 chunk 64（FLA 默认），L=4096 最优 128（比默认快 12%，比最差档快 1.8-2.9×）；chunk 只改结合顺序不改数学（1.2e-4 bf16 噪声） | ✅ | results/n11 |
| N9 | KV offload 账本：pinned vs pageable **25.5×**、跨NUMA 税 1.47×、等效重算带宽仅 91MB/s（**无条件 offload 划算**，pinned 往返=294×）；102MB cache 换出→跑B→换回 **bit-exact**，pinned 换入 10ms | ✅ | results/n9 |
| N1-L3 | continuous batching 调度器（CPU 模拟先行）：token-budget+KV预留+chunked prefill+抢占最后手段；**makespan 6.1× vs static**；3 个活锁 bug 墓志注释 | ✅ | examples/engine_np/scheduler.py |
| — | 手撕题库：MoE router（naive/向量化/负载失衡演示/dispatch）+ top-k/top-p 采样器（Gumbel vs 逆CDF，数值稳定），全部带一致性自测 | ✅ | examples/moe_router_sampler.py |

## 目录结构

- `docs/environment.md` — 环境盘点与已知限制
- `docs/experiment-plan.md` — 每个实验的设计、指标与面试讲解点
- `docs/interview-notes.md` — 面向面试的知识点索引（数据→结论→原理）
- `scripts/` — 全部可复跑脚本（serve/bench/监控）
- `results/` — 原始日志 + 汇总表 + 曲线数据
- `configs/` — 服务配置与 benchmark 参数

## 核心方法论

1. 所有实验脚本化、可重跑；原始日志全部归档
2. 指标三件套：**TTFT**（首 token 时延）、**TPOT/ITL**（单 token 时延）、**端到端吞吐**（tok/s）+ 成功率和 P50/P99
3. 每张数据表对应一个"为什么"——能讲机制才算完成
