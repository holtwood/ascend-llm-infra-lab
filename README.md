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
| E8b–e | 手撕算子/attention 变体（并行 lane） | ✅ | results/e8b–e8e |
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
| D3 | prefix-aware 路由 vs least_conn：命中率 54%→84% | ✅ | results/d3 |
| — | 手写 MHA+KV cache（0 diff 验证） | ✅ | examples/mha_from_scratch.py |
| — | SLO/成本换算模型 | ✅ | docs/slo-cost-model.md |

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
