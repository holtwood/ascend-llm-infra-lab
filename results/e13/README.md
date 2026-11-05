# E13 — Ascend 自研 BatchJobAware 调度器：同构/混合负载 A/B

> 面试原题："你们调度器还能怎么优化？" 本实验实测 vllm-ascend 独有的
> `BatchJobAwareScheduler`（LPT 最长任务优先 + KV 预留 + EWMA 解码长度估计）。
> 启用：`--additional-config '{"scheduler_config":{"batch_job_sched_config":{"enabled":true}}}'`

## 臂 A：同构批量负载（c256，E8 同款 KV 耗尽压测）

同负载、同 KV 池（util 0.5，441k tokens）：

| 调度器 | wall | lat_p50 | lat_p99 | 峰值KV | 峰值running | **抢占次数** |
|---|---|---|---|---|---|---|
| 默认 AsyncScheduler | 144.7s | 111.8 | 143.6 | 99.1% | 84 | **3** |
| BatchJobAware | 143.3s | 111.6 | 142.5 | 99.1% | 84 | **7** |

→ 同构负载下**吞吐/时延持平（噪声内），抢占反而更多（7 vs 3）**。
KV-reserve 机制在本负载（无 job 名、EWMA 无历史、全同构）下没有素材
可学——设计场景不匹配时收益为零，这是诚实的边界条件数据。

## 臂 B：混合长度 job（48×长 ~4200tok + 48×短 ~180tok，`#job_name[x]#`）

| job | 默认调度 mean/p50/p90 e2e | batch_job mean/p50/p90 e2e |
|---|---|---|
| long | 31.2 / 31.3 / 32.7 | **29.3 / 29.4 / 30.6** |
| short | 33.2 / 33.2 / 33.3 | **30.9 / 30.9 / 31.1** |

→ 两类 e2e 均 ~6% 改善。注意 LPT 并没有让 short 等更久——
"长任务先占坑、短任务填 decode 空隙"策略下两类**同时受益**。

## 机制解读（面试版）

- 它不是给在线 serving 的优化器，是给 **offline batch** 的：
  job 名分组（prompt 前缀 `#job_name[x]#`）+ EWMA 估计每 job 解码长度
  + 按预测解码量预留 KV block（`reserve_margin/max_blocks`）抑制抢占。
- 同构负载拿不到收益的原因：LPT 需要长度方差才有排序意义；
  EWMA 需要 job 名复现才有估计对象。我们两者都没给足。
- 工程细节：启用后自动走 `BatchJobAwareAsyncScheduler`（async 变体，
  说明 **Ascend 默认 async_scheduling 已开**——E11 的基线前提）。

数据：`pressure_c256o256u50_bj.json`（对照 `results/e8/pressure_c256o256u50.json`）、
`mixedjob_batchjob.txt` / `mixedjob_default.txt`、`serve_batchjob.log`。
