# E8h · vLLM /metrics 可观测性走读（被动采集）

> 他人在本机引擎上跑 c=64 压测期间，用 15s 间隔抓 `/metrics` 时序，
> 得到一次真实 burst 的调度器遥测 + 97 个 vllm: 指标清单。
> 快照见 `snap_*.txt` / `metrics_t0.txt`。

## 抓到的 burst 轨迹（14:18 前后）

| 时刻 | running | waiting | kv_cache_usage | preemptions |
|---|---|---|---|---|
| 基线 | 0 | 0 | 0 | 0 |
| 涌入中 | 17 | 15 | 2.9% | 0 |
| 峰值 | 64 | 0-6 | **11.0%** | 0 |
| 回落 | 0 | 0 | 0 | 0 |

- running=17 + waiting=15 的快照 = 涌入瞬间准入在补位；峰值 running=64
  与 `-max-num-seqs` 对齐（本次 serve 未显式限制，默认上限内）
- **KV 只用到 11%**：8192 max_model_len + random 512-tok prompt 的负载下
  容量远未成为瓶颈 → 该负载的约束在算力/调度不在 KV——容量规划要先看
  `kv_cache_usage_perc` 再决定加卡还是调参
- **零抢占**：与 E8f 模拟器呼应——KV 水位低时抢占计数恒 0，抢占只在
  `waiting_by_reason=capacity` 持续非零时才该报警

## 指标语义速查（面试"怎么监控 serving"直接答）

| 指标 | 看什么 | 告警姿势 |
|---|---|---|
| `num_requests_running` / `waiting` | 实时在批/排队 | waiting>0 持续 = 饱和或准入受限 |
| `num_requests_waiting_by_reason` | 排队原因（capacity/deferred） | capacity 涨 → KV 或 max_num_seqs 卡准入 |
| `kv_cache_usage_perc` | KV 池水位 | >85% → 抢占前夜，该扩 KV 或降 max_len |
| `num_preemptions_total` | 抢占次数 | 非零速率 = KV 容量不足 |
| `prefix_cache_queries/hits_total` | APC 命中率 | 命中率低时 cache-aware 路由或检查前缀形态 |
| `prompt_tokens_by_source_total` | prompt 来源（local_cache_hit 等） | 命中 vs 新算的比例账 |
| `inter_token_latency_seconds_*` | ITL/TPOT 直方图 | SLO 的 TPOT 口径 |
| `iteration_tokens_total_*` | 每步批 token 数分布 | decode 批大小画像 |
| `estimated_flops/read_bytes/write_bytes_per_gpu` | 算力/访存估计 | 判断 compute vs memory bound |
| `engine_sleep_state` | sleep/wake 状态 | 多租户弹性观测（E17） |
| `mm_cache_*` | 多模态缓存 | VLM 场景 |

## 教训

- **压测要看服务端口径**：客户端 TTFT/TPOT 与服务端 running/waiting/kv%
  是同一次实验的两面，只报客户端数等于少了一半证据
- 零值也是信息：`prefix_cache_hits=0` 暴露压测方用了 random 数据集——
  没共享前缀时 APC 指标必须为零，否则就是指标接错
