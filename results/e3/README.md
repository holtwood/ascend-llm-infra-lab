# E3 · 调度特性消融结果

## 实验 1：Prefix Caching（共享前缀负载）

- 负载：`prefix_repetition` 数据集 = 4 个不同的 2048-token 前缀，128 请求共享，
  每个请求另有 256-token 后缀 + 128-token 输出；c=32
- 对照：服务端默认开启 vs `--no-enable-prefix-caching` 重启

| 配置 | 成功 | 时长(s) | TTFT mean | TPOT mean | 输出吞吐 tok/s |
|---|---|---|---|---|---|
| PC ON | 128/128 | 20.9 | 1018.5 | 32.89 | 785.6 |
| PC OFF | 128/128 | ~31 | 1298.5 | 51.28 | 519.9 |
| **差值** | | **+48%** | **+28%** | **+56%** | **−34%** |

### 解读

- 128 请求仅 4 种前缀 → PC ON 只发生 4 次 2048-token 完整 prefill，
  其余 124 个请求命中缓存只需算 256-token 后缀 → TTFT 省 28%
- PC OFF 时所有请求全量 prefill（2304 tok），抢占算力还拖累 decode →
  TPOT 恶化 56%、整批吞吐降 34%
- **面试话术**：prefix caching 的收益 ∝ 负载的前缀共享率。真实场景：
  system prompt、多轮对话历史、RAG 模板都是天然共享前缀 → 开启是免费的午餐；
  但若负载前缀全不同，收益为零（只多一点点索引开销）

原始数据：`e3_pc_on.json` / `e3_pc_off.json`

## 实验 2：ngram Speculative Decoding

### 2a. random 数据集（负结果——方法论教训）

- 配置：`--speculative-config {"method":"ngram","num_speculative_tokens":5,prompt_lookup_max:4,prompt_lookup_min:1}`
- 负载：random 数据集 c=8（512in/128out）

| | TTFT | TPOT | 备注 |
|---|---|---|---|
| 基线（无 spec, E1 c=8） | 292ms | 23.4ms | |
| **ngram spec** | 1261ms | **86.9ms (3.7×恶化)** | 接受率仅 ~1-8% |

**根因**：随机 token 序列没有任何可复现 n-gram → draft 几乎全被拒
（serve 日志 `Avg Draft acceptance rate` 实测 0-8%），draft 前向纯属浪费算力。
**教训：spec decode 的收益 = f(接受率) = f(负载可预测性)**；ngram 只对
有重复模式的文本有效（代码/模板化输出/多轮重复），random 数据必然负收益。

### 2b. sonnet 数据集（真实文本，待补对照）

## 实验 3：Chunked Prefill / max_num_batched_tokens（待跑）
