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

### 2b. sonnet 数据集（真实文本对照）— spec 依然负收益

- 负载：sonnet.txt（莎士比亚十四行诗，真实自然语言），64 请求 c=8，512in/128out
- 目的：排除"random 数据对 ngram 不公平"的干扰——真实文本有语言规律性

| | TTFT mean | TPOT mean | 输出吞吐 tok/s | 接受率 |
|---|---|---|---|---|
| nospec sonnet c8 | 422.5ms | **14.65ms** | **448.2** | — |
| ngram spec sonnet c8 | 287.6ms | 88.46ms (6×恶化) | 87.9 (−80%) | **仅 8.2-11.1%**，接受长度 1.41-1.55 |

原始数据：`e3_spec_sonnet_c8.json` / `e3_nospec_sonnet_c8.json`；
接受率抄自 serve 日志：`serve_spec_ngram_c8.log`（Avg Draft acceptance rate / Mean acceptance length）

### 结论：ngram spec decode 在 vLLM-Ascend + Qwen3.5-4B 上全线负收益

1. **接受率是天花板**：真实文本也只有 ~10% 接受率（期望 ≥40% 才划算），
   接受长度 ~1.5 意味着 draft 5 个 token 平均只中不到半个
2. **draft 开销实打实**：每个 decode step 多做一次 5-token 前向 + 验证，
   拒绝的 draft 全部浪费 → TPOT 恶化 3.7-6 倍、吞吐掉 80%
3. **为什么自然语言接受率也这么低**：ngram 的 lookup 只匹配
   **当前请求自己已生成的上下文**（prompt_lookup），sonnet 每句都是新文本、
   无重复 n-gram；它只对"复读机"式输出（代码补全重复模式、模板化响应、
   多轮对话复读）有效。eagle/medusa 类用 draft 模型预测才是通用路线
4. **面试话术**：spec decode 不是免费午餐——
   `收益 = 接受率 × 每接受省一步 − draft+verify 开销`。
   上线前必须拿真实流量测接受率，低于 ~30-40% 的 ngram 配置直接关掉；
   且 `prompt_lookup_max` 越大 draft 越贵，本例 5-token draft 的固定开销
   已经吃掉了全部潜在收益

## 实验 3：Chunked Prefill / `max_num_batched_tokens` 扫描（E3b）

- 模型：qwen35-4b，单实例（M3:8000），`--max-model-len 8192`，其余默认
  （`max_num_seqs=128`，vLLM 0.30 默认 `max_num_batched_tokens=2048`、
  chunked prefill 常开）
- 每档重启服务后跑 `vllm bench serve`（random 数据集，`--ignore-eos`）
- 合法性边界：`mnb < max_model_len` 仅在 chunked prefill 开启时允许；
  `mnb` 必须 `≥ max_num_seqs`（128）→ 扫描档位 256~8192

### 主臂：长 prefill 负载（in=2048 / out=128 / c=32 / np=128）

| mnb | TTFT mean | TTFT p99 | TPOT mean | TPOT p99 | 输出吞吐 tok/s |
|---|---|---|---|---|---|
| 256 | 13093.6 | 24090.1 | 87.46 | 94.17 | 162.2 |
| 512 | 4686.2 | 21405.6 | **141.76** | 165.97 | 177.8 |
| 1024 | 3485.6 | 17433.3 | 116.07 | 139.28 | 220.6 |
| 2048 | 1499.6 | 6871.5 | 67.37 | 86.44 | 394.4 |
| 4096 | 1062.9 | 4249.1 | 57.59 | 77.57 | **470.1** |
| 8192 | 1097.2 | 4156.7 | 61.67 | 77.98 | 435.0 |

### 对照臂：短 prefill / decode 重负载（in=256 / out=512 / c=32 / np=128）

| mnb | TTFT mean | TTFT p99 | TPOT mean | TPOT p99 | 输出吞吐 tok/s |
|---|---|---|---|---|---|
| 256 | 655.0 | 3314.7 | **25.73** | 28.78 | **1108.1** |
| 8192 | 407.2 | 932.2 | 32.57 | 36.03 | 881.5 |

### 解读

1. **TTFT ∝ prefill 每步能吃的 token 数（长 prefill 臂）**
   in=2048 时 mnb=256 需要 ~8 个 chunk 才能完成一条 prefill，32 条并发排队
   → mean TTFT 13.1s。mnb 翻倍 → 排队深度减半，单调降到 mnb=4096 的 1.06s
   （约 12×）。mnb≥4096 后单条 prefill 已能一步装下（2048+decode 裕量），
   曲线进入平台期（1063→1097ms，±3% 噪声）——**mnb 调到 ~2×典型 prompt
   长度就吃饱了，再大无收益**。

2. **TPOT 在长 prefill 臂呈非单调（512 档最差 141.8ms）**
   chunked prefill 把 decode 和 prefill chunk 塞进同一个 token 预算：
   每步 = 32 条 decode + (mnb−32) 的 prefill chunk。步时延 ≈ prefill chunk
   的前向时间 → decode 的 ITL 被 chunk 大小直接绑架。
   - mnb=256：chunk 小（~224 tok）→ 步子轻，ITL 反而尚可（87ms）
   - mnb=512/1024：chunk 变大但仍需多步才完成一条 prefill →
     持续"重步"挤压 decode，TPOT 最差
   - mnb≥2048：单步装下整条 prefill，之后是纯 decode 步 → TPOT 恢复

3. **对照臂方向反转 = 教科书 tradeoff 的直接证据**
   decode 重负载下小 mnb 反而是优势配置：prefill 便宜（256 tok），
   小预算保护 decode 步不被大 chunk 绑架 → TPOT 25.7ms、吞吐 1108 tok/s；
   mnb=8192 让 32 条 prefill 挤进同一步形成巨型混合步 →
   TPOT +27%、吞吐 −20%，只换来 TTFT −38%。
   **结论：mnb 的最优点随负载 prefill/decode 配比移动，没有全局最优值。**

4. **面试话术**
   - chunked prefill 的本质：把"decode 优先 / prefill 独占"的离散调度，
     变成可配速的 token 预算旋钮；`max_num_batched_tokens` 就是那个旋钮
   - 调参公式：`mnb ≈ 2 × p50_prompt_len` 吃满 TTFT 收益；
     decode 敏感型负载（agent 长生成、streaming）反而要压低 mnb 保 TPOT
   - 观测 TPOT 尖峰是诊断"chunk 粒度与 prompt 长度失配"的信号

原始数据：`e3b_mnb{256,512,1024,2048,4096,8192}.json`、
`e3b_ctl{256,8192}.json`；驱动日志 `e3b_driver.log`；
服务日志 `serve_e3b_mnb256.log`（其余档位同目录 serve_e3b_mnb*.log）
