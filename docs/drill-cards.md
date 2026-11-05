# 追问链卡片 — 面试存活手册

每张卡 = 一个故事。结构固定：
**30 秒回答**（开场）→ **关键数据**（能被记住的数字）→
**三级追问**（L1 为什么 / L2 怎么证明 / L3 换个条件呢）→
**诚实缺口**（没测过的怎么答）。

> 用法：面试前过一遍 L1-L3，确保每层都有数字或机制；缺口部分准备
> "没测过，但我会这样验证"的答法，比硬编强。

---

## 卡 1：集群扩容与瓶颈定位（E4/E4b/E21/E22/E23）

**30 秒**：4 副本单卡集群，扩容到 N=4 吞吐不涨——我一开始判断是注入端，
后来交叉验证发现是压测工具自己的天花板，换工具后 4 副本真实上限
4.2-5.0k tok/s。

**关键数据**：单副本 ~1.1-1.5k tok/s；4 副本聚合 4.2-5.0k；
两工具同负载差 1.66×（vllm-bench 2193 vs bench_openai 3632）；
跨节点同配置差 11%。

**L1 为什么 N=4 不涨？**
→ 先用 vllm-bench 测出 N=3→4 平台 2100；换 bench_openai 单客户端
3632、双客户端 4371 → 平台是工具造成的。

**L2 怎么证明不是副本饱和？**
→ 三件事：客户端 CPU <1 核（ps 采样）；同集群两工具差 1.66×；
聚合 c256 单客户端最高 5002（多进程反而降）→ 副本在 c256 已饱和、
c128 时是工具限制。

**L3 换条件呢？**
→ c128 加第二个客户端 +20%（3632→4371），加第 4 个不涨（4365）——
副本侧在 c128 也见顶；c256 多进程降 15%（连接竞争+客户端与副本抢 CPU）。
→ 无 RDMA 的集群做不了 TP/PP（E18 实测），扩容只能副本级。

**L3 补充：四机并行做矩阵实验**
→ 调度器矩阵同刻四机（E26）：default 800.6 / SRF 665 / async-off 500 /
MTP 194 tok/s——差值 >11% 的才采信；KV 池扫描（E27）跨节点看似"池越小
越快"，**同节点对照方向反转** → 并行出假设、同机做验证。
长上下文并行扫描（E28）与 E16 串行结论互证：4k 178→63k 41 tok/s。

**诚实缺口**：没测网络带宽对注入的影响（节点间只有 SSH 隧道，带宽未量化）；
没测真实业务负载（都是合成 sonnet/random）。

---

## 卡 2：KV 容量与两类受限（E8/E9/E16）

**30 秒**：Qwen3.5-4B 是 hybrid 模型，32 层只有 8 层 full-attention 持 KV；
实测 48.7KB/token 比理论 32KB 高 48%，差额是 mamba 页对齐填充；
压测分离出"准入墙"和"容量墙"两种受限模式。

**关键数据**：KV 池 45.6GiB/1.006M tok；util0.92 c384 KV 峰值 73% 零抢占
（准入受限）；util0.5 c256 KV 99.1%、3 次抢占（容量受限）；64k 上下文
单请求 ~3.1GB。

**L1 为什么理论值和实测差 48%？**
→ vLLM 统一分页：mamba 页 +22.14% 填充到与 attention 640-token 页等大，
mamba 状态摊销进每 block；`kv_cache_size_tokens` 还随 max_model_len 变化
（8192→1.006M，81920→1.427M）。

**L2 怎么区分准入墙和容量墙？**
→ 看三个指标：running 稳态是否远低于 KV 容量、KV 占用是否接近 100%、
preemptions 是否非零。默认池 c384：running 142、KV 73%、0 抢占 = 准入；
缩池 c256：KV 99.1%、3 抢占 = 容量。

**L3 长上下文呢？**
→ 4k→64k：TTFT 1.39→12.2s 近线性，TPOT 19→62ms（prefill/decode 争抢），
KV 峰值 1.9→13.9%（c≤4）。长上下文瓶颈从调度准入切回 KV 字节。
→ 并行复现（E28，四机同刻）：4k 178→63k 41 tok/s，TTFT 527ms→4.59s，
TPOT 仅 14.4→20.5ms——**长上下文成本在 prefill 不在 decode**。

**L3 补充：KV 池是容量旋钮不是性能旋钮**
→ 四机扫 util 0.5-0.92 看似"池小更快"，同节点对照反转（u92 +7%）——
跨节点趋势是节点差异假象（E27）；池大小按峰值并发×序列长度定。

**诚实缺口**：没测 prefix cache 与 KV 池的交互（命中时池占用如何变化）；
没做 128k+ 上下文（模型支持 262k 但 KV 池不够）。

---

## 卡 3：量化精度代价（E2/E6）

**30 秒**：W8A8-27B 的精度代价在测量噪声内：QA 生成式 94.3%（vs bf16-4B 95.2%），
PPL 全量 27 chunk 配对 Δnll +0.023（9/27 chunk 反向），top-1 一致性 93.3%。

**关键数据**：w8a8 nll 1.6086/ppl 4.996 vs bf16 1.5873/4.891；2382 token；
QA 94.3%/96.2%（生成式/MCQ 打分）；top1 一致 93.3%。

**L1 为什么说"噪声内"？**
→ Δnll 跨度 −0.045..+0.123、9/27 chunk 反向——方向不一致说明差异被
"实现路径差异"（vLLM-NPU 量化 kernel vs transformers-CPU bf16）淹没。

**L2 怎么测的？**
→ 同一语料 27 chunk，serve 侧 prompt_logprobs 与 CPU 侧 logits 同方法学；
MCQ 打分按字母 token 自身 logprob 查（踩过 max-logprob 的坑）。

**L3 换模型/换量化呢？**
→ W8A16 在上游直接不支持（我提了 issue #21112）；W4A16 有 per-group
group_size 问题（#20973）。量化选型先查引擎支持矩阵。

**诚实缺口**：语料只有 2.4k token（置信区间宽）；没测下游任务级
（如代码生成 pass@1）；没测量化对长上下文的影响。

---

## 卡 4：spec decode 为什么亏（E3/E14/E15/E24）

**30 秒**：MTP 接受率 89%，但 c8-c64 全负：吞吐 0.45-0.55×、TPOT 2.1-2.2×。
原因：草稿头是完整 1 层 transformer，成本随 batch 同比例增长，不存在摊薄拐点。

**关键数据**：无 spec c64 1533 tok/s/TPOT 33ms vs MTP 846/58ms；
接受率 88.3-88.7%；ngram 方案接受率仅 8-11%。

**L1 为什么接受率高还亏？**
→ 收益 = 接受率 × 节省步数 − 草稿成本。k=1 时分子接近 0、分母固定；
k=3 时分子大但草稿 3 次串行前向。

**L2 怎么排除配置问题？**
→ k∈{1,2,3} × c∈{8,32,64} 全扫，单调负收益；接受率全程 88%。

**L3 什么条件能赚？**
→ 草稿成本 < target 单步成本 × 1/(k+1)；需要更小 draft（浅 MLP）、
独立 KV 组（当前警告 no KV cache group）、或共享权重的 early-exit。

**诚实缺口**：没试过真正的 EAGLE/Medusa 类小 draft 模型（词表不匹配，
Qwen3-0.6B vocab 151k vs 4B 248k）；没测投机解码对 TTFT 的影响。

---

## 卡 5：多租户换租（E10/E17）

**30 秒**：sleep level2 释放 97% 显存（31.2GB→1.1GB，1.77s），唤醒 0.139s、
首请求 0.129s 无 ramp——换租比重启快 1200×。边界：睡眠方仍占 stream/CQ，
共卡引擎只能 eager。

**关键数据**：冷启 169.4s；sleep2 1.77s；wake 0.139s；HBM 4480/65536MB；
B 在 A 睡出的空间冷启 114.2s 并共存。

**L1 为什么 wake 这么快？**
→ CaMem 按 tag（weights/kv_cache）搬显存到 CPU，allocator 元数据和 ACL graph
保留；wake = 回灌权重 + 恢复指针，不重新编译/capture。

**L2 怎么证明真的释放了？**
→ npu-smi Process memory 表：31994→1004MB；HBM 4480/65536。

**L3 多租户生产怎么用？**
→ RL rollout：训练与推理共享卡，rollout 时 wake、训练时 sleep；
或弹性伸缩：低峰 sleep 腾卡给批处理。坑：需 VLLM_SERVER_DEV_MODE=1、
rl_config.enabled（FRACTAL_NZ 回灌）。

**L3 唤醒后性能有损失吗？**
→ 无：warm 696.6 → post-wake 694.9 tok/s（TPOT 18.9→19.3ms），差异 −0.24%
在噪声内（E25）；成本只有 sleep 的 1.74s PCIe 拷贝。

**诚实缺口**：没测频繁 sleep/wake 循环的碎片化；没测 level1 与 level2
的切换成本差（只测了 level2）。

---

## 卡 6：网关层（D3/E20/E23）

**30 秒**：least_conn 对 LLM 不是最优——prefix 亲和路由把命中率 54%→84%，
但 p90 被排队稀释；再加 SRF 分道 + per-class 准入 cap 组成 mini 控制面，
被接纳请求 short p90 降 36%，过载请求 <1ms 收 429。

**关键数据**：命中率 53.8%→84.0%（6k 前缀）；准入后 short p90 4.85→3.10s；
429 响应 <1ms；cap=并发时边界竞态会产生大量 429（soak 踩到）。

**L1 为什么命中率没到 95%？**
→ 并发到达时"第二波才命中"——同模板请求在缓存写入前就发到同一后端；
上界 = (n−1)/n × 前缀占比。

**L2 准入 cap 怎么定？**
→ 按 E9 的 goodput 拐点：cap 对应 p99 满足 SLO 的并发；cap 必须大于
预期并发（等于会产生边界竞态 429）。

**L3 和引擎内调度器什么关系？**
→ 网关管"进不进"（准入/路由），引擎管"怎么排"（SRF/batch_job）；
E19 证明引擎侧 4 个调度器 3 个是 PD/PP 配套件，单机只有 SRF 有效。

**诚实缺口**：没做多租户权重/配额；没做动态 cap（按 p99 反馈调节）；
没测网关自身成为瓶颈的并发点（Go/Rust 实现才有意义）。

---

## 卡 7：PD 分离（E7/E18）

**30 秒**：同卡 PD 一半请求 10s 建链超时（HixlCSClientConnect）；跨节点
三层失败：ZMQ 元数据广播 pod IP（VLLM_HOST_IP+隧道修通）、TE-RPC 动态口
（再补隧道）、device-mem 传输物理不可行。最危险：失败请求仍返回 200 垃圾 token。

**关键数据**：同卡 P/D 各 0.35 util；跨节点元数据 172.17.0.5；TE RPC 16401/16352；
垃圾输出 `"!!!!!!!!!!!!!!!!"`。

**L1 为什么同卡失败？**
→ 同卡双 adxl 引擎资源受限，HixlCSClientConnect 10s 超时；成功请求
TPOT 95ms < mono 108ms（decode 隔离收益真实存在）。

**L2 跨节点为什么不行？**
→ 官方前置是 NPU RDMA 互联；本环境只有 SSH 端口，device-mem 传输
托不起 TCP 隧道（`Mooncake transfer failed ret:-1`）。

**L3 生产怎么部署 PD？**
→ 同 LAN + RDMA；P/D 配比按 prefill:decode 时间比；监控必须看输出质量
（传输失败仍 200）。

**诚实缺口**：没在真正有 RDMA 的环境验证过；没测 P/D 配比优化；
没测 KV 传输带宽对 TTFT 的影响（因为链路根本不通）。

---

## 卡 8：Ascend 特性画像（E19 + 探针）

**30 秒**：vllm-ascend 的调度器/特性多有部署前置：recompute/dyntra_lb 要
PD kv_consumer、profiling_chunk 要 PP>1、sparse_kv_offload 要 SFA 模型——
单机只有 short_request_first 有效（short −7.2%）。

**关键数据**：SRF short 28.73→26.66s（−7.2%），long +3.3%；3 个调度器
拒绝启动的报错原文；async scheduling 默认开（+60% 吞吐/−39% TPOT，E11）。

**L1 为什么这些调度器绑 PD？**
→ recompute 保 decode 步形状（配 finegrained_tp）、dyntra_lb 按 KV-block
均衡 D 节点气泡——它们是 PD 架构配套件。

**L2 SRF 怎么工作？**
→ 三通道等待队列（immediate/short/long），按 num_prompt_tokens ≤ threshold
分道 + long_max_wait_ms 老化防饿死；SRPT 的工程近似。

**L3 为什么 Ascend 默认开 async？**
→ E11 实测 async 吞吐 +60%、TPOT −39%；调度与模型步流水线重叠。
sync 模式 TTFT 反优（608 vs 817ms）——重叠会轻微干扰 prefill。

**诚实缺口**：没测 PD 场景下的 recompute/dyntra_lb（无 RDMA 起不来）；
没测 finegrained_tp；没读全部调度器源码（只读了 SRF 分道逻辑）。

---

## 卡 9：手撕机制（examples + E8b-e）

**30 秒**：手写 MHA+KV cache，naive 与 cached logits 0 diff、贪心逐 token
一致；RoPE 手写 vs 参考 diff 0（bf16 0.038）；GQA expand==repeat_interleave
位级一致；SWA 用 mask 比 is_causal 慢 26×——真 SWA 靠 kernel 跳过窗外 block。

**关键数据**：0 diff / 16 token 贪心一致 / KV 268→33.6MB（GQA 8×）/
SWA S8192 24.2ms vs 0.93ms。

**L1 KV cache 怎么算？**
→ 2 × layers × kv_heads × head_dim × bytes/token；4B hybrid 只有 8 层
full-attention → 32KiB/token；naive 32 层公式错 4 倍。

**L2 GQA 为什么省显存？**
→ KV heads 4 vs Q heads 16：KV 字节 1/4；expand 与 repeat_interleave
等价但前者零拷贝（位级一致验证）。

**L3 SWA 的坑？**
→ mask 方案永远 O(S²)；真 SWA 是 kernel 级跳过（FIA sparse_mode=4）。
负收益实验的价值：证明"算法对了实现不对也没用"。

**诚实缺口**：没写 AscendC/Triton 生产 kernel；没测 flash-attention 变体
在 NPU 的实际吞吐；没做 paged attention 的手写实现（只用了 vLLM 的）。

---

## 卡 10：开源贡献纪律（D1/D2）

**30 秒**：提了 2 个 issue（#21112 W8A16、#21113 cudagraph SQ/CQ 耗尽）
+ 1 个 comment（#18978 conv1d）；一个原计划的 PR 主动否决——因为查
mainline 历史发现目标符号被上游自己删了。

**关键数据**：#21112/#21113 均 open 无争议；#18978 是第三方先报的同缺陷；
上游 #5984→#8842→#14620 的符号生命周期。

**L1 为什么否决那个 PR？**
→ `causal_conv1d_update_npu` 被 #14620 主动删除，重写=重加被移除代码；
正确修法是升级版本/等 AscendC 路径。

**L2 提交前怎么防垃圾？**
→ 三步：search API 查重、git log --all --grep 查 mainline 历史、
最新 release 查是否已修；三步过才提。

**L3 issue 质量怎么保证？**
→ 环境/复现/期望/实际/根因/建议修复/临时方案 七段式；附源码行号并
逐行核对；模型路径实测存在。

**诚实缺口**：没有 merged PR（只有 issue）；没参与社区讨论/评审；
没做过跨版本回归测试。

---

## 总缺口清单（诚实作答模板）

| 缺口 | 面试怎么答 |
|---|---|
| 无 merged PR | "issue 也是贡献；我按 merged 标准写复现和根因，两个都被维护者接受无争议" |
| 无 RDMA 环境 | "PD/TP 的结论都是'此环境不可行'的边界数据，不是能力上限；换有 RDMA 的环境我会先验证 hccn" |
| 真实业务负载缺失 | "合成负载用于控制变量；真实负载要补 trace 回放（sharegpt 已有但没做长度分布对齐）" |
| 没写生产 kernel | "手撕是为了机制正确性（0 diff 验证），生产 kernel 需要 AscendC 工具链，我有 issue 级的源码阅读经验" |
| 成本模型是假设单价 | "¥25/h 是假设，方法（goodput×单价）可复用，换真实报价即可" |
| 没测 128k+ 上下文 | "模型支持 262k，KV 池限制在 64k 附近；要测得先扩池或做 KV offload（sparse_kv_offload 仅 SFA 模型）" |
