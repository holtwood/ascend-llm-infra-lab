# War Stories — 面试口述素材（每个都有实测数据背书）

按" STAR "结构整理。每条标了仓库证据位置，面试前扫一遍即可。

## 1. 容量排查：以为是调度慢，其实是 KV 墙（E8）

**情境**：长 prompt 高并发下延迟飙升。
**行动**：先算理论 KV（32 KiB/tok，hybrid 8 层全注意力），再实测
45.6 GiB/1.006M tok 验证页对齐损耗（mamba 页 +22% 填充），最后两组
压测分离**准入墙**（默认池 0 抢占、running 稳态 142）vs **容量墙**
（0.5 池 KV 99.1%、3 次抢占、waiting 吸收溢出）。
**结果**：拿到两类受限模式的完整判据——"并发上不去先看 running 稳态
是不是卡在准入速率，再看 KV 占用是否接近 100%"。
**证据**：`results/e8/`

## 2. 吞吐涨了但用户体验变差：goodput 才是北极星（E9）

**情境**：w8a8-27B 吞吐 396 tok/s 看着不错。
**行动**：从 bench JSON 的 per-request ttft/itl 字段重算 SLO 达标率。
**结果**：**93% 请求违反 SLO**——原始吞吐完全掩盖了尾部崩坏。
反过来 mnb 256→4096 让 goodput 从 0.02→3.06 rps（150×），
裸吞吐只反映 ~3×。
**证据**：`results/e9/` + `docs/slo-cost-model.md`

## 3. PD 分离踩坑实录：分层排障方法论（E7+E18）

**情境**：同卡 PD 一半请求超时，想跨节点解决。
**行动**：E7 定位到 `HixlCSClientConnect` 建链失败（同卡双 adxl 受限）；
E18 跨节点再试——ZMQ 元数据广播 pod IP（`VLLM_HOST_IP`+隧道修通）、
TE-RPC 动态口（再补隧道）、最终 device-mem 传输物理过不去。
**结果**：拿到三层失败画像 + **最危险发现：传输失败仍返回 200 垃圾
token**（静默劣化）。结论：PD 监控要盯输出质量不是状态码。
**证据**：`results/e7/` `results/e18/`

## 4. 接受率 89% 的 spec decode 是负收益（E14+E15）

**情境**：MTP 接受率实测 65-89%，看着能上。
**行动**：扫 k∈{1,2,3} + 对照 ngram（11%）。
**结果**：全部 TPOT 负收益（27.6-36.5 vs 15.6ms）——910B2 上 1 层
draft 头的固定前向成本是常量项，k 越大亏越多。**接受率从 11% 提到
89% 瓶颈不变**——证明瓶颈在草稿成本不在接受率。
**证据**：`results/e14/` `results/e15/`

## 5. 换租 150s→0.14s：sleep/wake 多租户（E17）

**情境**：多租户共享单卡，重启模型要 ~3 分钟。
**行动**：CaMem sleep level2（1.77s，HBM 4480/65536MB）→ 租户 B 在
腾出空间起服务 → B 在线时唤醒 A（**0.139s**，首请求 0.129s 无 ramp）。
**结果**：换租成本降 1200 倍。边界：睡眠方仍占 stream/CQ，共卡引擎
只能 eager（EE1023 同根因——和 issue2 对上）。
**证据**：`results/e17/`

## 6. 调度器动物园实测：4 选 1（E19）

**情境**：vllm-ascend 有 5+ 个调度器选项。
**行动**：逐一 A/B。
**结果**：recompute/dyntra_lb 需 PD kv_consumer、profiling_chunk 需
PP>1——**都是 PD/流水配套件**；单机只有 SRF 有效（short −7.2%）。
**证据**：`results/e19/`

## 7. 网关层工程：least_conn→prefix 亲和→准入控制（D3+E20）

**情境**：4 副本挂 nginx，prefix cache 命中稀烂。
**行动**：stdlib 写 LPR 路由（前缀 sha256 亲和+负载降级），命中率
54%→84%；再加 SRF 分道 + per-class 准入 cap 凑成 mini 控制面。
**结果**：被接纳 short p90 −36%；过载请求 <1ms 收 429 而不是排队
6 秒。"排队把成本均摊，拒绝把成本集中"的取舍实测。
**证据**：`results/d3/` `results/e20/` `scripts/e20_control_plane.py`

## 8. 开源贡献纪律：PR 前先查 mainline 历史（D1 否决）

**情境**：发现 `causal_conv1d_update_npu` fallback 每请求 host 同步。
**行动**：查上游 git log——符号被 #5984 实现→#8842 换 AscendC→
**#14620 主动删除**；同期第三方已提 #18978 同缺陷 issue。
**结果**：不重复造轮子也不污染上游——issue3 降为 comment，
issue1/2（w8a16、207005）保持独特可提。
**证据**：`docs/issues/`

## 9. MindIE 商用栈 bring-up（E5R 协作）

**情境**：商用引擎 MindIE 环境全绑死（glibc/CANN/py 版本）。
**行动**：9 步修复链到 warmup，最终定位 ATB `set_weight` hang——
device ack 永不到达（1Hz 32B 轮询），vendor 级故障。
**结果**：闭源栈的平台锁定 vs vLLM 源码可移植性——引擎选型隐性成本。
**证据**：`results/e5r/`

## 10. 手撕系列：机制级证明（examples + E8b-e）

**情境**：要证明懂底层不是会调 API。
**行动**：手写 MHA+KV cache（naive/cached logits 0 diff、贪心逐 token
一致）、RoPE/GQA/SWA 变体、attention 成本公式。
**结果**：KV/token 公式能默写，paging/填充损耗能实测，
SWA 负收益能讲清 kernel 级原因（mask 永远 O(S²)）。
**证据**：`examples/mha_from_scratch.py` `results/e8b-e8e`
