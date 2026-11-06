# 面试速查卡（环境A+B 全部实验的杀手数字，一页纸）

> 用法：面试前 10 分钟过一遍。每行 = 一个可抛出的数字 + 一句话机制。细节展开见 `interview-notes.md`（服务化）与 `interview-notes-b.md`（白盒）。

## 核心叙事主线

**prefill compute-bound / decode memory-bound 的教科书答案，我的三层实测修正：**
1. eager 下 decode 首先是 **launch-bound**（79% host 开销，device 每步只忙 24ms / 2935 次算子下发）→ 这正是 graph capture 存在的理由
2. decode batching 是**白嫖**（b32 每步 108.7ms ≈ b1 的 110.3ms → **32.5×吞吐**，固定 host 开销被摊薄）
3. "prefill 不能白嫖"要修正为"饱和点之后不能"（短 prompt 同样吃 10× 摊薄；8K ctx 2765 tok/s 才是 compute-bound）

## 数字弹药库（按主题）

| 主题 | 数字 | 一句话机制 |
|---|---|---|
| KV cache | 32 KB/token（hybrid）vs 128 全注意力 | 3:1 层布局，六长度点逐字节吻合 |
| hybrid state | 恒 49.5MB fp32 | GDN 递推数学必须 fp32（bf16 exp 溢出） |
| cache 干预 | 清 state 无影响 / 清 KV 塌缩 | GDN state 是递推器非存储器；全零 KV=均匀注意力 |
| 手写引擎 | L0/L2 = 4.4×→9.6× | cache 收益短上下文就兑现，随长度放大 |
| 引擎 vs 框架 | 快 8-10%（88-90 vs 96-98ms/步） | HF 封装/mask 构造的框架税 |
| continuous batching | 自研引擎 **4.99×**（5.3 vs 26.6s） | 调度器 6.1×（CPU 模拟）+ 真机验证 |
| TP 慢互联 | 裸 all-reduce **288µs**，16.3GB/s 饱和 | 4KB-256KB 与大小无关=barrier 税；TP=2 净负 −35%（27B 实测） |
| MTP | c=1 TPOT −39% / c=16 吞吐 **−15%** | acceptance 2.44；TPOT 改善≠吞吐改善 |
| W8A8 | per-channel int8 对称 + 动态激活，84.5% 覆盖 | GDN out_proj/lm_head 排除；1.73× 压缩 |
| KV int8 | −49.2% 显存 / 精度=打平步翻边 | 分叉时机由 top1=top2 gap=0 的步决定；32 步窗口不够 |
| KV offload | 等效重算带宽仅 **91 MB/s**，pinned=294× | 无条件 offload 划算；pinned vs pageable 25.5×；换出换回 bit-exact |
| PD 分离 | 32K 传 1GB=63ms≈0.6 decode 步 | DistServe 前提依赖 NVLink；慢互联层级传输握手税>传输 |
| 算子融合 | RMSNorm 链 100% host 开销 | compile 在 NPU **反慢 39-107%** → 原生 torchair/TileLang 路线 |
| GDN chunk | U 形：1k→64，4k→128（+12%） | launch∝L/chunk vs attn 矩阵∝chunk²·L 两项夹最优 |

## 独家故事（别人没有的）

1. **257=32×8+1 破案**：L3 引擎共享 seq_lens 被 8 个 attn 层各 +T——CPU 自测只有 1 个 attn 层所以 ×1 恰好对，NPU 4B 上炸出 257/256 广播错。教训：**共享状态 + per-layer 语义的 bug 会被"层数=1"的测试掩盖**
2. **left-padding 在 hybrid 上语义性失效**（实测）：GDN 无 attention-mask 概念，pad 进 conv/递推——vLLM 用 varlen cu_seqlens 的一手实证
3. **轴向与 arXiv 2609.04434 相反**（六条件干预）：KV 是唯一内容载体，state 管递推连续性
4. **bf16 batch 确定性翻边**：B=16 全槽第 10 步同步翻（GEMM tile 形状→归约顺序），同配置重跑 bit 一致——不是噪声是形状敏感
5. **三个调度器活锁**（dataclass 相等性 / 逐出 WAITING 无效 / 抢占震荡）——收录失败→等待后 6 轮干净收敛
6. DeepSeek 昇腾六件套 **5 个 950-only**（CANN 9.2+bisheng+UBMEM），tilelang-ascend 适配器是 910B 唯一活路（已在 36.151 验证 kernel 可跑）

## 服务化（环境A 精选，详见 interview-notes.md）

- prefix caching：p99 TTFT −55% 但 median +12%——**收益在尾部**；命中率 54%→84% 靠 prefix-aware 路由
- goodput：达标率与吞吐**分叉**（SLO 下有效吞吐≠裸吞吐）
- async_scheduling：吞吐 +60% TPOT −39%；MTP ngram 全负（接受率 8-11%）
- 多副本 1→4：瓶颈迁移到注入层（压测工具伪影曾虚报 2100 天花板，真实 4.2-5k）

## 手撕防身（examples/）

`mha_from_scratch.py`（0-diff）· `engine_np/`（hybrid 引擎全链）· `moe_router_sampler.py`（router+采样器，全部自测）· `scheduler.py`（continuous batching 调度器）
