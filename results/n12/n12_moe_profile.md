# N12: Qwen3.6-35B-A3B MoE 结构画像（36.151.162.7，本 fleet 唯一 MoE 真模型）

来源：/model/Qwen3.6-35B-A3B/config.json（model_type=qwen3_5_moe_text）。权重 67G bf16 / 38G w8a8。

## 结构速写

| 项 | 值 | 面试锚点 |
|---|---|---|
| 专家数 | **256**，每 token 激活 **top-8** | EP 分片的粒度；dispatch 字节 = 8×hidden×2B/token/层 |
| 共享专家 | 有（shared_expert_intermediate=512，恒激活） | DeepSeek 式 shared expert：每 token 的保底容量，缓解路由方差 |
| moe_intermediate | 512（小专家，hidden 2048） | 256 个小专家 vs 64 个大专家的设计取舍：负载更均衡、单专家 GEMM 更小 |
| 注意力 | **30 GDN + 10 full attn**（3:1，interval 4） | 与 Qwen3.5 同血统——N3/N3b 的全部结论直接迁移（KV 只在 10/40 层涨） |
| KV | 2 KV 头 × 256 dim | 16Q/2KV：极端 GQA，KV/token = 10层×2×2×256×2B = 20KB/token |
| MTP | 1 层 | 与 27B-w8a8 的 MTP 头同款（N6 的 acceptance 2.44 数据可参照） |
| ctx | 262,144 | hybrid+MoE 双省显存才撑得起 256K |

## 对应面试题（有这套结构就能讲的）

1. **EP vs TP 反转**：MoE 的 dispatch/combine 是 all-to-all（每 token 只去 8/256 个专家），与 TP 的每层 all-reduce 相反——N4 实测 TP 的 288µs barrier 税解释了为什么 dense TP 亏，而 EP 只要 all-to-all 可重叠就赚（DeepEP 的 low-latency mode）
2. **专家负载不均**：256×top-8 的路由方差 → EPLB/冗余专家；N5 手撕题库的 imbalance 系数（1.05→2.68 演示）是度量工具
3. **w8a8 版 38G 单卡可跑**：61G 卡放 38G 权重 + 20KB/token KV → 约 23 万 token 的 KV 预算（hybrid 帮忙）

（加载/generate 级：由 moe-profile agent 补充或下次会话完成。）
