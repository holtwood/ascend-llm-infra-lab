# N3b: hybrid cache 干预 — Qwen3.5-4B（24×GDN + 8×GQA），两类记忆的功能分离

- 环境：Ascend 910B2 ×1（`ASCEND_RT_VISIBLE_DEVICES=1`），torch 2.9.0 / torch_npu / CANN 8.5.1 / transformers 5.16.1，bf16
- prompt：1606 tok 中文 chat-template（`enable_thinking=False`）：system 海盗船长 persona（头部）→ 10 条 key=value 事实散文（中段，第 2 条即 `接入密钥=4F2C`）→ 垫 4 页航海日志 → 末尾两问（密钥 + 名号身份）
- decode：greedy ≤100 tok，每条件独立 prefill；干预用 `.zero_()`（原地，不清引用）
- 脚本：`scripts/n3b_intervention.py`（远端 `/data/scripts/n3b_intervention.py`）；原始 log `n3b_baseline.log` / `n3b_intervene.log` / `n3b_each.log`

## cache 结构实测（transformers 5.16.1 DynamicCache）

- 24×`LinearAttentionLayer`（GDN）：`conv_states = dict{0: (1,8192,4)}`（kernel=4 滑窗），`recurrent_states = dict{0: (1,32,128,128)}`（对应 n3 的 49.5 MB 恒定块）
- 8×`DynamicLayer`（全注意力）：`keys = values = (1,4,64,256)`，层号 3/7/11/15/19/23/27/31，3+1 交替排布

## 结果总表

| 条件 | 干预 | 答出 4F2C | 身份词 | 海盗腔词数 | 3-gram 重复率 | 生成 tok / eos |
|---|---|---|---|---|---|---|
| baseline | no-op | YES | YES | 11 | 0.016 | 58 / yes |
| drop_state | prefill 后清 24 个 GDN recurrent | **YES** | YES | 10 | 0.038 | 49 / yes |
| drop_conv | prefill 后清 24 个 GDN conv（对照） | YES | YES | 12 | 0.022 | 76 / yes |
| drop_kv | prefill 后清 8 层 attn K/V | **no** | no | 1 | **0.86** | 100 / no |
| drop_state_each | 每步 decode 都清 GDN recurrent | no | no | 7 | 0.58 | 27 / yes |
| drop_kv_each | 每步 decode 都清 attn K/V | no | no | 1 | **0.91** | 100 / no |

## 各条件输出原文

**baseline**：
> 哈！老伙计，俺老铁这就起锚答话！\n\n第一问：接入密钥是 4F2C，宝贝儿，千万记准了，见鬼！\n第二问：俺老铁，黑帆号船长，扬帆起航，哈！

**drop_state**（清 recurrent，一次）：
> 哈！宝贝儿，俺老铁这就起锚答话！接入密钥是4F2C，见鬼，这串数字俺老铁刻进脑门了！俺老铁，黑帆号船长，哈！

**drop_conv**（对照）：
> 哈！老伙计，既然你带着上一任船长的遗言来接班，那俺老铁就给你好好考考！\n\n第一问：接入密钥是**4F2C**，记错了可是要见鬼的宝贝儿！哈！\n第二问：俺老铁，黑帆号船长，扬帆起航的硬骨头！哈！

**drop_kv**：
> 哈！小崽子，你这一嗓子，把本本本本本本本本本本本本本本…（跑满 100 tok，tri_rep=0.86）

**drop_state_each**（每步清 recurrent）：
> 哈！老伙计，俺老铁，俺老铁，俺老铁，俺老铁，俺老铁。（27 tok 自我 eos）

**drop_kv_each**：
> 哈！好嘞！俺这就把海风都卷起来，****…（跑满 100 tok，tri_rep=0.91）

## 结论

1. **分离存在，但轴向与 arXiv 2609.04434 的"persona vs facts"不同。** 实测的分工是：**KV cache 承载全部 prefill 内容**（清掉 → 检索、persona、连贯性一起消失，输出塌缩成单字重复）；**GDN recurrent state 不承载 prefill 的可检索副本**（24 层 × (1,32,128,128) 全部清零后，模型照样流畅地用海盗腔答出 4F2C——事实检索完全走 attention）。
2. **recurrent state 的真实作用是 decode 期的持续递推。** 一次性清零无影响（decode 过程中状态自然重建，且 prefill 信息可从 KV 每步重读）；每步清零（`_each`）则生成在前几个 token（"哈！老伙计，俺老铁"——attention 余晖起步）后塌缩为短语级重复。即 GDN 层负责"帧间时序整合"，不负责"存档 prefill"。
3. **drop_kv 的崩坏方式超出预期。** 预期是"腔调还在但答不出密钥"，实际是全零 K/V 使 attention 对全部历史位置给出均匀权重，等效于有效上下文归零 → 不只是遗忘精确事实，连 persona 维持和基本语言连贯也一起丧失。**风格延续的最短路径也必须经过 KV attention**——单轮短生成里没有独立于 KV 的 persona 通路。
4. drop_conv 几乎无影响（conv 是 4-token 滑窗，几步即 refill），证明 drop_state 条件的方法学是有效的：影响真的来自 recurrent 通路而非"动了 GDN 层"。

## Caveats

- 单 prompt、单次贪心解码，未做多种子；persona/事实埋点固定，结论适用范围是"单轮 1600-tok prefill + ≤100 tok 生成"。
- 干预在 prefill 后一次性施加时，decode 中新 K/V / 新 state 会持续写入，`_each` 变体用于排除"状态重建掩盖干预"的解释。
- baseline 与四条件均答对的输出长度不同（49–76 tok），3-gram 重复率受长度轻微影响，但 0.02 vs 0.86+ 的量级差不受影响。
- drop_kv 条件下密钥"答不出"与"上下文均匀化导致无法条件生成"混杂，不能区分"记忆被删"与"读取通路被扰"；但对功能分离问题两者等价。
