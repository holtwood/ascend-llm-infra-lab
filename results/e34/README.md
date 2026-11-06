# E34 — Prefill→Decode 干扰（prefill stall）：块大小消融推翻半个教科书结论

> 面试题：**"一个长 prompt 进来为什么会卡 decode？chunked prefill 怎么调？"**
> 答：4 条持续 decode 流中注入一条 4k prefill，逐 token 记录 ITL——
> 基线 14.4ms，注入后 p99 **187ms（13×）**。但块大小消融给出反直觉结果：
> **小块 budget 没有降低尾部**——它是"浅而长"的干扰，大块是"一次深坑"，
> 且大块还让稳态 ITL 整体抬升 56%。

## 1. 方法

- 4 条流式 decode（max_tokens=300），SSE 逐 token 时间戳 → ITL 序列
- T=4s 注入一条 4096-token prompt（max_tokens=8）
- 指标：基线 ITL 中位、注入后 3s 窗内 p50/p99/max、超阈值（>2×基线）token 数、
  注入请求自身 TTFT
- 消融：`--max-num-batched-tokens` = 默认(2048) / 512 / 8192，同节点 M3 背靠背重启

## 2. 数据

| budget | 注入 TTFT | 基线 ITL med | post p50 | post p99 | post max | 超阈值 token 数 |
|---|---|---|---|---|---|---|
| 2048（默认） | 0.535s | 14.4ms | 14.9 | **186.8** | 191 | 60 |
| 512 | 1.395s | 14.1ms | 14.1 | 173.5 | 173 | 44 |
| 8192 | **0.366s** | **22.5ms** | 22.5 | **109.6** | 185 | **8** |

## 3. 机制解释

1. **stall 的本质**：prefill 与 decode 竞争同一个 step 的 token 预算。
   4k prefill 在一步中吃掉预算 → 并发 decode 的 ITL 出现尖峰。
2. **为什么 512 反而 p99 不差**：小 budget 把 4k 拆成 ~8 步，每步都在
   混跑（prefill 块 + decode）→ 干扰被摊到更多 token 上（44 个超阈值），
   形成"浅而长"的抬升；8192 一步装下全部 prefill → 只有 8 个 token 遭殃，
   其余立刻恢复。
3. **8192 的隐藏代价**：基线 ITL 22.5ms（比 14.4 高 56%）——大预算让调度器
   每个 step 能装更多并发工作，稳态每步更重。所以 p99 低的部分原因是
   分母变了。
4. **默认值最差**：2048 两头不占（尖峰 187ms + 60 个超阈值 token）。

## 4. 工程含义

- 教科书"chunk 越小 stall 越小"只对 **max 尖峰**成立；对 **p99/干扰时长**
  不成立。选 budget 要先定 SLO 关心什么：偶发深坑（重用户体验的请求级
  毛刺）还是持续抖动。
- 大 budget 换 TTFT（0.366s vs 1.395s）但牺牲稳态 ITL——在 PD 分离里
  这正是把 prefill 挪走的动机之一（E18 的失败从需求侧又获一条佐证）。
- 面试标准答案升级：**"chunked prefill 不是单调降 stall，它在
  尖峰深度 × 干扰时长 × 稳态 ITL × 注入 TTFT 四个维度上做权衡"**。

## 5. 局限

- 单请求注入、单节点、c4 流规模；未测 prefill 洪峰（多 prompt 并发）
- post 窗口固定 3s；不同 budget 下"窗口内尖峰占比"受 prefill 总时长影响
- 8192 基线抬升的具体构成（更大混合 batch vs 调度行为）未进一步拆解

## 6. 复现

```bash
python3 scripts/e34_prefill_stall.py --base-url http://127.0.0.1:8000 \
  --model qwen35-4b --streams 4 --out-tokens 300 --inject-at 4 \
  --prefill-tokens 4096 --out /tmp/e34.json
```
