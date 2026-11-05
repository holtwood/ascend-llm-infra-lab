# E19 — Ascend 调度器画像：4 个选项只有 1 个单机适用

> 面试题：**"vLLM-Ascend 有哪些调度策略？什么时候用什么？"**
> 数据回答：实测 4 个 scheduler_config 选项，**3 个是 PD 分离/流水专用**——
> `recompute_scheduler`/`dyntra_lb` 只认 `kv_role=kv_consumer`（PD 的 D 节点），
> `profiling_chunk` 要 PP>1。单机只有 `short_request_first` 可用且有效。

## A/B 数据（同负载：48 long(2800词) + 48 short(120词)，c96，out=256）

| 调度器 | 单机可用 | long e2e | short e2e | 结果 |
|---|---|---|---|---|
| 默认 AsyncScheduler | ✅ | 27.14s | 28.73s | 基线——**short 反而比 long 慢 1.6s**（排队倒转） |
| `short_request_first` | ✅ | 28.02s (+3.3%) | **26.66s (−7.2%)** | 倒转修复：short 现在快于 long 1.4s |
| `recompute_scheduler` | ❌ | — | — | 拒绝启动：需 `kv_role=kv_consumer`（PD D 节点） |
| `dyntra_lb` | ❌ | — | — | 拒绝启动：同上，且非 PD-mixed |
| `profiling_chunk` | ❌ | — | — | 校验拒绝：需 `pipeline_parallel_size > 1` |

SRF 激活日志：`ShortRequestFirstAsyncScheduler ... threshold=256,
long_max_wait_ms=0.000`；按 `num_prompt_tokens <= threshold` 分道。

## 机制解释

1. **SRF 为什么有效**：默认 FCFS 下 96 请求同时到达，48 个长 prompt 先进队
   吃掉 token 预算，短请求干等。SRF 三通道等待队列（immediate/short/long）
   + 简单老化（`long_max_wait_ms` 防饿死）把短请求提前——SRPT 思想的工程版。
   代价：long +3.3%（可接受，因为有老化兜底不会无限饿死）。
2. **recompute/dyntra_lb 为什么绑死 D 节点**：PD 分离下 D 节点只跑 decode，
   需要 recompute 保证"每步都是 decode 形状"（配合 finegrained_tp），
   需要 dyntra_lb 按 KV-block 负载单元在 D 间均衡气泡——**它们是 PD 架构
   的配套件，不是通用优化**。这也是 E13 笔记里 recompute 注释
   "keeps decode-node steps decode-shaped" 的出处。
3. **profiling_chunk 的 PP 绑定**：动态调 chunk 靠各 stage 的 profiling
   结果，单机没有 stage 边界，无从调起。

## 面试口径

- "Ascend 的调度器动物园大多是为 PD 分离部署服务的——单机该问的
  不是'用哪个调度器'，而是 SRF 够不够"（有拒绝日志+实测数据）
- "SRF 按 prompt 长度分道，不是按 output——输出预算未知时这是最
  现实的近似 SRPT"（读了源码，`num_prompt_tokens <= threshold`）
- "开调度器选项前先查部署前置：kv_role / PP>1 / PD-mixed 互斥，
  盲开会直接拒绝启动"（有 3 条 ValueError）

## 复现

```bash
# SRF
vllm serve /model/Qwen3.5-4B --served-model-name qwen35-4b --port 8000 \
  --max-model-len 8192 \
  --additional-config '{"scheduler_config":{"short_request_first_config":{"enabled":true,"threshold":256}}}'
python3 scripts/e13_mixedjob.py   # 48 long + 48 short, c96, out 256

# 验证拒绝信息（D 节点/PP 依赖）
--additional-config '{"scheduler_config":{"dyntra_lb_config":{"enabled":true}}}'
# → "DyntraLB is only supported on PD-disaggregated D nodes"
```

日志：`results/e19/serve_{srf,dyntra,recomp}.log`
