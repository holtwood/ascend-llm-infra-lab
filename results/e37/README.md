# E37 — 客户端断连 → KV 释放时序（面试题："客户端中途断连怎么办？"）

> 面试题：**"流式回复中客户端断连，引擎该怎么处理？"**
> 答（有数据）：8 个并发 SSE 流在 t=1.34s 全部断开 → t=1.68s
> `num_requests_running` 8→0、`kv_cache_usage_perc` 1.37%→0%——
> **0.34s 内全部中止并释放 KV**，不是"跑完 generation 才回收"。

## 1. 方法

- 8 个 http 客户端各开一条 SSE 流（max_tokens=2000），确认 generating 后
  **直接 close socket**（模拟断连）
- 每 0.2s 轮询 `/metrics`：`num_requests_running`、`kv_cache_usage_perc`
- 记录断连→归零时延

## 2. 数据（M2 / Qwen3.5-4B，两次测量）

| 测量 | 断连→全部释放 |
|---|---|
| run 1 | ~2s |
| run 2 | **0.34s**（t=1.34→1.68） |

- KV 同步归零（1.37%→0%）：**块立即回到空闲池**，不是延迟回收
- 两次差异来自 SSE 写节奏：断连是**写失败时**被发现的，不是读端探测

## 3. 机制解释

- 服务端感知断连的时机 = 下一次往 socket 写 token 失败（broken pipe）；
  所以释放时延 ≈ ITL 级别 + 清理路径，量级实测 0.3–2s
- 关键设计点（面试加分）：**cancel 必须即时释放 KV 块**，否则断连请求
  会继续占着缓存空转——vLLM 的 abort 路径直接还块
- 对比"客户端慢读"：proxy_buffering off 时服务端被反压，同样占 KV；
  这也是 streaming 服务需要 write timeout 的原因

## 4. 复现

```bash
python3 scripts/e37_cancel_kv.py --base-url http://127.0.0.1:8000 \
  --model qwen35-4b --clients 8 --settle 6 --out /tmp/e37.json
```
