#!/usr/bin/env python3
"""N6: vllm-ascend 简测 — 并发 1/4 两档，各 N 请求，报 TTFT/TPOT/吞吐 P50。
用法: python3 n6_vllm_bench.py --concurrency 1 --num-requests 12 --max-tokens 100 [--mtp]
"""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp

PROMPTS = [
    "请用大约一百字介绍昇腾910B芯片的主要特性。",
    "解释一下 Transformer 中的 KV cache 是什么，为什么解码阶段需要它。",
    "写一个 Python 函数，判断一个字符串是否是回文，并简单说明思路。",
    "为什么大模型推理的吞吐量会随着 batch size 增大而提升？请从计算强度角度分析。",
    "简述 TCP 三次握手的过程和每一步的作用。",
    "请解释计算机体系中 roofline 模型的含义，以及它对算子优化的指导意义。",
    "用一句话分别解释 latency 和 throughput 的区别，再举一个推理服务的例子。",
    "什么是 per-token 动态激活量化？它和静态量化相比有什么优缺点？",
    "简单介绍 All-Reduce 通信操作在张量并行中的作用。",
    "解释一下大模型推理中 prefill 和 decode 两个阶段的区别。",
    "为什么 HBM 带宽往往是 LLM 解码阶段的瓶颈？请给出简单推导。",
    "什么是投机解码（speculative decoding）？它能加速推理的前提条件是什么？",
    "写一句通顺的话描述今天的天气，要求包含温度和心情。",
    "解释一下 GPU/NPU 中 Tensor Core 的作用。",
    "简述操作系统中虚拟内存的作用。",
    "什么是 MoE（Mixture of Experts）？它和 dense 模型在推理上的差别是什么？",
]

def percentile(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]

async def one_request(session, url, model, prompt, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    usage_ntok = None
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        # 关闭思考模式（Qwen3.8 默认 <think>，否则 100 token 全被 think 吃掉）
        "chat_template_kwargs": {"enable_thinking": False},
        # 流式末尾返回 usage，用 completion_tokens 精确计数
        "stream_options": {"include_usage": True},
    }
    try:
        async with session.post(url, json=body) as resp:
            buf = b""
            async for raw in resp.content:
                buf += raw
                # 按 SSE 行解析（MTP 一次验证接受多 token 时多个 data: 行
                # 会落在同一个网络包里，按包迭代会把它们当成一行导致计数失真）
                while b"\n" in buf:
                    line_b, buf = buf.split(b"\n", 1)
                    line = line_b.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage_ntok = obj["usage"].get("completion_tokens")
                    if not obj.get("choices"):
                        continue
                    delta = obj["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        ntok += 1
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    total = time.perf_counter() - t0
    if usage_ntok:
        ntok = usage_ntok
    if ttft is None or ntok <= 1:
        return {"error": f"no tokens (ttft={ttft}, ntok={ntok})"}
    tpot = (total - ttft) / (ntok - 1)
    return {
        "ttft": ttft,
        "tpot": tpot,
        "ntok": ntok,
        "total": total,
        "tok_per_s": ntok / total,
    }

async def run(concurrency, num_requests, url, model, max_tokens):
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency + 4)
    timeout = aiohttp.ClientTimeout(total=600)
    results = []

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        async def worker(i):
            async with sem:
                r = await one_request(session, url, model, PROMPTS[i % len(PROMPTS)], max_tokens)
                results.append(r)

        t_start = time.perf_counter()
        await asyncio.gather(*[worker(i) for i in range(num_requests)])
        wall = time.perf_counter() - t_start

    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    out = {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "max_tokens": max_tokens,
        "ok": len(ok),
        "failed": len(bad),
        "wall_s": wall,
        "agg_tok_per_s": sum(r["ntok"] for r in ok) / wall if ok else 0.0,
    }
    for key in ("ttft", "tpot", "tok_per_s"):
        vals = [r[key] for r in ok]
        if vals:
            out[f"{key}_p50"] = statistics.median(vals)
            out[f"{key}_p50_ms"] = statistics.median(vals) * 1000 if key != "tok_per_s" else None
    out["ttft_p50_ms"] = percentile([r["ttft"] for r in ok], 50) * 1000 if ok else None
    out["ttft_p90_ms"] = percentile([r["ttft"] for r in ok], 90) * 1000 if ok else None
    out["tpot_p50_ms"] = percentile([r["tpot"] for r in ok], 50) * 1000 if ok else None
    out["tpot_p90_ms"] = percentile([r["tpot"] for r in ok], 90) * 1000 if ok else None
    out["tok_per_s_p50"] = percentile([r["tok_per_s"] for r in ok], 50) if ok else None
    out["errors"] = [r["error"] for r in bad][:5]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-requests", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    res = asyncio.run(run(args.concurrency, args.num_requests, url, args.model, args.max_tokens))
    res["tag"] = args.tag
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
