#!/usr/bin/env python3
"""Minimal streaming benchmark against an OpenAI-compatible endpoint.

Metrics mirror `vllm bench serve`: TTFT (ms), TPOT (ms), output throughput
(tok/s), plus p50/p99. Works against MindIE /v1/chat/completions and any
other OpenAI-compatible streaming server.

Usage:
    python3 bench_openai.py --base-url http://127.0.0.1:1025 \
        --model qwen3 --concurrency 16 --num-requests 128 \
        --input-tokens 512 --output-tokens 128 --out result.json
"""
import argparse
import asyncio
import json
import time

import aiohttp


def build_prompt(target_tokens: int) -> str:
    # ~1 token per 4 chars for mixed CN/EN; simple repeatable filler
    seg = "The quick brown fox jumps over the lazy dog. " * 4
    return (seg * (target_tokens * 4 // len(seg) + 1))[: target_tokens * 4]


async def one_request(session, url, model, prompt, out_tokens, results, idx):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": out_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    n_tok = 0
    try:
        async with session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                results[idx] = {"error": f"{resp.status}: {body[:200]}"}
                return
            async for line in resp.content:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                if line == b"data: [DONE]":
                    break
                try:
                    chunk = json.loads(line[5:])
                except Exception:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_tok += 1
    except Exception as e:  # noqa: BLE001
        results[idx] = {"error": str(e)[:200]}
        return
    e2e = time.perf_counter() - t0
    results[idx] = {
        "ttft_ms": (ttft or e2e) * 1000,
        "e2e_ms": e2e * 1000,
        "out_tokens": n_tok,
        "tpot_ms": (e2e - (ttft or e2e)) * 1000 / max(n_tok - 1, 1),
    }


def pct(vals, p):
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = max(int(len(vals) * p / 100) - 1, 0)
    return vals[k]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:1025")
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--num-requests", type=int, default=128)
    ap.add_argument("--input-tokens", type=int, default=512)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    url = args.base_url + args.endpoint
    prompt = build_prompt(args.input_tokens)
    results = [None] * args.num_requests
    sem = asyncio.Semaphore(args.concurrency)

    conn = aiohttp.TCPConnector(limit=args.concurrency + 8)
    async with aiohttp.ClientSession(connector=conn) as session:

        async def wrapped(i):
            async with sem:
                await one_request(
                    session, url, args.model, prompt, args.output_tokens, results, i
                )

        wall0 = time.perf_counter()
        await asyncio.gather(*(wrapped(i) for i in range(args.num_requests)))
        wall = time.perf_counter() - wall0

    ok = [r for r in results if r and "error" not in r]
    err = [r for r in results if r and "error" in r]
    ttfts = [r["ttft_ms"] for r in ok]
    tpots = [r["tpot_ms"] for r in ok]
    tot_out = sum(r["out_tokens"] for r in ok)
    summary = {
        "endpoint": url,
        "model": args.model,
        "concurrency": args.concurrency,
        "num_requests": args.num_requests,
        "completed": len(ok),
        "errors": len(err),
        "wall_s": round(wall, 2),
        "ttft_mean_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else 0,
        "ttft_p99_ms": round(pct(ttfts, 99), 1),
        "tpot_mean_ms": round(sum(tpots) / len(tpots), 1) if tpots else 0,
        "tpot_p99_ms": round(pct(tpots, 99), 1),
        "output_tok_per_s": round(tot_out / wall, 1),
        "err_sample": err[0]["error"] if err else "",
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "requests": results}, f, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
