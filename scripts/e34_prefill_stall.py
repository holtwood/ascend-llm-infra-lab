#!/usr/bin/env python3
"""E34 — prefill→decode interference (the "prefill stall") measurement.

Method: N persistent decode streams each request max_tokens tokens with
streaming; we record every token's arrival timestamp -> per-token ITL series.
At T_INJECT seconds we fire one long-prompt request (PREFILL_TOKENS input).

Output: baseline ITL vs spike magnitude vs recovery window, plus how the
injected request's TTFT compares. Repeat under different
--max-num-batched-tokens engine settings to show the chunked-prefill tradeoff.

Usage:
    python3 e34_prefill_stall.py --base-url http://127.0.0.1:8000 \
        --model qwen35-4b --streams 4 --out-tokens 400 \
        --inject-at 4 --prefill-tokens 4096 --out /tmp/e34.json
"""
import argparse, asyncio, json, time

import aiohttp

PROMPT_WORD = "lorem ipsum dolor sit amet consectetur adipiscing elit "


async def decode_stream(session, url, model, idx, out_tokens, t0, rec):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Write a long story about a mountain."}],
        "max_tokens": out_tokens,
        "temperature": 0.8,
        "stream": True,
    }
    times = []
    async with session.post(url + "/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.content:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                times.append(time.monotonic() - t0)
    rec[idx] = times


async def inject_prefill(session, url, model, at, in_tokens, t0, rec):
    await asyncio.sleep(at)
    t_start = time.monotonic() - t0
    body = {
        "model": model,
        "prompt": PROMPT_WORD * (in_tokens // 10),
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
    }
    async with session.post(url + "/v1/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.content:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                if "ttft" not in rec:
                    rec["ttft"] = time.monotonic() - t0 - t_start
    rec["t_start"] = t_start


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen35-4b")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--out-tokens", type=int, default=400)
    ap.add_argument("--inject-at", type=float, default=4.0)
    ap.add_argument("--prefill-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.monotonic()
    rec = {}
    streams = {}
    async with aiohttp.ClientSession() as s:
        tasks = [
            decode_stream(s, args.base_url, args.model, i, args.out_tokens, t0, streams)
            for i in range(args.streams)
        ]
        tasks.append(inject_prefill(s, args.base_url, args.model,
                                    args.inject_at, args.prefill_tokens, t0, rec))
        await asyncio.gather(*tasks)

    itls = {}  # idx -> list of ITL ms
    for i, ts in streams.items():
        itls[i] = [round((ts[j] - ts[j - 1]) * 1000, 2) for j in range(1, len(ts))]

    # per-millisecond binning is too fine; summary stats instead
    all_base = [v for i, vals in itls.items() for j, v in enumerate(vals)
                if streams[i][j + 1] < args.inject_at]
    all_post = [v for i, vals in itls.items() for j, v in enumerate(vals)
                if args.inject_at <= streams[i][j + 1] < args.inject_at + 3]
    post_sorted = sorted(all_post)
    summary = {
        "streams": args.streams,
        "inject_at_s": args.inject_at,
        "prefill_tokens": args.prefill_tokens,
        "inject_ttft_s": round(rec.get("ttft", -1), 3),
        "baseline_itl_med_ms": round(sorted(all_base)[len(all_base) // 2], 2) if all_base else None,
        "post_itl_p50_ms": round(post_sorted[len(post_sorted) // 2], 2) if post_sorted else None,
        "post_itl_p99_ms": round(post_sorted[int(len(post_sorted) * 0.99)], 2) if post_sorted else None,
        "post_itl_max_ms": round(max(all_post), 2) if all_post else None,
        "n_decode_tokens": sum(len(v) for v in streams.values()),
    }
    out = {"summary": summary, "itl_series": {str(k): v for k, v in itls.items()},
           "token_times": {str(k): [round(t, 3) for t in v] for k, v in streams.items()}}
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
