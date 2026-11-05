#!/usr/bin/env python3
"""Minimal OpenAI-API bench via urllib (no external deps).

Sends N requests at concurrency C to /v1/completions, records per-request
latency, computes throughput + percentiles. Output: JSON to stdout file.

Usage: python3 bench_min.py --port 8002 --c 16 --n 64 --max-tokens 64 --out out.json
"""
import argparse, json, threading, time, urllib.request


def req(port, prompt, max_tokens):
    body = json.dumps({"model": "m", "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                               data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=600) as resp:
        data = json.loads(resp.read())
    dt = time.perf_counter() - t0
    usage = data.get("usage", {})
    return dt, usage.get("completion_tokens", 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--c", type=int, default=16)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prompt = "benchmark " * args.prompt_tokens
    lats, toks, errs = [], [], [0]
    sem = threading.Semaphore(args.c)
    def job(i):
        with sem:
            try:
                dt, ct = req(args.port, prompt + str(i), args.max_tokens)
                lats.append(dt); toks.append(ct)
            except Exception:
                errs[0] += 1
    t0 = time.perf_counter()
    ths = [threading.Thread(target=job, args=(i,)) for i in range(args.n)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.perf_counter() - t0
    lats.sort()
    p = lambda q: lats[min(len(lats) - 1, int(len(lats) * q))] if lats else -1
    res = {"c": args.c, "n": args.n, "errors": errs[0], "wall_s": round(wall, 1),
           "lat_mean": round(sum(lats) / max(1, len(lats)), 3),
           "p50": round(p(0.5), 3), "p90": round(p(0.9), 3), "p99": round(p(0.99), 3),
           "tok_out_s": round(sum(toks) / wall, 1)}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))
