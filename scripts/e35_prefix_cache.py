#!/usr/bin/env python3
"""E35 — prefix-cache hit -> TTFT savings curve.

Protocol per prefix length P in the sweep:
  1. send one "warm" request with prefix P (populates the APC blocks)
  2. send K requests that share prefix P + unique suffix -> measure TTFT
  3. send K requests with a fully-unique P-length prompt -> baseline TTFT

Reports saved TTFT per P. Requires enable_prefix_caching (default on).

Usage:
    python3 e35_prefix_cache.py --base-url http://127.0.0.1:8000 \
        --model qwen35-4b --prefix-sizes 0,256,1024,2048,3584 \
        --reps 6 --out /tmp/e35.json
"""
import argparse, json, statistics, time, urllib.request

WORD = "the quick brown fox jumps over a lazy dog "  # ~9 tok/rep


def post(url, body):
    t0 = time.monotonic()
    req = urllib.request.Request(url + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    # stream for honest TTFT
    body["stream"] = True
    req = urllib.request.Request(url + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    ttft = None
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                if ttft is None:
                    ttft = time.monotonic() - t0
    return ttft or (time.monotonic() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen35-4b")
    ap.add_argument("--prefix-sizes", default="256,1024,2048,3584")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = {}
    for p in [int(x) for x in args.prefix_sizes.split(",")]:
        shared = WORD * (p // 9)
        # warm the cache
        post(args.base_url, {"model": args.model, "prompt": shared + " warm",
                             "max_tokens": 4, "temperature": 0})
        hit, miss = [], []
        for i in range(args.reps):
            hit.append(post(args.base_url, {
                "model": args.model,
                "prompt": shared + f" then story #{i}",
                "max_tokens": 4, "temperature": 0}))
        for i in range(args.reps):
            miss.append(post(args.base_url, {
                "model": args.model,
                "prompt": f"unique{i} " + WORD * (p // 9),
                "max_tokens": 4, "temperature": 0}))
        results[p] = {
            "hit_ttft_med_s": round(statistics.median(hit), 3),
            "miss_ttft_med_s": round(statistics.median(miss), 3),
            "saved_s": round(statistics.median(miss) - statistics.median(hit), 3),
            "saved_pct": round(100 * (1 - statistics.median(hit) / statistics.median(miss)), 1),
        }
        print(p, results[p])
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
