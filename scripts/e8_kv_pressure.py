#!/usr/bin/env python3
"""E8: KV-capacity stress test — drive concurrency beyond KV token budget.

Each request carries a UNIQUE long prompt (nonce + random words) so prefix
caching cannot collapse shared blocks; KV footprint is real per request.
Collects: per-request latency, /metrics preemption+usage deltas, peak usage.
"""
import argparse, asyncio, json, random, string, time
import httpx, urllib.request

WORDS = ("apple banana cedar delta ember fjord granite harbor ivory juniper "
         "kettle lumen mosaic nectar onyx prism quartz ridge saffron timber "
         "umbra violet walnut xenon yonder zephyr atlas beryl cobalt dune").split()

def get_metrics(port):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=5) as r:
            out = {}
            for line in r.read().decode().splitlines():
                if line.startswith("vllm:"):
                    k, _, v = line.partition(" ")
                    out[k] = v
            return out
    except Exception:
        return {}

def unique_prompt(n_words, rng):
    nonce = "".join(rng.choices(string.ascii_lowercase + string.digits, k=16))
    return f"{nonce} " + " ".join(rng.choices(WORDS, k=n_words))

async def one(client, port, prompt, max_tokens):
    t0 = time.time()
    try:
        r = await client.post(
            f"http://localhost:{port}/v1/completions",
            json={"model": "qwen35-4b", "prompt": prompt, "max_tokens": max_tokens,
                  "temperature": 0})
        return {"ok": r.status_code == 200, "lat": time.time() - t0,
                "status": r.status_code}
    except Exception as e:
        return {"ok": False, "lat": time.time() - t0, "err": str(e)[:120]}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=300)
    ap.add_argument("--prompt-words", type=int, default=3600,
                    help="approx prompt size in words (~1.1 tok/word)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/data/llm-infra-lab/results/e8")
    args = ap.parse_args()

    m0 = get_metrics(args.port)
    rng = random.Random(0)
    sem = asyncio.Semaphore(args.concurrency)
    peaks = {"usage": 0.0, "running": 0, "waiting": 0}
    stop = False

    def gauge_val(m, key):
        for k, v in m.items():
            if k.startswith(key):
                try:
                    return float(v.split()[-1])
                except Exception:
                    pass
        return 0.0

    async def poll():
        while not stop:
            m = get_metrics(args.port)
            peaks["usage"] = max(peaks["usage"], gauge_val(m, "vllm:kv_cache_usage_perc"))
            peaks["running"] = max(peaks["running"], gauge_val(m, "vllm:num_requests_running{"))
            peaks["waiting"] = max(peaks["waiting"], gauge_val(m, "vllm:num_requests_waiting{"))
            await asyncio.sleep(0.5)

    limits = httpx.Limits(max_connections=args.concurrency + 16)
    async with httpx.AsyncClient(limits=limits, timeout=1800) as client:
        async def task(i):
            async with sem:
                return await one(client, args.port,
                                 unique_prompt(args.prompt_words, rng),
                                 args.max_tokens)
        poll_t = asyncio.create_task(poll())
        t0 = time.time()
        res = await asyncio.gather(*(task(i) for i in range(args.concurrency)))
        stop = True
        await poll_t
        wall = time.time() - t0

    m1 = get_metrics(args.port)
    ok = sum(r["ok"] for r in res)
    lats = sorted(r["lat"] for r in res if r["ok"])
    def pct(p):
        return round(lats[int(p * (len(lats) - 1))], 2) if lats else None

    out = {
        "tag": args.tag, "concurrency": args.concurrency,
        "prompt_words": args.prompt_words, "max_tokens": args.max_tokens,
        "ok": ok, "fail": len(res) - ok, "wall_s": round(wall, 1),
        "lat_mean": round(sum(lats) / len(lats), 2) if lats else None,
        "lat_p50": pct(0.50), "lat_p90": pct(0.90), "lat_p99": pct(0.99),
        "peak_kv_usage": peaks["usage"], "peak_running": peaks["running"],
        "peak_waiting": peaks["waiting"],
        "preemptions_delta": (gauge_val(m1, "vllm:num_preemptions_total")
                              - gauge_val(m0, "vllm:num_preemptions_total")),
        "errors": [r["err"] for r in res if "err" in r][:10],
        "status_hist": {str(r["status"]): sum(1 for x in res if x.get("status") == r["status"])
                        for r in res if "status" in r},
    }
    path = f"{args.out}/pressure_{args.tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

asyncio.run(main())
