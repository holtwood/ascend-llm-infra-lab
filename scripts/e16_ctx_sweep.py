#!/usr/bin/env python3
"""E16: long-context sweep — TTFT/TPOT/KV-usage vs context length.

For each ctx level, fires `concurrency` requests with unique long prompts
(nonce + word list => real KV consumption, no prefix-cache sharing),
streaming to measure TTFT and per-request TPOT, while polling
/metrics for KV cache usage.
"""
import argparse, asyncio, json, random, string, time
import httpx, urllib.request

WORDS = ("apple banana cedar delta ember fjord granite harbor ivory juniper "
         "kettle lumen mosaic nectar onyx prism quartz ridge saffron timber "
         "umbra violet walnut xenon yonder zephyr atlas beryl cobalt dune "
         "otter pepper quail raven salmon tulip ursa vapor willow yucca").split()

def metrics(port):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=5) as r:
            out = {}
            for line in r.read().decode().splitlines():
                if line.startswith("vllm:kv_cache_usage_perc") or \
                   line.startswith("vllm:num_requests_running"):
                    k = line.split("{")[0]
                    out[k] = float(line.split()[-1])
            return out
    except Exception:
        return {}

def prompt(words, rng):
    nonce = "".join(rng.choices(string.ascii_lowercase + string.digits, k=12))
    return f"{nonce} " + " ".join(rng.choices(WORDS, k=words))

async def one(client, port, p, mt):
    t0 = time.time(); ttft = None; ntok = 0; p_tok = None
    try:
        async with client.stream(
            "POST", f"http://localhost:{port}/v1/completions",
            json={"model": "qwen35-4b", "prompt": p, "max_tokens": mt,
                  "temperature": 0, "stream": True}) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data:") or "[DONE]" in line:
                    continue
                try:
                    d = json.loads(line[5:])
                except Exception:
                    continue
                ch = d.get("choices", [{}])[0]
                if ch.get("text") and ttft is None:
                    ttft = time.time() - t0
                if ch.get("text"):
                    ntok += 1
                u = d.get("usage")
                if u:
                    p_tok = u.get("prompt_tokens")
        return {"ok": True, "ttft": ttft, "total": time.time() - t0,
                "ntok": ntok, "prompt_tokens": p_tok,
                "tpot": (time.time() - t0 - (ttft or 0)) / max(ntok - 1, 1) if ntok > 1 else None}
    except Exception as e:
        return {"ok": False, "err": str(e)[:150]}

async def arm(port, words, conc, mt, tag, outdir):
    rng = random.Random(words)
    peaks = {"kv": 0.0, "run": 0}
    stop = False
    async def poll():
        while not stop:
            m = metrics(port)
            peaks["kv"] = max(peaks["kv"], m.get("vllm:kv_cache_usage_perc", 0))
            peaks["run"] = max(peaks["run"], m.get("vllm:num_requests_running", 0))
            await asyncio.sleep(0.4)
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=conc + 8),
                                 timeout=3600) as c:
        async def t(i):
            async with sem:
                return await one(c, port, prompt(words, rng), mt)
        pt = asyncio.create_task(poll())
        res = await asyncio.gather(*(t(i) for i in range(conc)))
        stop = True; await pt
    ok = [r for r in res if r["ok"]]
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"])
    tpots = sorted(r["tpot"] for r in ok if r["tpot"])
    ptoks = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]]
    out = {
        "tag": tag, "target_words": words, "concurrency": conc,
        "prompt_tokens_mean": round(sum(ptoks) / len(ptoks)) if ptoks else None,
        "ok": len(ok), "fail": len(res) - len(ok),
        "ttft_mean_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
        "ttft_p50_s": round(ttfts[len(ttfts) // 2], 3) if ttfts else None,
        "tpot_mean_ms": round(1000 * sum(tpots) / len(tpots), 1) if tpots else None,
        "tpot_p50_ms": round(1000 * tpots[len(tpots) // 2], 1) if tpots else None,
        "peak_kv_usage": round(peaks["kv"], 4),
        "peak_running": peaks["run"],
        "errors": [r.get("err") for r in res if not r["ok"]][:5],
    }
    with open(f"{outdir}/ctx_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out))

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="/data/llm-infra-lab/results/e16")
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--arms", default="2600x4,10500x4,21000x3,42000x3",
                    help="words x concurrency list")
    args = ap.parse_args()
    import os; os.makedirs(args.out, exist_ok=True)
    for spec in args.arms.split(","):
        w, c = spec.split("x")
        await arm(args.port, int(w), int(c), args.out_tokens,
                  f"w{w}c{c}", args.out)

asyncio.run(main())
