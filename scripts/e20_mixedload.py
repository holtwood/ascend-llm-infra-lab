#!/usr/bin/env python3
"""E20 driver: mixed short/long load through a gateway.

- `--frac-short` of requests get ~SHORT_WORDS prompts; rest get ~LONG_WORDS.
- Measures per-class e2e latency + rejection (HTTP 429) count.
- Pure stdlib (ThreadPoolExecutor + urllib), same style as d3_sharedprefix.

Usage: python3 e20_mixedload.py --entry http://127.0.0.1:8092/v1/completions \
         --tag cp --n 64 --concurrency 16 --frac-short 0.5
"""
import argparse, json, random, string, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

WORDS = ("apple banana cedar delta ember fjord granite harbor ivory juniper "
         "kettle lumen mosaic nectar onyx prism quartz ridge saffron timber "
         "umbra violet walnut xenon yonder zephyr atlas beryl cobalt dune").split()


def mk(n, rng):
    return "".join(rng.choices(string.ascii_lowercase + string.digits, k=10)) \
        + " " + " ".join(rng.choices(WORDS, k=n))


def one(entry, model, prompt, mt, cls):
    t0 = time.time()
    req = urllib.request.Request(
        entry, data=json.dumps({"model": model, "prompt": prompt,
                                "max_tokens": mt, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
            return cls, time.time() - t0, 200
    except urllib.error.HTTPError as e:
        e.read()
        return cls, time.time() - t0, e.code
    except Exception as e:
        return cls, time.time() - t0, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--model", default="qwen35-4b")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--frac-short", type=float, default=0.5)
    ap.add_argument("--short-words", type=int, default=120)
    ap.add_argument("--long-words", type=int, default=2800)
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/root/results_e20")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    jobs = []
    for i in range(args.n):
        short = rng.random() < args.frac_short
        w = args.short_words if short else args.long_words
        jobs.append((mk(w, rng), "short" if short else "long"))

    import os
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    res = []
    with ThreadPoolExecutor(args.concurrency) as ex:
        futs = [ex.submit(one, args.entry, args.model, p,
                          args.out_tokens, c) for p, c in jobs]
        res = [f.result() for f in futs]
    wall = time.time() - t0

    rep = {"tag": args.tag, "entry": args.entry, "n": args.n,
           "concurrency": args.concurrency, "wall_s": round(wall, 1),
           "per_class": {}, "rejected": 0}
    for cls in ("short", "long"):
        xs = sorted(r[1] for r in res if r[0] == cls and r[2] == 200)
        ok = len(xs)
        rej = sum(1 for r in res if r[0] == cls and r[2] == 429)
        err = sum(1 for r in res if r[0] == cls and r[2] not in (200, 429))
        rep["per_class"][cls] = {"ok": ok, "rej429": rej, "err": err}
        rep["rejected"] += rej
        if xs:
            rep["per_class"][cls].update(
                mean=round(sum(xs)/len(xs), 2), p50=round(xs[len(xs)//2], 2),
                p90=round(xs[int(0.9*(len(xs)-1))], 2), max=round(xs[-1], 2))
    path = f"{args.out}/{args.tag}.json"
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
