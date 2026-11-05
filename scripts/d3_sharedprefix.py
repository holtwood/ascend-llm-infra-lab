#!/usr/bin/env python3
"""D3 driver: shared-prefix workload for prefix-aware-routing A/B.

T templates (each a long unique shared prefix) x R requests each
(unique suffix) -> through an entrypoint (nginx :8080 or router :8090).
Compares prefix_cache hit rate per backend + client latency.
Pure stdlib (ThreadPoolExecutor + urllib).
"""
import argparse, json, random, string, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

WORDS = ("apple banana cedar delta ember fjord granite harbor ivory juniper "
         "kettle lumen mosaic nectar onyx prism quartz ridge saffron timber "
         "umbra violet walnut xenon yonder zephyr atlas beryl cobalt dune").split()

def template_prefix(rng, words=600):
    return " ".join(rng.choices(WORDS, k=words))

def request(url, model, prompt, max_tokens=64):
    body = json.dumps({"model": model, "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            r.read()
        return time.time() - t0, None
    except Exception as e:
        return time.time() - t0, str(e)[:100]

def scrape_metrics(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",
                                    timeout=5) as r:
            out = {}
            for l in r.read().decode().splitlines():
                if l.startswith("vllm:prefix_cache_hits_total") or \
                   l.startswith("vllm:prefix_cache_queries_total") or \
                   l.startswith("vllm:time_to_first_token_seconds_sum") or \
                   l.startswith("vllm:time_to_first_token_seconds_count"):
                    k = l.split("{")[0]
                    out[k] = out.get(k, 0) + float(l.split()[-1])
            return out
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="http://127.0.0.1:8080/v1/completions")
    ap.add_argument("--backends", default="8010,8001,8002,8003")
    ap.add_argument("--templates", type=int, default=8)
    ap.add_argument("--per-template", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--prefix-words", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42,
                    help="template RNG seed — MUST differ between arms so "
                         "arm B cannot inherit arm A's cached prefixes")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/root/results_d3")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    templates = [template_prefix(rng, args.prefix_words)
                 for _ in range(args.templates)]
    jobs = []
    for t in templates:
        for i in range(args.per_template):
            suf = "".join(rng.choices(string.ascii_lowercase, k=16))
            jobs.append(t + f" request {suf} please continue:")

    ports = [p.strip() for p in args.backends.split(",")]
    m0 = {p: scrape_metrics(p) for p in ports}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        res = list(ex.map(lambda p: request(args.entry, "qwen35-4b", p), jobs))
    wall = time.time() - t0

    m1 = {p: scrape_metrics(p) for p in ports}
    lats = sorted(l for l, e in res if e is None)
    errs = [e for l, e in res if e]
    pct = lambda p: round(lats[int(p * (len(lats) - 1))], 2) if lats else None

    per_be = {}
    for p in ports:
        d = {k: m1[p].get(k, 0) - m0[p].get(k, 0) for k in
             ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
              "vllm:time_to_first_token_seconds_sum",
              "vllm:time_to_first_token_seconds_count")}
        d["hit_rate"] = (d["vllm:prefix_cache_hits_total"]
                         / max(d["vllm:prefix_cache_queries_total"], 1))
        d["ttft_mean_s"] = (d["vllm:time_to_first_token_seconds_sum"]
                            / max(d["vllm:time_to_first_token_seconds_count"], 1))
        per_be[p] = {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in d.items()}

    out = {"tag": args.tag, "entry": args.entry,
           "n_req": len(jobs), "templates": args.templates,
           "wall_s": round(wall, 1), "ok": len(lats), "err_n": len(errs),
           "lat_mean": round(sum(lats) / len(lats), 2) if lats else None,
           "lat_p50": pct(0.5), "lat_p90": pct(0.9),
           "per_backend": per_be,
           "tot_queries": sum(d["vllm:prefix_cache_queries_total"]
                              for d in per_be.values()),
           "tot_hits": sum(d["vllm:prefix_cache_hits_total"]
                           for d in per_be.values())}
    out["hit_rate_total"] = round(
        out["tot_hits"] / max(out["tot_queries"], 1), 4)
    import os; os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{args.tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

main()
