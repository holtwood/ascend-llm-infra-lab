#!/usr/bin/env python3
"""把 vllm bench serve 的 json 结果汇总成 markdown 表。"""
import glob, json, sys

rows = []
for f in sorted(glob.glob(sys.argv[1])):
    d = json.load(open(f))
    rows.append({
        "file": f.split("/")[-1],
        "conc": d["max_concurrency"],
        "n": d["num_prompts"],
        "ok": d["completed"],
        "dur": d["duration"],
        "ttft": d.get("mean_ttft_ms"),
        "ttft_p99": d.get("p99_ttft_ms"),
        "tpot": d.get("mean_tpot_ms"),
        "tpot_p99": d.get("p99_tpot_ms"),
        "out_tps": d.get("output_throughput"),
        "req_tps": d.get("request_throughput"),
    })

print("| conc | reqs | dur(s) | TTFT mean | TTFT p99 | TPOT mean | TPOT p99 | out tok/s | req/s |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['conc']} | {r['ok']}/{r['n']} | {r['dur']:.1f} | "
          f"{r['ttft']:.1f} | {(r['ttft_p99'] or 0):.0f} | "
          f"{r['tpot']:.2f} | {(r['tpot_p99'] or 0):.0f} | "
          f"{r['out_tps']:.0f} | {r['req_tps']:.2f} |")
