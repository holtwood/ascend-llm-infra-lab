#!/usr/bin/env python3
"""E9: goodput@SLO — per-request pass/fail vs raw throughput.

Scans vllm bench-serve JSONs (with per-request `ttfts`/`itls` arrays),
applies SLO = {ttft_ms <= T AND mean_itl_ms <= D}, reports goodput
(passing requests / wall time) next to raw output throughput, showing
where the two diverge.
"""
import argparse, glob, json, os

def load_runs(patterns):
    runs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            ttfts = d.get("ttfts") or []
            itls = d.get("itls") or []
            if not ttfts or not itls:
                continue
            # itls may be flat list or per-request nested lists
            per_req_itl = []
            if itls and isinstance(itls[0], (int, float)):
                per_req_itl = [float(x) for x in itls]          # flat: mean per req? treat as per-req
            else:
                for r in itls:
                    if r:
                        per_req_itl.append(sum(float(x) for x in r) / len(r))
            n = min(len(ttfts), len(per_req_itl)) if per_req_itl else len(ttfts)
            reqs = [{"ttft_ms": float(t) * (1000 if float(t) < 100 else 1),
                     "itl_ms": (per_req_itl[i] * (1000 if per_req_itl[i] < 10 else 1))
                              if i < len(per_req_itl) else None}
                    for i, t in enumerate(ttfts[:n])]
            dur = float(d.get("duration", 0)) or None
            runs.append({"file": f,
                         "concurrency": d.get("max_concurrency") or d.get("concurrency"),
                         "completed": d.get("completed", n),
                         "duration": dur,
                         "output_tps": d.get("output_throughput"),
                         "total_tps": d.get("total_token_throughput"),
                         "reqs": reqs})
    return runs

def goodput(run, ttft_slo_ms, itl_slo_ms):
    ok = [r for r in run["reqs"]
          if r["ttft_ms"] <= ttft_slo_ms
          and (r["itl_ms"] is None or r["itl_ms"] <= itl_slo_ms)]
    n, dur = len(run["reqs"]), run["duration"]
    return {"pass": len(ok), "rate": len(ok) / n if n else 0.0,
            "goodput_rps": (len(ok) / dur) if dur else None,
            "req_rps": (n / dur) if dur else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="glob patterns of bench jsons")
    ap.add_argument("--slos", default="2000x100,1000x100,2000x50",
                    help="comma list of TTFTmsxTPOTms")
    ap.add_argument("--out", default="results/e9/goodput.json")
    args = ap.parse_args()

    runs = load_runs(args.inputs)
    slos = []
    for s in args.slos.split(","):
        t, d = s.split("x"); slos.append((float(t), float(d)))

    report = {"slos": [{"ttft_ms": t, "tpot_ms": d} for t, d in slos], "runs": []}
    for run in runs:
        entry = {"file": run["file"], "concurrency": run["concurrency"],
                 "completed": run["completed"], "duration_s": run["duration"],
                 "output_tok_s": run["output_tps"], "n_req": len(run["reqs"]),
                 "ttft_p50": None, "ttft_p99": None, "itl_mean": None, "by_slo": []}
        ts = sorted(r["ttft_ms"] for r in run["reqs"])
        if ts:
            entry["ttft_p50"] = round(ts[len(ts) // 2], 1)
            entry["ttft_p99"] = round(ts[int(0.99 * (len(ts) - 1))], 1)
        il = [r["itl_ms"] for r in run["reqs"] if r["itl_ms"]]
        if il:
            entry["itl_mean"] = round(sum(il) / len(il), 1)
        for (t, d) in slos:
            g = goodput(run, t, d)
            entry["by_slo"].append({"slo": f"{t:g}x{d:g}", **g,
                                    "goodput_rps": g["goodput_rps"] and round(g["goodput_rps"], 2)})
        report["runs"].append(entry)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # console table
    for e in report["runs"]:
        gp = " | ".join(f"{b['slo']}:{b['rate']*100:.0f}%/{b['goodput_rps']}rps"
                        for b in e["by_slo"])
        print(f"{os.path.basename(e['file']):38s} c={str(e['concurrency']):4s} "
              f"out={e['output_tok_s'] and round(e['output_tok_s']):>5}tok/s "
              f"ttft_p99={e['ttft_p99']}ms {gp}")

if __name__ == "__main__":
    main()
