#!/usr/bin/env python3
"""E53 — KV block size: internal fragmentation vs paged-gather tax.

vLLM defaults to block_size=16. Two costs move in opposite directions:
  - small blocks: less tail waste (fragmentation), but more pages per sequence
    and a higher gather tax (measured in E8j-ext / paged_gather.json)
  - large blocks: less gather tax, but more wasted tokens in the last block

This script simulates vLLM's block allocator over a realistic length
distribution (no NPU needed for the fragmentation part) and combines it with
the MEASURED gather tax from results/overnight_20261105_160244/paged_gather.json
to pick the operating point.

Usage: python3 block_size_analysis.py --out out.json
"""
import argparse, json, random, statistics

# measured paged-gather tax at S=4096, B=4, NKV=8 (paged_gather.json, seq table)
# one harness, S=4096, B=4, NKV=8 (re-measured 2026-11-06, consistent set)
MEASURED_TAX = {16: 1.54, 32: 1.70, 64: 1.98, 128: 2.31, 256: 2.30}


def frag_waste(lengths, bs):
    """Fraction of allocated KV slots that are never used (tail waste)."""
    alloc = sum((L + bs - 1) // bs * bs for L in lengths)
    used = sum(lengths)
    return (alloc - used) / alloc, alloc, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dist", default="lognormal",
                    choices=["lognormal", "uniform", "bimodal"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", default="16,32,64,128,256")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.dist == "lognormal":
        lengths = [max(16, int(rng.lognormvariate(6.9, 0.8))) for _ in range(args.n)]
    elif args.dist == "uniform":
        lengths = [rng.randint(64, 2048) for _ in range(args.n)]
    else:
        lengths = [rng.choice([rng.randint(64, 256), rng.randint(1500, 3000)])
                   for _ in range(args.n)]
    res = {"dist": args.dist, "n": args.n,
           "mean_len": round(statistics.mean(lengths), 1), "rows": []}
    best = None
    for bs in [int(x) for x in args.blocks.split(",")]:
        waste, alloc, used = frag_waste(lengths, bs)
        tax = MEASURED_TAX.get(bs)
        # effective throughput proxy: useful tokens per unit capacity per unit time
        eff = (1 - waste) / tax if tax else None
        row = {"block": bs, "tail_waste_pct": round(100 * waste, 2),
               "allocated_tokens": alloc, "used_tokens": used,
               "gather_tax": tax,
               "efficiency_index": round(eff, 4) if eff else None}
        res["rows"].append(row)
        if eff and (best is None or eff > best["efficiency_index"]):
            best = row
        tax_s = f"{tax:.2f}" if tax else "-"
        eff_s = f"{eff:.4f}" if eff else "-"
        print(f"[E53] block={bs:3d} tail_waste={100*waste:5.2f}%  "
              f"gather_tax={tax_s:>5}  efficiency={eff_s}", flush=True)
    res["optimal_block"] = best["block"] if best else None
    print(f"[E53] optimal block (capacity x speed) = {res['optimal_block']}")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
