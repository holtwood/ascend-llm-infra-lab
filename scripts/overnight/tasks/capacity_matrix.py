#!/usr/bin/env python3
"""E64 — Capacity matrix: goodput at SLO across model size, context and SLO.

Runs the calibrated planner over a grid and prints a markdown table. All
coefficients are measured (see docs/capacity-planner.md):
  HBM 1412 GB/s, launch floor 29 ms (eager), attention efficiency 19/50/86%,
  prefill 0.10ms + 0.09us/token + 4.66e-8 ms/token^2 per layer.

Model presets use realistic shapes (layers, kv heads, head dim, weights).

Usage: python3 capacity_matrix.py [--graph-captured] [--slo 100]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from serving_capacity_planner import plan  # noqa: E402
import argparse as ap2  # noqa: E402


PRESETS = {
    "0.6B (28L, 8kv, bf16)": dict(weight_gb=1.5, layers=28, nkv=8, head_dim=128),
    "4B (36L, 8kv, bf16)":   dict(weight_gb=8.0, layers=36, nkv=8, head_dim=128),
    "8B (36L, 8kv, bf16)":   dict(weight_gb=16.0, layers=36, nkv=8, head_dim=128),
    "8B w8a16 (36L)":        dict(weight_gb=11.0, layers=36, nkv=8, head_dim=128),
    "32B (64L, 8kv, bf16)":  dict(weight_gb=64.0, layers=64, nkv=8, head_dim=128),
}


def build_args(preset, ctx, slo, graph):
    ns = ap2.Namespace()
    ns.weight_gb = preset["weight_gb"]
    ns.layers = preset["layers"]
    ns.nkv = preset["nkv"]
    ns.head_dim = preset["head_dim"]
    ns.hbm_gb = 61.0
    ns.overhead_gb = 2.0
    ns.ctx = ctx
    ns.prompt_len = min(ctx // 2, 4096)
    ns.target_tpot_ms = slo
    ns.graph_captured = graph
    return ns


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--ctxs", default="2048,8192,32768")
    a.add_argument("--slos", default="50,100,200")
    a.add_argument("--graph-captured", action="store_true")
    a.add_argument("--out", default=None)
    args = a.parse_args()
    lines = []
    mode = "graph-captured" if args.graph_captured else "eager (29ms floor)"
    lines.append(f"# 容量矩阵（goodput tok/s @ SLO，{mode}）\n")
    for slo in [float(s) for s in args.slos.split(",")]:
        lines.append(f"\n## TPOT SLO = {slo:.0f} ms\n")
        header = "| model | " + " | ".join(f"ctx {c}" for c in args.ctxs.split(",")) + " |"
        lines.append(header)
        lines.append("|---" * (len(args.ctxs.split(",")) + 1) + "|")
        for name, preset in PRESETS.items():
            cells = []
            for ctx in [int(c) for c in args.ctxs.split(",")]:
                r = plan(build_args(preset, ctx, slo, args.graph_captured))
                if "error" in r:
                    cells.append("OOM")
                elif r["slo_batch"] == 0:
                    cells.append("SLO✗")
                else:
                    cells.append(f"{r['goodput_tok_s']:,.0f} ({r['slo_batch']}seq)")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        open(args.out, "w").write(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
