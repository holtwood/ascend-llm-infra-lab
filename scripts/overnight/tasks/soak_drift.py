#!/usr/bin/env python3
"""Soak drift analysis: compare latency deciles over a long allreduce run.

Usage: python3 soak_drift.py soak.jsonl
"""
import json
import sys
from statistics import mean, median


def main(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        print("no data")
        return
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    span_h = (rows[-1]["ts"] - rows[0]["ts"]) / 3600
    us = [r["us"] for r in rows]
    dec = max(1, n // 10)
    first, last = us[:dec], us[-dec:]
    print(f"samples={n} span={span_h:.2f}h")
    print(f"median_us={median(us):.0f} p10={sorted(us)[n//10]:.0f} p90={sorted(us)[9*n//10]:.0f}")
    print(f"first10% median={median(first):.0f}  last10% median={median(last):.0f}  "
          f"drift={100*(median(last)-median(first))/median(first):+.1f}%")
    # hourly buckets
    t0 = rows[0]["ts"]
    buckets = {}
    for r in rows:
        h = int((r["ts"] - t0) // 3600)
        buckets.setdefault(h, []).append(r["us"])
    for h in sorted(buckets):
        b = buckets[h]
        print(f"h{h}: n={len(b):5d} median={median(b):7.0f}us max={max(b):8.0f}us")


if __name__ == "__main__":
    main(sys.argv[1])
