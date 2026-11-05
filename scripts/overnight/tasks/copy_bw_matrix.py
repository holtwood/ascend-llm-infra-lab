#!/usr/bin/env python3
"""DRAM<->HBM copy bandwidth matrix + copy-under-compute interference.

Pinned/pageable x H2D/D2H x size sweep; then repeat with a concurrent
matmul stream to measure contention (KV-offload / PD-plane relevance).

Usage: python3 copy_bw_matrix.py --out $OUT_DIR/copy_bw.json
"""
import argparse, json, time
import torch, torch_npu


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    results = []
    for sz_mb in [64, 256, 1024, 2048]:
        n = (sz_mb << 20) // 4
        for pinned in [True, False]:
            h = torch.empty(n, dtype=torch.float32, pin_memory=pinned)
            d = torch.empty(n, device="npu:0", dtype=torch.float32)
            for name, fn in [("h2d", lambda: d.copy_(h, non_blocking=True)),
                             ("d2h", lambda: h.copy_(d, non_blocking=True))]:
                dt = bench(fn)
                results.append({"dir": name, "pinned": pinned, "mb": sz_mb,
                                "gbps": round(sz_mb / 1024 / dt, 2), "s": dt})
                print(f"[copy] {name} pinned={pinned} {sz_mb}MB -> {sz_mb/1024/dt:.1f} GB/s", flush=True)
            del h, d
            torch.npu.empty_cache()
    # interference: D2H while 4k matmul spins on a second stream
    n = (1024 << 20) // 4
    h = torch.empty(n, dtype=torch.float32, pin_memory=True)
    d = torch.empty(n, device="npu:0", dtype=torch.float32)
    a = torch.randn(4096, 4096, device="npu:0", dtype=torch.bfloat16)
    s2 = torch.npu.Stream()
    def busy():
        with torch.npu.stream(s2):
            for _ in range(8):
                a @ a
    dt_idle = bench(lambda: h.copy_(d, non_blocking=True))
    def under_load():
        busy()
        h.copy_(d, non_blocking=True)
    dt_load = bench(under_load, iters=6, warmup=2)
    results.append({"dir": "d2h_under_matmul", "pinned": True, "mb": 1024,
                    "gbps_idle": round(1.0 / dt_idle, 2),
                    "gbps_load": round(1.0 / dt_load, 2)})
    print(f"[copy] d2h under matmul: {1/dt_idle:.1f} -> {1/dt_load:.1f} GB/s", flush=True)
    with open(args.out, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"[copy] wrote {args.out}")
