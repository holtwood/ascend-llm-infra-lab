"""E75: can transfers hide under compute? (stream overlap on 910B1)

Serving relies on overlap in three places:
  - KV offload D2H while decode computes (preemption)
  - weight H2D prefetch during run (sleep-wake / failover)
  - TP allreduce under GEMM (needs 2 ranks; proxy here = D2D copy)

Measure on two npu streams:
  seq   : GEMM chain then copy (or copy then GEMM)
  par   : copy on side stream + GEMM chain on compute stream, joined

speedup = seq/par; 1.0 -> zero overlap, ~2.0 -> perfect overlap.
Also report copy-alone and gemm-alone baselines.
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "npu:0"
    torch.npu.set_device(dev)

    # ~50ms GEMM chain: 14 x 8192^3 matmuls (~3.7ms each per E68)
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)

    def gemm_chain():
        z = a
        for _ in range(14):
            z = z @ b
        return z.sum()

    # ~40ms transfers: 940MB D2H and 940MB D2D
    host = torch.empty(940_000_000, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(940_000_000, dtype=torch.uint8, device=dev)
    src2 = torch.empty(235_000_000, dtype=torch.uint8, device=dev)
    dst2 = torch.empty(235_000_000, dtype=torch.uint8, device=dev)

    def d2d_chain():
        for _ in range(80):
            dst2.copy_(src2, non_blocking=True)

    side = torch.npu.Stream()
    res = {"rows": []}

    for tag, cp in [("d2h_940MB", lambda: dst.copy_(host, non_blocking=True)),
                    ("d2d_x80_235MB", d2d_chain)]:
        gemm_ms = bench(gemm_chain)
        copy_ms = bench(cp)

        def seq():
            gemm_chain()
            cp()

        def par():
            with torch.npu.stream(side):
                cp()
            r = gemm_chain()
            torch.npu.current_stream().wait_stream(side)
            return r

        seq_ms = bench(seq)
        par_ms = bench(par)
        row = {"transfer": tag, "gemm_ms": round(gemm_ms, 2),
               "copy_ms": round(copy_ms, 2), "seq_ms": round(seq_ms, 2),
               "par_ms": round(par_ms, 2),
               "overlap_efficiency": round((seq_ms - par_ms) / copy_ms * 100, 1)}
        res["rows"].append(row)
        print(f"[E75] {tag}: gemm={gemm_ms:.1f} copy={copy_ms:.1f} "
              f"seq={seq_ms:.1f} par={par_ms:.1f} "
              f"hidden={row['overlap_efficiency']}%", flush=True)

    res["note"] = ("overlap_efficiency = fraction of copy time hidden under "
                   "gemm; 100% = copy completely free")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[E75] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
