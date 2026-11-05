#!/usr/bin/env python3
"""E8d: HCCL allreduce microbenchmark on 2x 910B1.

Measures allreduce latency/bandwidth vs message size — the communication
primitive that dominates tensor-parallel decode. Compares against the
memory-bandwidth numbers from E8b to quantify when TP pays.

Run: torchrun --nproc_per_node=2 e8d_hccl.py --devices 2,3 --out e8d_hccl.json
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def bench_allreduce(nbytes_elems, iters=50, warmup=10):
    t = torch.ones(nbytes_elems, dtype=torch.bfloat16, device="npu")
    for _ in range(warmup):
        dist.all_reduce(t)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(t)
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    global LOCAL
    LOCAL = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    # ASCEND_RT_VISIBLE_DEVICES remaps physical -> logical ids; torch uses logical
    torch.npu.set_device(LOCAL)

    dist.init_process_group(backend="hccl")

    res = {}
    for exp in range(12, 31):  # 4KB .. 1GB elements*2B
        n = 1 << exp
        try:
            dt = bench_allreduce(n, iters=20 if n > 1 << 24 else 50)
        except RuntimeError:
            break
        mb = n * 2 / 1e6
        res[f"{mb:.0f}MB" if mb >= 1 else f"{mb*1000:.0f}KB"] = {
            "elems": n,
            "ms": round(dt * 1000, 3),
            "GBps_algbw": round(n * 2 / dt / 1e9, 1),
            "GBps_busbw": round(n * 2 / dt / 1e9, 1),
        }
        if rank == 0:
            print(res[list(res)[-1]], list(res)[-1], flush=True)

    if rank == 0 and args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1))
    dist.destroy_process_group()


LOCAL = 0
if __name__ == "__main__":
    main()
