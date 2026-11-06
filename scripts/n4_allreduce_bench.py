#!/usr/bin/env python3
"""N4: HCCL all-reduce microbenchmark on Ascend 910B2 (PCIe / SYS-crossed links).

Usage (2 ranks on physical NPUs 1,7; card 0 is reserved for the main experiment):
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_RT_VISIBLE_DEVICES=1,7
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=29501
    torchrun --nproc_per_node=2 --master_port=29501 n4_allreduce_bench.py

With ASCEND_RT_VISIBLE_DEVICES set, NPU devices are remapped to npu:0/npu:1
inside the process, so set_device(LOCAL_RANK) is correct.

For each bf16 message size: 10 warmup + 50 timed synchronous all_reduce calls,
wall-clocked with torch.npu.synchronize() around each call. Reports p50/p99
latency per rank (worst rank shown) and equivalent bandwidth
  algbw = bytes / latency
  busbw = 2*(N-1)/N * bytes / latency   (ring all-reduce traffic model)

A 1-element all_reduce is measured separately as the handshake/barrier floor.
"""
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  registers the npu / hccl backend

MSG_BYTES = [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
# Note on device IDs: in this container ASCEND_RT_VISIBLE_DEVICES takes
# container-local IDs, which map to host NPUs 0,1,7 as 0->0, 1->1, 2->7.
# Use ASCEND_RT_VISIBLE_DEVICES=1,2 (host cards 1 and 7; card 0 is reserved).
MSG_BYTES = os.environ.get("N4_SIZES", ",".join(map(str, MSG_BYTES)))
MSG_BYTES = [int(x) for x in MSG_BYTES.split(",") if int(x) >= 2]
WARMUP = int(os.environ.get("N4_WARMUP", 10))
ITERS = int(os.environ.get("N4_ITERS", 50))


def p99(lats):
    s = sorted(lats)
    return s[min(len(s) - 1, max(0, int(round(0.99 * len(s))) - 1))]


def bench(tensor):
    for _ in range(WARMUP):
        dist.all_reduce(tensor)
    torch.npu.synchronize()
    lats = []
    for _ in range(ITERS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(tensor)
        torch.npu.synchronize()
        lats.append((time.perf_counter() - t0) * 1e3)  # ms
    return lats


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    dev = torch.npu.current_device()
    dist.init_process_group(backend="hccl")

    if rank == 0:
        print(f"world={world} backend=hccl ASCEND_RT_VISIBLE_DEVICES="
              f"{os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} device={dev}", flush=True)

    results = []

    # 1-element all_reduce ~= 2 bytes: pure handshake / barrier floor
    one = torch.ones(1, dtype=torch.bfloat16, device=dev)
    lats = bench(one)
    results.append({"bytes": 2, "elems": 1, "p50_ms": statistics.median(lats),
                    "p99_ms": p99(lats), "min_ms": min(lats), "mean_ms": statistics.mean(lats)})
    del one

    for nbytes in MSG_BYTES:
        t = torch.full((nbytes // 2,), float(rank + 1), dtype=torch.bfloat16, device=dev)
        lats = bench(t)
        results.append({"bytes": nbytes, "elems": nbytes // 2,
                        "p50_ms": statistics.median(lats), "p99_ms": p99(lats),
                        "min_ms": min(lats), "mean_ms": statistics.mean(lats)})
        del t
        if rank == 0:
            r = results[-1]
            print(f"  {nbytes:>9} B  rank{rank} p50={r['p50_ms']*1e3:8.2f} us  "
                  f"p99={r['p99_ms']*1e3:8.2f} us", flush=True)

    obj = [None] * world
    dist.all_gather_object(obj, results)
    dist.barrier()

    if rank == 0:
        out = os.environ.get("N4_OUT", f"/data/results/n4_allreduce_raw_w{world}.json")
        with open(out, "w") as f:
            json.dump({"world": world, "vis": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                       "warmup": WARMUP, "iters": ITERS, "per_rank": obj}, f, indent=1)
        print(f"\nsaved raw json -> {out}")
        hdr = "     bytes  p50_us(worst) p99_us(worst)   algbw_GBps busbw_GBps"
        print(hdr)
        for i, r in enumerate(results):
            b = r["bytes"]
            worst_p50 = max(o[i]["p50_ms"] for o in obj) * 1e3
            worst_p99 = max(o[i]["p99_ms"] for o in obj) * 1e3
            lat_s = worst_p50 / 1e6
            algbw = b / lat_s / 1e9
            busbw = 2 * (world - 1) / world * b / lat_s / 1e9
            print(f"{b:>10}  {worst_p50:12.2f}  {worst_p99:12.2f}   {algbw:10.3f} {busbw:10.3f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
