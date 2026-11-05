#!/usr/bin/env python3
"""All-pairs p2p bandwidth/latency matrix on N NPUs (HCCL send/recv).

Reveals HCCS link topology: symmetric fast pairs vs slow/routed pairs.
Output: JSON with matrix[src][dst][size] = {us, gbps}.

Usage: python3 hccl_pairs.py --world 8 --out $OUT_DIR/hccl_pairs_w8.json
"""
import argparse, json, os, time
import torch, torch_npu, torch.distributed as dist
import torch.multiprocessing as mp

SIZES = [1 << 10, 64 << 10, 1 << 20, 16 << 20, 64 << 20]
ITERS, WARMUP = 15, 5


def worker(rank, world, result_q, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("hccl", rank=rank, world_size=world)
    torch_npu.npu.set_device(rank)
    dev = f"npu:{rank}"
    mine = {}  # dst -> size -> stats (only fills for rows where rank==src)
    for src in range(world):
        for dst in range(world):
            if src == dst:
                continue
            for sz in SIZES:
                n = max(1, sz // 4)
                buf = torch.ones(n, device=dev)
                if rank == src:
                    for _ in range(WARMUP):
                        dist.send(buf, dst)
                    torch.npu.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(ITERS):
                        dist.send(buf, dst)
                    torch.npu.synchronize()
                    dt = (time.perf_counter() - t0) / ITERS
                    mine.setdefault(dst, {})[sz] = {"us": dt * 1e6,
                                                    "gbps": sz / dt / 1e9}
                elif rank == dst:
                    for _ in range(WARMUP + ITERS):
                        dist.recv(buf, src)
                    torch.npu.synchronize()
                dist.barrier()
    rows = [None] * world
    dist.all_gather_object(rows, mine)
    if rank == 0:
        matrix = {s: r for s, r in enumerate(rows)}
        result_q.put({"world": world, "sizes": SIZES, "matrix": matrix})
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--port", type=int, default=29711)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=2700)
    args = ap.parse_args()
    avail = torch_npu.npu.device_count()
    assert args.world <= avail, f"world {args.world} > {avail} npus"
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    ps = [ctx.Process(target=worker, args=(r, args.world, q, args.port)) for r in range(args.world)]
    [p.start() for p in ps]
    try:
        res = q.get(timeout=args.timeout)
    except Exception:
        res = {"world": args.world, "error": "timeout waiting for rank0 result"}
    finally:
        for p in ps:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[hccl_pairs] wrote {args.out}")
