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


def worker(rank, world, result_q, port, out_dir):
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
    # no object collectives (HCCL support varies): each rank writes its row
    import json as _json
    with open(f"{out_dir}/hccl_pairs_rank{rank}.json", "w") as f:
        _json.dump({"rank": rank, "row": mine}, f)
    dist.destroy_process_group()
    if rank == 0:
        result_q.put({"world": world, "sizes": SIZES, "rows_written": world})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--port", type=int, default=29711)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=2700)
    args = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    avail = torch_npu.npu.device_count()
    assert args.world <= avail, f"world {args.world} > {avail} npus"
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    ps = [ctx.Process(target=worker, args=(r, args.world, q, args.port, out_dir))
          for r in range(args.world)]
    [p.start() for p in ps]
    try:
        res = q.get(timeout=args.timeout)
    except Exception as e:
        res = {"world": args.world, "error": f"{type(e).__name__}: {e}"}
    finally:
        for p in ps:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
    # merge per-rank rows into one matrix if all present
    rows = {}
    for r in range(args.world):
        p = f"{out_dir}/hccl_pairs_rank{r}.json"
        if os.path.exists(p):
            rows[r] = json.load(open(p))["row"]
    res["matrix"] = rows
    res["complete"] = len(rows) == args.world
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[hccl_pairs] wrote {args.out} complete={res['complete']}")
