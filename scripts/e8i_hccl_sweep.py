#!/usr/bin/env python3
"""E8i: HCCL collective sweep on N NPUs — extends E8d (2-card allreduce).

Measures: allreduce / all_gather / reduce_scatter / broadcast + device p2p
bandwidth, across 2/4/8 ranks, message sizes 256B..1GB.

Usage (single process per rank set):
  torchrun --nproc_per_node=N --master_port=29631 e8i_hccl_sweep.py --sizes ...

Simpler mode: spawn N procs in-process via mp.spawn (single node).
"""
import argparse, json, os, time
import torch, torch_npu, torch.distributed as dist
import torch.multiprocessing as mp


def bench_collective(fn, size_bytes, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt, size_bytes / dt / 1e9  # s, GB/s (payload/time)


def worker(rank, world, result_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")
    dist.init_process_group("hccl", rank=rank, world_size=world)
    torch_npu.npu.set_device(rank)
    dev = f"npu:{rank}"
    out = {"world": world, "rank": rank, "ops": {}}
    sizes = [256, 4096, 65536, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20]
    if rank == 0:
        print(f"[e8i] world={world} starting", flush=True)
    for sz in sizes:
        n = max(1, sz // 4)  # fp32 elems
        # allreduce
        x = torch.ones(n, device=dev)
        dt, bw = bench_collective(lambda: dist.all_reduce(x), sz)
        out["ops"].setdefault("allreduce", {})[sz] = {"s": dt * 1e6, "gbps": bw * world}  # bus bw
        # all_gather
        sg = torch.ones(n // world, device=dev)
        lst = [torch.empty_like(sg) for _ in range(world)]
        dt, _ = bench_collective(lambda: dist.all_gather(lst, sg), sz)
        out["ops"].setdefault("allgather", {})[sz] = {"s": dt * 1e6, "gbps": sz / dt / 1e9}
        # reduce_scatter
        rs = torch.ones(n, device=dev)
        rbuf = torch.empty(n // world, device=dev)
        dt, _ = bench_collective(lambda: dist.reduce_scatter_tensor(rbuf, rs), sz)
        out["ops"].setdefault("reduce_scatter", {})[sz] = {"s": dt * 1e6, "gbps": sz / dt / 1e9}
        # broadcast
        b = torch.ones(n, device=dev)
        dt, _ = bench_collective(lambda: dist.broadcast(b, 0), sz)
        out["ops"].setdefault("broadcast", {})[sz] = {"s": dt * 1e6, "gbps": sz / dt / 1e9}
        if rank == 0:
            print(f"[e8i] world={world} size={sz/1e6:.3f}MB done", flush=True)
    # p2p: rank0 -> rank1 send/recv (unidirectional)
    if world >= 2:
        for sz in [1 << 20, 16 << 20, 64 << 20, 256 << 20]:
            n = sz // 4
            buf = torch.ones(n, device=dev)
            if rank == 0:
                t0 = time.perf_counter()
                for _ in range(20):
                    dist.send(buf, 1)
                torch.npu.synchronize()
                dt = (time.perf_counter() - t0) / 20
                out["ops"].setdefault("p2p_send", {})[sz] = {"s": dt * 1e6, "gbps": sz / dt / 1e9}
            elif rank == 1:
                for _ in range(20):
                    dist.recv(buf, 0)
                torch.npu.synchronize()
            dist.barrier()
    if rank == 0:
        result_q.put(out)
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="2,4,8")
    ap.add_argument("--out", default="/tmp/e8i_hccl.json")
    args = ap.parse_args()
    avail = torch_npu.npu.device_count()
    print(f"[e8i] visible npus={avail}", flush=True)
    results = []
    for w in [int(x) for x in args.worlds.split(",")]:
        if w > avail:
            print(f"[e8i] skip world={w} > {avail}")
            continue
        ctx = mp.get_context("spawn")
        q = ctx.SimpleQueue()
        ps = [ctx.Process(target=worker, args=(r, w, q)) for r in range(w)]
        [p.start() for p in ps]
        r = q.get()
        results.append(r)
        [p.join() for p in ps]
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[e8i] wrote {args.out}")
