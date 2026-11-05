#!/usr/bin/env python3
"""Sustained allreduce soak — drift/thermal/degradation detector.

Runs 256MB allreduce on world=8 for --duration seconds, logging each
iteration's latency to JSONL. Compare first/last decile for throttling.

Usage: python3 hccl_soak.py --world 8 --duration 7200 --out $OUT_DIR/soak.jsonl
"""
import argparse, json, os, time
import torch, torch_npu, torch.distributed as dist
import torch.multiprocessing as mp

SIZE = 256 << 20


def worker(rank, world, duration, out_path, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("hccl", rank=rank, world_size=world)
    torch_npu.npu.set_device(rank)
    x = torch.ones(SIZE // 4, device=f"npu:{rank}")
    f = open(out_path, "a") if rank == 0 else None
    deadline = time.time() + duration
    for _ in range(5):
        dist.all_reduce(x)
    torch.npu.synchronize()
    i = 0
    while time.time() < deadline:
        t0 = time.perf_counter()
        dist.all_reduce(x)
        torch.npu.synchronize()
        dt = time.perf_counter() - t0
        if f:
            f.write(json.dumps({"i": i, "ts": time.time(), "us": dt * 1e6,
                                "gbps": SIZE / dt / 1e9 * world}) + "\n")
            if i % 60 == 0:
                f.flush()
        i += 1
        time.sleep(1.0)
    if f:
        f.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--duration", type=int, default=7200)
    ap.add_argument("--port", type=int, default=29721)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=worker, args=(r, args.world, args.duration, args.out, args.port))
          for r in range(args.world)]
    [p.start() for p in ps]
    for p in ps:
        p.join(timeout=args.duration + 600)
        if p.is_alive():
            p.terminate()
    print(f"[hccl_soak] done -> {args.out}")
