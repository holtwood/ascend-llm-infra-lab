"""E82: interview Q — "one rank gets a long prompt and every rank's decode
slows down. Why?" (DP-attention / TP collective straggler)

2-rank loop of [decode-ish work + allreduce]. On steps 10..17 rank0 runs an
extra ~300ms GEMM burst (the giant prompt). Both ranks time each step.
Rank1 does identical work every step — any inflation is PURE straggler wait.
"""
import argparse
import json
import os
import time

import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

STEPS = 30
SLOW_STEPS = range(10, 18)


def worker(rank, world, port, out_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("hccl", rank=rank, world_size=world)
    torch_npu.npu.set_device(rank)
    dev = f"npu:{rank}"

    a = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    b = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    msg = torch.ones(1024, dtype=torch.float32, device=dev)

    def decode_work(n=6):
        z = a
        for _ in range(n):
            z = z @ b
        return z.sum()

    def prefill_burst():
        z = a
        for _ in range(60):      # ~10x decode step => a giant prompt
            z = z @ b
        return z.sum()

    trace = []
    for t in range(STEPS):
        t0 = time.perf_counter()
        decode_work()
        if rank == 0 and t in SLOW_STEPS:
            prefill_burst()
        dist.all_reduce(msg)          # collective: rank1 must wait for rank0
        torch.npu.synchronize()
        trace.append({"t": t, "ms": round((time.perf_counter() - t0) * 1e3, 1)})

    dist.barrier()
    with open(f"{out_dir}/tp_straggler_rank{rank}.json", "w") as f:
        json.dump({"rank": rank, "trace": trace}, f)
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--port", type=int, default=29881)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = os.path.dirname(args.out)
    mp.spawn(worker, args=(args.world, args.port, out_dir),
             nprocs=args.world, join=True)
    res = {}
    for r in range(args.world):
        res[f"rank{r}"] = json.load(open(f"{out_dir}/tp_straggler_rank{r}.json"))["trace"]
    res["note"] = f"rank0 does extra ~10x-work prefill on steps {min(SLOW_STEPS)}..{max(SLOW_STEPS)}"
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    r1 = [x["ms"] for x in res["rank1"]]
    slow = [x["ms"] for x in res["rank1"] if x["t"] in SLOW_STEPS]
    fast = [x["ms"] for x in res["rank1"] if x["t"] not in SLOW_STEPS and x["t"] > 0]
    print(f"[E82] rank1 normal={sorted(fast)[len(fast)//2]:.1f}ms "
          f"during-rank0-prefill={sorted(slow)[len(slow)//2]:.1f}ms "
          f"max={max(r1):.1f}", flush=True)
    print(f"[E82] wrote {args.out}", flush=True)
