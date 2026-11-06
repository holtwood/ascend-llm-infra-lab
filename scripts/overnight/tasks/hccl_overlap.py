"""E77: can HCCL allreduce hide under compute? (2 ranks)

E75 showed D2H transfers hide ~99% under GEMM (dedicated DMA). HCCL uses the
HCCS/interconnect engine — if it also runs on dedicated hardware, an async
allreduce should hide under independent GEMM work; if it steals AI-core/HBM
resources, parallel should look like serial (or worse).

Per rank: 8192^3 GEMM chain (~50ms) + allreduce(msg).
  seq: allreduce(); gemm_chain()
  par: work=allreduce(async); gemm_chain(); work.wait()
hidden% = (seq - par)/ar_solo
"""
import argparse
import json
import os
import time

import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world, port, out_dir, mbytes, iters):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("hccl", rank=rank, world_size=world)
    torch_npu.npu.set_device(rank)
    dev = f"npu:{rank}"

    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)

    def gemm_chain(n=14):
        z = a
        for _ in range(n):
            z = z @ b
        return z.sum()

    msg = torch.ones(mbytes * 1000000 // 4, dtype=torch.float32, device=dev)

    def ar_sync():
        dist.all_reduce(msg)

    def ar_then_gemm():
        dist.all_reduce(msg)
        gemm_chain()

    def ar_overlap_gemm():
        work = dist.all_reduce(msg, async_op=True)
        gemm_chain()
        work.wait()

    def bench(fn, warm=3):
        for _ in range(warm):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    out = {}
    if rank == 0:
        gemm_ms = bench(gemm_chain)
        ar_ms = bench(ar_sync)
        seq_ms = bench(ar_then_gemm)
        par_ms = bench(ar_overlap_gemm)
        out = {"mbytes": mbytes, "gemm_ms": round(gemm_ms, 2),
               "ar_ms": round(ar_ms, 3), "seq_ms": round(seq_ms, 2),
               "par_ms": round(par_ms, 2),
               "hidden_pct": round((seq_ms - par_ms) / ar_ms * 100, 1)}
        print(f"[E77] msg={mbytes/1e6:.0f}MB ar={ar_ms:.2f} gemm={gemm_ms:.1f} "
              f"seq={seq_ms:.1f} par={par_ms:.1f} hidden={out['hidden_pct']}%",
              flush=True)
    else:
        # rank1 must run same collective sequence count
        for _ in range(3 + iters):
            pass
        gemm_ms = bench(gemm_chain)
        ar_ms = bench(ar_sync)
        seq_ms = bench(ar_then_gemm)
        par_ms = bench(ar_overlap_gemm)

    dist.barrier()
    with open(f"{out_dir}/hccl_overlap_rank{rank}.json", "w") as f:
        json.dump({"rank": rank, "result": out}, f)
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--port", type=int, default=29871)
    ap.add_argument("--mbytes", type=int, default=64)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = os.path.dirname(args.out)
    mp.spawn(worker, args=(args.world, args.port, out_dir, args.mbytes, args.iters),
             nprocs=args.world, join=True)
    if os.path.exists(f"{out_dir}/hccl_overlap_rank0.json"):
        res = json.load(open(f"{out_dir}/hccl_overlap_rank0.json"))["result"]
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[E77] wrote {args.out}", flush=True)
