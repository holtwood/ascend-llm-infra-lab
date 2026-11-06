#!/usr/bin/env python3
"""TP decode-step emulation with real Qwen3-0.6B weights (torch_npu, no vllm).

For tp in {1,2,4,8}: shard each linear layer's weights across ranks
(row-parallel style: out-dim split), run one decode step (bs=1, seq=1)
with an allreduce after each sharded projection, and measure:
  - per-step latency (compute + comm)
  - comm share of the step (allreduce time summed)
  - scaling vs tp=1

This is a torch-level answer to "what does TP cost per decode step",
complementing E8i (raw HCCL) and E4 (replica scaling).

Usage: python3 tp_decode_sim.py --model /data1/pytroch-7/models/Qwen3-0.6B \
         --tps 1,2,4,8 --out $OUT_DIR/tp_decode.json
"""
import argparse, json, os, time
import torch, torch_npu, torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

LAYERS, HIDDEN = 28, 1024
Q_HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128


def worker(rank, world, model_dir, out_q, port, iters=50, warmup=10):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if world > 1:
        dist.init_process_group("hccl", rank=rank, world_size=world)
    torch_npu.npu.set_device(rank)
    dev = f"npu:{rank}"

    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    # per-rank shard of one layer's projections (row-parallel on out-dim)
    shard = HIDDEN // world
    w_qkv = sd["model.layers.0.self_attn.q_proj.weight"].to(dev).to(torch.bfloat16)
    w_o = sd["model.layers.0.self_attn.o_proj.weight"].to(dev).to(torch.bfloat16)
    w_gate = sd["model.layers.0.mlp.gate_proj.weight"].to(dev).to(torch.bfloat16)
    w_down = sd["model.layers.0.mlp.down_proj.weight"].to(dev).to(torch.bfloat16)
    # shard: out-dim split for qkv/gate, in-dim split for o/down
    qkv_s = w_qkv[rank * shard:(rank + 1) * shard].contiguous()
    o_s = w_o[:, rank * shard:(rank + 1) * shard].contiguous()
    gate_s = w_gate[rank * shard:(rank + 1) * shard].contiguous()
    down_s = w_down[:, rank * shard:(rank + 1) * shard].contiguous()

    x = torch.randn(1, HIDDEN, device=dev, dtype=torch.bfloat16)

    def step():
        comm = 0.0
        t0 = time.perf_counter()
        # column-parallel qkv -> attention on local heads -> NO allreduce here
        h = x @ qkv_s.t()
        h = torch.nn.functional.silu(h) @ o_s.t()
        if world > 1:
            dist.all_reduce(h)
        g = torch.nn.functional.silu(x @ gate_s.t())
        y = g @ down_s.t()
        if world > 1:
            dist.all_reduce(y)
        torch.npu.synchronize()
        return time.perf_counter() - t0

    # pure-comm reference: same number of allreduces, no compute
    def comm_only():
        t0 = time.perf_counter()
        for _ in range(2):
            dist.all_reduce(x)
        torch.npu.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        step()
    t_step = []
    for _ in range(iters):
        t_step.append(step())
    t_step.sort()
    res = {"tp": world, "step_us": t_step[len(t_step) // 2] * 1e6}
    if world > 1:
        c = []
        for _ in range(20):
            c.append(comm_only())
        c.sort()
        res["comm2_us"] = c[len(c) // 2] * 1e6
        res["comm1_us"] = res["comm2_us"] / 2
    if rank == 0:
        out_q.put(res)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tps", default="1,2,4,8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port-base", type=int, default=29801)
    args = ap.parse_args()
    avail = torch_npu.npu.device_count()
    results = []
    for tp in [int(x) for x in args.tps.split(",")]:
        if tp > avail:
            continue
        os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = f"{61000 + tp * 100}-{61000 + tp * 100 + 99}"
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        ps = [ctx.Process(target=worker,
                          args=(r, tp, args.model, q, args.port_base + tp))
              for r in range(tp)]
        [p.start() for p in ps]
        try:
            results.append(q.get(timeout=1800))
        except Exception as e:
            results.append({"tp": tp, "error": f"{type(e).__name__}: {e}"})
        finally:
            for p in ps:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()
        print(f"[tpd] {results[-1]}", flush=True)
    base = next((r.get("step_us") for r in results if r.get("tp") == 1), None)
    for r in results:
        if base and "step_us" in r:
            r["vs_tp1"] = round(r["step_us"] / base, 3)
    with open(args.out, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"[tpd] wrote {args.out}")
