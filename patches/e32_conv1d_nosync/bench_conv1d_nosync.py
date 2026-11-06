#!/usr/bin/env python3
"""E32 benchmark: sync-free conv1d update vs the vllm-ascend PyTorch fallback.

The fallback calls ``conv_state_indices[i].item()`` per sequence, i.e. one host
read-back per request per layer. This script times both implementations on the
same device tensors for several batch sizes and reports ms/call, so the cost of
the sync is visible rather than asserted.

Run on an Ascend node:
    python3 patches/e32_conv1d_nosync/bench_conv1d_nosync.py [--iters 200]
"""
import argparse, os, sys, time

import torch

sys.path.insert(0, "/vllm-workspace/vllm-ascend")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conv1d_nosync import causal_conv1d_update_nosync  # noqa: E402
from vllm_ascend.ops.causal_conv1d import causal_conv1d_update as fallback  # noqa: E402

PAD = -1


def bench(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    dev = "npu:0" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    dim, width = args.dim, args.width
    torch.manual_seed(0)

    weight = (torch.randn(dim, width, device=dev, dtype=dtype) * 0.1)
    bias = (torch.randn(dim, device=dev, dtype=dtype) * 0.1)

    print(f"device={dev} dtype={args.dtype} dim={dim} width={width} iters={args.iters}")
    print(f"{'B':>4s} {'fallback ms':>12s} {'nosync ms':>10s} {'speedup':>8s}")
    for b in (1, 8, 32, 64, 128):
        slots = max(b * 2, 8)
        state = (torch.randn(slots, dim, width - 1, device=dev, dtype=dtype) * 0.1)
        x = (torch.randn(b, dim, device=dev, dtype=dtype) * 0.1)
        idx = torch.arange(b, device=dev, dtype=torch.int32)  # contiguous slots

        def run_fallback():
            fallback(x, state, weight, bias, "silu",
                     conv_state_indices=idx, pad_slot_id=PAD)

        def run_nosync():
            causal_conv1d_update_nosync(x, state, weight, bias, "silu",
                                        conv_state_indices=idx, pad_slot_id=PAD)

        tf = bench(run_fallback, args.iters)
        tn = bench(run_nosync, args.iters)
        print(f"{b:>4d} {tf:>12.3f} {tn:>10.3f} {tf / tn:>7.2f}x")


if __name__ == "__main__":
    main()
