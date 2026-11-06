"""E70: KV block swap timing — the offload decision boundary.

vLLM swaps paged KV to pinned CPU memory when preempting. Measure at real
KV-block granularity for the 0.6B GQA layout (kv/token = 114688 B):

  - D2H and H2D, pageable vs pinned
  - per-block (128 tokens) and per-sequence (2k/8k/32k) latency
  - effective GB/s vs the E59/copy_bw bulk numbers
  - decision: ms to swap out + swap in vs ms to recompute prefill (E31/E48)
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401

KV_PER_TOKEN = 114688  # bytes, E30 measured (K+V, 28 layers, 8 kv heads, hd=128)
BLOCK = 128


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def xfer_case(tokens, dev, pinned):
    bts = tokens * KV_PER_TOKEN
    src = torch.empty(bts, dtype=torch.uint8, device=dev)
    dst = torch.empty(bts, dtype=torch.uint8, pin_memory=pinned)
    d2h = bench(lambda: dst.copy_(src, non_blocking=True))
    del dst
    dst = torch.empty(bts, dtype=torch.uint8, pin_memory=pinned)
    h2d = bench(lambda: src.copy_(dst, non_blocking=True))
    return {
        "tokens": tokens, "mb": round(bts / 1e6, 1),
        "d2h_ms": round(d2h, 3), "h2d_ms": round(h2d, 3),
        "d2h_gbs": round(bts / d2h / 1e6, 1), "h2d_gbs": round(bts / h2d / 1e6, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "npu:0"
    res = {}
    for pinned in [False, True]:
        tag = "pinned" if pinned else "pageable"
        res[tag] = []
        for tok in [BLOCK, 2048, 8192, 32768]:
            r = xfer_case(tok, dev, pinned)
            res[tag].append(r)
            print(f"[E70] {tag:8s} tok={tok:<6} ({r['mb']:6.0f}MB) "
                  f"d2h={r['d2h_ms']:8.2f}ms h2d={r['h2d_ms']:8.2f}ms "
                  f"({r['d2h_gbs']:.1f}/{r['h2d_gbs']:.1f} GB/s)", flush=True)
            torch.npu.empty_cache()

    # per-block latency model: N small blocks vs one big transfer
    blk_bytes = BLOCK * KV_PER_TOKEN
    src = torch.empty(blk_bytes, dtype=torch.uint8, device=dev)
    dst = torch.empty(blk_bytes, dtype=torch.uint8, pin_memory=True)
    res["per_block_ms_pinned"] = round(bench(lambda: dst.copy_(src, non_blocking=True)), 4)
    # 64 sequential block swaps (a 8k seq evicted block-by-block)
    def multi():
        for _ in range(64):
            dst.copy_(src, non_blocking=True)
        torch.npu.synchronize()
    res["64_block_seq_ms"] = round(bench(multi, iters=5, warmup=2), 2)
    res["note"] = (f"kv/token={KV_PER_TOKEN}B (E30). Compare with recompute "
                   f"cost E48: prefill_ms(L)=~0.0112*L+4.66e-8*L^2 per layer*28")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[E70] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
