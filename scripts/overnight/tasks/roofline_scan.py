"""E68: roofline scan — measured TFLOPs vs arithmetic intensity on 910B1.

GEMM [M,K]x[K,N]: FLOPs=2*M*N*K, bytes≈2*(M*K+K*N+M*N) for bf16.
Sweep M (the decode/prefill axis) at fixed K,N matching Qwen3-0.6B shapes,
plus square-GEMM sweep to find the compute-bound knee.
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def gemm_case(m, n, k, dev):
    a = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    b = torch.randn(k, n, dtype=torch.bfloat16, device=dev)
    ms = bench(lambda: a @ b) * 1e3
    flops = 2 * m * n * k
    bts = 2 * (m * k + k * n + m * n)
    return {
        "m": m, "n": n, "k": k,
        "ms": round(ms, 4),
        "tflops": round(flops / (ms / 1e3) / 1e12, 2),
        "gbs": round(bts / (ms / 1e3) / 1e9, 1),
        "ai_flop_per_byte": round(flops / bts, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "npu:0"
    res = {"qwen3_shapes": [], "square_sweep": []}

    # decode axis: M = batch tokens; K,N from Qwen3-0.6B (hidden 1024, ffn 3072)
    for (n, k, tag) in [(1024, 1024, "proj"), (3072, 1024, "ffn_up"), (1024, 3072, "ffn_dn")]:
        for m in [1, 8, 32, 128, 512, 2048, 8192]:
            r = gemm_case(m, n, k, dev)
            r["shape"] = tag
            res["qwen3_shapes"].append(r)
            print(f"[E68] {tag:7s} m={m:<5} {r['ms']:8.3f}ms "
                  f"{r['tflops']:7.2f}TF {r['gbs']:7.0f}GB/s ai={r['ai_flop_per_byte']}",
                  flush=True)

    for s in [256, 512, 1024, 2048, 4096, 8192]:
        r = gemm_case(s, s, s, dev)
        res["square_sweep"].append(r)
        print(f"[E68] sq s={s:<5} {r['ms']:8.3f}ms {r['tflops']:7.2f}TF "
              f"ai={r['ai_flop_per_byte']}", flush=True)

    res["peak_note"] = "910B1 bf16 spec ~320 TFLOPS dense; HBM ~1.6TB/s"
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[E68] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
