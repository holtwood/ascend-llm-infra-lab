#!/usr/bin/env python3
"""E8: handwritten attention/KV-cache micro-experiments on a single Ascend NPU.

Three parts — all pure PyTorch on torch_npu:
  A) naive MHA (hand-written per-head attention) vs fused
     torch.nn.functional.scaled_dot_product_attention
  B) decode-step cost scaling: attention over growing KV length
  C) decode memory-bound check: bytes-moved/time vs 910B HBM bandwidth

Run inside the pytorch container:
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 e8_micro.py --out /data1/pytorch-7/results_e8.json
"""
import argparse
import json
import math
import time

import torch
import torch_npu  # noqa: F401  (device init)
import torch.nn.functional as F

DEV = "npu:0"
DT = torch.bfloat16
H, D = 32, 128  # heads, head_dim (Qwen3-8B-ish shape)


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def naive_mha(q, k, v):
    B, h, S, d = q.shape
    w = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d), dim=-1)
    return w @ v


def part_a():
    out = {}
    for S in [512, 2048, 8192]:
        q = torch.randn(1, H, S, D, dtype=DT, device=DEV)
        k = torch.randn(1, H, S, D, dtype=DT, device=DEV)
        v = torch.randn(1, H, S, D, dtype=DT, device=DEV)
        t_naive = bench(lambda: naive_mha(q, k, v))
        t_fused = bench(lambda: F.scaled_dot_product_attention(q, k, v))
        out[f"S{S}"] = {"naive_ms": round(t_naive, 3), "fused_ms": round(t_fused, 3),
                        "speedup": round(t_naive / t_fused, 2)}
    return out


def part_b():
    # decode step: q=[1,H,1,D], KV length L grows
    out = {}
    q = torch.randn(1, H, 1, D, dtype=DT, device=DEV)
    for L in [512, 2048, 8192, 32768]:
        k = torch.randn(1, H, L, D, dtype=DT, device=DEV)
        v = torch.randn(1, H, L, D, dtype=DT, device=DEV)
        t = bench(lambda: F.scaled_dot_product_attention(q, k, v))
        kv_bytes = 2 * H * L * D * 2  # K+V bf16
        out[f"L{L}"] = {"ms": round(t, 4), "kv_GB": round(kv_bytes / 1e9, 3),
                        "eff_GBps": round(kv_bytes / 1e9 / (t / 1e3), 0)}
    return out


def part_c():
    # weight-read bound: decode matvec vs matmul(prefill)
    W = torch.randn(4096, 4096, dtype=DT, device=DEV)
    x1 = torch.randn(1, 4096, dtype=DT, device=DEV)
    xB = torch.randn(256, 4096, dtype=DT, device=DEV)
    t1 = bench(lambda: x1 @ W)
    tB = bench(lambda: xB @ W)
    wbytes = W.numel() * 2
    return {
        "decode_bs1_ms": round(t1, 4),
        "batch256_ms": round(tB, 4),
        "decode_bw_GBps": round(wbytes / 1e9 / (t1 / 1e3), 0),
        "batch_bw_GBps": round(wbytes / 1e9 / (tB / 1e3), 0),
        "note": "decode reads same weights 256x fewer times amortized -> memory bound",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_e8.json")
    args = ap.parse_args()
    res = {"device": torch.npu.get_device_name(0) if torch.npu.is_available() else "?",
           "A_mha_naive_vs_fused": part_a(),
           "B_decode_step_vs_kvlen": part_b(),
           "C_decode_memory_bound": part_c()}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
