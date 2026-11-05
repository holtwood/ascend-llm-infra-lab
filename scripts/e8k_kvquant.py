#!/usr/bin/env python3
"""E8k: INT8-quantized KV cache — bandwidth saved vs accuracy lost.

Decode-step attention (q=[1,H,1,D]) where K/V are stored as int8
(per-out-channel scales, symmetric) vs bf16 baseline.
Measures:
  - decode step ms and effective KV-read bandwidth for bf16 vs int8 paths
    (int8 path: dequant on the fly — models a fused KV-quant kernel's input)
  - attention output max-diff int8 vs bf16 (accuracy cost)
  - per-channel vs per-token(group-of-64) scale granularity

KV len sweep: 8k / 32k / 64k.
"""
import json, math, time
import torch, torch_npu
import torch.nn.functional as F

DEV = "npu:0"
DT = torch.bfloat16
H, D = 32, 128


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def quant_int8(x, group=0):
    """Symmetric int8. group=0 -> per-channel (dim0 heads); group=g -> per-g-token groups."""
    if group == 0:
        scale = x.abs().amax(dim=2, keepdim=True).clamp_min(1e-6) / 127.0
        q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
        return q, scale
    # per-token-group: x [1,H,L,D] -> groups along L
    B, h, L, d = x.shape
    xg = x.reshape(B, h, L // group, group, d)
    scale = xg.abs().amax(dim=(3, 4), keepdim=True).clamp_min(1e-6) / 127.0
    q = torch.round(xg / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequant(q, scale, group=0):
    if group == 0:
        return (q.float() * scale.float()).to(DT)
    B, h, ng, g, d = q.shape
    return (q.float() * scale.float()).reshape(B, h, ng * g, d).to(DT)


def main():
    q = torch.randn(1, H, 1, D, dtype=DT, device=DEV)
    res = {"device": torch.npu.get_device_name(0), "rows": []}
    for L in [8192, 32768, 65536]:
        k = torch.randn(1, H, L, D, dtype=DT, device=DEV)
        v = torch.randn(1, H, L, D, dtype=DT, device=DEV)
        ref = F.scaled_dot_product_attention(q, k, v)
        t_bf16 = bench(lambda: F.scaled_dot_product_attention(q, k, v))
        kv_bytes = 2 * H * L * D * 2
        row = {"L": L, "bf16_ms": round(t_bf16, 4),
               "bf16_GBps": round(kv_bytes / 1e9 / (t_bf16 / 1e3), 0)}
        for grp in [0, 64]:
            qk, sk = quant_int8(k, grp)
            qv, sv = quant_int8(v, grp)
            def deq_attn():
                kd = dequant(qk, sk, grp)
                vd = dequant(qv, sv, grp)
                return F.scaled_dot_product_attention(q, kd, vd)
            t_q = bench(deq_attn)
            out_q = deq_attn()
            diff = (out_q - ref).abs().max().item()
            i8_bytes = kv_bytes // 2 + (sk.numel() + sv.numel()) * 2
            row[f"i8_g{grp}"] = {"ms": round(t_q, 4),
                                "kv_MB": round(i8_bytes / 1e6, 1),
                                "vs_bf16_MB": round(kv_bytes / 1e6, 1),
                                "max_diff": round(diff, 4),
                                "slow": round(t_q / t_bf16, 2)}
            del qk, sk, qv, sv
        res["rows"].append(row)
        print(f"[e8k] {row}", flush=True)
        del k, v, ref
        torch.npu.empty_cache()
    print(json.dumps(res, indent=1))
    with open("/tmp/e8k_kvquant.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
