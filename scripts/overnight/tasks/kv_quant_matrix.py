#!/usr/bin/env python3
"""int8-KV accuracy/cost matrix — extends E8k across S x B grid.

For each (seq_len, batch): build bf16 KV (GQA nkv=8, d=128), quantize
per-head int8, measure (a) max/mean attn-output error vs bf16 SDPA,
(b) naive dequantize-then-SDPA latency tax vs plain bf16 SDPA.

Usage: python3 kv_quant_matrix.py --out $OUT_DIR/kv_quant_matrix.json
"""
import argparse, json, time
import torch, torch_npu

NKV, D = 8, 128
NQ = 32  # query heads


def sdpa(q, k, v):
    # q: [B,NQ,S,1,D] decode step (Sq=1); expand KV heads for GQA
    k = k.repeat_interleave(NQ // NKV, dim=1)
    v = v.repeat_interleave(NQ // NKV, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))


def quant_i8(t):
    scale = t.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = scale.clamp_min(1e-8)
    return (t / scale).round().clamp(-127, 127).to(torch.int8), scale


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-kv-gb", type=float, default=6.0)
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    results = []
    for S in [1024, 2048, 4096, 8192, 16384]:
        for B in [1, 8, 32, 64]:
            kv_gb = 2 * B * S * NKV * D * 2 / 1e9
            if kv_gb > args.max_kv_gb:
                continue
            q = torch.randn(B, NQ, 1, D, device="npu:0", dtype=torch.bfloat16)
            k = torch.randn(B, NKV, S, D, device="npu:0", dtype=torch.bfloat16)
            v = torch.randn(B, NKV, S, D, device="npu:0", dtype=torch.bfloat16)
            ref = sdpa(q, k, v).float()
            kq, ks = quant_i8(k)
            vq, vs = quant_i8(v)
            mem_ratio = (kq.numel() + ks.numel() * 2 + vq.numel() + vs.numel() * 2) / (
                (k.numel() + v.numel()) * 2)  # int8+bf16scale bytes / bf16 bytes
            def dequant_sdpa():
                kd = kq.to(torch.bfloat16) * ks
                vd = vq.to(torch.bfloat16) * vs
                return sdpa(q, kd, vd)
            out = dequant_sdpa().float()
            err = (out - ref).abs()
            t_ref = bench(lambda: sdpa(q, k, v))
            t_naive = bench(dequant_sdpa)
            results.append({
                "S": S, "B": B, "kv_gb": round(kv_gb, 3),
                "mem_ratio": round(mem_ratio, 3),
                "err_max": err.max().item(), "err_mean": err.mean().item(),
                "t_ref_us": round(t_ref, 1), "t_naive_us": round(t_naive, 1),
                "naive_tax": round(t_naive / t_ref, 2)})
            print(f"[kvq] S={S} B={B} err_max={err.max().item():.4f} tax={t_naive/t_ref:.2f}x", flush=True)
            del q, k, v, kq, ks, vq, vs, ref, out
            torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"[kvq] wrote {args.out}")
