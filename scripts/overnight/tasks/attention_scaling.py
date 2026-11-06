#!/usr/bin/env python3
"""E37 — When does long-context attention overtake the weight read in decode?

At batch=1 the decode step cost = weight read (fixed, bandwidth-bound) +
attention over KV (grows with context). This measures the attention term per
layer across context lengths and finds the crossover where attention becomes
the dominant cost — i.e. the context length at which "decode is memory-bound
on weights" stops being true.

Per-layer weight bytes for Qwen3-0.6B ≈ 1.503GB/28 ≈ 53.7MB
-> weight-read time ≈ 53.7MB / 1400GB/s ≈ 38 µs (measured 1412 GB/s, E29).

Usage: python3 attention_scaling.py --out out.json
"""
import argparse, json, time
import torch, torch_npu

HBM_GBPS = 1400.0


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctxs", default="1024,4096,16384,32768,65536,131072")
    ap.add_argument("--nkv", type=int, default=8)
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--hd", type=int, default=128)
    ap.add_argument("--layer-weight-mb", type=float, default=53.7)
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    nkv, hq, hd = args.nkv, args.hq, args.hd
    G = hq // nkv
    wread_us = args.layer_weight_mb * 1e6 / HBM_GBPS / 1e3   # MB -> us
    res = {"layer_weight_mb": args.layer_weight_mb,
           "weight_read_us": round(wread_us, 1), "rows": []}
    q = torch.randn(1, hq, 1, hd, device=dev, dtype=torch.bfloat16) * 0.02
    for S in [int(x) for x in args.ctxs.split(",")]:
        try:
            k = torch.randn(1, nkv, S, hd, device=dev, dtype=torch.bfloat16) * 0.02
            v = torch.randn(1, nkv, S, hd, device=dev, dtype=torch.bfloat16) * 0.02
        except Exception as e:
            res["rows"].append({"ctx": S, "error": f"alloc: {e}"})
            print(f"[E37] ctx={S} alloc failed", flush=True)
            continue
        # GQA: query heads grouped per kv head; batch the kv heads (no expansion)
        qq = q.view(1, nkv, G, 1, hd).reshape(nkv, G, 1, hd)
        kk = k.view(1, nkv, S, hd).reshape(nkv, 1, S, hd)
        vv = v.view(1, nkv, S, hd).reshape(nkv, 1, S, hd)
        t = bench(lambda: torch.nn.functional.scaled_dot_product_attention(qq, kk, vv))
        kv_mb = 2 * nkv * S * hd * 2 / 1e6
        row = {"ctx": S, "kv_mb_per_layer": round(kv_mb, 1),
               "attn_us": round(t * 1e6, 1),
               "weight_read_us": round(wread_us, 1),
               "attn_over_weight": round(t * 1e6 / wread_us, 2),
               "kv_bw_gbps": round(kv_mb * 1e6 / t / 1e9, 1)}
        res["rows"].append(row)
        with open(args.out + "l", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[E37] ctx={S:7d} KV/layer={kv_mb:7.1f}MB attn={t*1e6:8.1f}us "
              f"({t*1e6/wread_us:5.2f}x weight-read, {row['kv_bw_gbps']:6.1f} GB/s)", flush=True)
        del k, v
        torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
