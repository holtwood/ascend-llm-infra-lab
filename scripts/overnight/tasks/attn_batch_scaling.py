#!/usr/bin/env python3
"""E39 — Batch scaling of decode attention at long context.

E37 showed long-context decode attention reads KV at only ~265 GB/s (19% of
HBM peak) at batch=1. Batching amortizes nothing on the KV side (each sequence
reads its own KV), but it does give the kernel more parallelism — so KV
bandwidth should climb with batch until it saturates.

Measures per-layer decode attention at ctx in {8k, 32k} x batch in {1,4,16,64}:
  - time, KV read bandwidth, and the batch at which bandwidth saturates
  - implied max concurrent long-context decodes before becoming compute-bound

Usage: python3 attn_batch_scaling.py --out out.json
"""
import argparse, json, time
import torch, torch_npu

HBM_GBPS = 1400.0


def bench(fn, iters=15, warmup=4):
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
    ap.add_argument("--ctxs", default="8192,32768")
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--nkv", type=int, default=8)
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--hd", type=int, default=128)
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    nkv, hq, hd = args.nkv, args.hq, args.hd
    G = hq // nkv
    res = {"rows": []}
    for S in [int(x) for x in args.ctxs.split(",")]:
        for bs in [int(b) for b in args.batches.split(",")]:
            try:
                q = torch.randn(bs, nkv, G, 1, hd, device=dev, dtype=torch.bfloat16) * 0.02
                k = torch.randn(bs, nkv, S, hd, device=dev, dtype=torch.bfloat16) * 0.02
                v = torch.randn(bs, nkv, S, hd, device=dev, dtype=torch.bfloat16) * 0.02
            except Exception as e:
                res["rows"].append({"ctx": S, "bs": bs, "error": f"alloc: {e}"})
                print(f"[E39] ctx={S} bs={bs} alloc failed", flush=True)
                continue
            qq = q.reshape(bs * nkv, G, 1, hd)
            kk = k.reshape(bs * nkv, 1, S, hd)
            vv = v.reshape(bs * nkv, 1, S, hd)
            t = bench(lambda: torch.nn.functional.scaled_dot_product_attention(qq, kk, vv))
            kv_mb = bs * 2 * nkv * S * hd * 2 / 1e6
            bw = kv_mb * 1e6 / t / 1e9
            row = {"ctx": S, "bs": bs, "kv_mb": round(kv_mb, 1),
                   "attn_us": round(t * 1e6, 1),
                   "kv_gbps": round(bw, 1),
                   "pct_of_hbm": round(100 * bw / HBM_GBPS, 1),
                   "us_per_seq": round(t * 1e6 / bs, 1)}
            res["rows"].append(row)
            with open(args.out + "l", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[E39] ctx={S:6d} bs={bs:3d} KV={kv_mb:8.1f}MB attn={t*1e6:9.1f}us "
                  f"{bw:7.1f} GB/s ({100*bw/HBM_GBPS:4.1f}% peak) {t*1e6/bs:8.1f}us/seq",
                  flush=True)
            del q, k, v
            torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
