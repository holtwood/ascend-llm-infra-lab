#!/usr/bin/env python3
"""E62 — GQA vs MHA vs MQA: what the KV-head ratio buys in serving.

Qwen3-0.6B uses hq=16, nkv=8 (GQA-2). Larger models use hq=32 with nkv=8
(GQA-4) or nkv=1 (MQA). The ratio decides KV memory AND attention cost, since
decode attention reads nkv*S*D per layer regardless of how many query heads
share it.

Measures at fixed hq=32, d=128, ctx in {8k}, batch in {1,8}:
  attention time, KV bytes per token, achieved bandwidth, and the
  memory/speed tradeoff vs full MHA.

Usage: python3 gqa_ratio.py --out out.json
"""
import argparse, json, time
import torch, torch_npu

HBM_GBPS = 1412.0


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
    ap.add_argument("--hq", type=int, default=32)
    ap.add_argument("--hd", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--nkv-list", default="32,8,4,1")
    ap.add_argument("--batches", default="1,8")
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    hq, hd, S = args.hq, args.hd, args.ctx
    res = {"hq": hq, "hd": hd, "ctx": S, "rows": []}
    base_time = None
    for nkv in [int(x) for x in args.nkv_list.split(",")]:
        G = hq // nkv
        for bs in [int(b) for b in args.batches.split(",")]:
            q = torch.randn(nkv * bs, G, 1, hd, device=dev, dtype=torch.bfloat16) * 0.02
            k = torch.randn(nkv * bs, 1, S, hd, device=dev, dtype=torch.bfloat16) * 0.02
            v = torch.randn(nkv * bs, 1, S, hd, device=dev, dtype=torch.bfloat16) * 0.02
            t = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
            kv_tok = 2 * nkv * hd * 2          # per layer, K+V
            kv_mb = bs * kv_tok * S / 1e6
            bw = kv_mb * 1e6 / t / 1e9
            if nkv == args.hq and bs == 1:
                base_time = t
            row = {"nkv": nkv, "group": G, "bs": bs,
                   "attn_us": round(t * 1e6, 1),
                   "kv_bytes_per_token_per_layer": kv_tok,
                   "kv_mb_read": round(kv_mb, 1),
                   "kv_gbps": round(bw, 1),
                   "pct_of_hbm": round(100 * bw / HBM_GBPS, 1),
                   "time_vs_mha": round(t / base_time, 2) if base_time else None}
            res["rows"].append(row)
            with open(args.out + "l", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[E62] nkv={nkv:2d} (G={G:2d}) bs={bs:2d} attn={t*1e6:7.1f}us "
                  f"KV={kv_tok:6d}B/tok  {bw:6.1f} GB/s  "
                  f"vs MHA {row['time_vs_mha'] if row['time_vs_mha'] else 1.0:4.2f}x",
                  flush=True)
            del q, k, v
            torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
