#!/usr/bin/env python3
"""E44 — Prefill/decode interference measured on hardware (not simulated).

In a mixed batch, a prefill chunk shares SMs and HBM with the decode steps
already running. This measures how much a concurrent prefill inflates decode
attention latency, and how much the decode batch slows the prefill — the
quantity that motivates chunked prefill and PD disaggregation.

Method: per-layer attention only (removes the 29ms launch floor of E29 so the
contention is visible).
  decode alone : bs_dec sequences, ctx=4096, 1 query each
  prefill alone: chunk of P tokens (causal)
  mixed        : both in one step

Usage: python3 pd_interference.py --out out.json
"""
import argparse, json, time
import torch, torch_npu


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
    ap.add_argument("--nkv", type=int, default=8)
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--hd", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--bs-decs", default="1,8")
    ap.add_argument("--chunks", default="512,2048,8192")
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    nkv, hq, hd, S = args.nkv, args.hq, args.hd, args.ctx
    G = hq // nkv
    res = {"ctx": S, "rows": []}

    def attn(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    for bs in [int(b) for b in args.bs_decs.split(",")]:
        qd = torch.randn(bs * nkv, G, 1, hd, device=dev, dtype=torch.bfloat16) * 0.02
        kd = torch.randn(bs * nkv, 1, S, hd, device=dev, dtype=torch.bfloat16) * 0.02
        vd = torch.randn(bs * nkv, 1, S, hd, device=dev, dtype=torch.bfloat16) * 0.02
        t_dec = bench(lambda: attn(qd, kd, vd))
        for P in [int(c) for c in args.chunks.split(",")]:
            qp = torch.randn(nkv, G, P, hd, device=dev, dtype=torch.bfloat16) * 0.02
            kp = torch.randn(nkv, 1, P, hd, device=dev, dtype=torch.bfloat16) * 0.02
            vp = torch.randn(nkv, 1, P, hd, device=dev, dtype=torch.bfloat16) * 0.02
            t_pre = bench(lambda: attn(qp, kp, vp))
            def mixed():                      # same stream: serialized (HOL blocking)
                attn(qd, kd, vd)
                attn(qp, kp, vp)
            t_mix = bench(mixed)
            s2 = torch.npu.Stream()
            def concurrent():                 # two streams: real SM/HBM contention
                with torch.npu.stream(s2):
                    attn(qp, kp, vp)
                attn(qd, kd, vd)
                torch.npu.current_stream().wait_stream(s2)
            t_conc = bench(concurrent)
            row = {"decode_bs": bs, "prefill_chunk": P,
                   "decode_us": round(t_dec * 1e6, 1),
                   "prefill_us": round(t_pre * 1e6, 1),
                   "sum_us": round((t_dec + t_pre) * 1e6, 1),
                   "mixed_us": round(t_mix * 1e6, 1),
                   "decode_inflation": round(t_mix / t_dec, 2),
                   "prefill_inflation": round(t_mix / t_pre, 2),
                   "mixed_concurrent_us": round(t_conc * 1e6, 1),
                   "decode_inflation_concurrent": round(t_conc / t_dec, 2)}
            res["rows"].append(row)
            with open(args.out + "l", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[E44] dec_bs={bs:2d} chunk={P:5d} decode={t_dec*1e6:7.1f}us "
                  f"prefill={t_pre*1e6:7.1f}us mixed={t_mix*1e6:7.1f}us "
                  f"-> decode x{t_mix/t_dec:4.2f}, prefill x{t_mix/t_pre:4.2f} "
                  f"| 2-stream decode x{t_conc/t_dec:4.2f}", flush=True)
            del qp, kp, vp
            torch.npu.empty_cache()
        del qd, kd, vd
        torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
