#!/usr/bin/env python3
"""E48 — Prefill cost model: separate the linear and quadratic terms.

Prefill time should be  T(L) = a + b*L + c*L^2  per layer
  a: launch floor   b: linear part (projections, MLP)   c: attention (O(L^2))
Fitting this from measurements gives a predictive model for capacity planning
("what is TTFT for a 16k prompt?") and tells you which term to optimize.

Measures per-layer attention / MLP / projections at L in {1k..16k} and fits
b and c.

Usage: python3 prefill_cost_model.py --model DIR --out out.json
"""
import argparse, json, os, time
import torch, torch_npu
from safetensors.torch import load_file


def bench(fn, iters=8, warmup=2):
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lens", default="1024,2048,4096,8192,16384")
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    sd = load_file(os.path.join(args.model, "model.safetensors"))
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    nl, hq = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", hq)
    hd = cfg.get("head_dim", cfg["hidden_size"] // hq)
    hidden = cfg["hidden_size"]
    G = hq // nkv
    p = "model.layers.0."
    Wq = sd[p + "self_attn.q_proj.weight"].to(dev)
    Wo = sd[p + "self_attn.o_proj.weight"].to(dev)
    Wg = sd[p + "mlp.gate_proj.weight"].to(dev)
    Wu = sd[p + "mlp.up_proj.weight"].to(dev)
    Wd = sd[p + "mlp.down_proj.weight"].to(dev)
    res = {"nlayers": nl, "rows": []}
    for L in [int(x) for x in args.lens.split(",")]:
        x = torch.randn(L, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        q = torch.randn(nkv, G, L, hd, device=dev, dtype=torch.bfloat16) * 0.02
        k = torch.randn(nkv, 1, L, hd, device=dev, dtype=torch.bfloat16) * 0.02
        v = torch.randn(nkv, 1, L, hd, device=dev, dtype=torch.bfloat16) * 0.02
        t_attn = bench(lambda: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False))
        def proj():
            return (x @ Wq.t()) @ Wo.t()
        def mlp():
            g = torch.nn.functional.silu(x @ Wg.t())
            u = x @ Wu.t()
            return (g * u) @ Wd.t()
        t_proj, t_mlp = bench(proj), bench(mlp)
        row = {"L": L, "attn_ms": round(t_attn * 1e3, 2), "proj_ms": round(t_proj * 1e3, 2),
               "mlp_ms": round(t_mlp * 1e3, 2),
               "attn_us_per_tok": round(t_attn * 1e6 / L, 2),
               "mlp_us_per_tok": round(t_mlp * 1e6 / L, 2)}
        res["rows"].append(row)
        with open(args.out + "l", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[E48] L={L:6d} attn={t_attn*1e3:7.2f}ms ({row['attn_us_per_tok']:6.2f}us/tok) "
              f"mlp={t_mlp*1e3:6.2f}ms ({row['mlp_us_per_tok']:5.2f}us/tok) "
              f"proj={t_proj*1e3:5.2f}ms", flush=True)
        del x, q, k, v
        torch.npu.empty_cache()
    # fit attn = c*L^2 (+a) using the two largest lengths, report c
    rows = res["rows"]
    if len(rows) >= 2:
        L1, L2 = rows[-2]["L"], rows[-1]["L"]
        t1, t2 = rows[-2]["attn_ms"], rows[-1]["attn_ms"]
        c = (t2 - t1) / (L2 ** 2 - L1 ** 2)          # ms per token^2
        res["quadratic_coef_ms_per_tok2"] = c
        res["attn_share_at_16k"] = round(
            c * 16384 ** 2 / (c * 16384 ** 2 + rows[-1]["mlp_ms"] + rows[-1]["proj_ms"]), 3)
        print(f"[E48] quadratic coef {c:.3e} ms/tok^2; attention share at 16k "
              f"= {res['attn_share_at_16k']:.0%}", flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
