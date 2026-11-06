#!/usr/bin/env python3
"""E45 — Speculative decoding verification economics (measured).

Spec decode drafts k tokens then verifies them in ONE forward pass. The claim
is that verification is nearly free because attention is memory-bound on KV,
not compute-bound on query rows. This measures the real cost curve:

  per-layer forward vs number of query rows q in {1,2,4,8,16} at ctx=4096
  breakdown: attention (KV-bound, ~flat) vs MLP (linear in q)

Then: verification overhead = t(q=k)/t(q=1), and the implied speedup ceiling
= k / (t(k)/t(1)) if acceptance were 100%.

Usage: python3 spec_verify_economics.py --model DIR --out out.json
"""
import argparse, json, os, time
import torch, torch_npu
from safetensors.torch import load_file


def bench(fn, iters=20, warmup=6):
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
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--qs", default="1,2,4,8,16")
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
    S = args.ctx
    p = "model.layers.0."
    Wq = sd[p + "self_attn.q_proj.weight"].to(dev)
    Wo = sd[p + "self_attn.o_proj.weight"].to(dev)
    Wg = sd[p + "mlp.gate_proj.weight"].to(dev)
    Wu = sd[p + "mlp.up_proj.weight"].to(dev)
    Wd = sd[p + "mlp.down_proj.weight"].to(dev)
    res = {"ctx": S, "rows": []}
    k = torch.randn(nkv, 1, S, hd, device=dev, dtype=torch.bfloat16) * 0.02
    v = torch.randn(nkv, 1, S, hd, device=dev, dtype=torch.bfloat16) * 0.02
    base = None
    for q in [int(x) for x in args.qs.split(",")]:
        qq = torch.randn(nkv, G, q, hd, device=dev, dtype=torch.bfloat16) * 0.02
        x = torch.randn(q, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        t_attn = bench(lambda: torch.nn.functional.scaled_dot_product_attention(qq, k, v))
        def mlp():
            h = x @ Wg.t()
            g = torch.nn.functional.silu(h)
            u = x @ Wu.t()
            return (g * u) @ Wd.t()
        t_mlp = bench(mlp)
        def proj():
            h = x @ Wq.t()
            return h @ Wo.t()
        t_proj = bench(proj)
        t_layer = t_attn + t_mlp + t_proj
        if base is None:
            base = t_layer
        row = {"q": q, "attn_us": round(t_attn * 1e6, 1), "mlp_us": round(t_mlp * 1e6, 1),
               "proj_us": round(t_proj * 1e6, 1), "layer_us": round(t_layer * 1e6, 1),
               "verify_overhead": round(t_layer / base, 2),
               "speedup_ceiling_if_100pct_accept": round(q / (t_layer / base), 2)}
        res["rows"].append(row)
        with open(args.out + "l", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[E45] q={q:2d} attn={t_attn*1e6:7.1f}us mlp={t_mlp*1e6:7.1f}us "
              f"proj={t_proj*1e6:7.1f}us layer={t_layer*1e6:7.1f}us "
              f"overhead x{t_layer/base:4.2f} -> ceiling {q/(t_layer/base):5.2f}x", flush=True)
        del qq, x
        torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
