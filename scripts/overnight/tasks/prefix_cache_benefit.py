#!/usr/bin/env python3
"""E33 — Prefix cache (APC) benefit measured at the mechanism level.

Scenario: N requests share a prefix of length P and differ in a short suffix S.
  cold  : every request re-prefills P+S tokens
  warm  : prefix KV computed once, each request prefills only S tokens but
          attends over [prefix KV | suffix] (exactly what APC does)

Reports per-request TTFT (prefill time), total wall time for N requests, the
speedup, and the KV bytes pinned by the cached prefix.

Usage: python3 prefix_cache_benefit.py --model DIR --out out.json
"""
import argparse, json, os, time
import torch, torch_npu
from safetensors.torch import load_file


def rms_norm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def bench(fn, iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def layer_forward(W, i, x, nkv, hq, hd, prefix_kv=None):
    """One layer. If prefix_kv given, k/v are concatenated in front (APC path)."""
    p = f"model.layers.{i}."
    G = hq // nkv
    h = rms_norm(x, W[p + "input_layernorm.weight"])
    q = h @ W[p + "self_attn.q_proj.weight"].t()
    k = h @ W[p + "self_attn.k_proj.weight"].t()
    v = h @ W[p + "self_attn.v_proj.weight"].t()
    bs, L, _ = q.shape
    qh = q.view(bs, L, hq, hd).transpose(1, 2)
    kh = k.view(bs, L, nkv, hd).transpose(1, 2)
    vh = v.view(bs, L, nkv, hd).transpose(1, 2)
    if prefix_kv is not None:
        pk, pv = prefix_kv
        kh = torch.cat([pk, kh], dim=2)
        vh = torch.cat([pv, vh], dim=2)
    Ltot = kh.shape[2]
    kh = kh.repeat_interleave(G, dim=1)
    vh = vh.repeat_interleave(G, dim=1)
    if prefix_kv is None:
        o = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
    else:
        # suffix attends to all prefix + causal within suffix
        P = Ltot - L
        mask = torch.ones(L, Ltot, dtype=torch.bool, device=qh.device)
        mask[:, :P] = True
        mask[:, P:] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=qh.device))
        o = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
    o = o.transpose(1, 2).reshape(bs, L, hq * hd)
    x = x + o @ W[p + "self_attn.o_proj.weight"].t()
    h2 = rms_norm(x, W[p + "post_attention_layernorm.weight"])
    g = torch.nn.functional.silu(h2 @ W[p + "mlp.gate_proj.weight"].t())
    u = h2 @ W[p + "mlp.up_proj.weight"].t()
    x = x + (g * u) @ W[p + "mlp.down_proj.weight"].t()
    return x, (kh0, vh0)   # return un-expanded k/v for caching


def prefill(W, x, nlayers, nkv, hq, hd, cache_prefix=None):
    caches = []
    for i in range(nlayers):
        x, kv = layer_forward(W, i, x, nkv, hq, hd,
                              cache_prefix[i] if cache_prefix else None)
        caches.append(kv)
    return x, caches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefixes", default="512,2048,8192")
    ap.add_argument("--suffix", type=int, default=128)
    ap.add_argument("--reqs", type=int, default=8)
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    sd = load_file(os.path.join(args.model, "model.safetensors"))
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    nlayers, hq = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", hq)
    hd = cfg.get("head_dim", cfg["hidden_size"] // hq)
    hidden = cfg["hidden_size"]
    W = {k: sd[k].to(dev) for k in sd if k.endswith("weight")}
    kv_tok = 2 * nlayers * nkv * hd * 2
    res = {"suffix": args.suffix, "reqs": args.reqs, "kv_bytes_per_token": kv_tok, "rows": []}
    for P in [int(x) for x in args.prefixes.split(",")]:
        S = args.suffix
        xf = torch.randn(1, P + S, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        t_cold = bench(lambda: prefill(W, xf, nlayers, nkv, hq, hd), iters=3, warmup=1)
        del xf
        torch.npu.empty_cache()
        xp = torch.randn(1, P, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        _, pfx = prefill(W, xp, nlayers, nkv, hq, hd)
        del xp
        torch.npu.empty_cache()
        xs = torch.randn(1, S, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        t_warm = bench(lambda: prefill(W, xs, nlayers, nkv, hq, hd, cache_prefix=pfx),
                       iters=3, warmup=1)
        row = {"prefix": P, "suffix": S,
               "cold_ms": round(t_cold * 1e3, 1), "warm_ms": round(t_warm * 1e3, 1),
               "speedup": round(t_cold / t_warm, 2),
               "cold_total_ms": round(t_cold * 1e3 * args.reqs, 1),
               "warm_total_ms": round((t_cold * 1e3 + t_warm * 1e3 * (args.reqs - 1)), 1),
               "total_speedup": round((t_cold * args.reqs) /
                                      (t_cold + t_warm * (args.reqs - 1)), 2),
               "prefix_kv_mb": round(kv_tok * P / 1e6, 1),
               "hit_ratio": round(P / (P + S), 3)}
        res["rows"].append(row)
        with open(args.out + "l", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[E33] P={P:5d} S={S} cold={t_cold*1e3:7.1f}ms warm={t_warm*1e3:6.1f}ms "
              f"per-req {t_cold/t_warm:5.2f}x total {row['total_speedup']:4.2f}x "
              f"(prefix KV {row['prefix_kv_mb']:.0f}MB)", flush=True)
        del xs, pfx
        torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
