#!/usr/bin/env python3
"""E34 — Chunked prefill: what does splitting a long prompt cost, and how
does chunk size bound per-step latency?

A long prompt can be prefilled in one shot (best total time, worst interference)
or in C-token chunks (each chunk is one scheduler step; decode steps from other
requests interleave between chunks).

Measures, for L=8192 and chunk sizes {8192,4096,2048,1024,512}:
  - total prefill time and overhead vs single-shot
  - per-chunk latency (= max time a chunk blocks other requests' decode)
  - implied "interference budget" for a decode step (compare with E29's ~29ms)

Usage: python3 chunked_prefill.py --model DIR --out out.json
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
    p = f"model.layers.{i}."
    G = hq // nkv
    h = rms_norm(x, W[p + "input_layernorm.weight"])
    q = h @ W[p + "self_attn.q_proj.weight"].t()
    k = h @ W[p + "self_attn.k_proj.weight"].t()
    v = h @ W[p + "self_attn.v_proj.weight"].t()
    bs, L, _ = q.shape
    qh = q.view(bs, L, hq, hd).transpose(1, 2)
    kh0 = k.view(bs, L, nkv, hd).transpose(1, 2)      # un-expanded, cacheable
    vh0 = v.view(bs, L, nkv, hd).transpose(1, 2)
    if prefix_kv is not None:
        kh0 = torch.cat([prefix_kv[0], kh0], dim=2)
        vh0 = torch.cat([prefix_kv[1], vh0], dim=2)
    Ltot = kh0.shape[2]
    kh = kh0.repeat_interleave(G, dim=1)
    vh = vh0.repeat_interleave(G, dim=1)
    if prefix_kv is None:
        o = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
    else:
        P = Ltot - L
        mask = torch.ones(L, Ltot, dtype=torch.bool, device=qh.device)
        mask[:, P:] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=qh.device))
        o = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
    o = o.transpose(1, 2).reshape(bs, L, hq * hd)
    x = x + o @ W[p + "self_attn.o_proj.weight"].t()
    h2 = rms_norm(x, W[p + "post_attention_layernorm.weight"])
    g = torch.nn.functional.silu(h2 @ W[p + "mlp.gate_proj.weight"].t())
    u = h2 @ W[p + "mlp.up_proj.weight"].t()
    x = x + (g * u) @ W[p + "mlp.down_proj.weight"].t()
    return x, (kh0, vh0)


def prefill_chunks(W, x, nlayers, nkv, hq, hd, chunk):
    """Prefill x [1, L, hidden] in `chunk`-token pieces, accumulating KV."""
    L = x.shape[1]
    caches = None
    per_chunk = []
    for s in range(0, L, chunk):
        xc = x[:, s:s + chunk]
        t0 = time.perf_counter()
        if caches is None:
            out, caches = [], None
        new_caches = []
        xc_in = xc
        for i in range(nlayers):
            xc_in, kv = layer_forward(W, i, xc_in, nkv, hq, hd,
                                      caches[i] if caches else None)
            new_caches.append(kv)
        torch.npu.synchronize()
        per_chunk.append(time.perf_counter() - t0)
        caches = new_caches
    return sum(per_chunk), per_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--len", type=int, default=8192)
    ap.add_argument("--chunks", default="8192,4096,2048,1024,512")
    ap.add_argument("--decode-step-ms", type=float, default=29.0)
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
    res = {"len": args.len, "rows": []}
    x = torch.randn(1, args.len, hidden, device=dev, dtype=torch.bfloat16) * 0.02
    prefill_chunks(W, x[:, :1024], nlayers, nkv, hq, hd, 1024)   # warmup: JIT/alloc
    torch.npu.synchronize()
    base = None
    for c in [int(v) for v in args.chunks.split(",")]:
        total, per = prefill_chunks(W, x, nlayers, nkv, hq, hd, c)
        if base is None:
            base = total
        row = {"chunk": c, "chunks": len(per),
               "total_ms": round(total * 1e3, 1),
               "overhead_vs_single": round(total / base, 2),
               "per_chunk_ms_mean": round(sum(per) / len(per) * 1e3, 1),
               "per_chunk_ms_max": round(max(per) * 1e3, 1),
               "decode_steps_blocked": round(sum(per) / (args.decode_step_ms / 1e3), 1)}
        res["rows"].append(row)
        with open(args.out + "l", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[E34] chunk={c:5d} x{len(per):2d} total={total*1e3:7.1f}ms "
              f"({total/base:4.2f}x) per-chunk mean={row['per_chunk_ms_mean']:6.1f}ms "
              f"max={row['per_chunk_ms_max']:6.1f}ms", flush=True)
        torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
