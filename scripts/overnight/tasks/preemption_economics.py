#!/usr/bin/env python3
"""E32 — Preemption economics: recompute vs swap, with real weights and KV.

vLLM preempts a sequence either by recomputing (re-prefill) or by swapping KV
to host. Which is cheaper depends on context length:
  recompute(L) ~ prefill(L)          (compute)
  swap(L)      ~ 2 * kv_bytes(L) / D2H_bandwidth   (host link)

This script measures both on real Qwen3-0.6B weights + real KV tensors and
reports the crossover length, plus the marginal per-token prefill cost
(launch floor removed) so the comparison is engine-realistic.

Usage: python3 preemption_economics.py --model DIR --out out.json
"""
import argparse, json, os, time
import torch, torch_npu
from safetensors.torch import load_file


def rms_norm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def bench(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def prefill(W, x, nlayers, nkv, hq, hd):
    G = hq // nkv
    for i in range(nlayers):
        p = f"model.layers.{i}."
        h = rms_norm(x, W[p + "input_layernorm.weight"])
        q = h @ W[p + "self_attn.q_proj.weight"].t()
        k = h @ W[p + "self_attn.k_proj.weight"].t()
        v = h @ W[p + "self_attn.v_proj.weight"].t()
        bs, L, _ = q.shape
        q = q.view(bs, L, hq, hd).transpose(1, 2)
        k = k.view(bs, L, nkv, hd).transpose(1, 2).repeat_interleave(G, dim=1)
        v = v.view(bs, L, nkv, hd).transpose(1, 2).repeat_interleave(G, dim=1)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(bs, L, hq * hd)
        x = x + o @ W[p + "self_attn.o_proj.weight"].t()
        h2 = rms_norm(x, W[p + "post_attention_layernorm.weight"])
        g = torch.nn.functional.silu(h2 @ W[p + "mlp.gate_proj.weight"].t())
        u = h2 @ W[p + "mlp.up_proj.weight"].t()
        x = x + (g * u) @ W[p + "mlp.down_proj.weight"].t()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lens", default="1024,4096,16384,32768")
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    sd = load_file(os.path.join(args.model, "model.safetensors"))
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    nlayers, hq = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", hq)
    hd = cfg.get("head_dim", cfg["hidden_size"] // hq)
    hidden = cfg["hidden_size"]
    kv_tok = 2 * nlayers * nkv * hd * 2
    W = {k: sd[k].to(dev) for k in sd if k.endswith("weight")}
    res = {"kv_bytes_per_token": kv_tok, "rows": []}
    for L in [int(x) for x in args.lens.split(",")]:
        x = torch.randn(1, L, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        t_pre = bench(lambda: prefill(W, x, nlayers, nkv, hq, hd), iters=3, warmup=1)
        del x
        torch.npu.empty_cache()
        # swap: KV block HBM -> pinned host -> HBM
        kv_bytes = kv_tok * L
        hb = torch.empty(kv_bytes // 2, dtype=torch.bfloat16, pin_memory=True)
        db = torch.empty(kv_bytes // 2, device=dev, dtype=torch.bfloat16)
        t_out = bench(lambda: hb.copy_(db, non_blocking=True), iters=5, warmup=2)
        t_in = bench(lambda: db.copy_(hb, non_blocking=True), iters=5, warmup=2)
        t_swap = t_out + t_in
        row = {"len": L, "kv_mb": round(kv_bytes / 1e6, 1),
               "recompute_ms": round(t_pre * 1e3, 1),
               "swap_ms": round(t_swap * 1e3, 1),
               "swap_d2h_gbps": round(kv_bytes / t_out / 1e9, 1),
               "swap_h2d_gbps": round(kv_bytes / t_in / 1e9, 1),
               "winner": "recompute" if t_pre < t_swap else "swap"}
        res["rows"].append(row)
        with open(args.out + "l", "a") as jf:      # incremental, survives timeout
            jf.write(json.dumps(row) + "\n")
        print(f"[E32] L={L:6d} KV={kv_bytes/1e6:7.1f}MB  recompute={t_pre*1e3:7.1f}ms  "
              f"swap={t_swap*1e3:7.1f}ms  -> {row['winner']}", flush=True)
        del hb, db
        torch.npu.empty_cache()
    # marginal prefill cost (remove launch floor) for an engine-realistic view
    base = min(r["recompute_ms"] for r in res["rows"] if r["len"] <= 1024) if res["rows"] else 0
    big = max(res["rows"], key=lambda r: r["len"])
    if base and big["len"] > 1024:
        marginal_us = (big["recompute_ms"] - base) * 1e3 / (big["len"] - 1024)
        res["marginal_prefill_us_per_token"] = round(marginal_us, 2)
        res["launch_floor_ms"] = round(base, 1)
        for r in res["rows"]:
            r["recompute_ms_no_floor"] = round(base + marginal_us * r["len"] / 1e3, 1)
            r["winner_no_floor"] = ("recompute"
                                    if r["recompute_ms_no_floor"] < r["swap_ms"] else "swap")
        print(f"[E32] marginal prefill {marginal_us:.2f} us/token, floor {base:.1f}ms",
              flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
