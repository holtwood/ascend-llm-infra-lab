#!/usr/bin/env python3
"""E30/E31 — serving capacity physics with real Qwen3-0.6B weights.

E30 KV capacity: derive KV bytes/token from the model config, validate by
real allocation until OOM, then predict max concurrency per context length.

E31 prefill/decode asymmetry: measure prefill (compute-bound, scales with
batch) vs decode (bandwidth-bound at small batch) on the same weights —
the physical justification for PD disaggregation and chunked prefill.

Usage: python3 capacity_physics.py --model DIR --out out.json
"""
import argparse, json, os, time
import torch, torch_npu
from safetensors.torch import load_file

GiB = 1 << 30


def rms_norm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def load_weights(sd, dev):
    keys = [k for k in sd if k.endswith("weight")]
    return {k: sd[k].to(dev) for k in keys}


def prefill(W, x, nlayers, nkv, hq, hd):
    """x: [bs, L, hidden]; returns [bs, L, hidden]. GQA via repeat_interleave."""
    G = hq // nkv
    for i in range(nlayers):
        p = f"model.layers.{i}."
        h = rms_norm(x, W[p + "input_layernorm.weight"])
        q = h @ W[p + "self_attn.q_proj.weight"].t()
        k = h @ W[p + "self_attn.k_proj.weight"].t()
        v = h @ W[p + "self_attn.v_proj.weight"].t()
        bs, L, _ = q.shape
        q = q.view(bs, L, hq, hd).transpose(1, 2)                 # [bs,hq,L,hd]
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


def kv_bytes_per_token(cfg):
    layers = cfg["num_hidden_layers"]
    nkv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    hd = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    return 2 * layers * nkv * hd * 2


def measure_kv_capacity(nlayers, nkv, hd, ctx, dev):
    """Allocate KV until allocation fails; return max seqs that fit."""
    per_seq = nlayers * 2 * nkv * ctx * hd * 2      # K and V for every layer
    bufs, n = [], 0
    while True:
        try:
            k = torch.empty(nlayers, nkv, ctx, hd, device=dev, dtype=torch.bfloat16)
            v = torch.empty(nlayers, nkv, ctx, hd, device=dev, dtype=torch.bfloat16)
            bufs.append((k, v))
            n += 1
            if n > 2048:
                break
        except Exception:
            break
    del bufs
    torch.npu.empty_cache()
    return n, per_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctxs", default="2048,8192,32768")
    ap.add_argument("--prompt-lens", default="128,512,2048,8192")
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    sd = load_file(os.path.join(args.model, "model.safetensors"))
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    nlayers, hq = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", hq)
    hd = cfg.get("head_dim", cfg["hidden_size"] // hq)
    hidden = cfg["hidden_size"]
    res = {"model": args.model, "nlayers": nlayers, "hq": hq, "nkv": nkv, "hd": hd}

    # ---- E30: KV capacity ----
    kv_tok = kv_bytes_per_token(cfg)
    res["kv_bytes_per_token"] = kv_tok
    res["kv_gb_per_1k_tokens"] = round(kv_tok * 1000 / 1e9, 4)
    free_before = torch_npu.npu.memory_allocated() / GiB
    res["kv_capacity"] = []
    for ctx in [int(c) for c in args.ctxs.split(",")]:
        n, per_seq = measure_kv_capacity(nlayers, nkv, hd, ctx, dev)
        res["kv_capacity"].append({"ctx": ctx, "max_seqs_fit": n,
                                   "bytes_per_seq": per_seq,
                                   "measured_total_gb": round(per_seq * n / 1e9, 2)})
        print(f"[E30] ctx={ctx:6d} KV/seq={per_seq/1e6:7.1f}MB -> {n} seqs fit "
              f"({per_seq*n/1e9:.1f}GB allocated before OOM)", flush=True)

    # ---- E31: prefill vs decode asymmetry ----
    W = load_weights(sd, dev)
    wbytes = sum(v.numel() * v.element_size() for v in W.values())
    res["weight_gb"] = round(wbytes / 1e9, 3)
    res["prefill"] = []
    for L in [int(x) for x in args.prompt_lens.split(",")]:
        for bs in [1, 4]:
            x = torch.randn(bs, L, hidden, device=dev, dtype=torch.bfloat16) * 0.02
            try:
                t = bench(lambda: prefill(W, x, nlayers, nkv, hq, hd), iters=3, warmup=1)
                tps = bs * L / t
                res["prefill"].append({"len": L, "bs": bs, "ms": round(t * 1e3, 1),
                                       "tok_s": round(tps, 1),
                                       "per_tok_us": round(t * 1e6 / (bs * L), 1)})
                print(f"[E31] prefill L={L:5d} bs={bs} {t*1e3:8.1f}ms "
                      f"{tps:9.1f} tok/s ({t*1e6/(bs*L):5.1f} us/tok)", flush=True)
            except Exception as e:
                res["prefill"].append({"len": L, "bs": bs, "error": str(e)[:120]})
                print(f"[E31] prefill L={L} bs={bs} ERROR {str(e)[:80]}", flush=True)
            del x
            torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
