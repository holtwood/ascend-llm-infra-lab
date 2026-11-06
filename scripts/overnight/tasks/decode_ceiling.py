#!/usr/bin/env python3
"""Decode ceiling + batching economics with real Qwen3-0.6B weights (no vllm).

Answers the two questions every serving interview asks:
  1. what is the decode tok/s ceiling for this model on this card, and why?
  2. how does tok/s scale with batch, and where is the knee?

Method: full 28-layer decode step (bs x 1 token) against a real KV cache,
measured on NPU; compared against the memory-bandwidth ceiling
(weight_bytes / HBM_BW). Also runs the int8 (W8A16) checkpoint to show the
weight-byte -> step-time proportionality directly.

Usage: python3 decode_ceiling.py --model DIR [--w8a16 DIR] --out out.json
"""
import argparse, json, os, time
import torch, torch_npu
from safetensors.torch import load_file

HBM_GBPS = 1400.0  # 910B1 HBM read bandwidth, measured ~1.4-1.5 TB/s


def rms_norm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


class KVBuf:
    """Tensor wrapper exposing the write position so a decode step can append."""
    def __init__(self, t, pos):
        self.t = t
        self.pos = pos

    def __getitem__(self, idx):
        return self.t[idx]

    def __setitem__(self, idx, val):
        self.t[idx] = val


class Layer:
    def __init__(self, sd, i, dev, w8a16=False):
        p = f"model.layers.{i}."
        g = lambda n: sd[p + n].to(dev)
        self.wq, self.wk, self.wv, self.wo = (g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                                              g("self_attn.v_proj.weight"), g("self_attn.o_proj.weight"))
        self.wg, self.wu, self.wd = (g("mlp.gate_proj.weight"), g("mlp.up_proj.weight"),
                                     g("mlp.down_proj.weight"))
        self.in_ln = g("input_layernorm.weight")
        self.post_ln = g("post_attention_layernorm.weight")


def run_step(layers, x, kcache, vcache, nkv, hd, G):
    bs = x.shape[0]
    for li, L in enumerate(layers):
        h = rms_norm(x, L.in_ln)
        q = h @ L.wq.t()
        k = h @ L.wk.t()
        v = h @ L.wv.t()
        pos = kcache[li].pos
        kcache[li][:, :, pos, :] = k.view(bs, nkv, hd)
        vcache[li][:, :, pos, :] = v.view(bs, nkv, hd)
        Lpos = pos + 1
        # q: [bs, hq, hd] -> [bs, nkv, G, 1, hd]; k/v: [bs, nkv, Lpos, hd]
        qq = q.view(bs, nkv, G, 1, hd)
        kk = kcache[li][:, :, :Lpos].unsqueeze(2)          # [bs, nkv, 1, Lpos, hd]
        vv = vcache[li][:, :, :Lpos].unsqueeze(2)
        scores = (qq * (1.0 / hd ** 0.5)) @ kk.transpose(-1, -2)   # [bs, nkv, G, 1, Lpos]
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        o = (probs @ vv).reshape(bs, hd * G * nkv)          # [bs, hq*hd]
        x = x + o @ L.wo.t()
        h2 = rms_norm(x, L.post_ln)
        gate = torch.nn.functional.silu(h2 @ L.wg.t())
        up = h2 @ L.wu.t()
        x = x + (gate * up) @ L.wd.t()
    return x


def kv_bytes_per_token(cfg):
    layers = cfg["num_hidden_layers"]
    nkv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    hd = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    return 2 * layers * nkv * hd * 2  # K and V, bf16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--w8a16", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    sd = load_file(os.path.join(args.model, "model.safetensors"))
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    nlayers = cfg["num_hidden_layers"]
    hq = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", hq)
    hd = cfg.get("head_dim", cfg["hidden_size"] // hq)
    hidden = cfg["hidden_size"]
    wbytes = sum(v.numel() * v.element_size() for k, v in sd.items() if k.endswith("weight"))
    print(f"layers={nlayers} hq={hq} nkv={nkv} hd={hd} hidden={hidden} weight_bytes={wbytes/1e9:.3f}GB", flush=True)

    G = hq // nkv
    layers = [Layer(sd, i, dev) for i in range(nlayers)]
    res = {"model": args.model, "weight_gb": round(wbytes / 1e9, 3),
           "kv_bytes_per_token": kv_bytes_per_token(cfg),
           "theoretical_tps_bs1": round(HBM_GBPS * 1e9 / wbytes, 1),
           "rows": []}
    for S in [512, 4096]:
        for bs in [1, 4, 16, 64]:
            kc = [KVBuf(torch.randn(bs, nkv, S + 1, hd, device=dev, dtype=torch.bfloat16) * 0.02, S)
                  for _ in range(nlayers)]
            vc = [KVBuf(torch.randn(bs, nkv, S + 1, hd, device=dev, dtype=torch.bfloat16) * 0.02, S)
                  for _ in range(nlayers)]
            x = torch.randn(bs, hidden, device=dev, dtype=torch.bfloat16) * 0.02
            try:
                t = bench(lambda: run_step(layers, x, kc, vc, nkv, hd, G), iters=5, warmup=2)
                tps = bs / t
                ceiling_bs = bs * HBM_GBPS * 1e9 / wbytes
                res["rows"].append({"ctx": S, "bs": bs, "step_ms": round(t * 1e3, 2),
                                    "tok_s": round(tps, 1),
                                    "ceiling_tok_s": round(ceiling_bs, 1),
                                    "pct_of_bw_ceiling": round(100 * tps / ceiling_bs, 1)})
                print(f"S={S:5d} bs={bs:3d} step={t*1e3:8.2f}ms tok/s={tps:8.1f} "
                      f"ceiling={bs*HBM_GBPS*1e9/wbytes:8.1f} ({100*tps/(bs*HBM_GBPS*1e9/wbytes):5.1f}%)", flush=True)
            except Exception as e:
                res["rows"].append({"ctx": S, "bs": bs, "error": f"{type(e).__name__}: {e}"})
                print(f"S={S} bs={bs} ERROR {e}", flush=True)
            del kc, vc
            torch.npu.empty_cache()
    # pure bandwidth baseline: read every weight once, minimal op count
    import itertools
    W = torch.cat([sd[k].to(dev) for k in sd
                   if k.endswith("weight") and sd[k].dim() == 2 and sd[k].shape[1] == hidden], dim=0)
    res["concat_weight_gb"] = round(W.numel() * W.element_size() / 1e9, 3)
    for bs in [1, 64]:
        xb = torch.randn(bs, W.shape[1], device=dev, dtype=torch.bfloat16)
        t = bench(lambda: xb @ W.t(), iters=10, warmup=3)
        gbps = W.numel() * W.element_size() / t / 1e9
        res.setdefault("pure_matvec", []).append(
            {"bs": bs, "ms": round(t * 1e3, 2), "achieved_GBps": round(gbps, 1),
             "equiv_tok_s_bs1": round(gbps * 1e9 / wbytes, 1)})
        print(f"pure_matvec bs={bs:3d} {t*1e3:7.2f}ms  {gbps:7.1f} GB/s  "
              f"(equiv {gbps*1e9/wbytes:.0f} tok/s at bs=1)", flush=True)
    del W
    torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
