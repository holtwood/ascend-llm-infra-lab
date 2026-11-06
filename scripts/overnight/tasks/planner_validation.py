#!/usr/bin/env python3
"""E52 — Validate the capacity planner against real full-model decode steps.

The planner (scripts/serving_capacity_planner.py) predicts
    step = weight_read + attention_read + launch_floor
with attention efficiency 19%->50% (E37/E39). This measures the real
full-model decode step at several (batch, ctx) points and compares.

Usage: python3 planner_validation.py --model DIR --out out.json
"""
import argparse, json, os, time
import torch, torch_npu
from safetensors.torch import load_file

HBM_GBPS = 1412.0
LAUNCH_FLOOR_MS = 29.0
EFF1, EFF16 = 0.19, 0.50


def eff(b):
    if b <= 1:
        return EFF1
    if b >= 16:
        return EFF16
    return EFF1 + (EFF16 - EFF1) * (b - 1) / 15.0


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", default="1:4096,8:4096,8:8192,16:8192")
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
    W = {k: sd[k].to(dev) for k in sd if k.endswith("weight")}
    wgb = sum(v.numel() * v.element_size() for v in W.values()) / 1e9
    kv_tok = 2 * nl * nkv * hd * 2
    weight_ms = wgb * 1e3 / HBM_GBPS

    def step(x, kc, vc):
        bs = x.shape[0]
        for i in range(nl):
            p = f"model.layers.{i}."
            h = rms_norm(x, W[p + "input_layernorm.weight"])
            q = h @ W[p + "self_attn.q_proj.weight"].t()
            k = h @ W[p + "self_attn.k_proj.weight"].t()
            v = h @ W[p + "self_attn.v_proj.weight"].t()
            # KV laid out as [nkv, bs, S, hd] so the 4-D reshape is a view
            kc[i][:, :, ctx, :] = k.view(bs, nkv, hd).permute(1, 0, 2)
            vc[i][:, :, ctx, :] = v.view(bs, nkv, hd).permute(1, 0, 2)
            qh = q.view(bs, nkv, G, 1, hd).permute(1, 0, 2, 3, 4).reshape(nkv * bs, G, 1, hd)
            kh = kc[i].reshape(nkv * bs, 1, ctx + 1, hd)
            vh = vc[i].reshape(nkv * bs, 1, ctx + 1, hd)
            o = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
            o = o.reshape(nkv, bs, G * hd).permute(1, 0, 2).reshape(bs, hq * hd)
            x = x + o @ W[p + "self_attn.o_proj.weight"].t()
            h2 = rms_norm(x, W[p + "post_attention_layernorm.weight"])
            g = torch.nn.functional.silu(h2 @ W[p + "mlp.gate_proj.weight"].t())
            u = h2 @ W[p + "mlp.up_proj.weight"].t()
            x = x + (g * u) @ W[p + "mlp.down_proj.weight"].t()
        return x

    res = {"weight_gb": round(wgb, 3), "kv_bytes_per_token": kv_tok, "rows": []}
    for pt in args.points.split(","):
        bs, ctx = [int(v) for v in pt.split(":")]
        kc = [torch.zeros(nkv, bs, ctx + 1, hd, device=dev, dtype=torch.bfloat16) for _ in range(nl)]
        vc = [torch.zeros(nkv, bs, ctx + 1, hd, device=dev, dtype=torch.bfloat16) for _ in range(nl)]
        x = torch.randn(bs, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        t = bench(lambda: step(x, kc, vc))
        kv_gb = bs * kv_tok * ctx / 1e9
        pred_attn = kv_gb * 1e3 / (HBM_GBPS * eff(bs))
        pred = weight_ms + pred_attn + LAUNCH_FLOOR_MS
        row = {"bs": bs, "ctx": ctx, "measured_ms": round(t * 1e3, 1),
               "pred_weight_ms": round(weight_ms, 2),
               "pred_attn_ms": round(pred_attn, 1),
               "pred_floor_ms": LAUNCH_FLOOR_MS,
               "predicted_ms": round(pred, 1),
               "error_pct": round(100 * (t * 1e3 - pred) / pred, 1)}
        res["rows"].append(row)
        with open(args.out + "l", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[E52] bs={bs:2d} ctx={ctx:5d} measured={t*1e3:7.1f}ms "
              f"predicted={pred:7.1f}ms (w {weight_ms:.1f} + attn {pred_attn:.1f} "
              f"+ floor {LAUNCH_FLOOR_MS}) error {row['error_pct']:+.0f}%", flush=True)
        del kc, vc, x
        torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
