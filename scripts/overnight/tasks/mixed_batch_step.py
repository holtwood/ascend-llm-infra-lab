#!/usr/bin/env python3
"""E63 — Full-model mixed batch: what one chunked-prefill step really costs.

E44 measured the interference on the attention op alone. This measures the
whole model step for a batch that contains BOTH decode sequences and a prefill
chunk — exactly what a chunked-prefill engine schedules in one iteration:

    rows = R_d decode tokens (each attending to ctx-long KV)
         + P prefill tokens (causal within the chunk)

Reports step time for decode-only, prefill-only and mixed, plus the implied
TPOT (for the decode sequences) and prefill throughput.

Usage: python3 mixed_batch_step.py --model DIR --out out.json
"""
import argparse, json, os, time
import torch, torch_npu
from safetensors.torch import load_file


def rms(x, w, eps=1e-6):
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
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--dec-bs", type=int, default=8)
    ap.add_argument("--prefills", default="512,2048")
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
    W = {k: sd[k].to(dev) for k in sd if k.endswith("weight")}
    Rd = args.dec_bs

    def step(P):
        """rows = Rd decode + P prefill (P=0 -> decode only)."""
        rows = Rd + P
        x = torch.randn(rows, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        kc = [torch.zeros(nkv, max(Rd, 1), S + 1, hd, device=dev, dtype=torch.bfloat16)
              for _ in range(nl)] if Rd else None
        vc = [torch.zeros(nkv, max(Rd, 1), S + 1, hd, device=dev, dtype=torch.bfloat16)
              for _ in range(nl)] if Rd else None
        for i in range(nl):
            p = f"model.layers.{i}."
            h = rms(x, W[p + "input_layernorm.weight"])
            q = h @ W[p + "self_attn.q_proj.weight"].t()
            k = h @ W[p + "self_attn.k_proj.weight"].t()
            v = h @ W[p + "self_attn.v_proj.weight"].t()
            outs = []
            if Rd:
                qd = q[:Rd].view(Rd, nkv, G, 1, hd).permute(1, 0, 2, 3, 4).reshape(nkv * Rd, G, 1, hd)
                kc[i][:, :, S, :] = k[:Rd].view(Rd, nkv, hd).permute(1, 0, 2)
                vc[i][:, :, S, :] = v[:Rd].view(Rd, nkv, hd).permute(1, 0, 2)
                od = torch.nn.functional.scaled_dot_product_attention(
                    qd, kc[i].reshape(nkv * Rd, 1, S + 1, hd),
                    vc[i].reshape(nkv * Rd, 1, S + 1, hd))
                outs.append(od.reshape(nkv, Rd, G * hd).permute(1, 0, 2).reshape(Rd, hq * hd))
            if P:
                # broadcast over G (no O(P^2) materialization): [nkv,G,P,hd] x [nkv,1,P,hd]
                qp = q[Rd:].view(P, nkv, G, hd).permute(1, 2, 0, 3)
                kp = k[Rd:].view(P, nkv, hd).permute(1, 0, 2).unsqueeze(1)
                vp = v[Rd:].view(P, nkv, hd).permute(1, 0, 2).unsqueeze(1)
                op = torch.nn.functional.scaled_dot_product_attention(qp, kp, vp, is_causal=True)
                outs.append(op.permute(0, 2, 1, 3).reshape(P, hq * hd))
            o = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
            x = x + o @ W[p + "self_attn.o_proj.weight"].t()
            h2 = rms(x, W[p + "post_attention_layernorm.weight"])
            g = torch.nn.functional.silu(h2 @ W[p + "mlp.gate_proj.weight"].t())
            u = h2 @ W[p + "mlp.up_proj.weight"].t()
            x = x + (g * u) @ W[p + "mlp.down_proj.weight"].t()
        return x

    res = {"ctx": S, "dec_bs": Rd, "rows": []}
    t_dec = bench(lambda: step(0))
    print(f"[E63] decode-only bs={Rd}: {t_dec*1e3:7.2f} ms", flush=True)
    for P in [int(x) for x in args.prefills.split(",")]:
        t_pre = bench(lambda: step(P)) if not Rd else None
        t_mix = bench(lambda: step(P))
        # prefill-only estimate: run with Rd temporarily 0 is not trivial; use mix - dec
        row = {"prefill_tokens": P, "decode_bs": Rd,
               "decode_only_ms": round(t_dec * 1e3, 2),
               "mixed_ms": round(t_mix * 1e3, 2),
               "prefill_cost_in_mix_ms": round((t_mix - t_dec) * 1e3, 2),
               "tpot_ms": round(t_mix * 1e3, 2),
               "prefill_tok_per_s": round(P / t_mix, 0)}
        res["rows"].append(row)
        print(f"[E63] mix dec={Rd} + prefill={P:5d}: {t_mix*1e3:7.2f} ms "
              f"(prefill adds {(t_mix-t_dec)*1e3:6.2f} ms) -> TPOT {t_mix*1e3:6.2f} ms, "
              f"prefill {P/t_mix:6.0f} tok/s", flush=True)
        del t_pre
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
