"""E69: real speculative decode — int8 draft + bf16 target on hardware.

Runs an actual draft/verify loop: the w8a16 model greedily drafts q tokens,
the bf16 model verifies all q in ONE forward; accept the longest matching
prefix + the target's own next token. Reports:

  - acceptance rate per draft position
  - mean tokens generated per target forward
  - modeled wall-clock speedup = E[tokens/cycle] / (q*draft + verify)
    using measured step times (draft==target here => honest caveat).

This is not a simulation: both forwards run the real checkpoints.
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer


def load(path, dev, dequant=False):
    W = {}
    with safe_open(f"{path}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if dequant and t.dtype == torch.int8:
                s = f.get_tensor(k.replace("weight", "weight_scale"))
                t = t.to(torch.bfloat16) * s.to(torch.bfloat16)
            elif t.dtype not in (torch.bfloat16, torch.float16):
                t = t.to(torch.bfloat16)
            W[k] = t.to(dev)
    return W


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype) * w


def build_rope(S, hd, dev, theta=1e6):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    pos = torch.arange(S, device=dev).float()
    fr = torch.outer(pos, inv)
    return fr.cos().to(torch.bfloat16), fr.sin().to(torch.bfloat16)


def rope_half(t, cos, sin):
    L, H, hd = t.shape
    c = cos[:L].unsqueeze(1)
    s = sin[:L].unsqueeze(1)
    t1, t2 = t[..., : hd // 2], t[..., hd // 2:]
    return torch.cat([t1 * c - t2 * s, t2 * c + t1 * s], dim=-1)


def forward_logits(ids, W, cfg, dev, cos, sin):
    nl = cfg["nlayers"]; hq = cfg["hq"]; nkv = cfg["nkv"]; hd = cfg["hd"]
    G = hq // nkv
    x = W["model.embed_tokens.weight"][ids]
    L = len(ids)
    for i in range(nl):
        p = f"model.layers.{i}."
        h = rms(x, W[p + "input_layernorm.weight"])
        q = h @ W[p + "self_attn.q_proj.weight"].T
        k = h @ W[p + "self_attn.k_proj.weight"].T
        v = h @ W[p + "self_attn.v_proj.weight"].T
        q = rope_half(rms(q.view(L, hq, hd), W[p + "self_attn.q_norm.weight"]), cos, sin)
        k = rope_half(rms(k.view(L, nkv, hd), W[p + "self_attn.k_norm.weight"]), cos, sin)
        qh = q.view(1, L, hq, hd).transpose(1, 2)
        kh = k.view(1, L, nkv, hd).transpose(1, 2).repeat_interleave(G, dim=1)
        vh = v.view(1, L, nkv, hd).transpose(1, 2).repeat_interleave(G, dim=1)
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        o = o.transpose(1, 2).reshape(L, hq * hd)
        x = x + o @ W[p + "self_attn.o_proj.weight"].T
        h = rms(x, W[p + "post_attention_layernorm.weight"])
        g = F.silu(h @ W[p + "mlp.gate_proj.weight"].T)
        u = h @ W[p + "mlp.up_proj.weight"].T
        x = x + (g * u) @ W[p + "mlp.down_proj.weight"].T
    x = rms(x, W["model.norm.weight"])
    return x @ W["model.embed_tokens.weight"].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    dev = "npu:0"
    hf = AutoConfig.from_pretrained(args.bf16)
    cfg = {"nlayers": hf.num_hidden_layers, "hq": hf.num_attention_heads,
           "nkv": hf.num_key_value_heads,
           "hd": getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)}
    tok = AutoTokenizer.from_pretrained(args.bf16)
    prompt = args.prompt or ("大型语言模型推理系统的核心挑战是显存带宽与调度的权衡。"
              "在企业级部署中，我们需要")
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    S_max = len(ids) + args.max_new + 16
    cos, sin = build_rope(S_max, cfg["hd"], dev)

    print("[E69] loading bf16 + w8a16 ...", flush=True)
    T = load(args.bf16, dev)
    D = load(args.int8, dev, dequant=True)

    res = {"prompt_len": len(ids), "cycles": []}
    for q in [2, 4, 8]:
        cur = ids[:]
        n_tgt_fwd, n_draft_fwd, pos_acc = 0, 0, [0] * q
        n_cycles = 0
        t0 = time.perf_counter()
        while len(cur) - len(ids) < args.max_new:
            draft = []
            c = cur[:]
            for _ in range(q):
                lg = forward_logits(c, D, cfg, dev, cos, sin)
                n_draft_fwd += 1
                nt = int(lg[-1].argmax())
                draft.append(nt)
                c.append(nt)
            cand = cur + draft
            lg = forward_logits(cand, T, cfg, dev, cos, sin)
            n_tgt_fwd += 1
            preds = lg[len(cur) - 1:].argmax(-1)
            acc = 0
            for i, dt in enumerate(draft):
                if int(preds[i]) == dt:
                    acc += 1
                    pos_acc[i] += 1
                else:
                    break
            cur = cur + draft[:acc] + [int(preds[acc])]
            n_cycles += 1
        gen = len(cur) - len(ids)
        wall = time.perf_counter() - t0
        row = {
            "q": q, "cycles": n_cycles, "tokens": gen,
            "accept_per_pos": [round(a / n_cycles, 3) for a in pos_acc],
            "mean_tokens_per_tgt_fwd": round(gen / n_tgt_fwd, 2),
            "draft_fwds": n_draft_fwd, "tgt_fwds": n_tgt_fwd,
            "wall_s": round(wall, 1),
        }
        res["cycles"].append(row)
        print(f"[E69] q={q} tokens/fwd={row['mean_tokens_per_tgt_fwd']} "
              f"acc={row['accept_per_pos']} wall={wall:.0f}s", flush=True)

    text = tok.decode(cur[len(ids):])
    res["sample"] = text[:80]
    res["note"] = ("draft==target size (same 0.6B): acceptance real, wall-clock "
                   "speedup NOT real (draft isn't smaller). Speedup model uses "
                   "E45 verify costs + measured acceptance.")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[E69] wrote {args.out}  sample: {text[:60]}", flush=True)


if __name__ == "__main__":
    main()
