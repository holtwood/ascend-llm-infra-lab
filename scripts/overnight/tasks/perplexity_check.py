#!/usr/bin/env python3
"""E66 — Quantization acceptance: does W8A16 hurt the model's perplexity?

Speed/capacity measurements (E41/E58) say W8A16 is capacity-only on this
stack. But the real acceptance question for serving is quality: how much
perplexity does int8 weight-only quantization cost?

Runs the same forward pass over the same text with (a) bf16 weights and
(b) int8 weights dequantized on the fly, and reports perplexity + top-1
agreement.

Usage: python3 perplexity_check.py --bf16 DIR --int8 DIR --out out.json
"""
import argparse, json, math, os
import torch, torch_npu
from safetensors.torch import load_file

TEXT = (
    "The deployment of large language models in production requires careful "
    "attention to latency, throughput and memory. A serving system must batch "
    "requests, manage a key-value cache, and admit work only while the "
    "scheduler can still meet its latency objective. Quantization reduces the "
    "memory footprint of weights and the bandwidth required to read them, but "
    "it can change the model's output distribution. "
    "在生产环境中部署大语言模型需要同时关注延迟、吞吐与显存占用。"
    "服务系统必须做请求批处理、管理 KV 缓存，并且在调度器仍能满足延迟目标时"
    "才接纳新的请求。量化可以降低权重的显存占用与读取带宽需求，但也可能改变"
    "模型的输出分布，因此必须用困惑度等指标验证量化后的质量。"
)


def rms_norm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def rope_cache(L, hd, theta, dev, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    t = torch.arange(L, device=dev).float()
    freqs = torch.outer(t, inv)                    # [L, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)        # [L, hd] (rotate-half layout)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)   # [L,1,hd] broadcast over heads
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)


def quant_kv_int8(t):
    """Per-token-per-head int8 quantization of the KV cache (symmetric)."""
    scale = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
    return (q.to(t.dtype) * scale)


def quant_kv_int4(t, group=32):
    """int4 quantization. group=None -> per-token-per-head scale (naive);
    group=32 -> per-group-of-32-channels scale (KIVI-style)."""
    if group is None:
        scale = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
        q = torch.round(t / scale).clamp(-7, 7).to(torch.int8)
        return (q.to(t.dtype) * scale)
    L, H, D = t.shape
    tg = t.view(L, H, D // group, group)
    scale = tg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = torch.round(tg / scale).clamp(-7, 7).to(torch.int8)
    return (q.to(t.dtype) * scale).view(L, H, D)


def forward_logits(ids, W, cfg, dev, nlayers, hq, nkv, hd, hidden, kv_int8=False,
                   kv_int4=False, kv4_group=None):
    G = hq // nkv
    x = W["model.embed_tokens.weight"][ids]              # [L, hidden]
    L = x.shape[0]
    cos, sin = rope_cache(L, hd, cfg.get("rope_theta", 1000000.0), dev, x.dtype)
    for i in range(nlayers):
        p = f"model.layers.{i}."
        h = rms_norm(x, W[p + "input_layernorm.weight"])
        q = h @ W[p + "self_attn.q_proj.weight"].t()
        k = h @ W[p + "self_attn.k_proj.weight"].t()
        v = h @ W[p + "self_attn.v_proj.weight"].t()
        # Qwen3: per-head RMSNorm on q/k, then RoPE
        q = rms_norm(q.view(L, hq, hd), W[p + "self_attn.q_norm.weight"]).view(L, -1)
        k = rms_norm(k.view(L, nkv, hd), W[p + "self_attn.k_norm.weight"]).view(L, -1)
        q, k = apply_rope(q.view(L, hq, hd), k.view(L, nkv, hd), cos, sin)
        if kv_int8:
            k = quant_kv_int8(k)
            v = quant_kv_int8(v.view(L, nkv, hd)).view(L, -1)
        elif kv_int4:
            k = quant_kv_int4(k, group=kv4_group)
            v = quant_kv_int4(v.view(L, nkv, hd), group=kv4_group).view(L, -1)
        qh = q.view(1, L, hq, hd).transpose(1, 2)
        kh = k.view(1, L, nkv, hd).transpose(1, 2).repeat_interleave(G, dim=1)
        vh = v.view(1, L, nkv, hd).transpose(1, 2).repeat_interleave(G, dim=1)
        o = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        o = o.transpose(1, 2).reshape(L, hq * hd)
        x = x + o @ W[p + "self_attn.o_proj.weight"].t()
        h2 = rms_norm(x, W[p + "post_attention_layernorm.weight"])
        g = torch.nn.functional.silu(h2 @ W[p + "mlp.gate_proj.weight"].t())
        u = h2 @ W[p + "mlp.up_proj.weight"].t()
        x = x + (g * u) @ W[p + "mlp.down_proj.weight"].t()
    x = rms_norm(x, W["model.norm.weight"])
    return x @ W["lm_head.weight"].t()


def load(path, dev, dequant=False):
    sd = load_file(os.path.join(path, "model.safetensors"))
    W = {}
    for k, v in sd.items():
        if k.endswith("weight_scale"):
            continue
        if v.dtype == torch.int8 and dequant:
            sc = sd[k.replace("weight", "weight_scale")].to(dev).float()
            v = (v.to(dev).float() * sc).to(torch.bfloat16)
        else:
            v = v.to(dev)
        W[k] = v
    return W


def ppl(logits, ids):
    lp = torch.log_softmax(logits[:-1].float(), dim=-1)
    tgt = ids[1:]
    nll = -lp[torch.arange(len(tgt)), tgt].mean().item()
    return math.exp(nll), nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.kv_int8 = True
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.bf16)
    ids = tok(TEXT, return_tensors="pt").input_ids[0].to(dev)
    cfg = json.load(open(os.path.join(args.bf16, "config.json")))
    nlayers, hq = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", hq)
    hd = cfg.get("head_dim", cfg["hidden_size"] // hq)
    hidden = cfg["hidden_size"]
    res = {"tokens": int(ids.shape[0])}
    outs = {}
    runs = [("bf16", args.bf16, False, 0),
            ("w8a16", args.int8, True, 0),
            ("bf16_kv8", args.bf16, False, 8),
            ("bf16_kv4", args.bf16, False, -4),
            ("bf16_kv4g32", args.bf16, False, -40),
            ("w8a16_kv8", args.int8, True, 8)]
    for tag, path, dq, kvb in runs:
        W = load(path, dev, dequant=dq)
        with torch.no_grad():
            lg = forward_logits(ids, W, cfg, dev, nlayers, hq, nkv, hd, hidden,
                                kv_int8=(kvb == 8), kv_int4=(kvb < 0),
                                kv4_group=(32 if kvb == -40 else None))
        p, nll = ppl(lg, ids)
        outs[tag] = lg
        res[f"{tag}_ppl"] = round(p, 3)
        res[f"{tag}_nll"] = round(nll, 4)
        print(f"[E66] {tag:5s} ppl={p:8.3f} nll={nll:.4f}", flush=True)
        del W
        torch.npu.empty_cache()
    base = outs["bf16"].argmax(-1)
    for tag in ["w8a16", "bf16_kv8", "bf16_kv4", "bf16_kv4g32", "w8a16_kv8"]:
        if tag in outs:
            agree = 100 * (base == outs[tag].argmax(-1)).float().mean().item()
            res[f"{tag}_top1_agree_pct"] = round(agree, 2)
            res[f"{tag}_ppl_delta_pct"] = round(
                100 * (res[f"{tag}_ppl"] - res["bf16_ppl"]) / res["bf16_ppl"], 2)
            print(f"[E66] {tag:10s} top-1 agree {agree:5.2f}%  "
                  f"ppl delta {res[f'{tag}_ppl_delta_pct']:+.2f}%", flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
