#!/usr/bin/env python3
"""E8e — hand-written attention variants on Ascend NPU.

Three from-scratch implementations, each verified against an independent
reference, then benchmarked:

  1. RoPE  — pairwise rotate-half rotary embedding (GPT-NeoX convention)
     vs an explicit complex-multiply formulation, plus algebraic property
     checks (norm preservation, relative-position invariance). Optionally
     cross-checked against HF transformers' apply_rotary_pos_emb.
  2. GQA   — expand-view vs repeat_interleave KV-head broadcasting:
     equivalence + KV-byte accounting (GQA's serving win is KV bytes, not FLOPs).
  3. SWA   — sliding-window attention via explicit band mask vs
     F.scaled_dot_product_attention with the same mask; window/seq sweeps.

Usage (pytorch container):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 e8e_attention_variants.py --device npu:1 --json-out /root/e8e.json
"""
import argparse, json, math, time

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


def sync(dev):
    if dev.startswith("npu") and HAS_NPU:
        torch.npu.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()


# ---------------------------------------------------------------- RoPE ----
def precompute_freqs(dim: int, max_pos: int, theta: float, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_pos, device=device).float()
    freqs = torch.outer(t, inv_freq)                    # [P, dim/2]
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def rope_manual(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX-convention RoPE. x: [B, S, H, D]; cos/sin: [S, D/2]."""
    cos = torch.cat((cos, cos), dim=-1)[None, :, None, :]
    sin = torch.cat((sin, sin), dim=-1)[None, :, None, :]
    return x * cos + rotate_half(x) * sin


def rope_complex_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Independent formulation: treat (x[...,j], x[...,j+D/2]) as a complex
    pair and multiply by e^{i p w}. Same NeoX pairing, different code path."""
    a = x.float()[..., : x.shape[-1] // 2]
    b = x.float()[..., x.shape[-1] // 2 :]
    c = cos.float()[None, :, None, :]
    s = sin.float()[None, :, None, :]
    return torch.cat((a * c - b * s, a * s + b * c), dim=-1)


def rope_hf_ref(x, cos, sin):
    """HF transformers apply_rotary_pos_emb, if importable."""
    from transformers.models.qwen3 import modeling_qwen3
    cos2 = torch.cat((cos, cos), dim=-1)[None, :, None, :].to(x.dtype)
    sin2 = torch.cat((sin, sin), dim=-1)[None, :, None, :].to(x.dtype)
    out, _ = modeling_qwen3.apply_rotary_pos_emb(x, x, cos2, sin2)
    return out


def run_rope(dev, S=4096, D=128, B=2, H=32):
    cos, sin = precompute_freqs(D, S, theta=1e6, device=dev)
    x = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)

    mine = rope_manual(x, cos, sin)
    ref = rope_complex_ref(x, cos, sin)
    diff = (mine.float() - ref).abs().max().item()

    hf_diff = None
    try:
        hf_diff = (mine.float() - rope_hf_ref(x, cos, sin).float()).abs().max().item()
    except Exception as e:
        hf_diff = f"n/a ({type(e).__name__})"

    # Property 1: rotation preserves per-head-vector norms
    norm_err = (x.float().norm(dim=-1) - mine.float().norm(dim=-1)).abs().max().item()

    # Property 2: <Rope_m(q), Rope_n(k)> depends only on m-n
    q = torch.randn(1, 1, H, D, device=dev)
    k = torch.randn(1, 1, H, D, device=dev)
    def rope_pos(v, p):
        c = torch.cat((cos[p:p+1], cos[p:p+1]), -1).float()
        s = torch.cat((sin[p:p+1], sin[p:p+1]), -1).float()
        return v.float() * c + rotate_half(v.float()) * s
    m, n = 100, 37
    lhs = (rope_pos(q, m) * rope_pos(k, n)).sum(-1)
    rhs = (rope_pos(q, m - n) * rope_pos(k, 0)).sum(-1)
    rel_err = (lhs - rhs).abs().max().item()

    return {"rope_max_abs_diff_vs_complex": diff,
            "rope_max_abs_diff_vs_hf": hf_diff,
            "rope_norm_preservation_err": norm_err,
            "rope_relative_position_err": rel_err,
            "config": {"S": S, "D": D, "H": H, "B": B, "theta": 1e6}}


# ---------------------------------------------------------------- GQA ----
def kv_repeat_expand(k: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B,S,Hkv,D] -> [B,S,Hkv*n_rep,D] zero-copy view expansion."""
    B, S, Hkv, D = k.shape
    return k.unsqueeze(3).expand(B, S, Hkv, n_rep, D).reshape(B, S, Hkv * n_rep, D)


def run_gqa(dev, B=2, S=8192, Hq=32, Hkv=4, D=128):
    n_rep = Hq // Hkv
    k = torch.randn(B, S, Hkv, D, device=dev)
    v = torch.randn(B, S, Hkv, D, device=dev)
    eq = torch.equal(kv_repeat_expand(k, n_rep), k.repeat_interleave(n_rep, dim=2))

    # correctness: replicated-KV MHA == native GQA path (and == manual expand)
    qs = torch.randn(B, min(64, S), Hq, D, device=dev).transpose(1, 2).float()
    ks = k[:, : min(64, S)].transpose(1, 2).float()
    vs = v[:, : min(64, S)].transpose(1, 2).float()
    o_exp = F.scaled_dot_product_attention(
        qs, kv_repeat_expand(k[:, : min(64, S)], n_rep).transpose(1, 2).float(),
        kv_repeat_expand(v[:, : min(64, S)], n_rep).transpose(1, 2).float())
    gqa_native_diff = None
    try:
        o_native = F.scaled_dot_product_attention(qs, ks, vs, enable_gqa=True)
        gqa_native_diff = (o_exp - o_native).abs().max().item()
    except Exception as e:
        gqa_native_diff = f"n/a ({type(e).__name__})"

    kv_mha = B * S * Hq * D * 2 * 2 / 1e6   # K+V, bf16, MB
    kv_gqa = B * S * Hkv * D * 2 * 2 / 1e6
    return {"gqa_expand_eq_repeat_interleave": eq,
            "gqa_vs_native_sdpa_max_diff": gqa_native_diff,
            "kv_bytes_per_req_mha_mb": round(kv_mha, 1),
            "kv_bytes_per_req_gqa_mb": round(kv_gqa, 1),
            "kv_reduction_x": n_rep,
            "config": {"B": B, "S": S, "Hq": Hq, "Hkv": Hkv, "D": D}}


# ---------------------------------------------------------------- SWA ----
def band_mask(S: int, w: int, device) -> torch.Tensor:
    i = torch.arange(S, device=device)[:, None]
    j = torch.arange(S, device=device)[None, :]
    keep = (j <= i) & (j > i - w)
    m = torch.full((S, S), float("-inf"), device=device)
    m[keep] = 0.0
    return m


def swa_manual(q, k, v, w):
    S, D = q.shape[-2], q.shape[-1]
    scores = q @ k.transpose(-1, -2) / math.sqrt(D)
    scores = scores + band_mask(S, w, q.device)
    return scores.softmax(-1) @ v


def run_swa(dev, B=1, H=8, D=64, dtype=torch.bfloat16):
    res = {"correctness": {}, "perf": []}
    S0 = 2048
    q = torch.randn(B, H, S0, D, device=dev, dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)
    for w in (128, 256, 512):
        m = band_mask(S0, w, dev)
        mine = swa_manual(q.float(), k.float(), v.float(), w)
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=m)
        res["correctness"][f"w{w}"] = (mine - ref).abs().max().item()

    def bench(fn, iters=5):
        for _ in range(2):
            fn()
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync(dev)
        return round((time.perf_counter() - t0) / iters * 1000, 3)

    for S in (2048, 4096, 8192):
        q = torch.randn(B, H, S, D, device=dev, dtype=dtype)
        k, v = torch.randn_like(q), torch.randn_like(q)
        row = {"S": S,
               "sdpa_full_causal_ms": bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))}
        for w in (256, 1024):
            m = band_mask(S, w, dev).to(dtype)
            row[f"sdpa_mask_w{w}_ms"] = bench(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=m))
        row["manual_masked_w256_ms"] = bench(lambda: swa_manual(q.float(), k.float(), v.float(), 256))
        res["perf"].append(row)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:1" if HAS_NPU else "cpu")
    ap.add_argument("--json-out", default="/root/e8e_attention_variants.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("npu"):
        torch.npu.set_device(int(dev.split(":")[-1]))

    out = {"device": dev, "torch": torch.__version__, "results": {}}
    print(f"[e8e] device={dev} torch={torch.__version__}")
    out["results"]["rope"] = run_rope(dev)
    print("[e8e] rope:", out["results"]["rope"])
    out["results"]["gqa"] = run_gqa(dev)
    print("[e8e] gqa:", {k: v for k, v in out["results"]["gqa"].items() if k != "config"})
    out["results"]["swa"] = run_swa(dev)
    print("[e8e] swa correctness:", out["results"]["swa"]["correctness"])
    for r in out["results"]["swa"]["perf"]:
        print("[e8e] swa perf:", r)
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print("[e8e] wrote", args.json_out)


if __name__ == "__main__":
    main()
