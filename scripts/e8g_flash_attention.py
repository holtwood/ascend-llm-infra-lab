#!/usr/bin/env python3
"""E8g — hand-written FlashAttention (online-softmax tiling) on Ascend NPU.

The canonical whiteboard question: implement fused attention without
materializing the S×S score matrix. This is the real algorithm — process
K/V in tiles, keep a running row max m_i and rescale denominator l_i,
output O_i accumulated in registers:

    for each K-tile j:
        s_ij = Q K_j^T / sqrt(d)              # tile-local scores
        m_new = max(m_i, rowmax(s_ij))
        P = exp(s_ij - m_new)
        l_i = l_i * exp(m_i - m_new) + rowsum(P)
        O_i = O_i * exp(m_i - m_new) + P V_j
        m_i = m_new
    out = O_i / l_i

Memory: O(S*d + Bc*S_tile) instead of O(S^2) — that's the whole trick.

Verified vs F.scaled_dot_product_attention (is_causal=True) on 910B1,
then benchmarked naive-materialize vs sdpa vs manual-tiled.

Usage: python3 e8g_flash_attention.py --device npu:1 --json-out e8g.json
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


def flash_manual(q, k, v, Bc=256, causal=True):
    """Online-softmax tiled attention. q,k,v: [B, H, S, D] same dtype.
    Returns fp32 accumulation upcast then downcast — matches sdpa numerics
    within bf16 tolerance."""
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    qf = q.float()
    O = torch.zeros(B, H, S, D, dtype=torch.float32, device=q.device)
    m = torch.full((B, H, S, 1), float("-inf"), dtype=torch.float32, device=q.device)
    l = torch.zeros(B, H, S, 1, dtype=torch.float32, device=q.device)

    for j0 in range(0, S, Bc):
        j1 = min(j0 + Bc, S)
        kj, vj = k[:, :, j0:j1].float(), v[:, :, j0:j1].float()
        s = (qf @ kj.transpose(-1, -2)) * scale              # [B,H,S,Bc]
        if causal:
            i = torch.arange(S, device=q.device)[:, None]
            j = torch.arange(j0, j1, device=q.device)[None, :]
            s = s.masked_fill(j > i, float("-inf"))
        m_new = torch.maximum(m, s.amax(-1, keepdim=True))   # running max
        p = torch.exp(s - m_new)
        alpha = torch.exp(m - m_new)
        l = l * alpha + p.sum(-1, keepdim=True)
        O = O * alpha + p @ vj
        m = m_new
    return (O / l).to(q.dtype)


def bench(fn, iters=5, dev="cpu"):
    for _ in range(2):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(dev)
    return round((time.perf_counter() - t0) / iters * 1000, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:1" if HAS_NPU else "cpu")
    ap.add_argument("--json-out", default="/root/e8g.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("npu"):
        torch.npu.set_device(int(dev.split(":")[-1]))

    out = {"device": dev, "correctness": {}, "perf": []}
    print(f"[e8g] device={dev} torch={torch.__version__}")

    # correctness: several shapes incl. non-multiple-of-Bc edges
    B, H, D = 2, 8, 64
    for S, Bc in ((512, 256), (1024, 128), (777, 256), (2048, 512)):
        q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        mine = flash_manual(q, k, v, Bc=Bc)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        d = (mine.float() - ref.float()).abs().max().item()
        out["correctness"][f"S{S}_Bc{Bc}"] = d
        print(f"[e8g] correctness S={S} Bc={Bc}: max_diff={d:.2e}")

    # non-causal variant check
    q = torch.randn(B, H, 1024, D, device=dev, dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    d = (flash_manual(q, k, v, causal=False).float()
         - F.scaled_dot_product_attention(q, k, v).float()).abs().max().item()
    out["correctness"]["noncausal_S1024"] = d
    print(f"[e8g] non-causal max_diff={d:.2e}")

    # perf: naive-materialize / sdpa / manual-tiled (fp32 accum)
    for S in (2048, 4096, 8192):
        q = torch.randn(1, 8, S, D, device=dev, dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        def naive():
            s = q.float() @ k.float().transpose(-1, -2) / math.sqrt(D)
            i = torch.arange(S, device=dev)[:, None]
            j = torch.arange(S, device=dev)[None, :]
            s = s.masked_fill(j > i, float("-inf"))
            return (s.softmax(-1) @ v.float()).to(q.dtype)
        row = {"S": S,
               "naive_materialize_ms": bench(naive, dev=dev),
               "sdpa_fused_ms": bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), dev=dev),
               "manual_tiled_Bc512_ms": bench(lambda: flash_manual(q, k, v, Bc=512), dev=dev)}
        out["perf"].append(row)
        print("[e8g] perf:", row)

    # memory estimate table (mechanism, not measured): peak score bytes
    for S in (2048, 4096, 8192):
        naive_bytes = S * S * 4                       # fp32 S×S matrix
        tile_bytes = S * 512 * 4                      # one Bc=512 tile at a time
        out.setdefault("mem_est", {})[f"S{S}"] = {
            "naive_score_mb": round(naive_bytes / 1e6, 1),
            "tiled_score_mb": round(tile_bytes / 1e6, 1),
            "ratio": round(naive_bytes / tile_bytes, 1)}

    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print("[e8g] wrote", args.json_out)


if __name__ == "__main__":
    main()
