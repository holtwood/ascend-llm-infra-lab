"""E76: fused SDPA vs manual attention — what the fused kernel buys.

Manual path materializes scores [H, L, S] + softmax + PV matmul — the
pre-FlashAttention algorithm. Fused path never materializes scores.
Sweep L (prefill) and (bs, L) decode-style.
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F


def bench(fn, iters=15, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def manual_attn(q, k, v, causal=True):
    # q,k,v: [1,H,L,hd]
    s = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
    if causal:
        L, S = s.shape[-2], s.shape[-1]
        mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=s.device),
                          diagonal=S - L + 1)
        s = s.masked_fill(mask, float("-inf"))
    p = F.softmax(s, dim=-1)
    return p @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "npu:0"
    H, HD = 16, 128
    res = {"prefill": [], "decode": []}

    for L in [512, 2048, 8192, 16384]:
        q = torch.randn(1, H, L, HD, dtype=torch.bfloat16, device=dev)
        k, v = q.clone(), q.clone()
        f_ms = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        m_ms = bench(lambda: manual_attn(q, k, v))
        res["prefill"].append({"L": L, "sdpa_ms": round(f_ms, 3),
                               "manual_ms": round(m_ms, 3),
                               "speedup": round(m_ms / f_ms, 2)})
        print(f"[E76] prefill L={L:<6} sdpa={f_ms:7.2f} manual={m_ms:8.2f} "
              f"-> {m_ms/f_ms:.2f}x", flush=True)
        del q, k, v
        torch.npu.empty_cache()

    for bs, ctx in [(1, 4096), (16, 4096), (64, 4096), (16, 32768)]:
        q = torch.randn(1, bs, H, 1, HD, dtype=torch.bfloat16, device=dev).squeeze(3)
        k = torch.randn(1, bs, H, ctx, HD, dtype=torch.bfloat16, device=dev)
        v = k.clone()
        qq = q.unsqueeze(3)
        f_ms = bench(lambda: F.scaled_dot_product_attention(qq, k, v))
        m_ms = bench(lambda: manual_attn(qq, k, v, causal=False))
        res["decode"].append({"bs": bs, "ctx": ctx,
                              "sdpa_ms": round(f_ms, 3), "manual_ms": round(m_ms, 3),
                              "speedup": round(m_ms / f_ms, 2)})
        print(f"[E76] decode bs={bs:<3} ctx={ctx:<6} sdpa={f_ms:6.2f} "
              f"manual={m_ms:7.2f} -> {m_ms/f_ms:.2f}x", flush=True)
        del q, k, v, qq
        torch.npu.empty_cache()

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[E76] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
