"""E78: interview Q — "a giant prompt is admitted while 64 users decode;
walk through their inter-token latency step by step" (Anthropic/GDM/Perplexity)

Real trace: bs=64 decodes at ctx=4096, then a 16384-token prompt is admitted.
Two policies:
  A) ungoverned  : the whole 16k prefill runs inside one decode step
  B) chunked 2048: prefill is split across steps
Report per-step TPOT trace; the spike IS the answer to the question.
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

NL, H, NKV, HD, HID, FFN = 28, 16, 8, 128, 1024, 3072
CTX, BS, GIANT, CHUNK = 4096, 64, 16384, 2048


def decode_step(x, kc, vc, w, pos):
    G = H // NKV
    b = x.shape[0]
    for wl in w:
        h = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * wl["ln1"]
        q = h @ wl["q"].T
        k = h @ wl["k"].T
        v = h @ wl["v"].T
        kc[:, :b, pos, :] = k.view(b, NKV, HD).permute(1, 0, 2)
        vc[:, :b, pos, :] = v.view(b, NKV, HD).permute(1, 0, 2)
        qh = q.view(b, NKV, G, HD).permute(1, 0, 2, 3)
        att = F.scaled_dot_product_attention(
            qh, kc[:, :b, :pos + 1, :], vc[:, :b, :pos + 1, :])
        att = att.permute(1, 0, 2, 3).reshape(b, -1)
        x = x + att @ wl["o"].T
        h = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * wl["ln2"]
        x = x + (F.silu(h @ wl["g"].T) * (h @ wl["u"].T)) @ wl["d"].T
    return x


def prefill_chunk(p, w):
    # one layer-equivalent of real prefill work per layer (no KV needed after)
    for wl in w:
        h = p * torch.rsqrt(p.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(p.dtype) * wl["ln1"]
        qh = (h @ wl["q"].T).view(-1, H, HD).transpose(0, 1).unsqueeze(0)
        kh = (h @ wl["k"].T).view(-1, NKV, HD).transpose(0, 1).unsqueeze(0).repeat_interleave(H // NKV, 1)
        vh = (h @ wl["v"].T).view(-1, NKV, HD).transpose(0, 1).unsqueeze(0).repeat_interleave(H // NKV, 1)
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        o = o.transpose(1, 2).reshape(p.shape[0], -1)
        p = p + o @ wl["o"].T
        h = p * torch.rsqrt(p.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(p.dtype) * wl["ln2"]
        p = p + (F.silu(h @ wl["g"].T) * (h @ wl["u"].T)) @ wl["d"].T
    return p


def run(dev, policy, steps=40, giant_at=15):
    w = [{k: torch.randn(s, dtype=torch.bfloat16, device=dev) * 0.02
          for k, s in [("ln1", HID), ("q", (H * HD, HID)), ("k", (NKV * HD, HID)),
                       ("v", (NKV * HD, HID)), ("o", (HID, H * HD)), ("ln2", HID),
                       ("g", (FFN, HID)), ("u", (FFN, HID)), ("d", (HID, FFN))]}
         for _ in range(NL)]
    kc = torch.randn(NKV, BS, CTX + 8, HD, dtype=torch.bfloat16, device=dev)
    vc = torch.randn(NKV, BS, CTX + 8, HD, dtype=torch.bfloat16, device=dev)
    giant = torch.randn(GIANT, HID, dtype=torch.bfloat16, device=dev)

    trace = []
    done = 0
    torch.npu.synchronize()
    for t in range(steps):
        x = torch.randn(BS, HID, dtype=torch.bfloat16, device=dev)
        t0 = time.perf_counter()
        x = decode_step(x, kc, vc, w, CTX + (t % 8))
        chunk = 0
        if t >= giant_at and done < GIANT:
            chunk = GIANT if policy == "ungoverned" else min(CHUNK, GIANT - done)
            giant[:chunk] = prefill_chunk(giant[:chunk], w)
            done += chunk
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        trace.append({"t": t, "ms": round(ms, 1), "chunk": chunk})
        del x
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "npu:0"
    res = {}
    for policy in ["ungoverned", "chunked2048"]:
        tr = run(dev, policy)
        res[policy] = tr
        base = sorted(r["ms"] for r in tr if r["chunk"] == 0)
        spike = [r for r in tr if r["chunk"] > 0]
        print(f"[E78] {policy}: baseline p50={base[len(base)//2]:.1f}ms "
              f"max={base[-1]:.1f} | prefill steps={len(spike)} "
              f"their ms={[r['ms'] for r in spike]}", flush=True)
    res["note"] = (f"bs={BS} decode ctx={CTX}; giant prompt={GIANT} tokens "
                   f"admitted at t=15; chunked={CHUNK}/step")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[E78] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
