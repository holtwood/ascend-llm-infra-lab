"""E72: tiered APC — cost of restoring a shared prefix by tier.

A prefix-cache hit restores KV without recompute. If HBM is full, the prefix
KV can live on CPU (pinned) and be reloaded. Three paths measured side by
side at real shapes (kv/token=114688B, 28-layer model):

  a) HBM hit        : ~0 (reference point)
  b) CPU-tier reload: pinned H2D of the prefix KV
  c) recompute      : real 28-layer prefill of L tokens

Output: ms per path x L in {2048, 8192, 32768}.
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

NL, H, NKV, HD, HID, FFN = 28, 16, 8, 128, 1024, 3072
KV_PER_TOKEN = 2 * NL * NKV * HD * 2  # bytes bf16


def bench(fn, iters=8, warmup=2):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def prefill(L, w, dev):
    x = torch.randn(L, HID, dtype=torch.bfloat16, device=dev)
    G = H // NKV
    for wl in w:
        h = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * wl["ln1"]
        q = (h @ wl["q"].T).view(L, H, HD)
        k = (h @ wl["k"].T).view(L, NKV, HD)
        v = (h @ wl["v"].T).view(L, NKV, HD)
        qh = q.transpose(0, 1).unsqueeze(0)
        kh = k.transpose(0, 1).unsqueeze(0).repeat_interleave(H // NKV, dim=1)
        vh = v.transpose(0, 1).unsqueeze(0).repeat_interleave(H // NKV, dim=1)
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        o = o.transpose(1, 2).reshape(L, -1)
        x = x + o @ wl["o"].T
        h = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * wl["ln2"]
        x = x + (F.silu(h @ wl["g"].T) * (h @ wl["u"].T)) @ wl["d"].T
    torch.npu.synchronize()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "npu:0"
    w = [{k: torch.randn(s, dtype=torch.bfloat16, device=dev) * 0.02
          for k, s in [("ln1", HID), ("q", (H * HD, HID)), ("k", (NKV * HD, HID)),
                       ("v", (NKV * HD, HID)), ("o", (HID, H * HD)), ("ln2", HID),
                       ("g", (FFN, HID)), ("u", (FFN, HID)), ("d", (HID, FFN))]}
         for _ in range(NL)]

    res = {"rows": []}
    for L in [2048, 8192, 32768]:
        kv_bytes = L * KV_PER_TOKEN
        host = torch.empty(kv_bytes, dtype=torch.uint8, pin_memory=True)
        dst = torch.empty(kv_bytes, dtype=torch.uint8, device=dev)
        reload_ms = bench(lambda: dst.copy_(host, non_blocking=True))
        recompute_ms = bench(lambda: prefill(L, w, dev))
        row = {"L": L, "kv_mb": round(kv_bytes / 1e6, 0),
               "hbm_hit_ms": 0.0,
               "cpu_reload_ms": round(reload_ms, 2),
               "recompute_ms": round(recompute_ms, 2),
               "reload_over_recompute": round(reload_ms / recompute_ms, 3)}
        res["rows"].append(row)
        print(f"[E72] L={L:<6} reload={reload_ms:8.2f}ms recompute={recompute_ms:8.2f}ms "
              f"ratio={row['reload_over_recompute']}", flush=True)
        del host, dst
        torch.npu.empty_cache()

    res["note"] = (f"kv/token={KV_PER_TOKEN}B. reload uses pinned H2D; "
                   "recompute = real 28-layer prefill (SDPA, causal)")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[E72] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
