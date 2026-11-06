"""E79: interview Q — "RadixAttention vs vLLM APC?" + "shared prefix differs
at position p — how much do you actually reuse?"

If two prompts share the first p tokens, the KV reuse saves the GEMM work of
p tokens BUT vLLM APC only reuses floor(p/16) blocks; radix reuses all p.
Measure the append-only prefill cost: L-p new tokens attending over L KV.
  saved(p) = prefill(L) - append(L-p new over L KV)
Sweep divergence point p on L=8192; report vLLM vs radix effective saving.
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

NL, H, NKV, HD, HID, FFN = 28, 16, 8, 128, 1024, 3072


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def layer_prefill(x, w, kcache, vcache, causal_offset):
    # x: [qn, HID] new tokens; kcache/vcache: [NKV, S, HD] cached+new concat
    qn = x.shape[0]
    G = H // NKV
    h = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * w["ln1"]
    q = (h @ w["q"].T).view(qn, NKV, G, HD).permute(1, 2, 0, 3)
    o = F.scaled_dot_product_attention(q, kcache, vcache)
    o = o.permute(2, 0, 1, 3).reshape(qn, -1)
    x = x + o @ w["o"].T
    h = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * w["ln2"]
    return x + (F.silu(h @ w["g"].T) * (h @ w["u"].T)) @ w["d"].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=8192)
    args = ap.parse_args()
    dev = "npu:0"
    L = args.L
    w = [{k: torch.randn(s, dtype=torch.bfloat16, device=dev) * 0.02
          for k, s in [("ln1", HID), ("q", (H * HD, HID)), ("k", (NKV * HD, HID)),
                       ("v", (NKV * HD, HID)), ("o", (HID, H * HD)), ("ln2", HID),
                       ("g", (FFN, HID)), ("u", (FFN, HID)), ("d", (HID, FFN))]}
         for _ in range(NL)]

    def prefill(qn, kvlen):
        x = torch.randn(qn, HID, dtype=torch.bfloat16, device=dev)
        kc = torch.randn(NKV, 1, kvlen, HD, dtype=torch.bfloat16, device=dev)
        vc = torch.randn(NKV, 1, kvlen, HD, dtype=torch.bfloat16, device=dev)
        for wl in w:
            # NOTE: for append we ignore causal correctness — cost shape is what matters
            x = layer_prefill(x, wl, kc, vc, 0)
        torch.npu.synchronize()
        del x, kc, vc
        torch.npu.empty_cache()

    full_ms = bench(lambda: prefill(L, L))
    res = {"L": L, "full_ms": round(full_ms, 2), "rows": []}
    for p in [1024, 2048, 4096, 6144, 7168, 8000, 8176]:
        qn = L - p
        ap_ms = bench(lambda: prefill(qn, L))
        saved = full_ms - ap_ms
        frac = saved / full_ms
        res["rows"].append({"shared_p": p, "new_tokens": qn,
                            "append_ms": round(ap_ms, 2),
                            "saved_ms": round(saved, 2),
                            "saved_frac": round(frac, 3)})
        print(f"[E79] p={p:<5} reuse={100*p/L:5.1f}% append={ap_ms:7.2f}ms "
              f"saved={saved:6.2f}ms ({100*frac:.0f}%)", flush=True)

    res["note"] = ("vLLM APC reuses floor(p/16)*16; radix reuses all p. "
                   "saved cost measured as full-prefill minus append-prefill")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[E79] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
