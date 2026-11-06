"""E80: interview Q — "does swapping out a victim's KV slow the running decodes?"

vLLM preempts by copying victim KV blocks to host. If the D2H copy runs on a
side stream (dedicated DMA, per E75), the running batch's TPOT should barely
move; if it steals bandwidth/SM time, TPOT inflates.

Real measurement: 60 decode steps (bs=32, ctx=4k); during steps 15..45 a side
stream continuously D2H-copies 15MB blocks. Report TPOT pre/during/post.
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

NL, H, NKV, HD, HID, FFN = 28, 16, 8, 128, 1024, 3072
CTX, BS, BLK_BYTES = 4096, 32, 15_000_000


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "npu:0"
    torch.npu.set_device(dev)
    w = [{k: torch.randn(s, dtype=torch.bfloat16, device=dev) * 0.02
          for k, s in [("ln1", HID), ("q", (H * HD, HID)), ("k", (NKV * HD, HID)),
                       ("v", (NKV * HD, HID)), ("o", (HID, H * HD)), ("ln2", HID),
                       ("g", (FFN, HID)), ("u", (FFN, HID)), ("d", (HID, FFN))]}
         for _ in range(NL)]
    kc = torch.randn(NKV, BS, CTX + 8, HD, dtype=torch.bfloat16, device=dev)
    vc = torch.randn(NKV, BS, CTX + 8, HD, dtype=torch.bfloat16, device=dev)
    src = torch.empty(BLK_BYTES, dtype=torch.uint8, device=dev)
    host = torch.empty(BLK_BYTES, dtype=torch.uint8, pin_memory=True)
    side = torch.npu.Stream()

    trace = []
    SW0, SW1 = 15, 45
    for t in range(60):
        x = torch.randn(BS, HID, dtype=torch.bfloat16, device=dev)
        swapping = SW0 <= t < SW1
        if swapping:
            with torch.npu.stream(side):
                host.copy_(src, non_blocking=True)  # one 15MB block per step
        t0 = time.perf_counter()
        x = decode_step(x, kc, vc, w, CTX + (t % 8))
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        trace.append({"t": t, "ms": round(ms, 1), "swap": swapping})
        del x

    def st(rows):
        ms = sorted(r["ms"] for r in rows if r["t"] > 0)
        return {"p50": ms[len(ms) // 2], "p90": ms[int(len(ms) * .9)], "max": ms[-1],
                "n": len(ms)}
    res = {"pre": st([r for r in trace if r["t"] < SW0]),
           "during": st([r for r in trace if SW0 <= r["t"] < SW1]),
           "post": st([r for r in trace if r["t"] >= SW1]),
           "blk_mb": BLK_BYTES / 1e6, "trace": trace}
    print(f"[E80] pre={res['pre']} during={res['during']} post={res['post']}",
          flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[E80] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
