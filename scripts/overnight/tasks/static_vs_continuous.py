"""E81: THE canonical question — static vs continuous batching, measured.

Workload: 32 requests, output lengths uniform [32,256] tokens, batch cap 16.
  static    : run batch until EVERY request finishes, then next batch
  continuous: finished seq leaves at its step; queued request joins instantly
Metric: total wall time, per-request e2e latency, mean active batch.
Caveat: uniform ctx=2048 for all (ctx dimension covered by E37/E39).
"""
import argparse
import json
import random
import time

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

NL, H, NKV, HD, HID, FFN = 28, 16, 8, 128, 1024, 3072
CTX, CAP, NREQ = 2048, 16, 32


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


def run(dev, mode, lengths, w):
    kc = torch.randn(NKV, CAP, CTX + 8, HD, dtype=torch.bfloat16, device=dev)
    vc = torch.randn(NKV, CAP, CTX + 8, HD, dtype=torch.bfloat16, device=dev)
    remaining = list(lengths)
    if mode == "static":
        batches = [remaining[i:i + CAP] for i in range(0, len(remaining), CAP)]
    else:
        batches = None
    lat = []
    t_start = time.perf_counter()
    active_bs = []

    if mode == "static":
        for bl in batches:
            todo = max(bl)
            while todo > 0:
                x = torch.randn(len(bl), HID, dtype=torch.bfloat16, device=dev)
                x = decode_step(x, kc, vc, w, CTX)
                torch.npu.synchronize()
                active_bs.append(len(bl))   # finished seqs keep occupying slots
                todo -= 1
                del x
    else:
        inflight = [[l, 0] for l in remaining[:CAP]]   # [remaining, steps]
        queue = remaining[CAP:]
        while inflight:
            x = torch.randn(len(inflight), HID, dtype=torch.bfloat16, device=dev)
            x = decode_step(x, kc, vc, w, CTX)
            torch.npu.synchronize()
            active_bs.append(len(inflight))
            nxt = []
            for rem, s in inflight:
                if rem - 1 > 0:
                    nxt.append([rem - 1, s + 1])
                else:
                    lat.append(s + 1)
            while queue and len(nxt) < CAP:
                nxt.append([queue.pop(0), 0])
            inflight = nxt
            del x
    total = time.perf_counter() - t_start
    lat.sort()
    return {"total_s": round(total, 1),
            "mean_active_bs": round(sum(active_bs) / len(active_bs), 1),
            "steps": len(active_bs),
            "lat_p50_steps": lat[len(lat) // 2] if lat else None,
            "lat_max_steps": lat[-1] if lat else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "npu:0"
    rng = random.Random(7)
    lengths = [rng.randint(32, 256) for _ in range(NREQ)]
    w = [{k: torch.randn(s, dtype=torch.bfloat16, device=dev) * 0.02
          for k, s in [("ln1", HID), ("q", (H * HD, HID)), ("k", (NKV * HD, HID)),
                       ("v", (NKV * HD, HID)), ("o", (HID, H * HD)), ("ln2", HID),
                       ("g", (FFN, HID)), ("u", (FFN, HID)), ("d", (HID, FFN))]}
         for _ in range(NL)]
    res = {"lengths": lengths}
    for mode in ["static", "continuous"]:
        r = run(dev, mode, lengths, w)
        res[mode] = r
        print(f"[E81] {mode}: {r}", flush=True)
    res["speedup"] = round(res["static"]["total_s"] / res["continuous"]["total_s"], 2)
    res["note"] = f"32 reqs, out_len U[32,256], cap={CAP}, ctx={CTX}"
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[E81] wrote {args.out} speedup={res['speedup']}x", flush=True)


if __name__ == "__main__":
    main()
