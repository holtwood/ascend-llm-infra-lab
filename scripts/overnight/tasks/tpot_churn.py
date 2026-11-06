"""E71: TPOT distribution under continuous-batching churn.

Real serving doesn't have a fixed batch — requests join and leave every step,
and prefill chunks share steps with decodes. TPOT is therefore a DISTRIBUTION,
not a constant; its tail sets the SLO margin.

Runs real 28-layer decode steps while batch size does a random walk in
[1,64]; scenario B additionally injects a 2048-token prefill chunk into a
step every ~12 steps (chunked-prefill style). Reports per-step ms trace,
p50/p90/p99/max, and mean concurrent batch.

Simplification (stated honestly): all active sequences share ctx=4096
(mid-life); the ctx dimension was characterized separately in E37/E39.
"""
import argparse
import json
import random
import time

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

NL, H, NKV, HD, HID, FFN = 28, 16, 8, 128, 1024, 3072
CTX = 4096
BMAX = 64


def layer_step(x, kc, vc, w, pos):
    G = H // NKV
    L = pos
    h = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * w["ln1"]
    q = h @ w["q"].T
    k = h @ w["k"].T
    v = h @ w["v"].T
    b = x.shape[0]
    kc[:, :b, pos, :] = k.view(b, NKV, HD).permute(1, 0, 2)
    vc[:, :b, pos, :] = v.view(b, NKV, HD).permute(1, 0, 2)
    qh = q.view(b, NKV, G, HD).permute(1, 0, 2, 3)
    att = F.scaled_dot_product_attention(
        qh, kc[:, :b, :pos + 1, :], vc[:, :b, :pos + 1, :])
    att = att.permute(1, 0, 2, 3).reshape(b, -1)
    x = x + att @ w["o"].T
    h = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * w["ln2"]
    x = x + (F.silu(h @ w["g"].T) * (h @ w["u"].T)) @ w["d"].T
    return x


def run_trace(dev, steps, inject_prefill, seed=0):
    rng = random.Random(seed)
    w = [{k: torch.randn(s, dtype=torch.bfloat16, device=dev) * 0.02
          for k, s in [("ln1", HID), ("q", (H * HD, HID)), ("k", (NKV * HD, HID)),
                       ("v", (NKV * HD, HID)), ("o", (HID, H * HD)), ("ln2", HID),
                       ("g", (FFN, HID)), ("u", (FFN, HID)), ("d", (HID, FFN))]}
         for _ in range(NL)]
    kc = torch.randn(NKV, BMAX, CTX + 1, HD, dtype=torch.bfloat16, device=dev)
    vc = torch.randn(NKV, BMAX, CTX + 1, HD, dtype=torch.bfloat16, device=dev)
    pf_w = torch.randn(HID, HID, dtype=torch.bfloat16, device=dev) if inject_prefill else None

    bs = 8
    trace = []
    torch.npu.synchronize()
    for t in range(steps):
        bs = max(1, min(BMAX, bs + rng.choice([-4, -2, -1, 0, 1, 2, 4])))
        do_pf = inject_prefill and (t % 12 == 5)
        x = torch.randn(bs, HID, dtype=torch.bfloat16, device=dev)
        t0 = time.perf_counter()
        for w_l in w:
            x = layer_step(x, kc, vc, w_l, CTX)
        if do_pf:
            p = torch.randn(2048, HID, dtype=torch.bfloat16, device=dev)
            for _ in range(NL):
                p = F.silu(p @ pf_w.T) @ pf_w.T
        torch.npu.synchronize()
        trace.append({"t": t, "bs": bs, "ms": round((time.perf_counter() - t0) * 1e3, 2),
                      "prefill": do_pf})
        if t % 20 == 0:
            print(f"[E71] step {t} bs={bs} ms={trace[-1]['ms']} pf={do_pf}", flush=True)
        del x
    return trace


def stats(trace, key_filter=None):
    ms = [r["ms"] for r in trace if key_filter is None or key_filter(r)]
    ms.sort()
    n = len(ms)
    q = lambda p: round(ms[min(n - 1, int(p * n))], 2)
    return {"n": n, "mean": round(sum(ms) / n, 2), "p50": q(.5), "p90": q(.9),
            "p99": q(.99), "max": ms[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()
    dev = "npu:0"

    for tag, inj in [("churn", False), ("churn_prefill", True)]:
        tr = run_trace(dev, args.steps, inj, seed=hash(tag) % 100)
        no_pf = stats(tr, lambda r: not r["prefill"])
        pf = stats(tr, lambda r: r["prefill"]) if inj else None
        all_s = stats(tr)
        print(f"[E71] {tag}: all={all_s}  no_pf={no_pf}  pf={pf}", flush=True)
        if tag not in [r.get("tag") for r in []]:
            pass
        globals().setdefault("_res", {}).setdefault(tag, {})
        _res[tag] = {"all": all_s, "decode_only": no_pf, "prefill_steps": pf,
                     "trace": tr}

    res = globals()["_res"]
    res["note"] = (f"ctx={CTX} fixed; bs random-walk [1,{BMAX}]; "
                   "prefill injection = 2048-token chunk every ~12 steps")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[E71] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
