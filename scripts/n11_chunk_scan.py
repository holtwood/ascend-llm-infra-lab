#!/usr/bin/env python3
"""N11: chunk-size sweep for chunked_gated_delta_rule (pure-torch GDN prefill kernel).

Standalone microbench: no model load, random tensors straight into the kernel.
q/k/v (1, L, 32, 128) bf16; L in {1024, 4096}; chunk in {32, 64, 128, 256}.
10 warmup + 50 timed per cell, torch synchronize bracketing.

The kernel already takes chunk_size as a keyword (gdn_kernels.py) so this is a
zero-diff sweep. Results stream to stdout as JSON lines so partial runs are
still usable if the box goes away.

Run: ASCEND_RT_VISIBLE_DEVICES=0 python3 scripts/n11_chunk_scan.py
"""

import json
import sys
import time

import torch

sys.path.insert(0, "examples/engine_np")
from gdn_kernels import chunked_gated_delta_rule

B, H, DK, DV = 1, 32, 128, 128
LENS = [1024, 4096]
CHUNKS = [32, 64, 128, 256]
WARMUP, ITERS = 10, 50


def bench_one(L, chunk):
    torch.manual_seed(0)
    q = torch.randn(B, L, H, DK, dtype=torch.bfloat16, device="npu")
    k = torch.randn(B, L, H, DK, dtype=torch.bfloat16, device="npu")
    v = torch.randn(B, L, H, DV, dtype=torch.bfloat16, device="npu")
    g = -torch.rand(B, L, H, dtype=torch.bfloat16, device="npu") * 0.1
    beta = torch.sigmoid(torch.randn(B, L, H, dtype=torch.bfloat16, device="npu"))

    for _ in range(WARMUP):
        out, state = chunked_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        out, state = chunked_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk)
    torch.npu.synchronize()
    wall = (time.perf_counter() - t0) / ITERS * 1e3

    # one extra sync before start so warmup latency can't bleed into the timer
    return {"L": L, "chunk": chunk, "ms_per_call": round(wall, 3),
            "n_chunks": L // chunk + (1 if L % chunk else 0)}


def main():
    assert torch.npu.is_available(), "no NPU"
    torch.npu.set_device(0)
    print(f"[n11] device={torch.npu.get_device_name(0)} "
          f"shape=({B},{H},{DK}/{DV}) warmup={WARMUP} iters={ITERS}", flush=True)

    # sanity: kernel result must not depend on chunk size (math equivalence)
    torch.manual_seed(0)
    L0 = 300  # exercises the pad branch for every chunk size
    q = torch.randn(B, L0, H, DK, dtype=torch.bfloat16, device="npu")
    k = torch.randn(B, L0, H, DK, dtype=torch.bfloat16, device="npu")
    v = torch.randn(B, L0, H, DV, dtype=torch.bfloat16, device="npu")
    g = -torch.rand(B, L0, H, dtype=torch.bfloat16, device="npu") * 0.1
    beta = torch.sigmoid(torch.randn(B, L0, H, dtype=torch.bfloat16, device="npu"))
    ref = None
    for c in CHUNKS:
        o, s = chunked_gated_delta_rule(q, k, v, g, beta, chunk_size=c)
        if ref is None:
            ref = (o.float(), s)
        else:
            d = (o.float() - ref[0]).abs().max().item()
            ds = (s - ref[1]).abs().max().item()
            print(f"[check] chunk={c}: max|out-ref|={d:.2e} max|state-ref|={ds:.2e}",
                  flush=True)

    results = []
    for L in LENS:
        for chunk in CHUNKS:
            try:
                r = bench_one(L, chunk)
            except Exception as e:  # record infeasible combos honestly
                r = {"L": L, "chunk": chunk, "error": f"{type(e).__name__}: {e}"[:200]}
            results.append(r)
            print("[row] " + json.dumps(r), flush=True)

    ok = [r for r in results if "ms_per_call" in r]
    for L in LENS:
        cells = [r for r in ok if r["L"] == L]
        if not cells:
            continue
        best = min(cells, key=lambda r: r["ms_per_call"])
        print(f"[best] L={L}: chunk={best['chunk']} ({best['ms_per_call']} ms)",
              flush=True)

    with open("results_n11.json", "w") as f:
        json.dump({"results": results, "done": len(results), "planned": len(LENS) * len(CHUNKS)},
                  f, indent=1)
    print("[n11] done", flush=True)


if __name__ == "__main__":
    main()
