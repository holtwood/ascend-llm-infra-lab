#!/usr/bin/env python3
"""E8j: where does paged-KV cost actually go? Decode-step attention over
block-table K/V vs contiguous K/V on Ascend NPU.

Layouts compared (single decode step, q=[1,H,1,D]):
  contiguous : K,V pre-materialized -> SDPA baseline
  paged      : logical KV lives in P pages of bs tokens; gather per step
               then SDPA (what a naive page-table impl pays)
  paged_mixed: gather only the newest page (incremental view) — models
               decode where old pages are already materialized... kept for ref

Across KV len L in {8k,32k,64k} x page size bs in {16,64,256}.
Answers: the gather tax ratio and whether page size matters.
"""
import json, math, time
import torch, torch_npu
import torch.nn.functional as F

DEV = "npu:0"
DT = torch.bfloat16
H, D = 32, 128


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    q = torch.randn(1, H, 1, D, dtype=DT, device=DEV)
    res = {"device": torch.npu.get_device_name(0), "rows": []}
    for L in [8192, 32768, 65536]:
        kc = torch.randn(1, H, L, D, dtype=DT, device=DEV)
        vc = torch.randn(1, H, L, D, dtype=DT, device=DEV)
        t_cont = bench(lambda: F.scaled_dot_product_attention(q, kc, vc))
        row = {"L": L, "contiguous_ms": round(t_cont, 4)}
        for bs in [16, 64, 256]:
            npages = (L + bs - 1) // bs
            # page pool: [npages, H, bs, D]; block_table: logical order
            pool_k = torch.randn(npages, H, bs, D, dtype=DT, device=DEV)
            pool_v = torch.randn(npages, H, bs, D, dtype=DT, device=DEV)
            bt = torch.randperm(npages, device=DEV)  # shuffled pages

            def paged_step():
                kg = pool_k[bt].permute(1, 0, 2, 3).reshape(1, H, -1, D)[:, :, :L]
                vg = pool_v[bt].permute(1, 0, 2, 3).reshape(1, H, -1, D)[:, :, :L]
                return F.scaled_dot_product_attention(q, kg, vg)

            t_paged = bench(paged_step)
            row[f"bs{bs}"] = {"ms": round(t_paged, 4),
                              "tax": round(t_paged / t_cont, 2)}
            del pool_k, pool_v
        res["rows"].append(row)
        print(f"[e8j] L={L}: {row}", flush=True)
        del kc, vc
        torch.npu.empty_cache()
    print(json.dumps(res, indent=1))
    with open("/tmp/e8j_paged.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
