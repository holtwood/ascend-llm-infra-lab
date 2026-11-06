#!/usr/bin/env python3
"""Paged-KV gather tax matrix — extends E8j across block_size x seq_len.

Paged KV stored as blocks; decode step needs gather(block_table) -> SDPA.
Measures gather+SDPA vs contiguous SDPA tax, varying page granularity.

Usage: python3 paged_gather_matrix.py --out $OUT_DIR/paged_gather.json
"""
import argparse, json, time
import torch, torch_npu

B, NKV, D = 4, 8, 128
NQ = 32


def sdpa(q, k, v):
    Bq, S = q.shape[0], k.shape[2]
    G = NQ // NKV
    qq = q.view(Bq, NKV, G, 1, D).reshape(Bq * NKV, G, 1, D)
    kk = k.reshape(Bq * NKV, 1, S, D)
    vv = v.reshape(Bq * NKV, 1, S, D)
    out = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv)
    return out.reshape(Bq, NQ, 1, D)


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    results = []
    q = torch.randn(B, NQ, 1, D, device="npu:0", dtype=torch.bfloat16)
    for S in [4096, 8192, 16384, 32768]:
        for bs in [16, 32, 64, 128]:
            nblocks_per_seq = (S + bs - 1) // bs
            nblocks = B * nblocks_per_seq
            # paged pool: [nblocks, bs, NKV, D]
            pool_k = torch.randn(nblocks, bs, NKV, D, device="npu:0", dtype=torch.bfloat16)
            pool_v = torch.randn(nblocks, bs, NKV, D, device="npu:0", dtype=torch.bfloat16)
            # scattered pages = real paged KV; sequential = best-case baseline
            table_seq = torch.arange(nblocks, device="npu:0").view(B, nblocks_per_seq)
            table_rnd = torch.randperm(nblocks, device="npu:0").view(B, nblocks_per_seq)
            # contiguous reference
            kc = pool_k[table_seq.view(-1)].view(B, nblocks_per_seq * bs, NKV, D)[:, :S].transpose(1, 2).contiguous()
            vc = pool_v[table_seq.view(-1)].view(B, nblocks_per_seq * bs, NKV, D)[:, :S].transpose(1, 2).contiguous()
            def gather_sdpa(table):
                kg = pool_k[table.view(-1)].view(B, nblocks_per_seq * bs, NKV, D)[:, :S].transpose(1, 2)
                vg = pool_v[table.view(-1)].view(B, nblocks_per_seq * bs, NKV, D)[:, :S].transpose(1, 2)
                return sdpa(q, kg, vg)
            t_ref = bench(lambda: sdpa(q, kc, vc))
            t_seq = bench(lambda: gather_sdpa(table_seq))
            t_rnd = bench(lambda: gather_sdpa(table_rnd))
            results.append({"S": S, "block": bs,
                            "t_ref_us": round(t_ref, 1),
                            "t_paged_seq_us": round(t_seq, 1),
                            "t_paged_rnd_us": round(t_rnd, 1),
                            "tax_seq": round(t_seq / t_ref, 2),
                            "tax_rnd": round(t_rnd / t_ref, 2)})
            print(f"[paged] S={S} block={bs} tax_seq={t_seq/t_ref:.2f}x tax_rnd={t_rnd/t_ref:.2f}x", flush=True)
            del pool_k, pool_v, kc, vc, table_seq, table_rnd
            torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump({"B": B, "NKV": NKV, "D": D, "results": results}, f, indent=2)
    print(f"[paged] wrote {args.out}")
