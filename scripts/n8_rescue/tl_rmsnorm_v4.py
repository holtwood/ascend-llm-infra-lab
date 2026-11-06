#!/usr/bin/env python3
"""v4: v1 structure, T.reduce_sum replaced by hand-unrolled tree reduce."""
import os
import time

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch
import torch_npu

import tilelang
import tilelang.language as T

EPS = 1e-6
N = 2560
M_SCAN = [64, 2048]
WARMUP = 10
ITERS = 50
DT = "bfloat16"


def rms_chain(x, w):
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    out = out * (1.0 + w.float())
    return out.type_as(x)


def bench_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / ITERS


@tilelang.jit(out_idx=[-1], target="npuir")
def rms_kern(M: int, N: int, bM: int):
    @T.prim_func
    def kern(X: T.Tensor((M, N), "bfloat16"), W: T.Tensor((N,), "bfloat16"),
             O: T.Tensor((M, N), "bfloat16")):
        with T.Kernel(T.ceildiv(M, bM), is_npu=True) as (bx, _):
            X_s = T.alloc_shared((bM, N), "float32")
            W_s = T.alloc_shared((N,), "float32")
            ss = T.alloc_fragment((bM, N), "float32")
            inv_r = T.alloc_fragment((bM,), "float32")
            O_s = T.alloc_shared((bM, N), "bfloat16")
            T.copy(X[bx * bM, 0], X_s)
            T.copy(W[0], W_s)
            for i, j in T.Parallel(bM, N):
                ss[i, j] = X_s[i, j] * X_s[i, j]
            # tree reduce row-sum into ss[:, 0], N=2560 -> 11 halving steps
            for i, j in T.Parallel(bM, 1280):
                ss[i, j] = ss[i, j] + ss[i, j + 1280]
            for i, j in T.Parallel(bM, 640):
                ss[i, j] = ss[i, j] + ss[i, j + 640]
            for i, j in T.Parallel(bM, 320):
                ss[i, j] = ss[i, j] + ss[i, j + 320]
            for i, j in T.Parallel(bM, 160):
                ss[i, j] = ss[i, j] + ss[i, j + 160]
            for i, j in T.Parallel(bM, 80):
                ss[i, j] = ss[i, j] + ss[i, j + 80]
            for i, j in T.Parallel(bM, 40):
                ss[i, j] = ss[i, j] + ss[i, j + 40]
            for i, j in T.Parallel(bM, 20):
                ss[i, j] = ss[i, j] + ss[i, j + 20]
            for i, j in T.Parallel(bM, 10):
                ss[i, j] = ss[i, j] + ss[i, j + 10]
            for i, j in T.Parallel(bM, 5):
                ss[i, j] = ss[i, j] + ss[i, j + 5]
            for i, j in T.Parallel(bM, 2):
                ss[i, j] = ss[i, j] + ss[i, j + 2]
            for i, j in T.Parallel(bM, 1):
                ss[i, j] = ss[i, j] + ss[i, j + 1]
            for i in T.Parallel(bM):
                inv_r[i] = T.rsqrt(ss[i, 0] / N + 1e-6)
            for i, j in T.Parallel(bM, N):
                O_s[i, j] = T.cast(X_s[i, j] * inv_r[i] * (1.0 + W_s[j]), "bfloat16")
            T.copy(O_s, O[bx * bM, 0])
    return kern


def main():
    torch.manual_seed(0)
    print(f"v4 hand-unrolled tree  tilelang {tilelang.__version__}  bf16 N={N} "
          f"{WARMUP}wu+{ITERS}it", flush=True)
    rows = []
    for M in M_SCAN:
        x = (torch.randn(M, N, device="npu", dtype=torch.bfloat16) * 0.5)
        w = (torch.randn(N, device="npu", dtype=torch.bfloat16) * 0.1)
        ref = rms_chain(x, w)
        print(f"\n-- M={M} --", flush=True)
        te = bench_us(lambda: rms_chain(x, w))
        print(f"  eager.rms_chain   {te:10.1f} us", flush=True)
        bm = int(os.environ.get("TL_BM", "4"))
        kern = rms_kern(M, N, bm)
        out = kern(x, w)
        torch.npu.synchronize()
        diff = (out.float() - ref.float()).abs().max().item()
        tt = bench_us(lambda: kern(x, w))
        sp = te / tt
        print(f"  tilelang.rmsnorm  {tt:10.1f} us  max|diff|={diff:.3e}  "
              f"speedup={sp:.2f}x", flush=True)
        rows.append((M, te, tt, diff, sp))
    print("\n=== summary (us/iter, bf16, N=2560) ===")
    print(f"{'M':>6} {'eager':>10} {'tilelang':>10} {'max|diff|':>10} {'speedup':>8}")
    for M, te, tt, diff, sp in rows:
        print(f"{M:>6} {te:>10.1f} {tt:>10.1f} {diff:>10.3e} {sp:>7.2f}x")


if __name__ == "__main__":
    main()
