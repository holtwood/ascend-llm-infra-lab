#!/usr/bin/env python3
"""v6: T.rsqrt is broken on 0.1.1.30-npuir (K_b dump); use 1.0/T.sqrt instead.
K_d: sqrt smoke. K_e: full fused RMSNorm (tree reduce, fp32 math, bf16 io)
     + correctness + eager-vs-tilelang timing (10wu+50it sync)."""
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
def k_d(M: int, N: int):
    @T.prim_func
    def kern(X: T.Tensor((M, N), "bfloat16"), O: T.Tensor((M, N), "bfloat16")):
        with T.Kernel(M, is_npu=True) as (bx, _):
            X_s = T.alloc_shared((N,), "bfloat16")
            O_s = T.alloc_shared((N,), "bfloat16")
            xf = T.alloc_fragment((N,), "float32")
            T.copy(X[bx, 0], X_s)
            for j in T.Parallel(N):
                xf[j] = 1.0 / T.sqrt(T.cast(X_s[j], "float32") + 2.0)
            for j in T.Parallel(N):
                O_s[j] = T.cast(xf[j], "bfloat16")
            T.copy(O_s, O[bx, 0])
    return kern


@tilelang.jit(out_idx=[-1], target="npuir")
def k_e(M: int, N: int):
    """Fused RMSNorm, one row per block: bf16 io, fp32 math, tree reduce,
    inv = 1/sqrt(mean+eps) (T.rsqrt broken in this build)."""
    @T.prim_func
    def kern(X: T.Tensor((M, N), "bfloat16"), W: T.Tensor((N,), "bfloat16"),
             O: T.Tensor((M, N), "bfloat16")):
        with T.Kernel(M, is_npu=True) as (bx, _):
            X_s = T.alloc_shared((N,), "bfloat16")
            W_s = T.alloc_shared((N,), "bfloat16")
            O_s = T.alloc_shared((N,), "bfloat16")
            xf = T.alloc_fragment((N,), "float32")
            T.copy(X[bx, 0], X_s)
            T.copy(W[0], W_s)
            for j in T.Parallel(N):
                xf[j] = T.cast(X_s[j], "float32") * T.cast(X_s[j], "float32")
            for j in T.Parallel(N // 2):
                xf[j] = xf[j] + xf[j + N // 2]
            for j in T.Parallel(N // 4):
                xf[j] = xf[j] + xf[j + N // 4]
            for j in T.Parallel(N // 8):
                xf[j] = xf[j] + xf[j + N // 8]
            for j in T.Parallel(N // 16):
                xf[j] = xf[j] + xf[j + N // 16]
            for j in T.Parallel(N // 32):
                xf[j] = xf[j] + xf[j + N // 32]
            for j in T.Parallel(N // 64):
                xf[j] = xf[j] + xf[j + N // 64]
            for j in T.Parallel(N // 128):
                xf[j] = xf[j] + xf[j + N // 128]
            for j in T.Parallel(N // 256):
                xf[j] = xf[j] + xf[j + N // 256]
            for j in T.Parallel(N // 512):
                xf[j] = xf[j] + xf[j + N // 512]
            for j in T.Parallel(N // 1024):
                xf[j] = xf[j] + xf[j + N // 1024]
            for j in T.Parallel(N // 2048):
                xf[j] = xf[j] + xf[j + N // 2048]
            for j in T.Parallel(N):
                O_s[j] = T.cast(T.cast(X_s[j], "float32")
                                * (1.0 / T.sqrt(xf[j] / N + 1e-6))
                                * (1.0 + T.cast(W_s[j], "float32")), "bfloat16")
            T.copy(O_s, O[bx, 0])
    return kern


def main():
    torch.manual_seed(0)
    print(f"v6 sqrt-bypass  tilelang {tilelang.__version__} N={N}", flush=True)

    x = torch.randn(8, N, device="npu", dtype=torch.bfloat16) * 0.5
    ref_d = 1.0 / torch.sqrt(x.float() + 2.0)
    od = k_d(8, N)(x)
    torch.npu.synchronize()
    print(f"K_d sqrt smoke max|diff|={(od.float()-ref_d).abs().max().item():.3e}",
          flush=True)

    rows = []
    for M in M_SCAN:
        x = torch.randn(M, N, device="npu", dtype=torch.bfloat16) * 0.5
        w = torch.randn(N, device="npu", dtype=torch.bfloat16) * 0.1
        ref = rms_chain(x, w)
        kern = k_e(M, N)
        out = kern(x, w)
        torch.npu.synchronize()
        diff = (out.float() - ref.float()).abs().max().item()
        te = bench_us(lambda: rms_chain(x, w))
        tt = bench_us(lambda: kern(x, w))
        sp = te / tt
        print(f"M={M}: eager {te:8.1f} us | tilelang {tt:8.1f} us | "
              f"max|diff|={diff:.3e} | speedup {sp:.2f}x", flush=True)
        rows.append((M, te, tt, diff, sp))
    print("\n=== summary (us/iter, bf16, N=2560, 10wu+50it sync) ===")
    print(f"{'M':>6} {'eager':>10} {'tilelang':>10} {'max|diff|':>10} {'speedup':>8}")
    for M, te, tt, diff, sp in rows:
        print(f"{M:>6} {te:>10.1f} {tt:>10.1f} {diff:>10.3e} {sp:>7.2f}x")


if __name__ == "__main__":
    main()
