#!/usr/bin/env python3
"""Bisect the core dump: assemble RMSNorm from verified add-kernel primitives,
adding back one suspect at a time. Kernels run in order; each prints+syncs
before the next, so a dump preserves everything up to the culprit."""
import os
import time

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch
import torch_npu

import tilelang
import tilelang.language as T

N = 2560
M = 64


def chk(tag, out, ref):
    torch.npu.synchronize()
    d = (out.float() - ref.float()).abs().max().item()
    print(f"  [{tag}] PASS max|diff|={d:.3e}", flush=True)


# K_a: bf16 copy + fp32 fragment square + (1+w) cast-out — no reduce, no rsqrt
@tilelang.jit(out_idx=[-1], target="npuir")
def k_a(M: int, N: int):
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
            for j in T.Parallel(N):
                O_s[j] = T.cast(xf[j] * (1.0 + T.cast(W_s[j], "float32")), "bfloat16")
            T.copy(O_s, O[bx, 0])
    return kern


# K_b: K_a + T.rsqrt on a fragment row (no reduce; rsqrt of x^2-row-mean stand-in)
@tilelang.jit(out_idx=[-1], target="npuir")
def k_b(M: int, N: int):
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
            for j in T.Parallel(N):
                O_s[j] = T.cast(T.rsqrt(xf[j] + 1e-6) * (1.0 + T.cast(W_s[j], "float32")),
                                "bfloat16")
            T.copy(O_s, O[bx, 0])
    return kern


# K_c: full shape — row vector (1 row per block), tree reduce, rsqrt, (1+w)
@tilelang.jit(out_idx=[-1], target="npuir")
def k_c(M: int, N: int):
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
            for j in T.Parallel(N // 2, ):
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
                                * T.rsqrt(xf[j] / N + 1e-6)
                                * (1.0 + T.cast(W_s[j], "float32")), "bfloat16")
            T.copy(O_s, O[bx, 0])
    return kern


def main():
    torch.manual_seed(0)
    x = torch.randn(M, N, device="npu", dtype=torch.bfloat16) * 0.5
    w = torch.randn(N, device="npu", dtype=torch.bfloat16) * 0.1
    x32 = x.float()
    ref_rms = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
               * (1.0 + w.float())).to(torch.bfloat16)

    print(f"bisect tilelang {tilelang.__version__} M={M} N={N}", flush=True)
    print("K_a: fragment+cast, no reduce/rsqrt", flush=True)
    oa = k_a(M, N)(x, w)
    chk("a", oa, x32.pow(2) * (1.0 + w.float()))   # same math, fp32 ref cast

    print("K_b: + T.rsqrt elementwise", flush=True)
    ob = k_b(M, N)(x, w)
    chk("b", ob, torch.rsqrt(x32.pow(2) + 1e-6) * (1.0 + w.float()))

    print("K_c: full rmsnorm (tree reduce + rsqrt + 1+w)", flush=True)
    oc = k_c(M, N)(x, w)
    chk("c", oc, ref_rms)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
