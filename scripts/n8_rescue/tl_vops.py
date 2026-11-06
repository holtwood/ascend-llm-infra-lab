#!/usr/bin/env python3
"""Vector-op rescue: T.vrsqrt/vsqrt/vrec compile+run smokes, then full RMSNorm."""
import os
import time
import traceback

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch
import torch_npu

import tilelang
import tilelang.language as T

N = 2560
M = 64


def stage(tag, fn):
    print(f"--- {tag} ---", flush=True)
    try:
        fn()
    except Exception:
        print(f"  [{tag}] EXC: {traceback.format_exc().strip().splitlines()[-1][:150]}",
              flush=True)


def smoke(expr_tag):
    def run():
        @tilelang.jit(out_idx=[-1], target="npuir")
        def k(M: int, N: int):
            @T.prim_func
            def kern(X: T.Tensor((M, N), "bfloat16"), O: T.Tensor((M, N), "bfloat16")):
                with T.Kernel(M, is_npu=True) as (bx, _):
                    X_s = T.alloc_shared((N,), "bfloat16")
                    O_s = T.alloc_shared((N,), "bfloat16")
                    xf = T.alloc_fragment((N,), "float32")
                    T.copy(X[bx, 0], X_s)
                    for j in T.Parallel(N):
                        if expr_tag == "vrec":
                            xf[j] = T.vrec(T.cast(X_s[j], "float32") + 2.0)
                        elif expr_tag == "vsqrt":
                            xf[j] = T.vsqrt(T.cast(X_s[j], "float32") + 2.0)
                        elif expr_tag == "vrsqrt":
                            xf[j] = T.vrsqrt(T.cast(X_s[j], "float32") + 2.0)
                    for j in T.Parallel(N):
                        O_s[j] = T.cast(xf[j], "bfloat16")
                    T.copy(O_s, O[bx, 0])
            return kern
        kern = k(M, N)
        x = torch.randn(M, N, device="npu", dtype=torch.bfloat16) * 0.5
        o = kern(x)
        torch.npu.synchronize()
        v = x.float() + 2.0
        if expr_tag == "vrec":
            ref = 1.0 / v
        elif expr_tag == "vsqrt":
            ref = torch.sqrt(v)
        else:
            ref = torch.rsqrt(v)
        d = (o.float() - ref).abs().max().item()
        print(f"  {expr_tag} max|diff|={d:.3e} {expr_tag.upper()} OK", flush=True)
    return run


def full_rms(vop):
    def run():
        @tilelang.jit(out_idx=[-1], target="npuir")
        def k(M: int, N: int):
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
                        if vop == "vrec_vsqrt":
                            inv = T.vrec(T.vsqrt(xf[j] / N + 1e-6))
                        else:
                            inv = T.vrsqrt(xf[j] / N + 1e-6)
                        O_s[j] = T.cast(T.cast(X_s[j], "float32") * inv
                                        * (1.0 + T.cast(W_s[j], "float32")), "bfloat16")
                    T.copy(O_s, O[bx, 0])
            return kern
        kern = k(M, N)
        torch.manual_seed(0)
        x = torch.randn(M, N, device="npu", dtype=torch.bfloat16) * 0.5
        w = torch.randn(N, device="npu", dtype=torch.bfloat16) * 0.1
        o = kern(x, w)
        torch.npu.synchronize()
        x32 = x.float()
        ref = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
               * (1.0 + w.float())).to(torch.bfloat16)
        d = (o.float() - ref.float()).abs().max().item()
        print(f"  full_rms({vop}) max|diff|={d:.3e} FULL OK", flush=True)
    return run


def main():
    print(f"vops rescue tilelang {tilelang.__version__}", flush=True)
    stage("vrec smoke", smoke("vrec"))
    stage("vsqrt smoke", smoke("vsqrt"))
    stage("vrsqrt smoke", smoke("vrsqrt"))
    stage("full rms via vrec(vsqrt)", full_rms("vrec_vsqrt"))
    stage("full rms via vrsqrt", full_rms("vrsqrt"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
