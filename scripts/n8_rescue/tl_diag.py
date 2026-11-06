import os
os.environ["TILELANG_ASCEND_MODE"] = "Developer"
import torch, torch_npu
import tilelang, tilelang.language as T
M, N = 8, 2560

@tilelang.jit(out_idx=[-1], target="npuir")
def k_log(M: int, N: int):
    @T.prim_func
    def kern(X: T.Tensor((M, N), "bfloat16"), O: T.Tensor((M, N), "bfloat16")):
        with T.Kernel(M, is_npu=True) as (bx, _):
            X_s = T.alloc_shared((N,), "bfloat16")
            O_s = T.alloc_shared((N,), "bfloat16")
            xf = T.alloc_fragment((N,), "float32")
            T.copy(X[bx, 0], X_s)
            for j in T.Parallel(N):
                xf[j] = T.log(T.cast(X_s[j], "float32") + 2.0)
            for j in T.Parallel(N):
                O_s[j] = T.cast(xf[j], "bfloat16")
            T.copy(O_s, O[bx, 0])
    return kern

@tilelang.jit(out_idx=[-1], target="npuir")
def k_exp(M: int, N: int):
    @T.prim_func
    def kern(X: T.Tensor((M, N), "bfloat16"), O: T.Tensor((M, N), "bfloat16")):
        with T.Kernel(M, is_npu=True) as (bx, _):
            X_s = T.alloc_shared((N,), "bfloat16")
            O_s = T.alloc_shared((N,), "bfloat16")
            xf = T.alloc_fragment((N,), "float32")
            T.copy(X[bx, 0], X_s)
            for j in T.Parallel(N):
                xf[j] = T.exp(-0.5 * T.cast(X_s[j], "float32"))
            for j in T.Parallel(N):
                O_s[j] = T.cast(xf[j], "bfloat16")
            T.copy(O_s, O[bx, 0])
    return kern

torch.manual_seed(0)
x = torch.randn(M, N, device="npu", dtype=torch.bfloat16) * 0.5
o1 = k_log(M, N)(x); torch.npu.synchronize()
r1 = torch.log(x.float() + 2.0)
print("log  max|diff|:", (o1.float() - r1).abs().max().item(), flush=True)
o2 = k_exp(M, N)(x); torch.npu.synchronize()
r2 = torch.exp(-0.5 * x.float())
print("exp  max|diff|:", (o2.float() - r2).abs().max().item(), flush=True)
print("log sample:", o1.float()[0, :4].tolist(), "vs", r1[0, :4].tolist(), flush=True)
print("exp sample:", o2.float()[0, :4].tolist(), "vs", r2[0, :4].tolist(), flush=True)
