import os
os.environ["TILELANG_ASCEND_MODE"] = "Developer"
import torch, torch_npu
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1], target="npuir")
def vec_add(n, dtype="float16"):
    @T.prim_func
    def add(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, 1024), is_npu=True) as (bx, _):
            A_s = T.alloc_shared((1024,), dtype)
            B_s = T.alloc_shared((1024,), dtype)
            C_s = T.alloc_shared((1024,), dtype)
            T.copy(A[bx * 1024], A_s)
            T.copy(B[bx * 1024], B_s)
            for i in T.Parallel(1024):
                C_s[i] = A_s[i] + B_s[i]
            T.copy(C_s, C[bx * 1024])
    return add

n = 4096
k = vec_add(n)
a = torch.randn(n).npu().half(); b = torch.randn(n).npu().half()
c = k(a, b)
ref = a + b
print("max diff:", (c - ref).abs().max().item())
assert torch.equal(c, ref), "MISMATCH"
print("tl_add PASS")
