#!/usr/bin/env python3
"""N8: fusion payoff benchmark — eager AIV chains vs TileLang kernel vs torch.compile.

Question (from N2): eager prefill spends 68.9% on AIV small ops (Mul/Slice/Add/
ReduceMean...), decode is 79% host overhead. What does fusing the RMSNorm
family — the dominant AIV chain in every decoder layer — actually buy?

Sections (each independently guarded; one failing doesn't kill the rest):

  [eager]    the engine-exact RMSNorm chain (9 ops) and add+rmsnorm chain
             (10 ops, the decoder-layer residual+norm pattern), M-scan at
             N=2560 (Qwen3.5-4B hidden) + N=9728 spot check.
  [tilelang] boundary-safe fused add+rmsnorm kernel (bf16 in, fp32 norm math,
             zero-init delta weight `out = rmsnorm(x)*(1+w)` per Qwen3.5),
             M-scan; host-dispatch-only time recorded alongside end-to-end.
  [compile]  torch.compile over the same eager chains, backend auto-probed
             (inductor / torchair on NPU).

Timing: 10 warmup + 100 timed iters, torch.npu.synchronize brackets.
  t_sync  = end-to-end per-iter wall time (device work included)
  t_async = host-dispatch-only per-iter time (no sync inside; queue drained
            before/after, so it measures pure Python+launch overhead)
Correctness: max|diff| against the fp32 eager chain (the engine's reference).

Run (NPU host):  cd /data/n8 && python3 n8_rmsnorm_bench.py
Output:          n8_rmsnorm_bench.json + stdout table.
"""

import json
import statistics
import time
import traceback

import torch

EPS = 1e-6
M_SCAN = [1, 8, 64, 512, 2048]
N_MAIN = 2560
N_SPOT = 9728
WARMUP = 10
ITERS = 100
OUT_JSON = "n8_rmsnorm_bench.json"


def t_sync(fn):
    """Per-iter ms, sync-bracketed."""
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3 / ITERS


def t_async(fn):
    """Host-dispatch-only ms/iter: no sync inside the loop; queue drained
    before and after. First call of the loop does a sync to guarantee the
    queue starts empty."""
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    return (t1 - t0) * 1e3 / ITERS


def bench(tag, fn, results):
    fn()                                   # compile/first-touch outside timing
    fn()
    ts, ta = t_sync(fn), t_async(fn)
    results[tag] = {"t_sync_ms": ts, "t_async_ms": ta}
    print(f"    {tag:34s} sync {ts*1e3:9.1f} us | host-only {ta*1e3:9.1f} us",
          flush=True)


# --------------------------------------------------------------------------- #
# eager chains (engine-exact: examples/engine_np/model_qwen35.py RMSNorm)
# --------------------------------------------------------------------------- #

def rms_chain(x, w):
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    out = out * (1.0 + w.float())
    return out.type_as(x)


def add_rms_chain(x1, x2, w):
    """Decoder-layer residual+norm: hidden = residual + delta; norm(hidden)."""
    s = x1 + x2
    return s, rms_chain(s, w)


def ref_add_rms(x1, x2, w):
    """fp64 ground truth for correctness of every path."""
    s = (x1.double() + x2.double())
    inv = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + EPS)
    return s, (s * inv * (1.0 + w.double())).to(x1.dtype)


# --------------------------------------------------------------------------- #
# TileLang kernel: boundary-safe fused add+rmsnorm
# --------------------------------------------------------------------------- #

PASS_CONFIGS = None      # filled lazily (needs tilelang import)
_VEC_NUM = 2


def _tiling(M, N):
    """block_M/block_N keeping the fp32 UB footprint under ~160 KB/VEC lane."""
    block_N = min(512, N)
    while N % block_N != 0:
        block_N -= 128
    max_rows = max(1, 160 * 1024 // (12 * block_N * 4))   # ~12 live fp32 tiles
    block_M = 2                                           # decode floor: ROWS=1
    while block_M < max_rows and (block_M * 2) <= max_rows:
        block_M *= 2
    return block_M, block_N


def make_tl_add_rmsnorm(M, N, eps, dtype="bfloat16"):
    import tilelang
    from tilelang import language as T

    global PASS_CONFIGS
    if PASS_CONFIGS is None:
        PASS_CONFIGS = {
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        }

    block_M, block_N = _tiling(M, N)
    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    ROWS = block_M // _VEC_NUM
    tile_elems = ROWS * block_N
    is_aligned = (M % block_M == 0)          # Python-level, baked at codegen
    assert N % block_N == 0, "only the M boundary is handled"
    acc, CAST_IN, CAST_OUT = "float32", "CAST_NONE", "CAST_RINT"

    @tilelang.jit(out_idx=[3, 4], pass_configs=PASS_CONFIGS)
    def kern(x1: T.Tensor((M, N), dtype), x2: T.Tensor((M, N), dtype),
             gamma: T.Tensor((N,), dtype), x_out: T.Tensor((M, N), dtype),
             y: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            in0 = T.alloc_ub([ROWS, block_N], dtype)
            in1 = T.alloc_ub([ROWS, block_N], dtype)
            xf = T.alloc_ub([ROWS, block_N], acc)              # x1+x2 in fp32
            tmp = T.alloc_ub([ROWS, block_N], acc)
            sq_acc = T.alloc_ub([ROWS, block_N], acc)
            sq_row = T.alloc_ub([ROWS, 1], acc)
            inv_rms = T.alloc_ub([ROWS, 1], acc)
            inv_tile = T.alloc_ub([ROWS, block_N], acc)
            g1d = T.alloc_ub([block_N], dtype)
            g1f = T.alloc_ub([block_N], acc)
            g_one = T.alloc_ub([block_N], acc)                 # ones for (1+gamma)
            g_bc = T.alloc_ub([ROWS, block_N], acc)
            xout_cast = T.alloc_ub([ROWS, block_N], dtype)

            with T.Scope("V"):
                row0 = cid * block_M + vid * ROWS
                # device-side row clamp for the possibly-partial last tile
                remaining = M - cid * block_M
                vid_rem = T.if_then_else(remaining > vid * ROWS,
                                         remaining - vid * ROWS, 0)
                valid_rows = T.min(ROWS, vid_rem)

                T.tile.fill(sq_acc, 0.0)
                T.tile.fill(g_one, 1.0)

                if is_aligned:
                    # PASS 1: x_out = x1+x2 (write back to GM), accumulate x_out^2
                    for by in T.serial(n_num):
                        c0 = by * block_N
                        T.copy(x1[row0:row0 + ROWS, c0:c0 + block_N], in0)
                        T.copy(x2[row0:row0 + ROWS, c0:c0 + block_N], in1)
                        T.tile.cast(xf, in0, mode=CAST_IN, count=tile_elems)
                        T.tile.cast(tmp, in1, mode=CAST_IN, count=tile_elems)
                        T.tile.add(xf, xf, tmp)
                        T.tile.mul_add_dst(sq_acc, xf, xf)
                        T.tile.cast(xout_cast, xf, mode=CAST_OUT, count=tile_elems)
                        T.copy(xout_cast, x_out[row0:row0 + ROWS, c0:c0 + block_N])

                    # inter-pass: inv_rms = rsqrt(mean(x_out^2) + eps)
                    T.reduce_sum(sq_acc, sq_row, dim=-1)
                    T.tile.mul(sq_row, sq_row, T.cast(1.0 / N, acc))
                    T.tile.add(sq_row, sq_row, T.cast(eps, acc))
                    T.tile.rsqrt(inv_rms, sq_row)
                    T.tile.broadcast(inv_tile, inv_rms)

                    # PASS 2: y = x_out * inv_rms * (1 + gamma), engine fp32 math
                    for by in T.serial(n_num):
                        c0 = by * block_N
                        T.copy(x_out[row0:row0 + ROWS, c0:c0 + block_N], in0)
                        T.tile.cast(xf, in0, mode=CAST_IN, count=tile_elems)
                        T.tile.mul(xf, xf, inv_tile)
                        T.copy(gamma[c0:c0 + block_N], g1d)
                        T.tile.cast(g1f, g1d, mode=CAST_IN, count=block_N)
                        T.tile.add(g1f, g1f, g_one)            # 1 + gamma
                        T.tile.broadcast(g_bc, g1f)
                        T.tile.mul(xf, xf, g_bc)
                        T.tile.cast(xout_cast, xf, mode=CAST_OUT, count=tile_elems)
                        T.copy(xout_cast, y[row0:row0 + ROWS, c0:c0 + block_N])
                else:
                    # boundary path: zero-fill + row-clamped partial copies
                    for by in T.serial(n_num):
                        c0 = by * block_N
                        T.tile.fill(in0, 0.0)
                        T.tile.fill(in1, 0.0)
                        T.copy(x1[row0:row0 + valid_rows, c0:c0 + block_N],
                               in0[0:valid_rows, 0:block_N])
                        T.copy(x2[row0:row0 + valid_rows, c0:c0 + block_N],
                               in1[0:valid_rows, 0:block_N])
                        T.tile.cast(xf, in0, mode=CAST_IN, count=tile_elems)
                        T.tile.cast(tmp, in1, mode=CAST_IN, count=tile_elems)
                        T.tile.add(xf, xf, tmp)
                        T.tile.mul_add_dst(sq_acc, xf, xf)
                        T.tile.cast(xout_cast, xf, mode=CAST_OUT, count=tile_elems)
                        T.copy(xout_cast[0:valid_rows, 0:block_N],
                               x_out[row0:row0 + valid_rows, c0:c0 + block_N])

                    T.reduce_sum(sq_acc, sq_row, dim=-1)
                    T.tile.mul(sq_row, sq_row, T.cast(1.0 / N, acc))
                    T.tile.add(sq_row, sq_row, T.cast(eps, acc))
                    T.tile.rsqrt(inv_rms, sq_row)
                    T.tile.broadcast(inv_tile, inv_rms)

                    for by in T.serial(n_num):
                        c0 = by * block_N
                        T.tile.fill(in0, 0.0)
                        T.copy(x_out[row0:row0 + valid_rows, c0:c0 + block_N],
                               in0[0:valid_rows, 0:block_N])
                        T.tile.cast(xf, in0, mode=CAST_IN, count=tile_elems)
                        T.tile.mul(xf, xf, inv_tile)
                        T.copy(gamma[c0:c0 + block_N], g1d)
                        T.tile.cast(g1f, g1d, mode=CAST_IN, count=block_N)
                        T.tile.add(g1f, g1f, g_one)            # 1 + gamma
                        T.tile.broadcast(g_bc, g1f)
                        T.tile.mul(xf, xf, g_bc)
                        T.tile.cast(xout_cast, xf, mode=CAST_OUT, count=tile_elems)
                        T.copy(xout_cast[0:valid_rows, 0:block_N],
                               y[row0:row0 + valid_rows, c0:c0 + block_N])

    return kern


def main():
    torch.manual_seed(0)
    import torch_npu                        # noqa: F401
    torch.npu.set_device(0)
    results = {"eps": EPS, "m_scan": M_SCAN, "warmup": WARMUP, "iters": ITERS,
               "sections": {}}

    for N in (N_MAIN, N_SPOT):
        print(f"\n== N={N} (bf16, eps={EPS}) ==", flush=True)
        sec = results["sections"].setdefault(f"N{N}", {})

        for M in M_SCAN:
            if N == N_SPOT and M != 64:
                continue
            print(f"  -- M={M} --", flush=True)
            x1 = torch.randn(M, N, device="npu", dtype=torch.bfloat16) * 0.5
            x2 = torch.randn(M, N, device="npu", dtype=torch.bfloat16) * 0.5
            w = torch.randn(N, device="npu", dtype=torch.bfloat16) * 0.1

            try:
                bench("eager.rms_chain", lambda: rms_chain(x1, w), sec)
                bench("eager.add_rms_chain", lambda: add_rms_chain(x1, x2, w), sec)
            except Exception:
                traceback.print_exc()

            # TileLang fused kernel
            try:
                kern = make_tl_add_rmsnorm(M, N, EPS, "bfloat16")
                x_out, y = kern(x1, x2, w)
                ref_s, ref_y = ref_add_rms(x1, x2, w)
                dy = (y.double() - ref_y.double()).abs().max().item()
                ds = (x_out.double() - ref_s.double()).abs().max().item()
                print(f"    [tl] fused add+rmsnorm  max|dy|={dy:.3e} max|ds|={ds:.3e}",
                      flush=True)
                bench("tilelang.fused_add_rms", lambda: kern(x1, x2, w), sec)
            except Exception as e:
                sec["tilelang.error"] = repr(e)[:300]
                print(f"    [tl] FAILED: {e}", flush=True)

            # torch.compile over the same chain
            try:
                cf = torch.compile(lambda a, b, g: add_rms_chain(a, b, g),
                                   dynamic=False)
                s_c, y_c = cf(x1, x2, w)
                ref_s, ref_y = ref_add_rms(x1, x2, w)
                dc = (y_c.double() - ref_y.double()).abs().max().item()
                print(f"    [compile] add_rms_chain max|dy|={dc:.3e}", flush=True)
                bench("compile.add_rms_chain", lambda: cf(x1, x2, w), sec)
            except Exception as e:
                sec["compile.error"] = repr(e)[:300]
                print(f"    [compile] FAILED: {e}", flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n[done] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
