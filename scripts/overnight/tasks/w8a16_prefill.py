#!/usr/bin/env python3
"""E58 — W8A16 under prefill load: quantization helps compute-bound, not
launch-bound work.

E41 showed int8 weight-only is a *negative* speedup for decode (per-layer
matvec is launch-bound at ~31us). Prefill is the opposite regime: large M
(L tokens), compute/bandwidth bound. This measures the MLP block (gate/up/down)
at prefill sizes, bf16 vs int8 weights.

Usage: python3 w8a16_prefill.py --bf16 DIR --int8 DIR --out out.json
"""
import argparse, json, os, time
import torch, torch_npu
from safetensors.torch import load_file


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lens", default="1024,4096,8192,16384")
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    sb = load_file(os.path.join(args.bf16, "model.safetensors"))
    si = load_file(os.path.join(args.int8, "model.safetensors"))
    p = "model.layers.0.mlp."
    Wg = sb[p + "gate_proj.weight"].to(dev)
    Wu = sb[p + "up_proj.weight"].to(dev)
    Wd = sb[p + "down_proj.weight"].to(dev)
    Gi = si[p + "gate_proj.weight"].to(dev).t().contiguous()      # [K, N]
    Ui = si[p + "up_proj.weight"].to(dev).t().contiguous()
    Di = si[p + "down_proj.weight"].to(dev).t().contiguous()
    def sc(name):
        t = si[p + name].to(dev)
        return t.reshape(1, -1) if t.dim() == 2 else t
    Sg, Su, Sd = sc("gate_proj.weight_scale"), sc("up_proj.weight_scale"), sc("down_proj.weight_scale")
    hidden = Wg.shape[1]
    res = {"rows": []}
    for L in [int(x) for x in args.lens.split(",")]:
        x = torch.randn(L, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        def mlp_bf16():
            g = torch.nn.functional.silu(x @ Wg.t())
            u = x @ Wu.t()
            return (g * u) @ Wd.t()
        def mlp_int8():
            g = torch.nn.functional.silu(
                torch_npu.npu_weight_quant_batchmatmul(x, Gi, Sg))
            u = torch_npu.npu_weight_quant_batchmatmul(x, Ui, Su)
            return torch_npu.npu_weight_quant_batchmatmul(g * u, Di, Sd)
        t_bf = bench(mlp_bf16)
        try:
            t_i8 = bench(mlp_int8)
            row = {"L": L, "bf16_ms": round(t_bf * 1e3, 3),
                   "int8_ms": round(t_i8 * 1e3, 3),
                   "speedup": round(t_bf / t_i8, 2),
                   "bf16_us_per_tok": round(t_bf * 1e6 / L, 2),
                   "int8_us_per_tok": round(t_i8 * 1e6 / L, 2)}
        except Exception as e:
            row = {"L": L, "bf16_ms": round(t_bf * 1e3, 3), "error": str(e)[:100]}
        res["rows"].append(row)
        with open(args.out + "l", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[E58] L={L:6d} bf16={row['bf16_ms']:7.3f}ms "
              f"({row['bf16_us_per_tok']:5.2f}us/tok) "
              f"int8={row.get('int8_ms','-')}ms "
              f"({row.get('int8_us_per_tok','-')}us/tok) x{row.get('speedup','-')}",
              flush=True)
        del x
        torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
