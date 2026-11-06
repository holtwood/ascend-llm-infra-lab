#!/usr/bin/env python3
"""E59 — Cold start decomposition: what does "scale up a replica" cost?

Breaks the scale-up latency into the parts an engineer can actually attack:
  1. disk read      (safetensors -> host memory)
  2. host->device   (weights to HBM)
  3. first call     (kernel/JIT warmup, one-time)
  4. steady state   (for contrast)

E17 measured 169s cold start end-to-end on vLLM (model load + ACLGraph capture);
this decomposes where that time can go on the bare torch stack.

Usage: python3 cold_start.py --model DIR --out out.json
"""
import argparse, json, os, time
import torch, torch_npu
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    path = os.path.join(args.model, "model.safetensors")
    size_gb = os.path.getsize(path) / 1e9
    res = {"model": args.model, "file_gb": round(size_gb, 3)}

    t0 = time.perf_counter()
    sd = load_file(path)
    t_disk = time.perf_counter() - t0
    res["disk_read_s"] = round(t_disk, 3)
    res["disk_gbps"] = round(size_gb / t_disk, 2)

    t0 = time.perf_counter()
    W = {k: v.to(dev) for k, v in sd.items() if k.endswith("weight")}
    torch.npu.synchronize()
    t_h2d = time.perf_counter() - t0
    wgb = sum(v.numel() * v.element_size() for v in W.values()) / 1e9
    res["h2d_s"] = round(t_h2d, 3)
    res["h2d_gbps"] = round(wgb / t_h2d, 2)
    res["weights_gb"] = round(wgb, 3)

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    hidden = cfg["hidden_size"]
    x = torch.randn(1, hidden, device=dev, dtype=torch.bfloat16) * 0.02
    p = "model.layers.0."

    def one_layer():
        # two representative matvecs of correct shapes (q_proj, gate_proj)
        a = x @ W[p + "self_attn.q_proj.weight"].t()
        b = x @ W[p + "mlp.gate_proj.weight"].t()
        return a, b

    t0 = time.perf_counter()
    one_layer()
    torch.npu.synchronize()
    res["first_call_s"] = round(time.perf_counter() - t0, 4)
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        one_layer()
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    res["steady_call_ms"] = round(ts[len(ts) // 2] * 1e3, 2)
    res["first_over_steady"] = round(res["first_call_s"] * 1e3 / res["steady_call_ms"], 1)
    res["scale_up_total_s"] = round(t_disk + t_h2d + res["first_call_s"], 3)

    print(f"[E59] file {size_gb:.2f}GB  disk {t_disk:.2f}s ({res['disk_gbps']} GB/s)  "
          f"H2D {t_h2d:.2f}s ({res['h2d_gbps']} GB/s)  "
          f"first call {res['first_call_s']*1e3:.0f}ms ({res['first_over_steady']}x steady "
          f"{res['steady_call_ms']}ms)")
    print(f"[E59] scale-up total (weights only) = {res['scale_up_total_s']} s")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
