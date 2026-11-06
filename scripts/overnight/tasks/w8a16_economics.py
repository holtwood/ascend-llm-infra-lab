#!/usr/bin/env python3
"""E41 — W8A16 weight-only quantization: what does it actually buy serving?

Compares the bf16 and int8 weight-only checkpoints on three axes:
  1. memory: weight bytes and the KV capacity they free up (E30 formula)
  2. speed: per-layer matvec time bf16 vs npu_weight_quant_batchmatmul
  3. accuracy: relative error of one real layer's output

Usage: python3 w8a16_economics.py --bf16 DIR --int8 DIR --out out.json
"""
import argparse, json, os, time
import torch, torch_npu
from safetensors.torch import load_file


def bench(fn, iters=30, warmup=8):
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
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    dev = "npu:0"
    sb = load_file(os.path.join(args.bf16, "model.safetensors"))
    si = load_file(os.path.join(args.int8, "model.safetensors"))
    cfg = json.load(open(os.path.join(args.bf16, "config.json")))
    nl, hq = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", hq)
    hd = cfg.get("head_dim", cfg["hidden_size"] // hq)
    kv_tok = 2 * nl * nkv * hd * 2
    wb = sum(v.numel() * v.element_size() for v in sb.values())
    wi = sum(v.numel() * v.element_size() for v in si.values())
    res = {"bf16_weight_gb": round(wb / 1e9, 3), "int8_weight_gb": round(wi / 1e9, 3),
           "weight_saving_gb": round((wb - wi) / 1e9, 3), "kv_bytes_per_token": kv_tok}
    # capacity freed at 8k ctx (E30 formula)
    freed_kv = (wb - wi)
    res["extra_seqs_at_8k"] = round(freed_kv / (kv_tok * 8192), 2)
    print(f"[E41] weights bf16 {wb/1e9:.3f}GB -> int8 {wi/1e9:.3f}GB "
          f"(saves {(wb-wi)/1e9:.3f}GB = +{res['extra_seqs_at_8k']} seqs @8k ctx)", flush=True)

    # speed: representative layer matvecs, bs=1 and bs=64
    res["speed"] = []
    for name, key in [("q_proj", "model.layers.0.self_attn.q_proj.weight"),
                      ("gate_proj", "model.layers.0.mlp.gate_proj.weight"),
                      ("down_proj", "model.layers.0.mlp.down_proj.weight")]:
        Wb = sb[key].to(dev)
        Wi = si[key].to(dev)                      # int8, shape [N, K]
        sc = si[key.replace("weight", "weight_scale")].to(dev)
        if sc.dim() == 2:
            sc = sc.reshape(1, -1)
        K = Wb.shape[1]
        for bs in [1, 64]:
            x = torch.randn(bs, K, device=dev, dtype=torch.bfloat16) * 0.02
            t_b = bench(lambda: x @ Wb.t())
            WiT = Wi.t().contiguous()             # op wants [K, N]
            s1 = sc if sc.shape[0] == 1 else sc.t().contiguous()
            try:
                t_i = bench(lambda: torch_npu.npu_weight_quant_batchmatmul(x, WiT, s1))
                rel = None
                try:
                    y_i = torch_npu.npu_weight_quant_batchmatmul(x, WiT, s1)
                    y_b = x @ Wb.t()
                    rel = round((y_i.float() - y_b.float()).abs().max().item() /
                                (y_b.float().abs().max().item() + 1e-6), 5)
                except Exception:
                    pass
                row = {"tensor": name, "bs": bs, "bf16_us": round(t_b * 1e6, 1),
                       "int8_us": round(t_i * 1e6, 1),
                       "speedup": round(t_b / t_i, 2), "rel_err": rel}
            except Exception as e:
                row = {"tensor": name, "bs": bs, "bf16_us": round(t_b * 1e6, 1),
                       "error": str(e)[:100]}
            res["speed"].append(row)
            print(f"[E41] {name:10s} bs={bs:2d} bf16={row.get('bf16_us')}us "
                  f"int8={row.get('int8_us')}us x{row.get('speedup')} "
                  f"rel={row.get('rel_err')}", flush=True)
            del x, WiT
            torch.npu.empty_cache()
        del Wb, Wi
        torch.npu.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
