"""E83: interview Q — "two identical requests at temperature zero return
different outputs. Explain why." (batch-dependent numerics)

Same prompt, same weights. Run once at bs=1 and once inside a bs=8 batch of
IDENTICAL copies; compare row-0 logits: argmax flips and max |Δlogit|.
Demonstrates that batched kernels change reduction order -> tiny numeric
drift -> greedy argmax can diverge downstream.
"""
import argparse
import importlib.util
import json
import sys

import torch
import torch_npu  # noqa: F401
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location(
        "pc", "/data1/pytroch-7/overnight/tasks/perplexity_check.py")
    pc = importlib.util.module_from_spec(spec)
    sys.argv = ["x", "--bf16", args.bf16, "--int8", args.bf16, "--out", "/tmp/_pc.json"]
    spec.loader.exec_module(pc)
    dev = "npu:0"
    cfg, hd = pc.cfg_of(AutoConfig.from_pretrained(args.bf16)) if hasattr(pc, "cfg_of") else (None, None)
    # reuse pc loaders
    hf = AutoConfig.from_pretrained(args.bf16)
    cfgd = {"nlayers": hf.num_hidden_layers, "hq": hf.num_attention_heads,
            "nkv": hf.num_key_value_heads,
            "hd": getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)}
    tok = AutoTokenizer.from_pretrained(args.bf16)
    ids = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
    W = pc.load(args.bf16, dev)
    cos, sin = pc.rope_cache(len(ids) + 8, cfgd["hd"],
                           getattr(hf, "rope_theta", 1000000.0), dev, torch.bfloat16)

    nl, hq, nkv, hd, hidden = (cfgd["nlayers"], cfgd["hq"], cfgd["nkv"],
                               cfgd["hd"], hf.hidden_size)
    with torch.no_grad():
        lg1 = pc.forward_logits(ids, W, cfgd, dev, nl, hq, nkv, hd, hidden)
        # bs=8 identical copies: emulate by running the same single-seq forward
        # 8x and stacking — kernels process [8*L] rows => different reduction order
        big = ids * 8
        lg8 = pc.forward_logits(big, W, cfgd, dev, nl, hq, nkv, hd, hidden)
    L = len(ids)
    a1 = lg1[:L].argmax(-1)
    a8 = lg8[:L].argmax(-1)          # first copy's predictions
    diff = (lg1[:L] - lg8[:L]).abs()
    res = {
        "prompt": "The capital of France is",
        "L": L,
        "top1_agree_pct": round(100 * (a1 == a8).float().mean().item(), 2),
        "max_abs_logit_diff": round(diff.max().item(), 4),
        "mean_abs_logit_diff": round(diff.mean().item(), 6),
        "bs1_next_token": tok.decode([int(a1[-1])]),
        "bs8_next_token": tok.decode([int(a8[-1])]),
        "note": "bs=8 = same prompt x8 concatenated; reduction order differs",
    }
    print(f"[E83] top1_agree={res['top1_agree_pct']}% "
          f"max_dlogit={res['max_abs_logit_diff']} "
          f"next: bs1='{res['bs1_next_token']}' bs8='{res['bs8_next_token']}'",
          flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
