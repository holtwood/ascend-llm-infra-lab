#!/usr/bin/env python3
"""N10b: longer-generation cross-check for the counter-intuitive N10 result.

N10 found: at 8k ctx BOTH per-token and per-channel int8 KV stayed argmax-identical to
bf16 for 32 greedy steps, while at 1.5k ctx per-token diverged at token 5 — the opposite
of the "error accumulates with context" expectation. This script runs 128-step greedy
generations at 8k to confirm or refute, and logs the per-step top1-top2 logit gap to
identify the fragile (small-margin) steps.

Usage:  python3 n10b_long_gen_check.py
Output: stdout + /data/results/n10b_long_gen.json
"""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import importlib.util
spec = importlib.util.spec_from_file_location("n10", "/data/scripts/n10_kv_int8.py")
n10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n10)  # reuse SYSTEM/FACTS/PAD/build_ids/q_roundtrip/kv_roundtrip

M = "/model/Qwen3.5-4B"
DEV = "npu"
N_STEPS = 128
OUT_JSON = "/data/results/n10b_long_gen.json"


@torch.no_grad()
def run(model, ids, mode):
    out = model(ids, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    if mode in ("per_token", "per_channel"):
        n10.kv_roundtrip(cache, mode)
    toks = torch.argmax(out.logits[0, -1]).view(1, 1)
    gen, gaps = [int(toks.item())], []
    for _ in range(N_STEPS):
        o = model(toks, past_key_values=cache, use_cache=True, logits_to_keep=1)
        lg = o.logits[0, -1]
        top2 = torch.topk(lg, 2)
        gaps.append(float((top2.values[0] - top2.values[1]).item()))
        toks = torch.argmax(lg).view(1, 1)
        gen.append(int(toks.item()))
        if mode in ("per_token", "per_channel"):
            n10.kv_roundtrip(cache, mode)
    del cache
    return gen, gaps


def main():
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
    ids = n10.build_ids(tok, 8192)
    print(f"[load] ctx={ids.shape[1]}", flush=True)

    ref, ref_gaps = run(model, ids, "bf16")
    res = {"ctx": ids.shape[1], "n_steps": N_STEPS, "variants": {}}
    res["variants"]["bf16"] = {"min_top1_top2_gap": round(min(ref_gaps), 4),
                               "fragile_steps_bottom5":
                                   sorted(range(len(ref_gaps)), key=lambda i: ref_gaps[i])[:5]}
    print(f"[bf16] min gap {min(ref_gaps):.4f} at steps "
          f"{sorted(range(len(ref_gaps)), key=lambda i: ref_gaps[i])[:5]}", flush=True)
    for mode in ("per_token", "per_channel"):
        g, gaps = run(model, ids, mode)
        div = next((i for i, (a, b) in enumerate(zip(ref, g)) if a != b), None)
        mism = sum(a != b for a, b in zip(ref, g)) / len(ref)
        res["variants"][mode] = {"first_div_token": div, "mismatch_ratio": round(mism, 4),
                                 "min_top1_top2_gap": round(min(gaps), 4)}
        print(f"[{mode}] first_div={div} mismatch={mism:.3f} min_gap={min(gaps):.4f}", flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[N10B] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
