#!/usr/bin/env python3
"""N2b probe: characterize the equal-length batch divergence seen at B>=16.

From n2b_batch_scan.py: identical prompts, zero padding — B<=8 slots match the B=1
reference bit-for-bit, B>=16 diverge. Two candidate mechanisms, distinguishable:
  A) batch-shape-dependent numerics: bf16 GEMM tiling/reduction order changes with B;
     same B re-run must then be bit-identical (deterministic per shape).
  B) true nondeterminism: same B re-run differs (scheduler / async dispatch).
Also measures: first diverging decode step per slot, mismatch ratio over the 31 tokens,
and prefill first-step logits drift (max abs diff vs B=1, top-1 agreement).

Usage:  python3 n2b_divergence_probe.py
Output: stdout + /data/results/n2b_divergence.json
"""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import importlib.util
spec = importlib.util.spec_from_file_location(
    "n2b", "/data/scripts/n2b_batch_scan.py")
n2b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n2b)  # reuse SYSTEM/USER/build_base_ids/make_batch

M = "/model/Qwen3.5-4B"
DEV = "npu"
BATCHES = [8, 16]
N_STEPS = 30
OUT_JSON = "/data/results/n2b_divergence.json"


@torch.no_grad()
def run(model, ids, mask, B):
    out = model(ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    first_logits = out.logits[:, -1, :].clone()
    toks = torch.argmax(first_logits, dim=-1).view(B, 1)
    m = mask
    gens = [toks.clone()]
    for _ in range(N_STEPS):
        o = model(toks, attention_mask=m, past_key_values=cache,
                  use_cache=True, logits_to_keep=1)
        m = torch.cat([m, torch.ones(B, 1, dtype=m.dtype, device=DEV)], dim=1)
        toks = torch.argmax(o.logits[:, -1, :], dim=-1).view(B, 1)
        gens.append(toks.clone())
    del cache
    return first_logits, torch.cat(gens, dim=1)  # (B, N_STEPS+1)


def main():
    tok = AutoTokenizer.from_pretrained(M)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
    base = n2b.build_base_ids(tok)
    print(f"[load] base {base.shape[1]} tok", flush=True)

    _, ref_gen = run(model, base, torch.ones_like(base, dtype=torch.long), 1)
    ref_logits = None

    res = {"base_prompt_tok": base.shape[1], "batches": {}}
    for B in BATCHES:
        ids, mask = n2b.make_batch(base, [0] * B, pad_id)
        lg_a, gen_a = run(model, ids, mask, B)
        lg_b, gen_b = run(model, ids, mask, B)  # independent re-run, same B, same shapes

        # ref first-step logits for drift comparison
        o1 = model(base, attention_mask=torch.ones_like(base, dtype=torch.long),
                   use_cache=True, logits_to_keep=1)
        ref_logits = o1.logits[:, -1, :].clone()
        del o1

        drift = (lg_a[0].float() - ref_logits[0].float()).abs().max().item()
        top1_same = int(torch.argmax(lg_a[0]) == torch.argmax(ref_logits[0]))

        first_div = []
        for i in range(B):
            d = (gen_a[i] != ref_gen[0]).nonzero()
            first_div.append(int(d[0]) if d.numel() else None)
        n_div_slots = sum(1 for f in first_div if f is not None)
        mism_ratio = float((gen_a != ref_gen[0]).float().mean())

        rerun_bit_equal = bool(torch.equal(gen_a, gen_b))
        cross_slot_bit_equal = bool(all(torch.equal(gen_a[0], gen_a[i]) for i in range(B)))

        res["batches"][str(B)] = {
            "first_div_step_per_slot": first_div[:8],
            "slots_diverging": n_div_slots,
            "mismatch_token_ratio_after_all_steps": round(mism_ratio, 4),
            "prefill_logits_max_abs_diff_slot0_vs_b1": round(drift, 5),
            "prefill_top1_slot0_same_as_b1": bool(top1_same),
            "rerun_same_B_bit_equal": rerun_bit_equal,
            "slots_bit_equal_within_run": cross_slot_bit_equal,
        }
        print(f"[B={B}] slots_diverging={n_div_slots}/{B} "
              f"first_div={first_div[:4]}... mismatch_ratio={mism_ratio:.3f} | "
              f"logits_drift={drift:.4f} top1_same={top1_same} | "
              f"rerun_bit_equal={rerun_bit_equal} within_run_equal={cross_slot_bit_equal}",
              flush=True)
        del ids, mask, lg_a, gen_a, lg_b, gen_b
        torch.npu.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[PROBE] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
