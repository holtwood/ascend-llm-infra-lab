#!/usr/bin/env python3
"""E8c: hand-written speculative decoding — draft/verify loop in pure PyTorch.

Measures how speculative speedup depends on workload predictability:
  - draft: prompt-lookup ngram proposer (no extra model, like vLLM ngram spec)
  - verify: single batched forward of the target model over k candidates
  - workloads: repetitive text (high ngram hit) vs random tokens (low hit)

Run in pytorch container (HF transformers + torch_npu):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 e8c_specdecode.py --model /data1/pytorch-7/models/Qwen3-0.6B \
      --out /data1/pytorch-7/results_e8/e8c_spec.json
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompts(tokenizer):
    # instruction to repeat -> model output itself becomes periodic, so the
    # ngram table built from generated tokens keeps hitting (the regime where
    # prompt-lookup speculation pays off)
    rep = (
        'Repeat the following sentence exactly 50 times: "The quick brown fox '
        'jumps over the lazy dog." Output only the repeated sentences.\n\n'
        "1. The quick brown fox jumps over the lazy dog.\n"
        "2. The quick brown fox jumps over the lazy dog.\n"
        "3. The quick brown fox jumps over the lazy dog.\n4."
    )
    # open-ended prompt: model produces varied text, ngram lookups rarely hit
    rnd = (
        "Write a diverse essay about unusual words: xylophone quasar meridian "
        "fathom cobalt jigsaw vortex beagle prism amber neutron zephyr kaleido "
    )
    texts = {
        "repetitive": rep,
        "random": rnd,
    }
    return {k: tokenizer(t, return_tensors="pt").input_ids for k, t in texts.items()}


@torch.no_grad()
def decode_baseline(model, ids, n_new, device):
    ids = ids.to(device)
    out = model(input_ids=ids, use_cache=True)  # prefill
    kv = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    ids = torch.cat([ids, nxt], dim=1)
    t0 = time.perf_counter()
    steps = 1
    for _ in range(n_new - 1):
        out = model(input_ids=ids[:, -1:], past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        steps += 1
    torch.npu.synchronize() if device.startswith("npu") else None
    return {"tokens": n_new, "fw_calls": steps, "sec": time.perf_counter() - t0}


def ngram_draft(ids, table, k):
    """Propose k tokens by ngram lookup on the whole sequence."""
    props = []
    seq = ids[0].tolist()
    for _ in range(k):
        for n in (4, 3, 2):
            ctx = tuple(seq[-n:])
            if ctx in table and seq[-n:] == list(ctx):
                props.append(table[ctx])
                seq.append(table[ctx])
                break
        else:
            break
    return props


@torch.no_grad()
def decode_spec(model, ids, n_new, device, k=4):
    ids = ids.to(device)
    table = {}  # ngram -> next token, built from prompt + generated text
    seq0 = ids[0].tolist()
    for i in range(len(seq0) - 1):
        for n in (4, 3, 2):
            if i + 1 - n >= 0:
                table[tuple(seq0[i + 1 - n:i + 1])] = seq0[i + 1]
    # prefill: writes KV for the whole prompt, yields greedy first token `bonus`
    out = model(input_ids=ids, use_cache=True)
    kv = out.past_key_values
    bonus = out.logits[0, -1, :].argmax().item()
    t0 = time.perf_counter()
    fw = 0
    produced = 0
    n_props = 0
    n_acc = 0
    while produced < n_new:
        props = ngram_draft(torch.cat([ids, torch.tensor([[bonus]], device=device)], 1),
                            table, k)
        n_props += len(props)
        # feed [bonus, d1..dk]: bonus is the committed token lacking a KV entry;
        # logits[j] = distribution after in_ids[j], verifying d_{j+1} vs greedy
        in_ids = torch.tensor([[bonus] + props], device=device)
        out = model(input_ids=in_ids, past_key_values=kv, use_cache=True)
        fw += 1
        kv = out.past_key_values
        logits = out.logits[0]
        greedy = logits.argmax(-1).tolist()
        acc = 0
        for j, p in enumerate(props):
            if greedy[j] == p:
                acc += 1
            else:
                break
        n_acc += acc
        bonus = greedy[acc]
        committed = props[:acc] + [bonus]
        ids = torch.cat([ids, torch.tensor([committed], device=device)], dim=1)
        for t in committed:
            seq0.append(t)
            for n in (4, 3, 2):
                if len(seq0) > n:
                    table[tuple(seq0[-n - 1:-1])] = seq0[-1]
        produced += acc + 1
        # drop KV entries of rejected draft tokens: keep bonus + accepted
        kv.crop(kv.get_seq_length() - (len(props) - acc))
    if device.startswith("npu"):
        torch.npu.synchronize()
    return {
        "tokens": produced,
        "fw_calls": fw,
        "sec": time.perf_counter() - t0,
        "proposed": n_props,
        "accepted": n_acc,
        "accept_rate": round(n_acc / n_props, 3) if n_props else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(args.device).eval()

    res = {}
    for name, ids in build_prompts(tok).items():
        b = decode_baseline(model, ids, args.n_new, args.device)
        entry = {"baseline": {**b, "tok_per_s": round(b["tokens"] / b["sec"], 1)}}
        for k in (2, 4, 8):
            s = decode_spec(model, ids, args.n_new, args.device, k)
            entry["spec_k%d" % k] = {
                **s,
                "tok_per_s": round(s["tokens"] / s["sec"], 1),
                "fw_per_tok": round(s["fw_calls"] / s["tokens"], 3),
            }
        res[name] = entry
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
