"""E67: sampling overhead per decode step.

The LM head produces [bs, vocab] logits; converting logits to a token has a
cost that rides on TPOT but is rarely measured. Sweep batch x sampler:

  argmax          : trivial reduce
  softmax+sample  : full softmax over vocab (151936), then multinomial
  top-p sort      : sort vocab descending, cumsum, mask (what vLLM does)

Also measure the LM head GEMM itself so sampling can be expressed as a
fraction of "the last layer's" cost.
"""
import argparse
import json
import os
import time

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def top_p_sample(logits, p=0.9):
    probs = F.softmax(logits, dim=-1)
    sprobs, sidx = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sprobs, dim=-1)
    mask = cum - sprobs > p
    sprobs = sprobs.masked_fill(mask, 0)
    sprobs = sprobs / sprobs.sum(-1, keepdim=True)
    pick = torch.multinomial(sprobs, 1)
    return sidx.gather(-1, pick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "npu:0"
    vocab = 151936  # Qwen3 vocab
    hidden = 1024
    torch.manual_seed(0)

    w = torch.randn(hidden, vocab, dtype=torch.bfloat16, device=dev)
    res = {"vocab": vocab, "rows": []}
    for bs in [1, 8, 16, 32, 64, 128]:
        x = torch.randn(bs, hidden, dtype=torch.bfloat16, device=dev)
        logits = x @ w
        row = {
            "bs": bs,
            "lm_head_ms": round(bench(lambda: x @ w), 4),
            "argmax_ms": round(bench(lambda: logits.argmax(-1)), 4),
            "softmax_ms": round(bench(lambda: F.softmax(logits, -1)), 4),
            "greedy_ms": round(bench(lambda: logits.argmax(-1)), 4),
            "sample_ms": round(bench(lambda: torch.multinomial(
                F.softmax(logits, -1), 1)), 4),
            "top_p_ms": round(bench(lambda: top_p_sample(logits)), 4),
        }
        row["sample_over_lm_head"] = round(row["sample_ms"] / row["lm_head_ms"], 2)
        row["top_p_over_lm_head"] = round(row["top_p_ms"] / row["lm_head_ms"], 2)
        res["rows"].append(row)
        print(f"[E67] bs={bs:<4} lm={row['lm_head_ms']:.3f}ms "
              f"argmax={row['argmax_ms']:.3f} softmax={row['softmax_ms']:.3f} "
              f"sample={row['sample_ms']:.3f} top_p={row['top_p_ms']:.3f} "
              f"top_p/lm={row['top_p_over_lm_head']}x", flush=True)
        del x, logits
        torch.npu.empty_cache()

    res["note"] = ("vocab=151936 Qwen3; top_p = softmax+sort+cumsum+mask+"
                   "multinomial (vLLM-style); lm_head = [bs,1024]x[1024,vocab]")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[E67] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
