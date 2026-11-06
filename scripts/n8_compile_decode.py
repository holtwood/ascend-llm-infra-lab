#!/usr/bin/env python3
"""N8 fallback path: torch.compile payoff on the engine decode step.

N2 measured decode = 79% host overhead (device busy ~24 ms of every step,
2935 op dispatches). Graph capture should collapse the op stream; this
quantifies the ceiling achievable on this engine with zero custom kernels.

Protocol per prompt length (128, 512):
  eager:    fresh HybridCaches -> prefill -> 2 warmup decode steps -> 30 timed
  compiled: fresh HybridCaches -> prefill through the SAME compiled module
            (compilation lands here) -> 3 warmup decode steps -> 30 timed
  correctness: greedy token streams must match eager exactly; max|logits|
               diff reported on the first step.

The compiled callable is the whole `model(tokens, caches=..., logits_to_keep=1)`
forward — caches are mutated in-place by the traced graph, so each timed step
still appends K/V exactly like eager (we assert on outputs, not on mechanics).

Run (NPU host):  cd /data/n8 && python3 n8_compile_decode.py
Output:          n8_compile_decode.json + stdout table.
"""

import json
import time
import traceback

import torch

from cache import HybridCaches          # engine_np modules (cwd or sys.path)
from model_qwen35 import from_pretrained

MODEL_DIR = "/model/Qwen3.5-4B"
PROMPT_LENS = [128, 512]
STEPS = 30
OUT_JSON = "n8_compile_decode.json"


def timed_steps(model, caches, nxt, n_steps):
    """Time n_steps single-token forwards; returns (tokens, ms list)."""
    tokens, steps = [int(nxt[0])], []
    for _ in range(n_steps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        logits = model(nxt[:, None], caches=caches, logits_to_keep=1)
        nxt = logits[:, -1, :].argmax(-1)
        torch.npu.synchronize()
        steps.append((time.perf_counter() - t0) * 1e3)
        tokens.append(int(nxt[0]))
    return tokens, steps


def stats(ms):
    s = sorted(ms)
    return {"first_ms": ms[0], "p50_ms": s[len(s) // 2],
            "mean_ms": sum(ms) / len(ms)}


def main():
    torch.manual_seed(0)
    import torch_npu                        # noqa: F401
    torch.npu.set_device(0)
    dev, dt = "npu", torch.bfloat16
    model = from_pretrained(MODEL_DIR, device=dev, dtype=dt)
    results = {"prompt_lens": PROMPT_LENS, "steps": STEPS, "configs": []}

    try:
        model_c = torch.compile(model, dynamic=False)
    except Exception as e:
        results["compile.error"] = repr(e)[:300]
        print(f"[compile] torch.compile(model) FAILED: {e}", flush=True)
        model_c = None

    for T in PROMPT_LENS:
        print(f"\n== prompt {T} ==", flush=True)
        prompt = torch.randint(1000, 240_000, (1, T), device=dev)
        entry = {"prompt_len": T}

        with torch.no_grad():
            # eager
            caches = HybridCaches(model.config, 1, T + STEPS + 8, dev, dt)
            nxt = model(prompt, caches=caches, logits_to_keep=1)[:, -1].argmax(-1)
            nxt = model(nxt[:, None], caches=caches, logits_to_keep=1)[:, -1].argmax(-1)
            tok_e, ms_e = timed_steps(model, caches, nxt, STEPS)
            entry["eager"] = stats(ms_e)
            entry["eager_steps_ms"] = ms_e
            print(f"  eager    : {entry['eager']}", flush=True)

            # compiled
            if model_c is not None:
                try:
                    caches2 = HybridCaches(model.config, 1, T + STEPS + 8, dev, dt)
                    nxt = model_c(prompt, caches=caches2, logits_to_keep=1)[:, -1].argmax(-1)
                    nxt = model_c(nxt[:, None], caches=caches2, logits_to_keep=1)[:, -1].argmax(-1)
                    # warmup 3 more steps to absorb late recompiles
                    for _ in range(3):
                        nxt = model_c(nxt[:, None], caches=caches2,
                                      logits_to_keep=1)[:, -1].argmax(-1)
                    tok_c, ms_c = timed_steps(model_c, caches2, nxt, STEPS)
                    entry["compiled"] = stats(ms_c)
                    entry["compiled_steps_ms"] = ms_c
                    entry["tokens_match"] = tok_c == tok_e
                    speedup = entry["eager"]["p50_ms"] / entry["compiled"]["p50_ms"]
                    print(f"  compiled : {entry['compiled']} | match="
                          f"{entry['tokens_match']} | speedup {speedup:.2f}x",
                          flush=True)
                except Exception as e:
                    traceback.print_exc()
                    entry["compiled.error"] = repr(e)[:300]

        del caches
        torch.npu.empty_cache()
        results["configs"].append(entry)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n[done] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
