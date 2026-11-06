#!/usr/bin/env python3
"""N1b: L0 vs L2 ladder benchmark for the hand-written engine on Ascend 910B2.

L0 = generate_no_cache semantics: re-feed the FULL sequence every step (O(T)/token).
L2 = generate() semantics: prefill once into HybridCaches, then 1-token decode steps.

For every prompt length in {128, 512, 2048} and 30 greedy decode steps we record:
  - per-step wall time, bracketed by torch.npu.synchronize(): first step vs p50
  - prefill time (L2 only)
  - peak allocator memory per config (torch.npu.max_memory_allocated) against the
    post-load weights baseline

Warmup: each (mode, prompt_len) runs untimed forwards of the exact shapes it will
time first, so op compilation / fusion caching lands outside the timed region.
The L0/L2 instrumented loops replicate generate_no_cache()/generate() op-for-op
(temperature=0 -> argmax); we assert the two token streams agree as a sanity check.

Run (NPU host):  cd /data/engine_np && python3 bench_ladder.py
Output:          bench_ladder_out.json + stdout tables.
"""

import json
import statistics
import time

import torch

from cache import HybridCaches
from engine_naive import NaiveEngine
from model_qwen35 import from_pretrained

MODEL_DIR = "/model/Qwen3.5-4B"
PROMPT_LENS = [128, 512, 2048]
STEPS = 30
OUT_JSON = "bench_ladder_out.json"


def timed(fn):
    """Run fn with NPU sync brackets; return (result, wallclock ms)."""
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.npu.synchronize()
    return out, (time.perf_counter() - t0) * 1e3


def mem(mib=True):
    """Currently allocated NPU memory (MiB), -1 if the API is unavailable."""
    f = getattr(torch.npu, "memory_allocated", None)
    if f is None:
        return -1.0
    return f() / 2**20 if mib else f()


def peak_mem_mib():
    f = getattr(torch.npu, "max_memory_allocated", None)
    return f() / 2**20 if f else -1.0


def reset_peak():
    f = getattr(torch.npu, "reset_peak_memory_stats", None)
    if f:
        f()


def step_stats(steps_ms):
    """First step, p50, and the mean of the last 10 steps (growth indicator)."""
    s = sorted(steps_ms)
    p50 = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
    return {
        "first_ms": steps_ms[0],
        "p50_ms": p50,
        "mean_ms": statistics.fmean(steps_ms),
        "last10_mean_ms": statistics.fmean(steps_ms[-10:]),
        "min_ms": s[0],
        "max_ms": s[-1],
    }


@torch.no_grad()
def bench_l0(model, prompt_ids):
    """generate_no_cache, instrumented: time each full-sequence forward."""
    seq = prompt_ids
    # warmup: two steps of the exact op sequence (compile/fuse this shape family)
    for _ in range(2):
        logits = model(seq, logits_to_keep=1)
        nxt = logits[:, -1, :].argmax(-1)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    torch.npu.synchronize()

    seq = prompt_ids
    tokens, steps_ms = [], []
    for _ in range(STEPS):
        logits, ms = timed(lambda: model(seq, logits_to_keep=1))
        nxt = logits[:, -1, :].argmax(-1)
        torch.npu.synchronize()
        tokens.append(int(nxt[0]))
        steps_ms.append(ms)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    return tokens, steps_ms


def bench_l2(engine, config, prompt_ids, device, dtype):
    """generate(), instrumented: prefill once, time each 1-token decode step."""
    total = prompt_ids.shape[1] + STEPS + 8
    caches = HybridCaches(config, batch=1, max_seq_len=total,
                          device=device, dtype=dtype)
    try:
        # warmup: full prefill + 2 decode steps (compiles both op paths)
        nxt = engine.prefill(prompt_ids, caches)
        nxt = engine.decode_step(nxt[:, None], caches)
        engine.decode_step(nxt[:, None], caches)
        torch.npu.synchronize()
        caches.reset()

        prompt = prompt_ids
        nxt, prefill_ms = timed(lambda: engine.prefill(prompt, caches))
        tokens, steps_ms = [int(nxt[0])], []
        for _ in range(STEPS):
            n, ms = timed(lambda: engine.decode_step(nxt[:, None], caches))
            nxt = n
            steps_ms.append(ms)
            tokens.append(int(nxt[0]))
    finally:
        del caches
        torch.npu.empty_cache()
    return tokens, prefill_ms, steps_ms


def main():
    torch.manual_seed(0)
    import torch_npu                     # noqa: F401
    torch.npu.set_device(0)
    device, dtype = "npu", torch.bfloat16

    t0 = time.perf_counter()
    model = from_pretrained(MODEL_DIR, device=device, dtype=dtype)
    load_s = time.perf_counter() - t0
    engine = NaiveEngine(model)
    baseline_mib = mem()
    print(f"[load] {load_s:.1f}s, weights baseline {baseline_mib:.0f} MiB", flush=True)

    results = {"model_dir": MODEL_DIR, "dtype": "bfloat16", "steps": STEPS,
               "weights_baseline_mib": baseline_mib, "configs": []}
    for T in PROMPT_LENS:
        prompt_ids = torch.randint(1000, 240_000, (1, T), device=device)
        reset_peak()

        tok_l0, ms_l0 = bench_l0(model, prompt_ids)
        l0_peak = peak_mem_mib()
        torch.npu.synchronize()

        reset_peak()                         # L2 peak must not inherit L0's
        tok_l2, prefill_ms, ms_l2 = bench_l2(engine, model.config, prompt_ids,
                                             device, dtype)
        l2_peak = peak_mem_mib()

        agree = tok_l0 == tok_l2[:len(tok_l0)]
        entry = {
            "prompt_len": T,
            "L0": step_stats(ms_l0), "L0_steps_ms": ms_l0,
            "L0_peak_mib": l0_peak,
            "L2_prefill_ms": prefill_ms,
            "L2": step_stats(ms_l2), "L2_steps_ms": ms_l2,
            "L2_peak_mib": l2_peak,
            "peak_over_baseline_mib": {"L0": l0_peak - baseline_mib,
                                       "L2": l2_peak - baseline_mib},
            "greedy_tokens_match_L0_vs_L2": bool(agree),
        }
        results["configs"].append(entry)
        print(f"\n== prompt {T} (greedy tokens L0==L2: {agree}) ==", flush=True)
        print(f"  L0: first {entry['L0']['first_ms']:.1f} ms | p50 "
              f"{entry['L0']['p50_ms']:.1f} ms | last10 {entry['L0']['last10_mean_ms']:.1f} ms"
              f" | peak {l0_peak:.0f} MiB (+{l0_peak - baseline_mib:.0f})")
        print(f"  L2: prefill {prefill_ms:.1f} ms | decode first "
              f"{entry['L2']['first_ms']:.1f} ms | p50 {entry['L2']['p50_ms']:.1f} ms"
              f" | last10 {entry['L2']['last10_mean_ms']:.1f} ms"
              f" | peak {l2_peak:.0f} MiB (+{l2_peak - baseline_mib:.0f})")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n[done] wrote {OUT_JSON}, total {time.perf_counter() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
