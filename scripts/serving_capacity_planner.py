#!/usr/bin/env python3
"""Serving capacity planner — all constants measured on Ascend 910B1.

Every coefficient below comes from a real measurement in this repo
(results/overnight_20261105_160244/, summarised in docs/serving-physics.md):

  HBM_ACHIEVED_GBPS   1412   one big bf16 matvec over 1.5GB weights   (E29)
  LAUNCH_FLOOR_MS     29     eager decode step, ~420 kernel launches  (E29)
  ATTN_EFF_BS1        0.19   decode attention KV read / HBM peak      (E37)
  ATTN_EFF_BS16       0.50   same at batch>=16                        (E39)
  ATTN_EFF_BS64       0.86   same at batch>=64 (E52 back-solved)      (E52)
  PREFILL_QUAD        4.66e-8 ms/token^2 per layer                    (E48)
  PREFILL_LIN_US      0.09   µs/token per layer (MLP+proj)            (E48)
  PREFILL_FLOOR_MS    0.10   per layer                                (E48)

The model is deliberately simple and auditable; each term names its source so
a reviewer can check it against the measurement.

Usage:
  python3 serving_capacity_planner.py --weight-gb 1.5 --layers 28 --nkv 8 \
      --head-dim 128 --hbm-gb 61 --ctx 8192 --prompt-len 2048
"""
import argparse
import json

HBM_ACHIEVED_GBPS = 1412.0
LAUNCH_FLOOR_MS = 29.0
ATTN_EFF_BS1 = 0.19
ATTN_EFF_BS16 = 0.50
ATTN_EFF_BS64 = 0.86
PREFILL_QUAD = 4.66e-8      # ms/token^2 per layer
PREFILL_LIN_US = 0.09       # us/token per layer
PREFILL_FLOOR_MS = 0.10     # ms per layer


def attn_eff(batch):
    """Piecewise ramp calibrated against measurements (E37/E39/E52).

    bs=1 -> 19% (E37), bs=16 -> 50% (E39), bs>=64 -> 86% (E52 back-solved
    from the measured 54.6ms step at bs=64/ctx=4096).
    """
    if batch <= 1:
        return ATTN_EFF_BS1
    if batch < 16:
        return ATTN_EFF_BS1 + (ATTN_EFF_BS16 - ATTN_EFF_BS1) * (batch - 1) / 15.0
    if batch < 64:
        return ATTN_EFF_BS16 + (ATTN_EFF_BS64 - ATTN_EFF_BS16) * (batch - 16) / 48.0
    return ATTN_EFF_BS64


def plan(args):
    kv_tok = 2 * args.layers * args.nkv * args.head_dim * 2
    usable_gb = args.hbm_gb - args.weight_gb - args.overhead_gb
    if usable_gb <= 0:
        return {"error": "model does not fit: weights exceed usable HBM"}
    max_seqs = int(usable_gb * 1e9 / (kv_tok * args.ctx))
    max_seqs = max(1, max_seqs)

    # decode: weight read (amortized over batch) + attention KV read
    weight_ms = args.weight_gb * 1e3 / HBM_ACHIEVED_GBPS
    kv_read_gb = max_seqs * kv_tok * args.ctx / 1e9
    attn_ms = kv_read_gb * 1e3 / (HBM_ACHIEVED_GBPS * attn_eff(max_seqs))
    floor_ms = 0.0 if args.graph_captured else LAUNCH_FLOOR_MS
    step_ms = max(weight_ms, 0) + attn_ms + floor_ms
    decode_tps = max_seqs * 1e3 / step_ms
    per_seq_tps = 1e3 / step_ms

    # prefill: per-layer linear + quadratic
    L = args.prompt_len
    prefill_ms = args.layers * (PREFILL_FLOOR_MS + PREFILL_LIN_US * L / 1e3
                                + PREFILL_QUAD * L * L)
    if not args.graph_captured:
        prefill_ms += LAUNCH_FLOOR_MS * 0.5      # partial floor on prefill path
    ttft_ms = prefill_ms + step_ms

    # find the largest batch whose step still meets the TPOT SLO
    def step_at(b):
        kv_gb = b * kv_tok * args.ctx / 1e9
        return weight_ms + kv_gb * 1e3 / (HBM_ACHIEVED_GBPS * attn_eff(b)) + floor_ms

    slo_batch, slo_tps = 0, 0.0
    for b in range(1, max_seqs + 1):
        st = step_at(b)
        if st <= args.target_tpot_ms:
            slo_batch, slo_tps = b, b * 1e3 / st
    tpot_ok = step_ms <= args.target_tpot_ms
    goodput = slo_tps
    return {
        "kv_bytes_per_token": kv_tok,
        "usable_hbm_gb": round(usable_gb, 1),
        "max_concurrency": max_seqs,
        "weight_read_ms": round(weight_ms, 2),
        "attn_ms": round(attn_ms, 2),
        "launch_floor_ms": floor_ms,
        "step_ms": round(step_ms, 2),
        "decode_tok_s_aggregate": round(decode_tps, 1),
        "decode_tok_s_per_seq": round(per_seq_tps, 1),
        "prefill_ms": round(prefill_ms, 1),
        "ttft_ms": round(ttft_ms, 1),
        "meets_tpot_slo": tpot_ok,
        "slo_batch": slo_batch,
        "slo_step_ms": round(step_at(slo_batch), 2) if slo_batch else None,
        "goodput_tok_s": round(goodput, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-gb", type=float, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--nkv", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--hbm-gb", type=float, default=61.0)
    ap.add_argument("--overhead-gb", type=float, default=2.0)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--target-tpot-ms", type=float, default=50.0)
    ap.add_argument("--graph-captured", action="store_true",
                    help="assume ACLGraph removes the launch floor")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    r = plan(args)
    if args.json:
        print(json.dumps(r, indent=2))
        return
    print(f"model: {args.weight_gb:.1f}GB weights, {args.layers} layers, "
          f"{args.nkv} KV heads x {args.head_dim}  |  ctx={args.ctx}  "
          f"{'graph-captured' if args.graph_captured else 'eager'}")
    print(f"  KV/token            {r['kv_bytes_per_token']:,} B")
    print(f"  usable HBM          {r['usable_hbm_gb']} GB")
    print(f"  max concurrency     {r['max_concurrency']} seqs @ {args.ctx} ctx")
    print(f"  decode step         {r['step_ms']} ms  "
          f"(weight {r['weight_read_ms']} + attn {r['attn_ms']} + floor {r['launch_floor_ms']})")
    print(f"  decode throughput   {r['decode_tok_s_aggregate']:,.0f} tok/s aggregate, "
          f"{r['decode_tok_s_per_seq']:,.1f} tok/s per seq")
    print(f"  prefill ({args.prompt_len})     {r['prefill_ms']} ms -> TTFT {r['ttft_ms']} ms")
    print(f"  TPOT SLO ({args.target_tpot_ms}ms)  "
          f"{'PASS at max concurrency' if r['meets_tpot_slo'] else 'FAIL at max concurrency'}")
    if r["slo_batch"]:
        print(f"  SLO operating point {r['slo_batch']} seqs "
              f"(step {r['slo_step_ms']} ms) -> goodput {r['goodput_tok_s']:,.0f} tok/s")
    else:
        print(f"  SLO unreachable: even batch=1 step ({step_at(1):.1f} ms) exceeds "
              f"{args.target_tpot_ms} ms")


if __name__ == "__main__":
    main()
