#!/usr/bin/env python3
"""E51 — Latency vs load: discrete-event simulation driven by MEASURED service times.

This is a simulator, not an engine measurement — but every service time comes
from the measurements in this repo (E29 decode step, E48 prefill model), so the
shape of the curve reflects real hardware behaviour rather than guesses.

Model (vLLM-like continuous batching):
  - requests arrive Poisson(lambda)
  - each has prompt length ~ U[lo, hi] and fixed output length
  - scheduler: admit while batch < max_batch and KV fits; prefill is chunked
    into C tokens per step; decode steps run for all admitted sequences
  - step cost: weight read + attention KV read (efficiency 19%->50% by batch,
    E37/E39) + launch floor (E29)
  - metrics: TTFT p50/p99, TPOT p50/p99, throughput, goodput at SLO

Usage: python3 latency_vs_load.py --weight-gb 1.5 --layers 28 --ctx 8192 \
        --rates 1,2,4,8,16 --out out.json
"""
import argparse, json, random, statistics

HBM_ACHIEVED_GBPS = 1412.0
LAUNCH_FLOOR_MS = 29.0
ATTN_EFF_BS1, ATTN_EFF_BS16, ATTN_EFF_BS64 = 0.19, 0.50, 0.86
PREFILL_QUAD = 4.66e-8
PREFILL_LIN_US = 0.09
PREFILL_FLOOR_MS = 0.10


def attn_eff(b):
    """Calibrated against E37 (bs=1:19%), E39 (bs=16:50%), E52 (bs=64:86%)."""
    if b <= 1:
        return ATTN_EFF_BS1
    if b < 16:
        return ATTN_EFF_BS1 + (ATTN_EFF_BS16 - ATTN_EFF_BS1) * (b - 1) / 15.0
    if b < 64:
        return ATTN_EFF_BS16 + (ATTN_EFF_BS64 - ATTN_EFF_BS16) * (b - 16) / 48.0
    return ATTN_EFF_BS64


class Sim:
    def __init__(self, a):
        self.a = a
        self.kv_tok = 2 * a.layers * a.nkv * a.head_dim * 2
        self.weight_ms = a.weight_gb * 1e3 / HBM_ACHIEVED_GBPS
        self.usable_gb = a.hbm_gb - a.weight_gb - a.overhead_gb
        self.max_by_kv = max(1, int(self.usable_gb * 1e9 / (self.kv_tok * a.ctx)))

    def step_ms(self, batch, max_ctx):
        kv_gb = batch * self.kv_tok * max_ctx / 1e9
        return (self.weight_ms + kv_gb * 1e3 / (HBM_ACHIEVED_GBPS * attn_eff(batch))
                + (0 if self.a.graph_captured else LAUNCH_FLOOR_MS))

    def prefill_ms(self, L):
        return self.a.layers * (PREFILL_FLOOR_MS + PREFILL_LIN_US * L / 1e3
                                + PREFILL_QUAD * L * L)

    def run(self, lam, seed=0):
        rng = random.Random(seed)
        a = self.a
        t, n_arrivals, reqs = 0.0, 0, []
        # time unit is milliseconds: convert rate per second -> per ms
        next_arrival = rng.expovariate(lam / 1000.0) if lam > 0 else float("inf")
        active, done = [], []
        horizon = a.sim_seconds * 1000.0
        while t < horizon:
            # admit as many as KV/batch allow
            while reqs and len(active) < min(a.max_batch, self.max_by_kv):
                r = reqs.pop(0)
                active.append(r)
            if not active and next_arrival > horizon:
                break
            batch = max(1, len(active))
            ctx_now = max((r["prompt"] + r["gen"] for r in active), default=a.ctx // 2)
            cost = self.step_ms(batch, min(ctx_now, a.ctx))
            # one step: everyone in batch either finishes prefill chunk or decodes
            for r in list(active):
                if r["prefilled"] < r["prompt"]:
                    chunk = min(a.chunk, r["prompt"] - r["prefilled"])
                    r["prefilled"] += chunk
                    if r["prefilled"] >= r["prompt"]:
                        r["ttft"] = t + cost - r["t_arrive"]
                else:
                    r["gen"] += 1
                    if r["gen"] >= r["out_len"]:
                        r["tpot"] = cost
                        r["lat"] = t + cost - r["t_arrive"]
                        active.remove(r)
                        done.append(r)
            t += cost
            # arrivals during this step
            while next_arrival <= t:
                reqs.append({"t_arrive": next_arrival,
                             "prompt": rng.randint(a.prompt_lo, a.prompt_hi),
                             "gen": 0, "prefilled": 0, "out_len": a.out_len})
                n_arrivals += 1
                next_arrival += (rng.expovariate(lam / 1000.0) if lam > 0 else float("inf"))
        if not done:
            return None
        ttft = sorted(r["ttft"] for r in done)
        tpot = sorted(r["tpot"] for r in done)
        p = lambda v, q: v[min(len(v) - 1, int(len(v) * q))]
        tok = sum(r["out_len"] for r in done)
        wall = horizon / 1000.0
        return {"rate": lam, "completed": len(done),
                "ttft_p50": round(p(ttft, 0.5), 1), "ttft_p99": round(p(ttft, 0.99), 1),
                "tpot_p50": round(p(tpot, 0.5), 1), "tpot_p99": round(p(tpot, 0.99), 1),
                "tok_s": round(tok / wall, 1),
                "goodput_tok_s": round(sum(r["out_len"] for r in done
                                           if r["tpot"] <= a.slo_ms
                                           and r["ttft"] <= a.slo_ttft_ms) / wall, 1),
                "slo_met_frac": round(sum(1 for r in done
                                          if r["tpot"] <= a.slo_ms
                                          and r["ttft"] <= a.slo_ttft_ms) / len(done), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-gb", type=float, default=1.5)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--nkv", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--hbm-gb", type=float, default=61.0)
    ap.add_argument("--overhead-gb", type=float, default=2.0)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--prompt-lo", type=int, default=512)
    ap.add_argument("--prompt-hi", type=int, default=2048)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--rates", default="0.5,1,2,4,8,16")
    ap.add_argument("--slo-ms", type=float, default=100.0, help="TPOT SLO")
    ap.add_argument("--slo-ttft-ms", type=float, default=2000.0, help="TTFT SLO")
    ap.add_argument("--sim-seconds", type=float, default=300.0)
    ap.add_argument("--graph-captured", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sim = Sim(args)
    print(f"kv/token={sim.kv_tok:,}B  max_by_kv={sim.max_by_kv}  "
          f"weight_read={sim.weight_ms:.2f}ms  "
          f"{'graph' if args.graph_captured else 'eager'}")
    rows = []
    for lam in [float(x) for x in args.rates.split(",")]:
        r = sim.run(lam)
        if r:
            rows.append(r)
            print(f"λ={lam:5.1f}/s done={r['completed']:4d} "
                  f"TTFT p50={r['ttft_p50']:7.1f} p99={r['ttft_p99']:8.1f} | "
                  f"TPOT p50={r['tpot_p50']:6.1f} p99={r['tpot_p99']:6.1f} | "
                  f"{r['tok_s']:7.1f} tok/s, goodput {r['goodput_tok_s']:7.1f} "
                  f"(SLO met {r['slo_met_frac']:.0%})")
    with open(args.out, "w") as f:
        json.dump({"params": vars(args), "rows": rows}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
