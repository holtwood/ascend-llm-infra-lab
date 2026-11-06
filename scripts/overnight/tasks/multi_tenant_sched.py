#!/usr/bin/env python3
"""E54 — Multi-tenant scheduling: FCFS vs SJF vs SLO-admission vs priority.

Two classes of tenants share one engine:
  short: prompt 128-256  (chat / classification)
  long : prompt 2048-4096 (RAG / summarisation)

Policies:
  fcfs        admit in arrival order
  sjf         admit shortest prompt first (reorders the queue)
  slo_admit   admit only if projected TTFT stays under the SLO, else reject
  priority    short class always admitted first

Service times come from the measured model (E29/E37/E39/E48/E52), same as
latency_vs_load.py. Reports per-class TTFT p50/p99, goodput and rejection rate.

Usage: python3 multi_tenant_sched.py --out out.json --rate 12
"""
import argparse, json, random

HBM_GBPS = 1412.0
LAUNCH_FLOOR_MS = 29.0
EFF = {1: 0.19, 16: 0.50, 64: 0.86}


def eff(b):
    if b <= 1:
        return EFF[1]
    if b < 16:
        return EFF[1] + (EFF[16] - EFF[1]) * (b - 1) / 15.0
    if b < 64:
        return EFF[16] + (EFF[64] - EFF[16]) * (b - 16) / 48.0
    return EFF[64]


class Engine:
    def __init__(self, a):
        self.a = a
        self.kv_tok = 2 * a.layers * a.nkv * a.head_dim * 2
        self.weight_ms = a.weight_gb * 1e3 / HBM_GBPS
        self.max_by_kv = max(1, int((a.hbm_gb - a.weight_gb - 2.0) * 1e9
                                    / (self.kv_tok * a.ctx)))

    def step_ms(self, batch, ctx):
        kv_gb = batch * self.kv_tok * ctx / 1e9
        return (self.weight_ms + kv_gb * 1e3 / (HBM_GBPS * eff(batch))
                + (0 if self.a.graph_captured else LAUNCH_FLOOR_MS))

    def prefill_ms(self, L):
        return self.a.layers * (0.10 + 0.09 * L / 1e3 + 4.66e-8 * L * L)


def run(a, policy, lam):
    rng = random.Random(1234)
    eng = Engine(a)
    horizon = a.sim_seconds * 1000.0
    t = 0.0
    queue, active, done = [], [], []
    next_arr = rng.expovariate(lam / 1000.0)
    rejected = 0
    while t < horizon:
        # admission
        while queue and len(active) < min(a.max_batch, eng.max_by_kv):
            if policy == "sjf":
                queue.sort(key=lambda r: r["prompt"])
            elif policy == "priority":
                queue.sort(key=lambda r: (r["cls"] != "short", r["t_arrive"]))
            cand = queue[0]
            if policy == "slo_admit":
                # include queue depth: every queued request ahead costs a step
                proj_ttft = (eng.prefill_ms(cand["prompt"])
                             + (len(queue) + 1) * eng.step_ms(
                                 min(len(active) + 1, a.max_batch), a.ctx))
                if proj_ttft > a.slo_ttft_ms:
                    queue.pop(0)
                    rejected += 1
                    continue
            queue.pop(0)
            active.append(cand)
        if not active and next_arr > horizon:
            break
        batch = max(1, len(active))
        ctx_now = min(a.ctx, max((r["prompt"] + r["gen"] for r in active),
                                 default=a.ctx // 2))
        cost = eng.step_ms(batch, ctx_now)
        for r in list(active):
            if r["prefilled"] < r["prompt"]:
                r["prefilled"] += min(a.chunk, r["prompt"] - r["prefilled"])
                if r["prefilled"] >= r["prompt"]:
                    r["ttft"] = t + cost - r["t_arrive"]
            else:
                r["gen"] += 1
                if r["gen"] >= a.out_len:
                    r["tpot"] = cost
                    active.remove(r)
                    done.append(r)
        t += cost
        while next_arr <= t:
            cls = "short" if rng.random() < 0.7 else "long"
            pl = (rng.randint(128, 256) if cls == "short"
                  else rng.randint(2048, 4096))
            queue.append({"t_arrive": next_arr, "prompt": pl, "gen": 0,
                          "prefilled": 0, "cls": cls, "out_len": a.out_len})
            next_arr += rng.expovariate(lam / 1000.0)
    out = {"policy": policy, "rate": lam, "rejected": rejected, "classes": {}}
    for cls in ("short", "long"):
        rows = [r for r in done if r["cls"] == cls]
        if not rows:
            continue
        ttft = sorted(r["ttft"] for r in rows)
        p = lambda v, q: v[min(len(v) - 1, int(len(v) * q))]
        ok = [r for r in rows if r["ttft"] <= a.slo_ttft_ms and r["tpot"] <= a.slo_ms]
        out["classes"][cls] = {
            "n": len(rows), "ttft_p50": round(p(ttft, 0.5), 1),
            "ttft_p99": round(p(ttft, 0.99), 1),
            "slo_met_pct": round(100 * len(ok) / len(rows), 1)}
    out["tok_s"] = round(sum(r["out_len"] for r in done) / (horizon / 1000.0), 1)
    out["goodput"] = round(sum(r["out_len"] for r in done
                               if r["ttft"] <= a.slo_ttft_ms
                               and r["tpot"] <= a.slo_ms) / (horizon / 1000.0), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-gb", type=float, default=1.5)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--nkv", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--hbm-gb", type=float, default=61.0)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--rate", type=float, default=12.0)
    ap.add_argument("--slo-ms", type=float, default=100.0)
    ap.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    ap.add_argument("--sim-seconds", type=float, default=300.0)
    ap.add_argument("--graph-captured", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for pol in ["fcfs", "sjf", "slo_admit", "priority"]:
        r = run(args, pol, args.rate)
        rows.append(r)
        s, l = r["classes"].get("short", {}), r["classes"].get("long", {})
        print(f"[E54] {pol:10s} λ={args.rate:5.1f} short p99={s.get('ttft_p99','-'):>9} "
              f"({s.get('slo_met_pct','-')}% met) long p99={l.get('ttft_p99','-'):>9} "
              f"({l.get('slo_met_pct','-'):>5}% met) rejected={r['rejected']:3d} "
              f"goodput={r['goodput']:7.1f}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"params": vars(args), "rows": rows}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
