#!/usr/bin/env python3
"""E8f — continuous batching + paged-KV mechanics simulator (pure stdlib).

A discrete-step simulation of the vLLM V1 scheduling loop
(schedule → budget → allocate → preempt), not a kernel benchmark. Its job
is to make the *mechanisms* in docs/request-walkthrough.md §4-5 measurable:

  static      : legacy batching — N reqs, one shared prefill, decode for
                max(out_len) steps (dead slots idle = the waste CB removes)
  continuous  : per-step FCFS admission, token budget (mnb) shared between
                decode slots and chunked prefill
  + kv pool   : finite block pool, tail-eviction preemption on exhaustion
                (vLLM drops computed state — recomputes on re-admit)
  + prefix    : hash-chain prefix cache — shared template heads skip prefill

Workload: Poisson arrivals over T seconds, 4 shared system-prompt templates
(~2000 tok) + short unique tails — same shape as E3/E16 prefix workloads.

Run: python3 e8f_batching_sim.py --json-out e8f.json
"""
import argparse, json, math, random
from dataclasses import dataclass, field


@dataclass
class Req:
    rid: int
    arrival: int          # arrival step
    in_len: int
    out_len: int
    tpl: int              # template id (-1 = unique)
    computed: int = 0     # tokens whose KV exists (prompt computed so far)
    generated: int = 0
    ttft: int = -1
    preempts: int = 0


def workload(seed=7, n=128, steps=400):
    """128 reqs: 75% share one of 4 ~2000-tok templates, 25% unique 300-tok."""
    rng = random.Random(seed)
    reqs = []
    t = 0
    for i in range(n):
        t += rng.expovariate(n / steps)          # Poisson arrivals over window
        if rng.random() < 0.75:
            tpl = rng.randrange(4)
            in_len = 2000 + rng.randrange(-80, 81)
        else:
            tpl = -1
            in_len = 300 + rng.randrange(-60, 61)
        reqs.append(Req(i, int(t), in_len, 96 + rng.randrange(-32, 33), tpl))
    return reqs


def sim_static(reqs, max_seqs=64):
    """Baseline: batch reqs by arrival into groups of max_seqs; each group
    prefills everything then decodes max(out_len) steps."""
    reqs = sorted(reqs, key=lambda r: r.arrival)
    step = 0
    i = 0
    tot_out = 0
    while i < len(reqs):
        grp = reqs[i:i + max_seqs]
        step = max(step, grp[-1].arrival)
        # shared prefill: serial per req (cost ~ 1 step each, conservative)
        for r in grp:
            r.ttft = step + 1
            step += 1                            # prefill step per request
        max_out = max(r.out_len for r in grp)
        step += max_out                          # dead slots idle — the waste
        tot_out += sum(r.out_len for r in grp)
        i += max_seqs
    ttfts = [r.ttft - r.arrival for r in reqs]
    return {"makespan": step, "throughput_tok_per_step": round(tot_out / step, 2),
            "ttft_mean": round(sum(ttfts) / len(ttfts), 1),
            "ttft_p99": sorted(ttfts)[int(len(ttfts) * .99) - 1]}


def sim_continuous(reqs, mnb=4096, max_seqs=64, kv_blocks=None, block=128,
                   prefix_cache=False):
    """vLLM V1 loop, one decode-step granularity.

    Per step: budget = mnb tokens; running decoders each consume 1;
    leftover budget prefills admitted waiting reqs (chunked across steps).
    KV pool: blocks allocated lazily as computed+generated grows; full pool
    → evict the last-admitted running req (computed reset to 0 = recompute
    preemption, matching vLLM). Prefix cache: template prefix hit makes the
    first `tpl_len` tokens already-computed.
    """
    waiting = sorted(reqs, key=lambda r: r.arrival)
    running, done, cache = [], [], {}            # tpl -> prefix tokens cached
    step = tot_out = preempts = hits = 0
    blocks_used = lambda: sum((r.computed + r.generated + block - 1) // block
                             for r in running)

    while (waiting or running) and step < 200000:
        arrived = [r for r in waiting if r.arrival <= step]
        waiting = [r for r in waiting if r.arrival > step]

        # --- admission: FCFS, capped by max_seqs (scheduler L124 analog)
        for r in arrived:
            if len(running) >= max_seqs:
                waiting.append(r)
                continue
            if prefix_cache and r.tpl >= 0 and r.tpl in cache and r.computed == 0:
                r.computed = min(cache[r.tpl], r.in_len)
                if r.computed >= r.in_len and r.ttft < 0:
                    r.ttft = step               # hit = first token this step
                hits += 1
            running.append(r)

        budget = mnb
        # --- phase 1: decode slots claim 1 token each (only after prefill done)
        for r in running:
            if r.computed >= r.in_len:
                r.generated += 1
                tot_out += 1
                budget -= 1

        # --- phase 2: leftover budget -> chunked prefill, FCFS order
        for r in running:
            if budget <= 0:
                break
            if r.computed < r.in_len:
                need = r.in_len - r.computed
                take = min(need, budget)
                r.computed += take
                budget -= take
                if r.computed >= r.in_len and r.ttft < 0:
                    r.ttft = step               # first token emitted this step

        # --- finish + prefix-cache commit
        finished = [r for r in running
                    if r.computed >= r.in_len and r.generated >= r.out_len]
        for r in finished:
            running.remove(r)
            done.append(r)
            if prefix_cache and r.tpl >= 0:
                cache[r.tpl] = r.in_len         # full prompt committed

        # --- kv pool exhaustion -> tail-evict, recompute preemption
        if kv_blocks is not None:
            while blocks_used() > kv_blocks and running:
                victim = running[-1]             # tail evict (vLLM: lowest pri)
                running.remove(victim)
                victim.computed = 0              # KV freed -> must recompute
                victim.preempts += 1
                preempts += 1
                waiting.insert(0, victim)        # back to queue head

        step += 1

    ttfts = [r.ttft - r.arrival for r in done]
    return {"makespan": step, "throughput_tok_per_step": round(tot_out / step, 2),
            "ttft_mean": round(sum(ttfts) / len(ttfts), 1),
            "ttft_p99": sorted(ttfts)[max(0, int(len(ttfts) * .99) - 1)],
            "preemptions": preempts,
            "prefix_hit_reqs": hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="/root/e8f.json")
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()
    reqs = workload(n=args.n)

    def fresh():
        return [Req(r.rid, r.arrival, r.in_len, r.out_len, r.tpl) for r in reqs]

    res = {}
    res["static_baseline"] = sim_static(fresh())
    for mnb in (256, 1024, 4096):
        res[f"continuous_mnb{mnb}"] = sim_continuous(fresh(), mnb=mnb)
    res["continuous_apc"] = sim_continuous(fresh(), mnb=4096, prefix_cache=True)
    res["continuous_apc_mnb256"] = sim_continuous(fresh(), mnb=256, prefix_cache=True)
    res["continuous_kv_tight"] = sim_continuous(fresh(), mnb=4096, kv_blocks=160)
    res["workload"] = {"n": len(reqs),
                       "shared_tpl_reqs": sum(1 for r in reqs if r.tpl >= 0),
                       "in_len_mean": round(sum(r.in_len for r in reqs) / len(reqs))}
    for k, v in res.items():
        if k != "workload":
            print(f"{k:24s}", v)
    print("workload:", res["workload"])
    with open(args.json_out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.json_out)


if __name__ == "__main__":
    main()
